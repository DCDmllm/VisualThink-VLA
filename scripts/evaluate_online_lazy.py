#!/usr/bin/env python3
"""Benchmark OpenVLA / FullSoft / VisualThink-VLA.

VisualThink-VLA:
1) run a cheap-context LazyGate student on the current frame + instruction;
2) lazily load only selected evidence vectors from the per-step NPZ;
3) decode the action with a trained VisualThink-VLA adapter;
4) emit side-path evidence-trace text from the selected route.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualthink_vla.models.evidence_router import (
    FEATURE_DIMS,
    INSTANCE_AMBIGUITY_DIM,
    LearnedEvidencePolicy,
    STAGES,
    bbox_vector,
    edge_vector,
    hashed_bow,
    infer_stage,
    instruction_meta_vector,
    load_image_tensor,
    load_local_tokenizer,
    motion_vector,
    relation_vector,
    route_bank_tensor,
    sample_budget_topk_gates,
    sample_route_mixture_gates,
    stage_to_one_hot,
    tokenize_to_arrays,
)
from visualthink_vla.models.soft_evidence_adapter import (
    EvidenceTokenAdapter,
    EvidenceTokenBatch,
    make_openvla_prompt,
    predict_action_with_soft_evidence,
)


def load_rows(manifest_path: str) -> list[dict]:
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty manifest: {manifest_path}")
    return rows


def step_ratio_map(rows: list[dict]) -> dict[tuple[int, int, str], float]:
    step_max = {}
    for row in rows:
        ep = int(row["episode_idx"])
        step_max[ep] = max(step_max.get(ep, 0), int(row["step_idx"]))
    out = {}
    for row in rows:
        ep = int(row["episode_idx"])
        step = int(row["step_idx"])
        out[(ep, step, str(row["npz_path"]))] = float(step) / float(max(1, step_max[ep]))
    return out


def action_l1(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, dtype=np.float32) - np.asarray(gt, dtype=np.float32))))


def predict_action_original(vla, processor, image: Image.Image, instruction: str, model_path: str, device: str, dtype, unnorm_key: str):
    prompt = make_openvla_prompt(instruction, model_path)
    inputs = processor(prompt, image).to(device, dtype=dtype)
    start = time.time()
    with torch.inference_mode():
        action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return np.asarray(action, dtype=np.float32), time.time() - start


def load_soft_adapter(
    checkpoint_dir: str,
    channels: tuple[str, ...],
    hidden_size: int,
    device: torch.device,
) -> tuple[EvidenceTokenAdapter, np.ndarray]:
    ckpt_dir = Path(checkpoint_dir)
    resolved = json.loads((ckpt_dir / "resolved_config.json").read_text(encoding="utf-8"))
    adapter_cfg = resolved["config"]["adapter"]
    adapter = EvidenceTokenAdapter(
        channel_dims={ch: int(FEATURE_DIMS[ch]) for ch in channels},
        channels=channels,
        hidden_size=hidden_size,
        num_global_tokens=int(adapter_cfg["num_global_tokens"]),
        proj_dim=int(adapter_cfg["proj_dim"]),
        dropout=float(adapter_cfg.get("dropout", 0.1)),
    ).to(device)
    state = torch.load(ckpt_dir / "adapter.pt", map_location=device)
    missing, unexpected = adapter.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing adapter keys when loading {ckpt_dir}: {missing}", flush=True)
    if unexpected:
        print(f"[warn] unexpected adapter keys when loading {ckpt_dir}: {unexpected}", flush=True)
    adapter.eval()
    masks = np.load(ckpt_dir / "channel_masks.npy")
    return adapter, masks


def load_online_lazy_gate(checkpoint_dir: str, device: torch.device) -> tuple[LearnedEvidencePolicy, dict, object | None]:
    ckpt_dir = Path(checkpoint_dir)
    resolved = json.loads((ckpt_dir / "resolved_config.json").read_text(encoding="utf-8"))
    cfg = resolved["config"]
    model_cfg = cfg["model"]
    channels = tuple(resolved["channels"])
    budget_values = tuple(int(x) for x in resolved["budget_values"])
    text_cfg = dict(model_cfg.get("text_encoder", {}))
    text_encoder_type = str(text_cfg.get("type", "bow"))
    text_vocab_size = 0
    tokenizer = None
    if text_encoder_type == "sequence":
        tokenizer = load_local_tokenizer(str(text_cfg["tokenizer_path"]))
        text_vocab_size = int(len(tokenizer))
    gate_type = str(model_cfg.get("gate_type", "stage_conditioned"))
    if gate_type == "stage_conditioned":
        if bool(cfg.get("route_mixture", {}).get("enabled", False)):
            gate_type = "route_mixture"
        elif bool(cfg.get("latent_phase", {}).get("enabled", False)):
            gate_type = "latent_phase"
    policy = LearnedEvidencePolicy(
        bow_dim=int(model_cfg["bow_dim"]),
        query_bow_dim=int(model_cfg.get("query_bow_dim", 64)),
        ctx_dim=int(model_cfg["ctx_dim"]),
        channel_dim=int(model_cfg["channel_dim"]),
        teacher_hidden=int(model_cfg["teacher_hidden"]),
        student_hidden=int(model_cfg["student_hidden"]),
        channels=channels,
        budget_values=budget_values,
        text_encoder_type=text_encoder_type,
        text_vocab_size=text_vocab_size,
        instruction_max_len=int(text_cfg.get("instruction_max_len", 24)),
        query_max_len=int(text_cfg.get("query_max_len", 12)),
        text_embed_dim=int(text_cfg.get("embed_dim", 96)),
        text_hidden_dim=int(text_cfg.get("hidden_dim", 128)),
        gate_type=gate_type,
        latent_phase_slots=int(model_cfg.get("latent_phase_slots", 8)),
        route_bank=tuple(tuple(r) for r in cfg.get("route_mixture", {}).get("route_bank", [])),
    ).to(device)
    policy.context.load_state_dict(torch.load(ckpt_dir / "context.pt", map_location=device))
    if getattr(policy, "instruction_text_encoder", None) is not None and (ckpt_dir / "instruction_text_encoder.pt").exists():
        policy.instruction_text_encoder.load_state_dict(torch.load(ckpt_dir / "instruction_text_encoder.pt", map_location=device))
    if getattr(policy, "query_text_encoder", None) is not None and (ckpt_dir / "query_text_encoder.pt").exists():
        policy.query_text_encoder.load_state_dict(torch.load(ckpt_dir / "query_text_encoder.pt", map_location=device))
    policy.gate.load_state_dict(torch.load(ckpt_dir / "gate.pt", map_location=device))
    policy.eval()
    return policy, resolved, tokenizer


def make_sequence_ids(
    tokenizer,
    text: str,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids, attention_mask = tokenize_to_arrays(tokenizer, text, max_len)
    return (
        torch.tensor(input_ids.tolist(), dtype=torch.long),
        torch.tensor(attention_mask.tolist(), dtype=torch.float32),
    )


def build_online_batch(
    row: dict,
    stage: str,
    ratio: float,
    policy: LearnedEvidencePolicy,
    resolved: dict,
    lazy_tokenizer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model_cfg = resolved["config"]["model"]
    instruction = row.get("instruction", "")
    query_text = " ".join(row.get("query_words") or [])
    image = load_image_tensor(str(row["image_path"]), int(model_cfg["image_size"]))
    batch = {
        "image": image.unsqueeze(0).to(device=device, dtype=torch.float32),
        "bow": torch.tensor(hashed_bow(instruction, int(model_cfg["bow_dim"])).tolist(), dtype=torch.float32, device=device).unsqueeze(0),
        "query_bow": torch.tensor(
            hashed_bow(query_text, int(model_cfg.get("query_bow_dim", 64))).tolist(),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0),
        "instruction_meta": torch.tensor(
            instruction_meta_vector(instruction, row.get("query_words")).tolist(),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0),
        "ambiguity_vec": torch.zeros((1, INSTANCE_AMBIGUITY_DIM), dtype=torch.float32, device=device),
        "stage_one_hot": torch.tensor(stage_to_one_hot(stage).tolist(), dtype=torch.float32, device=device).unsqueeze(0),
        "step_ratio": torch.tensor([[float(ratio)]], dtype=torch.float32, device=device),
    }
    if policy.use_text_sequence:
        text_cfg = dict(model_cfg.get("text_encoder", {}))
        instruction_ids, instruction_mask = make_sequence_ids(
            lazy_tokenizer,
            instruction,
            int(text_cfg.get("instruction_max_len", 24)),
        )
        query_ids, query_mask = make_sequence_ids(
            lazy_tokenizer,
            query_text,
            int(text_cfg.get("query_max_len", 12)),
        )
        batch["instruction_ids"] = instruction_ids.unsqueeze(0).to(device)
        batch["instruction_mask"] = instruction_mask.unsqueeze(0).to(device)
        batch["query_ids"] = query_ids.unsqueeze(0).to(device)
        batch["query_mask"] = query_mask.unsqueeze(0).to(device)
    else:
        batch["instruction_ids"] = torch.zeros((1, 1), dtype=torch.long, device=device)
        batch["instruction_mask"] = torch.zeros((1, 1), dtype=torch.float32, device=device)
        batch["query_ids"] = torch.zeros((1, 1), dtype=torch.long, device=device)
        batch["query_mask"] = torch.zeros((1, 1), dtype=torch.float32, device=device)
    return batch


def infer_online_mask(
    policy: LearnedEvidencePolicy,
    resolved: dict,
    batch: dict[str, torch.Tensor],
    soft_mask_blend: float,
) -> tuple[np.ndarray, dict[str, bool]]:
    cfg = resolved["config"]
    channels = tuple(resolved["channels"])
    budget_values = tuple(int(x) for x in resolved["budget_values"])
    gate_cfg = cfg["gating"]
    stage_conditioning_enabled = bool(cfg.get("stage_conditioning", {}).get("enabled", True))
    route_mixture_cfg = cfg.get("route_mixture", {})
    route_mixture_enabled = bool(route_mixture_cfg.get("enabled", False))
    route_bank_masks = None
    if route_mixture_enabled:
        route_bank = tuple(tuple(route) for route in route_mixture_cfg.get("route_bank", []))
        route_bank_masks = route_bank_tensor(route_bank, channels, batch["image"].device)

    with torch.inference_mode():
        ctx = policy.encode_context(batch)
        stage_one_hot = batch["stage_one_hot"] if stage_conditioning_enabled else torch.zeros_like(batch["stage_one_hot"])
        gate_out = policy.forward_gate(
            ctx,
            batch,
            stage_one_hot,
            phase_temperature=float(cfg.get("latent_phase", {}).get("temperature_end", gate_cfg["temperature_end"])),
            hard_phase=bool(cfg.get("latent_phase", {}).get("hard_assignment", False)),
            training=False,
        )
        if route_bank_masks is not None:
            hard_gates, soft_gates, _, _, _, _ = sample_route_mixture_gates(
                gate_out["route_logits"],
                route_bank_masks,
                budget_values=budget_values,
                temperature=float(gate_cfg["temperature_end"]),
                training=False,
            )
        else:
            hard_gates, channel_probs, _, budget_probs = sample_budget_topk_gates(
                gate_out["channel_logits"],
                gate_out["budget_logits"],
                budget_values=budget_values,
                temperature=float(gate_cfg["temperature_end"]),
                training=False,
            )
            budget_tensor = torch.tensor(budget_values, dtype=budget_probs.dtype, device=budget_probs.device).unsqueeze(0)
            soft_budget = torch.sum(budget_probs * budget_tensor, dim=1)
            denom = torch.clamp(channel_probs.sum(dim=1, keepdim=True), min=1e-6)
            soft_gates = torch.clamp(channel_probs * (soft_budget.unsqueeze(1) / denom), 0.0, 1.0)
        blend = float(np.clip(soft_mask_blend, 0.0, 1.0))
        effective = torch.clamp((1.0 - blend) * hard_gates + blend * soft_gates, 0.0, 1.0)[0].cpu().numpy()
        hard = hard_gates[0].cpu().numpy() > 0.5
    return effective.astype(np.float32), {ch: bool(hard[i]) for i, ch in enumerate(channels)}


def expand_dependencies(gates: dict[str, bool]) -> dict[str, bool]:
    plan = dict(gates)
    if plan.get("relation", False):
        plan["bbox"] = True
    return plan


def load_lazy_features(row: dict, channels: tuple[str, ...], gates: dict[str, bool]) -> tuple[dict[str, np.ndarray], float]:
    plan = expand_dependencies(gates)
    start = time.time()
    out = {ch: np.zeros((FEATURE_DIMS[ch],), dtype=np.float32) for ch in channels}
    with np.load(row["npz_path"], allow_pickle=True) as npz:
        if plan.get("bbox", False) and "bbox" in channels:
            out["bbox"] = bbox_vector(npz)
        if plan.get("edge", False) and "edge" in channels:
            out["edge"] = edge_vector(npz)
        if plan.get("motion", False) and "motion" in channels:
            out["motion"] = motion_vector(npz)
        if plan.get("relation", False) and "relation" in channels:
            out["relation"] = relation_vector(npz, row=row)
    return out, time.time() - start


def build_evidence_batch(
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    channels: tuple[str, ...],
    stage: str,
    ratio: float,
    device: torch.device,
) -> EvidenceTokenBatch:
    return EvidenceTokenBatch(
        channel_features={
            ch: torch.tensor(features[ch].tolist(), dtype=torch.float32, device=device).unsqueeze(0)
            for ch in channels
        },
        channel_mask=torch.tensor(mask.tolist(), dtype=torch.float32, device=device).unsqueeze(0),
        stage_one_hot=torch.tensor(stage_to_one_hot(stage).tolist(), dtype=torch.float32, device=device).unsqueeze(0),
        step_ratio=torch.tensor([[float(ratio)]], dtype=torch.float32, device=device),
    )


def trace_text(gates: dict[str, bool], features: dict[str, np.ndarray], stage: str) -> tuple[list[str], str]:
    selected = [ch for ch, on in gates.items() if on]
    if not selected:
        selected = ["edge"]
    details = []
    for ch in selected:
        vec = features.get(ch)
        if vec is None:
            continue
        if ch == "bbox":
            details.append(f"bbox count={vec[0]:.1f}, top_center=({vec[8]:.2f},{vec[9]:.2f})")
        elif ch == "edge":
            details.append(f"edge density={vec[0]:.3f}, left/right=({vec[3]:.3f},{vec[4]:.3f})")
        elif ch == "motion":
            details.append(f"motion density={vec[2]:.3f}, center={vec[3]:.3f}")
        elif ch == "relation":
            details.append(f"relation goal_dist={vec[17]:.3f}, nearest_other={vec[8]:.3f}")
    return selected, f"At the {stage} stage, selected evidence is {', '.join(selected)}; " + "; ".join(details)


def summarize(rows: list[dict], channels: tuple[str, ...]) -> dict:
    out = {
        "n": int(len(rows)),
        "success_rate": float(np.mean([1.0 if row.get("success") else 0.0 for row in rows])) if rows else 0.0,
        "action_fps": float(np.mean([float(row.get("action_fps", 0.0)) for row in rows])) if rows else 0.0,
        "end2end_fps": float(np.mean([float(row.get("end2end_fps", 0.0)) for row in rows])) if rows else 0.0,
        "avg_extract_time_ms": 1000.0 * float(np.mean([float(row.get("extract_time", 0.0)) for row in rows])) if rows else 0.0,
        "avg_selected_channels": float(np.mean([float(row.get("selected_channels", 0.0)) for row in rows])) if rows else 0.0,
    }
    for ch in channels:
        out[f"{ch}_keep_rate"] = float(np.mean([1.0 if row.get("gates", {}).get(ch, False) else 0.0 for row in rows])) if rows else 0.0
    return out


def write_stage_summary(orig_rows: list[dict], full_rows: list[dict], lazy_rows: list[dict], out_dir: Path, channels: tuple[str, ...]) -> None:
    lines = [
        "| Stage | Model | N | Success | ActionFPS | End2EndFPS | ExtractMS | AvgSelected | "
        + " | ".join(ch.title() for ch in channels)
        + " |",
        "|---|---|---:|---:|---:|---:|---:|---:|" + "---:|" * len(channels),
    ]
    for stage in STAGES:
        for name, rows in [("OpenVLA", orig_rows), ("FullSoft", full_rows), ("VisualThink-VLA", lazy_rows)]:
            sub = [row for row in rows if row["stage"] == stage]
            if not sub:
                continue
            stats = summarize(sub, channels)
            lines.append(
                f"| {stage} | {name} | {stats['n']} | {stats['success_rate']:.4f} | "
                f"{stats['action_fps']:.4f} | {stats['end2end_fps']:.4f} | "
                f"{stats['avg_extract_time_ms']:.4f} | {stats['avg_selected_channels']:.4f} | "
                + " | ".join(f"{stats[f'{ch}_keep_rate']:.4f}" for ch in channels)
                + " |"
            )
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "stage_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(orig_rows: list[dict], full_rows: list[dict], lazy_rows: list[dict], out_dir: Path, channels: tuple[str, ...]) -> None:
    orig = summarize(orig_rows, channels)
    full = summarize(full_rows, channels)
    lazy = summarize(lazy_rows, channels)
    lines = [
        "| Metric | OpenVLA | FullSoft | VisualThink-VLA | Full-Orig | Lazy-Orig |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    metrics = ["success_rate", "action_fps", "end2end_fps", "avg_extract_time_ms", "avg_selected_channels"] + [
        f"{ch}_keep_rate" for ch in channels
    ]
    for key in metrics:
        ov = float(orig.get(key, 0.0))
        fv = float(full.get(key, 0.0))
        lv = float(lazy.get(key, 0.0))
        lines.append(f"| {key} | {ov:.4f} | {fv:.4f} | {lv:.4f} | {fv - ov:+.4f} | {lv - ov:+.4f} |")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_stage_summary(orig_rows, full_rows, lazy_rows, out_dir, channels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--full_checkpoint_dir", required=True)
    parser.add_argument("--visualthink_checkpoint_dir", required=True)
    parser.add_argument("--lazy_gate_checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--success_l1_thresh", type=float, default=0.08)
    parser.add_argument("--unnorm_key", default="bridge_orig")
    parser.add_argument("--soft_mask_blend", type=float, default=0.35)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_rows(args.feature_manifest)
    ratios = step_ratio_map(rows)
    full_resolved = json.loads((Path(args.full_checkpoint_dir) / "resolved_config.json").read_text(encoding="utf-8"))
    channels = tuple(full_resolved["channels"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    vla = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        attn_implementation="sdpa",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.eval()
    hidden_size = int(vla.get_input_embeddings().weight.shape[1])

    full_adapter, full_masks = load_soft_adapter(args.full_checkpoint_dir, channels, hidden_size, device)
    lazy_adapter, _ = load_soft_adapter(args.visualthink_checkpoint_dir, channels, hidden_size, device)
    lazy_gate, lazy_resolved, lazy_tokenizer = load_online_lazy_gate(args.lazy_gate_checkpoint_dir, device)

    indices = list(range(len(rows)))
    if args.limit > 0:
        indices = indices[: args.limit]

    orig_rows = []
    full_rows = []
    lazy_rows = []

    for out_idx, idx in enumerate(indices):
        row = rows[idx]
        key = (int(row["episode_idx"]), int(row["step_idx"]), str(row["npz_path"]))
        ratio = float(ratios[key])
        stage = infer_stage(ratio)
        image = Image.open(row["image_path"]).convert("RGB")
        gt = np.asarray(row["action"], dtype=np.float32)

        ov_action, ov_t = predict_action_original(
            vla=vla,
            processor=processor,
            image=image,
            instruction=row.get("instruction", ""),
            model_path=args.model_path,
            device=str(device),
            dtype=dtype,
            unnorm_key=args.unnorm_key,
        )
        ov_l1 = action_l1(ov_action, gt)
        orig_rows.append(
            {
                "idx": out_idx,
                "stage": stage,
                "success": ov_l1 <= float(args.success_l1_thresh),
                "l1": ov_l1,
                "action_fps": 1.0 / max(1e-6, ov_t),
                "end2end_fps": 1.0 / max(1e-6, ov_t),
                "extract_time": 0.0,
                "selected_channels": 0.0,
                "gates": {ch: False for ch in channels},
            }
        )

        full_features, full_extract_t = load_lazy_features(row, channels, gates={ch: True for ch in channels})
        full_batch = build_evidence_batch(
            full_features,
            np.asarray(full_masks[idx], dtype=np.float32),
            channels,
            stage,
            ratio,
            device,
        )
        t0 = time.time()
        full_action = predict_action_with_soft_evidence(
            vla=vla,
            processor=processor,
            tokenizer=tokenizer,
            adapter=full_adapter,
            image=image,
            instruction=row.get("instruction", ""),
            evidence_batch=full_batch,
            model_path=args.model_path,
            device=device,
            dtype=dtype,
            unnorm_key=args.unnorm_key,
        )
        full_action_t = time.time() - t0
        full_l1 = action_l1(full_action, gt)
        full_rows.append(
            {
                "idx": out_idx,
                "stage": stage,
                "success": full_l1 <= float(args.success_l1_thresh),
                "l1": full_l1,
                "action_fps": 1.0 / max(1e-6, full_action_t),
                "end2end_fps": 1.0 / max(1e-6, full_action_t + full_extract_t),
                "extract_time": full_extract_t,
                "selected_channels": float(np.sum(full_masks[idx] > 0.5)),
                "gates": {ch: bool(full_masks[idx][i] > 0.5) for i, ch in enumerate(channels)},
            }
        )

        gate_batch = build_online_batch(row, stage, ratio, lazy_gate, lazy_resolved, lazy_tokenizer, device)
        lazy_mask, lazy_binary = infer_online_mask(
            policy=lazy_gate,
            resolved=lazy_resolved,
            batch=gate_batch,
            soft_mask_blend=float(args.soft_mask_blend),
        )
        lazy_features, lazy_extract_t = load_lazy_features(row, channels, gates=lazy_binary)
        lazy_batch = build_evidence_batch(
            lazy_features,
            lazy_mask,
            channels,
            stage,
            ratio,
            device,
        )
        t0 = time.time()
        lazy_action = predict_action_with_soft_evidence(
            vla=vla,
            processor=processor,
            tokenizer=tokenizer,
            adapter=lazy_adapter,
            image=image,
            instruction=row.get("instruction", ""),
            evidence_batch=lazy_batch,
            model_path=args.model_path,
            device=device,
            dtype=dtype,
            unnorm_key=args.unnorm_key,
        )
        lazy_action_t = time.time() - t0
        lazy_l1 = action_l1(lazy_action, gt)
        selected, rationale = trace_text(lazy_binary, lazy_features, stage)
        lazy_rows.append(
            {
                "idx": out_idx,
                "stage": stage,
                "success": lazy_l1 <= float(args.success_l1_thresh),
                "l1": lazy_l1,
                "action_fps": 1.0 / max(1e-6, lazy_action_t),
                "end2end_fps": 1.0 / max(1e-6, lazy_action_t + lazy_extract_t),
                "extract_time": lazy_extract_t,
                "selected_channels": float(sum(1 for v in lazy_binary.values() if v)),
                "gates": lazy_binary,
                "selected_evidence": selected,
                "visual_rationale": rationale,
            }
        )

        if (out_idx + 1) % 10 == 0 or (out_idx + 1) == len(indices):
            print(f"[progress] {out_idx + 1}/{len(indices)}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "openvla_eval.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in orig_rows),
        encoding="utf-8",
    )
    (out_dir / "fullsoft_eval.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in full_rows),
        encoding="utf-8",
    )
    (out_dir / "online_lazy_visualthink_vla_eval.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in lazy_rows),
        encoding="utf-8",
    )
    write_summary(orig_rows, full_rows, lazy_rows, out_dir, channels)
    print("[ok] openvla:", summarize(orig_rows, channels))
    print("[ok] full_soft:", summarize(full_rows, channels))
    print("[ok] online_lazy_visualthink_vla:", summarize(lazy_rows, channels))
    print(f"[ok] report dir: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
