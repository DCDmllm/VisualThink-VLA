#!/usr/bin/env python3
"""Train a evidence-token adapter for real OpenVLA evidence injection.

This script freezes OpenVLA and only trains a lightweight adapter that maps
structured evidence channels to learned soft tokens. The soft tokens are
inserted between image patch embeddings and the task prompt tokens.
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
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from visualthink_vla.models.evidence_router import (
    FEATURE_DIMS,
    LearnedEvidencePolicy,
    LearnedGatingDataset,
    load_yaml,
    move_batch_to_device,
    resolve_budget_values,
    resolve_channels,
    route_bank_tensor,
    sample_route_mixture_gates,
    sample_budget_topk_gates,
)
from visualthink_vla.models.soft_evidence_adapter import (
    EvidenceTokenAdapter,
    actions_to_token_ids,
    build_multimodal_inputs,
    build_training_labels,
    ensure_action_start_token,
    make_openvla_prompt,
    preprocess_image,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    val_n = max(1, int(round(n * val_ratio))) if n > 1 else 0
    val_idx = idx[:val_n]
    train_idx = idx[val_n:] if val_n < n else idx[:]
    if not train_idx:
        train_idx = idx[:]
        val_idx = idx[: min(1, len(idx))]
    return train_idx, val_idx


def load_rows(manifest_path: str, limit: int = 0) -> list[dict]:
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError(f"empty manifest: {manifest_path}")
    return rows


def load_trace_rows(trace_path: str, limit: int = 0) -> list[dict]:
    if not trace_path:
        return []
    rows = []
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError(f"empty VisualEvidence-Set manifest: {trace_path}")
    return rows


def trace_selected_channels(trace_row: dict) -> list[str]:
    selected = trace_row.get("selected_evidence")
    if isinstance(selected, list):
        return [str(ch) for ch in selected]
    selected = trace_row.get("selected_channels")
    if isinstance(selected, list):
        return [str(ch) for ch in selected]
    return []


def trace_top_utility_channel(trace_row: dict) -> str:
    utility_rank = trace_row.get("utility_rank_records")
    if not isinstance(utility_rank, list) or not utility_rank:
        utility_rank = trace_row.get("utility_rank")
    if not isinstance(utility_rank, list) or not utility_rank:
        return ""
    top_item = utility_rank[0]
    if isinstance(top_item, dict):
        return str(top_item.get("channel", ""))
    return str(top_item)


def trace_alignment_key(trace_row: dict) -> tuple[object, object, str]:
    episode_idx = trace_row.get("episode_idx", trace_row.get("episode_id"))
    step_idx = trace_row.get("step_idx", trace_row.get("step_id"))
    npz_path = trace_row.get("npz_path") or trace_row.get("feature_ref") or ""
    return episode_idx, step_idx, str(npz_path)


def rows_are_aligned(row: dict, trace_row: dict) -> bool:
    row_key = (row.get("episode_idx"), row.get("step_idx"), str(row.get("npz_path", "")))
    trace_key = trace_alignment_key(trace_row)
    if row_key == trace_key:
        return True
    return row_key[:2] == trace_key[:2] and Path(row_key[2]).name == Path(trace_key[2]).name


def trace_target_from_row(trace_row: dict, channels: tuple[str, ...], device: torch.device) -> torch.Tensor:
    if "route_mask" in trace_row and isinstance(trace_row["route_mask"], dict):
        values = [1.0 if trace_row["route_mask"].get(ch, False) else 0.0 for ch in channels]
    else:
        selected = set(trace_selected_channels(trace_row))
        values = [1.0 if ch in selected else 0.0 for ch in channels]
    if not any(values):
        top_ch = trace_top_utility_channel(trace_row)
        values = [1.0 if ch == top_ch else 0.0 for ch in channels]
    return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)


def validate_trace_alignment(rows: list[dict], trace_rows: list[dict]) -> None:
    if not trace_rows:
        return
    if len(rows) != len(trace_rows):
        raise RuntimeError(f"row/trace mismatch: {len(rows)} vs {len(trace_rows)}")
    for idx, (row, trace_row) in enumerate(zip(rows, trace_rows)):
        if not rows_are_aligned(row, trace_row):
            row_key = (row.get("episode_idx"), row.get("step_idx"), row.get("npz_path"))
            trace_key = trace_alignment_key(trace_row)
            raise RuntimeError(f"trace alignment mismatch at idx={idx}: row={row_key} trace={trace_key}")


def load_gate_policy(
    checkpoint_dir: str,
    device: torch.device,
) -> LearnedEvidencePolicy:
    ckpt_dir = Path(checkpoint_dir)
    resolved = json.loads((ckpt_dir / "resolved_config.json").read_text(encoding="utf-8"))
    cfg = resolved["config"]
    model_cfg = resolved["config"]["model"]
    channels = tuple(resolved["channels"])
    budget_values = tuple(int(x) for x in resolved["budget_values"])
    text_cfg = dict(model_cfg.get("text_encoder", {}))
    text_encoder_type = str(text_cfg.get("type", "bow"))
    gate_type = str(model_cfg.get("gate_type", "stage_conditioned"))
    if gate_type == "stage_conditioned":
        if bool(cfg.get("route_mixture", {}).get("enabled", False)):
            gate_type = "route_mixture"
        elif bool(cfg.get("latent_phase", {}).get("enabled", False)):
            gate_type = "latent_phase"
    route_bank = ()
    if gate_type == "route_mixture":
        route_bank = tuple(tuple(route) for route in cfg.get("route_mixture", {}).get("route_bank", []))
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
        text_vocab_size=int(text_cfg.get("vocab_size", 32000)),
        instruction_max_len=int(text_cfg.get("instruction_max_len", 24)),
        query_max_len=int(text_cfg.get("query_max_len", 12)),
        text_embed_dim=int(text_cfg.get("embed_dim", 96)),
        text_hidden_dim=int(text_cfg.get("hidden_dim", 128)),
        gate_type=gate_type,
        latent_phase_slots=int(model_cfg.get("latent_phase_slots", 8)),
        route_bank=route_bank,
    ).to(device)
    policy.context.load_state_dict(torch.load(ckpt_dir / "context.pt", map_location=device))
    if getattr(policy, "instruction_text_encoder", None) is not None and (ckpt_dir / "instruction_text_encoder.pt").exists():
        policy.instruction_text_encoder.load_state_dict(
            torch.load(ckpt_dir / "instruction_text_encoder.pt", map_location=device)
        )
    if getattr(policy, "query_text_encoder", None) is not None and (ckpt_dir / "query_text_encoder.pt").exists():
        policy.query_text_encoder.load_state_dict(
            torch.load(ckpt_dir / "query_text_encoder.pt", map_location=device)
        )
    for ch in channels:
        getattr(policy, f"{ch}_encoder").load_state_dict(torch.load(ckpt_dir / f"{ch}_encoder.pt", map_location=device))
    policy.gate.load_state_dict(torch.load(ckpt_dir / "gate.pt", map_location=device))
    policy.eval()
    return policy


def compute_routing_masks(
    manifest_path: str,
    channels: tuple[str, ...],
    gate_checkpoint_dir: str,
    gate_config: str,
    device: torch.device,
    limit: int = 0,
    batch_size: int = 64,
    mask_mode: str = "hard",
    soft_mask_blend: float = 0.0,
    min_mask_floor: float = 0.0,
) -> np.ndarray:
    resolved = json.loads((Path(gate_checkpoint_dir) / "resolved_config.json").read_text(encoding="utf-8"))
    cfg = resolved["config"]
    model_cfg = cfg["model"]
    gate_cfg = cfg["gating"]
    stage_conditioning_cfg = cfg.get("stage_conditioning", {})
    stage_conditioning_enabled = bool(stage_conditioning_cfg.get("enabled", True))
    latent_phase_cfg = cfg.get("latent_phase", {})
    route_mixture_cfg = cfg.get("route_mixture", {})
    route_mixture_enabled = bool(route_mixture_cfg.get("enabled", False))
    resolved_channels = tuple(resolved["channels"])
    budget_values = tuple(int(x) for x in resolved["budget_values"])
    if tuple(channels) != resolved_channels:
        raise RuntimeError(
            f"Channel mismatch between soft-interface request {channels} and gate checkpoint {resolved_channels}"
        )
    dataset = LearnedGatingDataset(
        manifest_path,
        image_size=int(model_cfg["image_size"]),
        bow_dim=int(model_cfg["bow_dim"]),
        query_bow_dim=int(model_cfg.get("query_bow_dim", 64)),
        limit=limit,
        channels=channels,
        text_encoder_type=str(model_cfg.get("text_encoder", {}).get("type", "bow")),
        tokenizer_path=model_cfg.get("text_encoder", {}).get("tokenizer_path"),
        instruction_max_len=int(model_cfg.get("text_encoder", {}).get("instruction_max_len", 24)),
        query_max_len=int(model_cfg.get("text_encoder", {}).get("query_max_len", 12)),
        phase_proxy_cfg=cfg.get("phase_proxy", {}),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    policy = load_gate_policy(
        checkpoint_dir=gate_checkpoint_dir,
        device=device,
    )
    route_bank = tuple(tuple(route) for route in route_mixture_cfg.get("route_bank", []))
    route_bank_masks = route_bank_tensor(route_bank, channels, device) if route_mixture_enabled else None
    masks = []
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            ctx = policy.encode_context(batch)
            gate_stage = batch["stage_one_hot"] if stage_conditioning_enabled else torch.zeros_like(batch["stage_one_hot"])
            gate_out = policy.forward_gate(
                ctx,
                batch,
                gate_stage,
                phase_temperature=float(latent_phase_cfg.get("temperature_end", gate_cfg["temperature_end"])),
                hard_phase=bool(latent_phase_cfg.get("hard_assignment", False)),
                training=False,
            )
            if route_bank_masks is not None:
                hard_gates, gate_probs, _, _, _, _ = sample_route_mixture_gates(
                    gate_out["route_logits"],
                    route_bank_masks,
                    budget_values=budget_values,
                    temperature=float(gate_cfg["temperature_end"]),
                    training=False,
                )
                soft_mask = gate_probs
            else:
                hard_gates, channel_probs, _, budget_probs = sample_budget_topk_gates(
                    gate_out["channel_logits"],
                    gate_out["budget_logits"],
                    budget_values=budget_values,
                    temperature=float(gate_cfg["temperature_end"]),
                    training=False,
                )
                budget_tensor = torch.tensor(budget_values, dtype=budget_probs.dtype, device=device).unsqueeze(0)
                soft_budget_values = torch.sum(budget_probs * budget_tensor, dim=1)
                denom = torch.clamp(channel_probs.sum(dim=1, keepdim=True), min=1e-6)
                soft_mask = torch.clamp(channel_probs * (soft_budget_values.unsqueeze(1) / denom), 0.0, 1.0)
            hard_mask = hard_gates.detach()
            if mask_mode == "soft":
                effective_mask = soft_mask
            elif mask_mode == "blend":
                blend = float(np.clip(soft_mask_blend, 0.0, 1.0))
                effective_mask = (1.0 - blend) * hard_mask + blend * soft_mask
            else:
                effective_mask = hard_mask
            if min_mask_floor > 0.0:
                floor = float(np.clip(min_mask_floor, 0.0, 1.0))
                effective_mask = floor + (1.0 - floor) * effective_mask
            effective_mask = torch.clamp(effective_mask, 0.0, 1.0)
            masks.append(effective_mask.cpu().numpy().astype(np.float32))
    return np.concatenate(masks, axis=0)


def sample_to_adapter_inputs(sample: dict, channels: tuple[str, ...], device: torch.device, mask: np.ndarray) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    features = {ch: sample[ch].unsqueeze(0).to(device=device, dtype=torch.float32) for ch in channels}
    channel_mask = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
    stage_one_hot = sample["stage_one_hot"].unsqueeze(0).to(device=device, dtype=torch.float32)
    step_ratio = sample["step_ratio"].unsqueeze(0).to(device=device, dtype=torch.float32)
    return features, channel_mask, stage_one_hot, step_ratio


def load_soft_adapter(
    checkpoint_dir: str,
    channels: tuple[str, ...],
    hidden_size: int,
    device: torch.device,
) -> tuple[EvidenceTokenAdapter, dict]:
    ckpt_dir = Path(checkpoint_dir)
    resolved = json.loads((ckpt_dir / "resolved_config.json").read_text(encoding="utf-8"))
    cfg = resolved["config"]
    adapter_cfg = cfg["adapter"]
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
    return adapter, resolved


def load_adapter_init_weights(
    adapter: EvidenceTokenAdapter,
    checkpoint_dir: str,
    channels: tuple[str, ...],
    device: torch.device,
) -> dict:
    ckpt_dir = Path(checkpoint_dir)
    resolved = json.loads((ckpt_dir / "resolved_config.json").read_text(encoding="utf-8"))
    init_channels = tuple(resolved.get("channels", ()))
    if init_channels != channels:
        raise RuntimeError(
            f"Init adapter channel mismatch: expected {channels}, got {init_channels}"
        )
    state = torch.load(ckpt_dir / "adapter.pt", map_location=device)
    missing, unexpected = adapter.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing init adapter keys when loading {ckpt_dir}: {missing}", flush=True)
    if unexpected:
        print(f"[warn] unexpected init adapter keys when loading {ckpt_dir}: {unexpected}", flush=True)
    return resolved


def forward_outputs(
    vla,
    processor,
    tokenizer,
    adapter: EvidenceTokenAdapter,
    row: dict,
    sample: dict,
    channels: tuple[str, ...],
    mask: np.ndarray,
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    unnorm_key: str,
    fusion_mode: str,
    evidence_scale: float,
) -> dict[str, torch.Tensor]:
    image = Image.open(row["image_path"]).convert("RGB")
    prompt = make_openvla_prompt(row["instruction"], model_path)
    tok = tokenizer(prompt, truncation=True, return_tensors="pt")
    prompt_ids = tok.input_ids.to(device)
    prompt_attn = tok.attention_mask.to(device)
    prompt_ids, prompt_attn = ensure_action_start_token(tokenizer, prompt_ids, prompt_attn)
    action_token_ids = actions_to_token_ids(vla, row["action"], unnorm_key=unnorm_key, device=device)
    full_text_ids = torch.cat([prompt_ids, action_token_ids], dim=1)
    full_text_attn = torch.cat(
        [
            prompt_attn,
            torch.ones((1, action_token_ids.shape[1]), dtype=prompt_attn.dtype, device=device),
        ],
        dim=1,
    )
    pixel_values = preprocess_image(processor, image, device=device, dtype=dtype)
    features, channel_mask, stage_one_hot, step_ratio = sample_to_adapter_inputs(sample, channels, device, mask)
    evidence_tokens = adapter(
        channel_features=features,
        channel_mask=channel_mask,
        stage_one_hot=stage_one_hot,
        step_ratio=step_ratio,
    ).to(dtype=dtype)
    evidence_tokens = evidence_tokens * float(evidence_scale)
    mm_embeds, mm_attn = build_multimodal_inputs(
        vla,
        full_text_ids,
        full_text_attn,
        pixel_values,
        evidence_tokens,
        fusion_mode=fusion_mode,
    )
    evidence_len = 0 if str(fusion_mode).lower() == "visual_residual" else evidence_tokens.shape[1]
    patch_len = mm_embeds.shape[1] - full_text_ids.shape[1] - evidence_len
    labels = build_training_labels(prompt_ids, action_token_ids, patch_len=patch_len, evidence_len=evidence_len, device=device)
    outputs = vla.language_model(
        inputs_embeds=mm_embeds,
        attention_mask=mm_attn,
        labels=labels,
        use_cache=False,
    )
    action_mask = labels.ne(-100)
    return {
        "loss": outputs.loss,
        "logits": outputs.logits,
        "labels": labels,
        "action_mask": action_mask,
    }


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    action_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    mask = action_mask.bool()
    if mask.ndim != 2:
        raise ValueError(f"Expected action_mask to have shape [batch, seq], got {tuple(mask.shape)}")
    student_sel = student_logits[mask]
    teacher_sel = teacher_logits[mask]
    if student_sel.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=student_logits.device)
    temp = max(1e-4, float(temperature))
    student_log_probs = F.log_softmax(student_sel.float() / temp, dim=-1)
    teacher_probs = F.softmax(teacher_sel.float() / temp, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temp * temp)


def trace_aux_loss(
    adapter: EvidenceTokenAdapter,
    sample: dict,
    channels: tuple[str, ...],
    mask: np.ndarray,
    trace_row: dict,
    device: torch.device,
    positive_weight: float,
) -> torch.Tensor:
    features, channel_mask, stage_one_hot, step_ratio = sample_to_adapter_inputs(sample, channels, device, mask)
    trace_logits = adapter.predict_trace_logits(
        channel_features=features,
        channel_mask=channel_mask,
        stage_one_hot=stage_one_hot,
        step_ratio=step_ratio,
    )
    target = trace_target_from_row(trace_row, channels, device=device)
    pos_weight = torch.full((len(channels),), max(1e-6, float(positive_weight)), dtype=torch.float32, device=device)
    return F.binary_cross_entropy_with_logits(trace_logits.float(), target, pos_weight=pos_weight)


def evaluate_loss(
    vla,
    processor,
    tokenizer,
    adapter: EvidenceTokenAdapter,
    teacher_adapter: EvidenceTokenAdapter | None,
    rows: list[dict],
    dataset: LearnedGatingDataset,
    indices: list[int],
    masks: np.ndarray,
    channels: tuple[str, ...],
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    unnorm_key: str,
    teacher_masks: np.ndarray | None = None,
    distill_weight: float = 0.0,
    distill_temperature: float = 1.0,
    trace_rows: list[dict] | None = None,
    trace_aux_weight: float = 0.0,
    trace_positive_weight: float = 1.0,
    fusion_mode: str = "insert",
    evidence_scale: float = 1.0,
) -> float:
    adapter.eval()
    if teacher_adapter is not None:
        teacher_adapter.eval()
    total = 0.0
    with torch.inference_mode():
        for idx in indices:
            student_out = forward_outputs(
                vla=vla,
                processor=processor,
                tokenizer=tokenizer,
                adapter=adapter,
                row=rows[idx],
                sample=dataset[idx],
                channels=channels,
                mask=masks[idx],
                model_path=model_path,
                device=device,
                dtype=dtype,
                unnorm_key=unnorm_key,
                fusion_mode=fusion_mode,
                evidence_scale=evidence_scale,
            )
            total_loss = student_out["loss"]
            if teacher_adapter is not None and teacher_masks is not None and distill_weight > 0.0:
                teacher_out = forward_outputs(
                    vla=vla,
                    processor=processor,
                    tokenizer=tokenizer,
                    adapter=teacher_adapter,
                    row=rows[idx],
                    sample=dataset[idx],
                    channels=channels,
                    mask=teacher_masks[idx],
                    model_path=model_path,
                    device=device,
                    dtype=dtype,
                    unnorm_key=unnorm_key,
                    fusion_mode=fusion_mode,
                    evidence_scale=evidence_scale,
                )
                kd = distillation_loss(
                    student_logits=student_out["logits"],
                    teacher_logits=teacher_out["logits"],
                    action_mask=student_out["action_mask"],
                    temperature=distill_temperature,
                )
                total_loss = total_loss + float(distill_weight) * kd
            if trace_rows is not None and trace_aux_weight > 0.0:
                aux = trace_aux_loss(
                    adapter=adapter,
                    sample=dataset[idx],
                    channels=channels,
                    mask=masks[idx],
                    trace_row=trace_rows[idx],
                    device=device,
                    positive_weight=trace_positive_weight,
                )
                total_loss = total_loss + float(trace_aux_weight) * aux
            total += float(total_loss.item())
    return total / max(1, len(indices))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "visualthink_adapter.yaml"))
    parser.add_argument("--mode", choices=["full", "visualthink"], required=True)
    parser.add_argument("--unnorm_key", default="bridge_orig")
    parser.add_argument("--gate_checkpoint_dir", default="")
    parser.add_argument("--gate_config", default=str(ROOT / "configs" / "evidence_router.yaml"))
    parser.add_argument("--teacher_adapter_dir", default="")
    parser.add_argument("--init_adapter_dir", default="")
    parser.add_argument("--evidence_trace_manifest", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--fusion_mode", choices=["insert", "visual_residual"], default="insert")
    parser.add_argument("--evidence_scale", type=float, default=1.0)
    parser.add_argument("--base_distill_weight", type=float, default=0.0)
    parser.add_argument("--base_distill_temperature", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    cfg = load_yaml(args.config)
    channels = tuple(cfg["channel_order"])
    train_cfg = cfg["training"]
    adapter_cfg = cfg["adapter"]
    routing_mask_cfg = cfg.get("routing_mask", {})
    print(f"[stage] loading feature manifest rows: {args.feature_manifest}", flush=True)
    rows = load_rows(args.feature_manifest, limit=int(args.limit))
    print(f"[ok] loaded feature rows={len(rows)}", flush=True)
    trace_rows = load_trace_rows(args.evidence_trace_manifest, limit=int(args.limit)) if args.evidence_trace_manifest else []
    validate_trace_alignment(rows, trace_rows)
    print(f"[stage] building evidence-token dataset total={len(rows)}", flush=True)
    dataset = LearnedGatingDataset(
        args.feature_manifest,
        image_size=64,
        bow_dim=256,
        limit=int(args.limit),
        channels=channels,
    )
    if len(rows) != len(dataset):
        raise RuntimeError(f"row/dataset mismatch: {len(rows)} vs {len(dataset)}")
    print(f"[ok] dataset built rows={len(dataset)}", flush=True)

    train_idx, val_idx = split_indices(len(rows), val_ratio=float(train_cfg["val_ratio"]), seed=args.seed)

    print(f"[stage] loading OpenVLA model from {args.model_path}", flush=True)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=False)
    tokenizer = processor.tokenizer
    vla = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        attn_implementation="sdpa",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(device)
    print(f"[ok] OpenVLA loaded device={device} dtype={dtype}", flush=True)
    vla.eval()
    for param in vla.parameters():
        param.requires_grad = False

    hidden_size = int(vla.get_input_embeddings().weight.shape[1])
    adapter = EvidenceTokenAdapter(
        channel_dims={ch: int(FEATURE_DIMS[ch]) for ch in channels},
        channels=channels,
        hidden_size=hidden_size,
        num_global_tokens=int(adapter_cfg["num_global_tokens"]),
        proj_dim=int(adapter_cfg["proj_dim"]),
        dropout=float(adapter_cfg.get("dropout", 0.1)),
    ).to(device)
    init_adapter_resolved = None
    if args.init_adapter_dir:
        print(f"[stage] loading init adapter from {args.init_adapter_dir}", flush=True)
        init_adapter_resolved = load_adapter_init_weights(
            adapter=adapter,
            checkpoint_dir=args.init_adapter_dir,
            channels=channels,
            device=device,
        )
        print(f"[ok] init adapter loaded from {args.init_adapter_dir}", flush=True)

    distill_cfg = cfg.get("distillation", {})
    distill_weight = float(distill_cfg.get("weight", 0.0))
    distill_temperature = float(distill_cfg.get("temperature", 1.0))
    trace_aux_cfg = cfg.get("evidence_trace_aux", {})
    trace_aux_weight = float(trace_aux_cfg.get("weight", 0.0)) if bool(trace_aux_cfg.get("enabled", False)) else 0.0
    trace_positive_weight = float(trace_aux_cfg.get("positive_weight", 1.0))
    if trace_aux_weight > 0.0 and not trace_rows:
        print("[warn] evidence_trace_aux is enabled but --evidence_trace_manifest is empty; disabling trace auxiliary loss.", flush=True)
        trace_aux_weight = 0.0
    teacher_adapter = None
    teacher_masks = None
    if args.mode == "visualthink" and args.teacher_adapter_dir and distill_weight > 0.0:
        teacher_adapter, teacher_resolved = load_soft_adapter(
            checkpoint_dir=args.teacher_adapter_dir,
            channels=channels,
            hidden_size=hidden_size,
            device=device,
        )
        teacher_adapter.eval()
        for param in teacher_adapter.parameters():
            param.requires_grad = False
        teacher_masks = np.load(Path(args.teacher_adapter_dir) / "channel_masks.npy")
        teacher_channels = tuple(teacher_resolved["channels"])
        if teacher_channels != channels:
            raise RuntimeError(
                f"Teacher adapter channel mismatch: expected {channels}, got {teacher_channels}"
            )

    if args.mode == "visualthink":
        if not args.gate_checkpoint_dir:
            raise RuntimeError("--gate_checkpoint_dir is required for mode=visualthink")
        print(f"[stage] computing VisualThink masks from {args.gate_checkpoint_dir}", flush=True)
        masks = compute_routing_masks(
            manifest_path=args.feature_manifest,
            channels=channels,
            gate_checkpoint_dir=args.gate_checkpoint_dir,
            gate_config=args.gate_config,
            device=device,
            limit=int(args.limit),
            batch_size=int(train_cfg.get("gate_batch_size", 64)),
            mask_mode=str(routing_mask_cfg.get("mode", "hard")).lower(),
            soft_mask_blend=float(routing_mask_cfg.get("soft_mask_blend", 0.0)),
            min_mask_floor=float(routing_mask_cfg.get("min_mask_floor", 0.0)),
        )
        print(f"[ok] VisualThink masks ready shape={tuple(masks.shape)}", flush=True)
    else:
        masks = np.ones((len(rows), len(channels)), dtype=np.float32)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "channel_masks.npy", masks)
    resolved = {
        "feature_manifest": args.feature_manifest,
        "model_path": args.model_path,
        "mode": args.mode,
        "channels": list(channels),
        "limit": int(args.limit),
        "seed": args.seed,
        "dataset_size": len(rows),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "config_path": args.config,
        "gate_checkpoint_dir": args.gate_checkpoint_dir,
        "gate_config": args.gate_config,
        "teacher_adapter_dir": args.teacher_adapter_dir,
        "init_adapter_dir": args.init_adapter_dir,
        "evidence_trace_manifest": args.evidence_trace_manifest,
        "fusion_mode": args.fusion_mode,
        "evidence_scale": float(args.evidence_scale),
        "base_distill_weight": float(args.base_distill_weight),
        "base_distill_temperature": float(args.base_distill_temperature),
        "init_adapter_resolved_config": init_adapter_resolved,
        "config": cfg,
    }
    (out_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")

    opt = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    grad_accum = int(train_cfg.get("grad_accum_steps", 1))
    epochs = int(train_cfg["epochs"])
    max_train_steps = int(train_cfg.get("max_train_steps", 0))

    metrics_path = out_dir / "train_metrics.jsonl"
    best_val = float("inf")
    global_step = 0

    with metrics_path.open("w", encoding="utf-8") as mf:
        for epoch in range(1, epochs + 1):
            adapter.train()
            epoch_indices = train_idx[:]
            random.shuffle(epoch_indices)
            running = 0.0
            steps = 0
            start_ts = time.time()
            opt.zero_grad(set_to_none=True)
            print(f"[train] mode={args.mode} epoch={epoch}/{epochs} steps={len(epoch_indices)}", flush=True)
            for local_step, idx in enumerate(epoch_indices, start=1):
                student_out = forward_outputs(
                    vla=vla,
                    processor=processor,
                    tokenizer=tokenizer,
                    adapter=adapter,
                    row=rows[idx],
                    sample=dataset[idx],
                    channels=channels,
                    mask=masks[idx],
                    model_path=args.model_path,
                    device=device,
                    dtype=dtype,
                    unnorm_key=args.unnorm_key,
                    fusion_mode=args.fusion_mode,
                    evidence_scale=float(args.evidence_scale),
                )
                loss = student_out["loss"]
                if float(args.base_distill_weight) > 0.0:
                    if args.fusion_mode != "visual_residual":
                        raise RuntimeError("--base_distill_weight requires --fusion_mode visual_residual")
                    with torch.inference_mode():
                        base_teacher_out = forward_outputs(
                            vla=vla,
                            processor=processor,
                            tokenizer=tokenizer,
                            adapter=adapter,
                            row=rows[idx],
                            sample=dataset[idx],
                            channels=channels,
                            mask=masks[idx],
                            model_path=args.model_path,
                            device=device,
                            dtype=dtype,
                            unnorm_key=args.unnorm_key,
                            fusion_mode=args.fusion_mode,
                            evidence_scale=0.0,
                        )
                    base_kd = distillation_loss(
                        student_logits=student_out["logits"],
                        teacher_logits=base_teacher_out["logits"],
                        action_mask=student_out["action_mask"],
                        temperature=float(args.base_distill_temperature),
                    )
                    loss = loss + float(args.base_distill_weight) * base_kd
                if teacher_adapter is not None and teacher_masks is not None and distill_weight > 0.0:
                    with torch.inference_mode():
                        teacher_out = forward_outputs(
                            vla=vla,
                            processor=processor,
                            tokenizer=tokenizer,
                            adapter=teacher_adapter,
                            row=rows[idx],
                            sample=dataset[idx],
                            channels=channels,
                            mask=teacher_masks[idx],
                            model_path=args.model_path,
                            device=device,
                            dtype=dtype,
                            unnorm_key=args.unnorm_key,
                            fusion_mode=args.fusion_mode,
                            evidence_scale=float(args.evidence_scale),
                        )
                    kd = distillation_loss(
                        student_logits=student_out["logits"],
                        teacher_logits=teacher_out["logits"],
                        action_mask=student_out["action_mask"],
                        temperature=distill_temperature,
                    )
                    loss = loss + distill_weight * kd
                if trace_rows and trace_aux_weight > 0.0:
                    aux = trace_aux_loss(
                        adapter=adapter,
                        sample=dataset[idx],
                        channels=channels,
                        mask=masks[idx],
                        trace_row=trace_rows[idx],
                        device=device,
                        positive_weight=trace_positive_weight,
                    )
                    loss = loss + trace_aux_weight * aux
                (loss / grad_accum).backward()
                running += float(loss.item())
                steps += 1
                global_step += 1
                if (local_step % grad_accum) == 0:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                if local_step % args.log_every == 0 or local_step == len(epoch_indices):
                    elapsed = max(1e-6, time.time() - start_ts)
                    speed = local_step / elapsed
                    eta = (len(epoch_indices) - local_step) / max(1e-6, speed)
                    print(
                        f"[progress] epoch={epoch}/{epochs} step={local_step}/{len(epoch_indices)} "
                        f"loss={running/max(1, steps):.4f} speed={speed:.2f} it/s eta={eta:.0f}s",
                        flush=True,
                    )
                if max_train_steps > 0 and global_step >= max_train_steps:
                    break
            if (steps % grad_accum) != 0:
                opt.step()
                opt.zero_grad(set_to_none=True)

            val_loss = evaluate_loss(
                vla=vla,
                processor=processor,
                tokenizer=tokenizer,
                adapter=adapter,
                teacher_adapter=teacher_adapter,
                rows=rows,
                dataset=dataset,
                indices=val_idx,
                masks=masks,
                channels=channels,
                model_path=args.model_path,
                device=device,
                dtype=dtype,
                unnorm_key=args.unnorm_key,
                teacher_masks=teacher_masks,
                distill_weight=distill_weight,
                distill_temperature=distill_temperature,
                trace_rows=trace_rows if trace_rows else None,
                trace_aux_weight=trace_aux_weight,
                trace_positive_weight=trace_positive_weight,
                fusion_mode=args.fusion_mode,
                evidence_scale=float(args.evidence_scale),
            )
            train_loss = running / max(1, steps)
            record = {
                "epoch": epoch,
                "mode": args.mode,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            mf.write(json.dumps(record, ensure_ascii=False) + "\n")
            mf.flush()
            print(record, flush=True)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(adapter.state_dict(), out_dir / "adapter.pt")
                print(f"[best] val_loss={best_val:.4f}", flush=True)
            if max_train_steps > 0 and global_step >= max_train_steps:
                break

    print(f"[ok] saved adapter: {out_dir / 'adapter.pt'}", flush=True)


if __name__ == "__main__":
    main()
