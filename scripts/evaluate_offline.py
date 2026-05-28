#!/usr/bin/env python3
"""Benchmark real OpenVLA evidence interface with evidence tokens.

Modes:
1. OpenVLA-only: original instruction prompt only.
2. FullSoft: frozen OpenVLA + full evidence adapter.
3. VisualThink-VLA: frozen OpenVLA + routed evidence adapter/masks.
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

from visualthink_vla.models.evidence_router import FEATURE_DIMS, LearnedGatingDataset, STAGES
from visualthink_vla.models.soft_evidence_adapter import (
    EvidenceTokenAdapter,
    EvidenceTokenBatch,
    make_openvla_prompt,
    predict_action_with_soft_evidence,
)


def add_disturbance(image: Image.Image, sigma: float = 22.0) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    noise = np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def action_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def load_rows(manifest_path: str, limit: int = 0) -> list[dict]:
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit > 0 and len(rows) >= limit:
                    break
    if not rows:
        raise RuntimeError(f"empty manifest: {manifest_path}")
    return rows


def load_norm_stats(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_action_original(vla, processor, image: Image.Image, prompt: str, device: str, dtype, unnorm_key: str):
    start = time.perf_counter()
    inputs = processor(prompt, image).to(device, dtype=dtype)
    with torch.inference_mode():
        action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    elapsed = time.perf_counter() - start
    return np.asarray(action, dtype=np.float32), elapsed


def summarize(rows: list[dict], channels: tuple[str, ...]) -> dict:
    success = np.array([1.0 if r.get("success", False) else 0.0 for r in rows], dtype=np.float32)
    fps = np.array([float(r.get("fps", 0.0)) for r in rows], dtype=np.float32)
    latency = np.array([float(r.get("inference_time_s", 0.0)) for r in rows], dtype=np.float32)
    selected = np.array([float(r.get("selected_channels", 0.0)) for r in rows], dtype=np.float32)
    out = {
        "n": int(len(rows)),
        "success_rate": float(success.mean()) if len(rows) else 0.0,
        "avg_step_latency_s": float(latency.mean()) if len(rows) else 0.0,
        "inference_fps": float(fps.mean()) if len(rows) else 0.0,
        "avg_selected_channels": float(selected.mean()) if len(rows) else 0.0,
    }
    for ch in channels:
        out[f"{ch}_keep_rate"] = float(np.mean([1.0 if r.get("gates", {}).get(ch, False) else 0.0 for r in rows])) if rows else 0.0
    return out


def episode_proxy_summary(rows: list[dict], channels: tuple[str, ...]) -> dict:
    episodes: dict[int, list[dict]] = {}
    for row in rows:
        ep = int(row.get("episode_idx", row.get("idx", 0)))
        episodes.setdefault(ep, []).append(row)
    if not episodes:
        out = {
            "n_episodes": 0,
            "success_rate_episode_proxy": 0.0,
            "avg_completion_time_s": None,
            "timeout_penalized_completion_time_s": None,
        }
        for ch in channels:
            out[f"{ch}_keep_rate_episode"] = 0.0
        return out

    proxy_completion = []
    proxy_success_completion = []
    episode_successes = []
    episode_keep_rates = {ch: [] for ch in channels}
    for episode_rows in episodes.values():
        ordered = sorted(episode_rows, key=lambda item: int(item.get("step_idx", item.get("idx", 0))))
        comp = float(sum(float(row.get("inference_time_s", 0.0)) for row in ordered))
        success = all(bool(row.get("success", False)) for row in ordered)
        proxy_completion.append(comp)
        episode_successes.append(1.0 if success else 0.0)
        if success:
            proxy_success_completion.append(comp)
        for ch in channels:
            episode_keep_rates[ch].append(
                float(np.mean([1.0 if row.get("gates", {}).get(ch, False) else 0.0 for row in ordered]))
            )
    out = {
        "n_episodes": int(len(episodes)),
        "success_rate_episode_proxy": float(np.mean(episode_successes)),
        "avg_completion_time_s": float(np.mean(proxy_success_completion)) if proxy_success_completion else None,
        "timeout_penalized_completion_time_s": float(np.mean(proxy_completion)) if proxy_completion else None,
    }
    for ch in channels:
        out[f"{ch}_keep_rate_episode"] = float(np.mean(episode_keep_rates[ch])) if episode_keep_rates[ch] else 0.0
    return out


def stage_summary(rows: list[dict], channels: tuple[str, ...]) -> dict[str, dict]:
    out = {}
    for stage in STAGES:
        sub = [r for r in rows if r.get("stage") == stage]
        if not sub:
            continue
        row = {
            "n": len(sub),
            "success_rate": float(np.mean([1.0 if r["success"] else 0.0 for r in sub])),
            "avg_step_latency_s": float(np.mean([float(r.get("inference_time_s", 0.0)) for r in sub])),
            "avg_selected_channels": float(np.mean([float(r.get("selected_channels", 0.0)) for r in sub])),
        }
        for ch in channels:
            row[f"{ch}_keep_rate"] = float(np.mean([1.0 if r.get("gates", {}).get(ch, False) else 0.0 for r in sub]))
        out[stage] = row
    return out


def write_stage_summary(orig_rows: list[dict], full_rows: list[dict], vt_rows: list[dict], out_dir: Path, channels: tuple[str, ...]) -> None:
    orig = stage_summary(orig_rows, channels)
    full = stage_summary(full_rows, channels)
    dyn = stage_summary(vt_rows, channels)
    lines = [
        "| Stage | Model | N | Success | StepLatency(s) | AvgSelected | " + " | ".join(ch.title() for ch in channels) + " |",
        "|---|---|---:|---:|---:|---:|" + "---:|" * len(channels),
    ]
    for stage in STAGES:
        for name, data in [("OpenVLA", orig.get(stage)), ("FullSoft", full.get(stage)), ("VisualThink-VLA", dyn.get(stage))]:
            if data is None:
                continue
            lines.append(
                f"| {stage} | {name} | {data['n']} | {data['success_rate']:.4f} | {data['avg_step_latency_s']:.4f} | "
                f"{data['avg_selected_channels']:.4f} | "
                + " | ".join(f"{data[f'{ch}_keep_rate']:.4f}" for ch in channels)
                + " |"
            )
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "stage_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_report(orig_rows: list[dict], full_rows: list[dict], vt_rows: list[dict], out_dir: Path, channels: tuple[str, ...]) -> None:
    orig = summarize(orig_rows, channels)
    full = summarize(full_rows, channels)
    dyn = summarize(vt_rows, channels)
    orig_ep = episode_proxy_summary(orig_rows, channels)
    full_ep = episode_proxy_summary(full_rows, channels)
    vt_ep = episode_proxy_summary(vt_rows, channels)
    metrics = [
        "success_rate",
        "success_rate_episode_proxy",
        "avg_completion_time_s",
        "timeout_penalized_completion_time_s",
        "avg_step_latency_s",
        "inference_fps",
        "avg_selected_channels",
    ] + [f"{ch}_keep_rate" for ch in channels]
    lines = [
        "| Metric | OpenVLA | FullSoft | VisualThink-VLA | Full-Orig | VT-Orig |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in metrics:
        if key in orig_ep:
            ov = orig_ep.get(key)
            fv = full_ep.get(key)
            dv = vt_ep.get(key)
        else:
            ov = orig.get(key, 0.0)
            fv = full.get(key, 0.0)
            dv = dyn.get(key, 0.0)
        def fmt(value):
            return "--" if value is None else f"{float(value):.4f}"
        def diff(a, b):
            if a is None or b is None:
                return "--"
            return f"{float(b)-float(a):+.4f}"
        lines.append(f"| {key} | {fmt(ov)} | {fmt(fv)} | {fmt(dv)} | {diff(ov, fv)} | {diff(ov, dv)} |")
    (out_dir / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_stage_summary(orig_rows, full_rows, vt_rows, out_dir, channels)


def load_adapter(checkpoint_dir: str, channels: tuple[str, ...], hidden_size: int, device: torch.device) -> tuple[EvidenceTokenAdapter, np.ndarray, dict]:
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
    adapter.eval()
    masks = np.load(ckpt_dir / "channel_masks.npy")
    return adapter, masks, resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--full_checkpoint_dir", required=True)
    parser.add_argument("--visualthink_checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--success_l1_thresh", type=float, default=0.08)
    parser.add_argument("--disturb_ratio", type=float, default=0.0)
    parser.add_argument("--disturb_scale", type=float, default=22.0)
    parser.add_argument("--skip_empty_instruction", action="store_true")
    parser.add_argument("--unnorm_key", default="bridge_orig")
    parser.add_argument("--norm_stats", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_limit = args.limit if args.limit > 0 and not args.skip_empty_instruction else 0
    rows = load_rows(args.feature_manifest, limit=dataset_limit)
    full_resolved = json.loads((Path(args.full_checkpoint_dir) / "resolved_config.json").read_text(encoding="utf-8"))
    channels = tuple(full_resolved["channels"])
    dataset = LearnedGatingDataset(
        args.feature_manifest,
        image_size=64,
        bow_dim=256,
        limit=dataset_limit,
        channels=channels,
    )
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
    if args.norm_stats:
        extra_norm_stats = load_norm_stats(args.norm_stats)
        base_norm_stats = dict(getattr(vla, "norm_stats", {}) or {})
        base_norm_stats.update(extra_norm_stats)
        vla.norm_stats = base_norm_stats
    vla.eval()
    hidden_size = int(vla.get_input_embeddings().weight.shape[1])

    full_adapter, full_masks, _ = load_adapter(args.full_checkpoint_dir, channels, hidden_size, device)
    vt_adapter, vt_masks, _ = load_adapter(args.visualthink_checkpoint_dir, channels, hidden_size, device)

    candidate_indices = []
    for i, row in enumerate(rows):
        if args.skip_empty_instruction and not (row.get("instruction") or "").strip():
            continue
        candidate_indices.append(i)
    if args.limit > 0:
        candidate_indices = candidate_indices[: args.limit]
    if not candidate_indices:
        raise RuntimeError("No samples selected for benchmark.")

    orig_rows = []
    full_rows = []
    vt_rows = []

    for out_idx, idx in enumerate(candidate_indices):
        row = rows[idx]
        sample = dataset[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        disturbed = random.random() < args.disturb_ratio
        image_in = add_disturbance(image, sigma=args.disturb_scale) if disturbed else image
        gt = np.asarray(row["action"], dtype=np.float32)

        prompt = make_openvla_prompt(row["instruction"], args.model_path)
        orig_action, orig_t = predict_action_original(vla, processor, image_in, prompt, str(device), dtype, args.unnorm_key)
        orig_l1 = action_l1(orig_action, gt)
        orig_rows.append(
            {
                "idx": out_idx,
                "episode_idx": int(dataset.records[idx].episode_idx),
                "step_idx": int(dataset.records[idx].step_idx),
                "stage": dataset.records[idx].stage,
                "success": orig_l1 <= args.success_l1_thresh,
                "inference_time_s": orig_t,
                "fps": 1.0 / max(1e-6, orig_t),
                "l1": orig_l1,
                "selected_channels": 0.0,
                "gates": {ch: False for ch in channels},
            }
        )

        start = time.perf_counter()
        full_batch = EvidenceTokenBatch(
            channel_features={ch: sample[ch].unsqueeze(0) for ch in channels},
            channel_mask=torch.tensor(full_masks[idx], dtype=torch.float32).unsqueeze(0),
            stage_one_hot=sample["stage_one_hot"].unsqueeze(0),
            step_ratio=sample["step_ratio"].unsqueeze(0),
        )
        full_action = predict_action_with_soft_evidence(
            vla=vla,
            processor=processor,
            tokenizer=tokenizer,
            adapter=full_adapter,
            image=image_in,
            instruction=row["instruction"],
            evidence_batch=full_batch,
            model_path=args.model_path,
            device=device,
            dtype=dtype,
            unnorm_key=args.unnorm_key,
        )
        full_t = time.perf_counter() - start
        full_l1 = action_l1(full_action, gt)
        full_rows.append(
            {
                "idx": out_idx,
                "episode_idx": int(dataset.records[idx].episode_idx),
                "step_idx": int(dataset.records[idx].step_idx),
                "stage": dataset.records[idx].stage,
                "success": full_l1 <= args.success_l1_thresh,
                "inference_time_s": full_t,
                "fps": 1.0 / max(1e-6, full_t),
                "l1": full_l1,
                "selected_channels": float(np.sum(full_masks[idx] > 0.5)),
                "gates": {ch: bool(full_masks[idx][i] > 0.5) for i, ch in enumerate(channels)},
            }
        )

        start = time.perf_counter()
        vt_batch = EvidenceTokenBatch(
            channel_features={ch: sample[ch].unsqueeze(0) for ch in channels},
            channel_mask=torch.tensor(vt_masks[idx], dtype=torch.float32).unsqueeze(0),
            stage_one_hot=sample["stage_one_hot"].unsqueeze(0),
            step_ratio=sample["step_ratio"].unsqueeze(0),
        )
        vt_action = predict_action_with_soft_evidence(
            vla=vla,
            processor=processor,
            tokenizer=tokenizer,
            adapter=vt_adapter,
            image=image_in,
            instruction=row["instruction"],
            evidence_batch=vt_batch,
            model_path=args.model_path,
            device=device,
            dtype=dtype,
            unnorm_key=args.unnorm_key,
        )
        vt_t = time.perf_counter() - start
        vt_l1 = action_l1(vt_action, gt)
        vt_rows.append(
            {
                "idx": out_idx,
                "episode_idx": int(dataset.records[idx].episode_idx),
                "step_idx": int(dataset.records[idx].step_idx),
                "stage": dataset.records[idx].stage,
                "success": vt_l1 <= args.success_l1_thresh,
                "inference_time_s": vt_t,
                "fps": 1.0 / max(1e-6, vt_t),
                "l1": vt_l1,
                "selected_channels": float(np.sum(vt_masks[idx] > 0.5)),
                "gates": {ch: bool(vt_masks[idx][i] > 0.5) for i, ch in enumerate(channels)},
            }
        )
        if (out_idx + 1) % 10 == 0 or out_idx + 1 == len(candidate_indices):
            print(f"[progress] {out_idx + 1}/{len(candidate_indices)}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "openvla_eval.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in orig_rows), encoding="utf-8")
    (out_dir / "fullsoft_eval.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in full_rows), encoding="utf-8")
    (out_dir / "visualthink_vla_eval.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in vt_rows), encoding="utf-8")
    save_report(orig_rows, full_rows, vt_rows, out_dir, channels)
    print("[ok] openvla:", summarize(orig_rows, channels))
    print("[ok] full_soft:", summarize(full_rows, channels))
    print("[ok] visualthink_vla:", summarize(vt_rows, channels))
    print(f"[ok] report dir: {out_dir}")


if __name__ == "__main__":
    main()
