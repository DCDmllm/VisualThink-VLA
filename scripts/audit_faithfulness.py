#!/usr/bin/env python3
"""Evaluate counterfactual faithfulness of VisualThink-VLA evidence routing and traces."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualthink_vla.models.evidence_router import FEATURE_DIMS, LearnedGatingDataset, STAGES
from visualthink_vla.models.soft_evidence_adapter import EvidenceTokenAdapter, EvidenceTokenBatch, predict_action_with_soft_evidence


CHANNELS = ("bbox", "edge", "motion", "relation")


def load_jsonl(path: Path, limit: int = 0) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError(f"empty jsonl: {path}")
    return rows


def load_norm_stats(path: str) -> dict:
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if "norm_stats" in raw and isinstance(raw["norm_stats"], dict):
        return raw["norm_stats"]
    return raw


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



def validate_trace_alignment(rows: list[dict], trace_rows: list[dict]) -> None:
    if not trace_rows:
        return
    if len(rows) != len(trace_rows):
        raise RuntimeError(f"row/trace mismatch: {len(rows)} vs {len(trace_rows)}")
    for idx, (row, trace_row) in enumerate(zip(rows, trace_rows)):
        if not rows_are_aligned(row, trace_row):
            row_key = (row.get("episode_idx"), row.get("step_idx"), row.get("npz_path"))
            trace_key = trace_alignment_key(trace_row)
            raise RuntimeError(f"trace mismatch at idx={idx}: row={row_key} trace={trace_key}")



def action_l1(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, dtype=np.float32) - np.asarray(gt, dtype=np.float32))))



def load_adapter(checkpoint_dir: str, channels: tuple[str, ...], hidden_size: int, device: torch.device) -> tuple[EvidenceTokenAdapter, np.ndarray]:
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
    return adapter, np.load(ckpt_dir / "channel_masks.npy")



def predict_with_sample_mask(
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
) -> tuple[np.ndarray, float]:
    batch = EvidenceTokenBatch(
        channel_features={ch: sample[ch].unsqueeze(0) for ch in channels},
        channel_mask=torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
        stage_one_hot=sample["stage_one_hot"].unsqueeze(0),
        step_ratio=sample["step_ratio"].unsqueeze(0),
    )
    start = time.time()
    action = predict_action_with_soft_evidence(
        vla=vla,
        processor=processor,
        tokenizer=tokenizer,
        adapter=adapter,
        image=Image.open(row["image_path"]).convert("RGB"),
        instruction=row["instruction"],
        evidence_batch=batch,
        model_path=model_path,
        device=device,
        dtype=dtype,
        unnorm_key=unnorm_key,
    )
    return action, time.time() - start


def safe_predict_with_sample_mask(*, gt: np.ndarray, label: str, idx: int, **kwargs) -> tuple[np.ndarray, float, bool]:
    try:
        action, elapsed = predict_with_sample_mask(**kwargs)
        return action, elapsed, True
    except Exception as exc:
        print(f"[warn] prediction failed idx={idx} label={label}: {type(exc).__name__}: {exc}", flush=True)
        return np.full_like(gt, np.inf, dtype=np.float32), 0.0, False



def clone_sample(sample: dict) -> dict:
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in sample.items()}


def trace_utility_rank(trace_row: dict | None) -> list:
    if not trace_row:
        return []
    utility_rank = trace_row.get("utility_rank_records")
    if isinstance(utility_rank, list) and utility_rank:
        return utility_rank
    utility_rank = trace_row.get("utility_rank")
    return utility_rank if isinstance(utility_rank, list) else []


def utility_item_channel(item) -> str:
    if isinstance(item, dict):
        return str(item.get("channel", ""))
    return str(item)



def top_channel_from_trace(trace_row: dict | None, mask: np.ndarray, channels: tuple[str, ...], threshold: float) -> str:
    route_mask = {ch: bool(float(mask[i]) > threshold) for i, ch in enumerate(channels)}
    utility_rank = trace_utility_rank(trace_row)
    if utility_rank:
        for item in utility_rank:
            ch = utility_item_channel(item)
            if ch in route_mask and route_mask[ch]:
                return ch
        top_ch = utility_item_channel(utility_rank[0])
        if top_ch in channels:
            return top_ch
    if np.max(mask) > 0:
        return channels[int(np.argmax(mask))]
    return channels[0]



def shuffle_channels_in_sample(sample: dict, donor_sample: dict, shuffle_channels: tuple[str, ...]) -> dict:
    out = clone_sample(sample)
    for ch in shuffle_channels:
        if ch in out and ch in donor_sample:
            out[ch] = donor_sample[ch].clone()
    return out



def parse_rationale_channels(text: str, channels: tuple[str, ...]) -> set[str]:
    lower = (text or "").lower()
    return {ch for ch in channels if re.search(rf"\b{re.escape(ch)}\b", lower)}



def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))



def aggregate_rows(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {
            "n": 0,
            "visualthink_success": 0.0,
            "top_remove_success": 0.0,
            "shuffle_success": 0.0,
            "top_remove_drop": 0.0,
            "shuffle_drop": 0.0,
            "route_rationale_jaccard": 0.0,
            "top_utility_mentioned_rate": 0.0,
            "avg_selected_channels": 0.0,
            "avg_visualthink_step_latency_s": 0.0,
        }
    visualthink_success = float(np.mean([1.0 if row["visualthink_success"] else 0.0 for row in rows]))
    top_remove_success = float(np.mean([1.0 if row["top_remove_success"] else 0.0 for row in rows]))
    shuffle_success = float(np.mean([1.0 if row["shuffle_success"] else 0.0 for row in rows]))
    return {
        "n": len(rows),
        "visualthink_success": visualthink_success,
        "top_remove_success": top_remove_success,
        "shuffle_success": shuffle_success,
        "top_remove_drop": visualthink_success - top_remove_success,
        "shuffle_drop": visualthink_success - shuffle_success,
        "route_rationale_jaccard": float(np.mean([float(row.get("route_rationale_jaccard", 0.0)) for row in rows])),
        "top_utility_mentioned_rate": float(np.mean([1.0 if row.get("top_utility_mentioned", False) else 0.0 for row in rows])),
        "avg_selected_channels": float(np.mean([float(row.get("selected_channels", 0.0)) for row in rows])),
        "avg_visualthink_step_latency_s": float(np.mean([float(row.get("visualthink_step_latency_s", 0.0)) for row in rows])),
    }



def write_markdown_tables(out_dir: Path, eval_rows: list[dict]) -> None:
    metric_keys = [
        "visualthink_success",
        "top_remove_success",
        "top_remove_drop",
        "shuffle_success",
        "shuffle_drop",
        "route_rationale_jaccard",
        "top_utility_mentioned_rate",
        "avg_selected_channels",
        "avg_visualthink_step_latency_s",
    ]
    overall = aggregate_rows(eval_rows)
    overall_lines = ["| Metric | Value |", "|---|---:|"]
    for key in ["n"] + metric_keys:
        val = overall[key]
        overall_lines.append(f"| {key} | {val:.4f} |" if isinstance(val, float) else f"| {key} | {val} |")
    (out_dir / "summary_table.md").write_text("\n".join(overall_lines) + "\n", encoding="utf-8")

    stage_lines = [
        "| Stage | N | VisualThinkSuccess | TopRemoveSuccess | TopRemoveDrop | ShuffleSuccess | ShuffleDrop | RouteRationaleJaccard | TopUtilityMentioned | AvgSelected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        stats = aggregate_rows([row for row in eval_rows if row.get("stage") == stage])
        if stats["n"] == 0:
            continue
        stage_lines.append(
            f"| {stage} | {stats['n']} | {stats['visualthink_success']:.4f} | {stats['top_remove_success']:.4f} | {stats['top_remove_drop']:+.4f} | "
            f"{stats['shuffle_success']:.4f} | {stats['shuffle_drop']:+.4f} | {stats['route_rationale_jaccard']:.4f} | "
            f"{stats['top_utility_mentioned_rate']:.4f} | {stats['avg_selected_channels']:.4f} |"
        )
    (out_dir / "stage_summary.md").write_text("\n".join(stage_lines) + "\n", encoding="utf-8")

    grouped = defaultdict(list)
    for row in eval_rows:
        grouped[str(row.get("instruction", ""))].append(row)
    inst_lines = [
        "| Instruction | N | VisualThinkSuccess | TopRemoveDrop | ShuffleDrop | RouteRationaleJaccard | TopUtilityMentioned |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for instruction, sub in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        stats = aggregate_rows(sub)
        clean_instruction = instruction.replace("|", "/")
        inst_lines.append(
            f"| {clean_instruction} | {stats['n']} | {stats['visualthink_success']:.4f} | {stats['top_remove_drop']:+.4f} | "
            f"{stats['shuffle_drop']:+.4f} | {stats['route_rationale_jaccard']:.4f} | {stats['top_utility_mentioned_rate']:.4f} |"
        )
    (out_dir / "instruction_summary.md").write_text("\n".join(inst_lines) + "\n", encoding="utf-8")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--visualthink_checkpoint_dir", required=True)
    parser.add_argument("--evidence_trace_manifest", default="")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--norm_stats", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--unnorm_key", default="bridge_orig")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--success_l1_thresh", type=float, default=0.08)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--shuffle_channels", default="relation,motion")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_jsonl(Path(args.feature_manifest), limit=int(args.limit))
    trace_rows = load_jsonl(Path(args.evidence_trace_manifest), limit=int(args.limit)) if args.evidence_trace_manifest else []
    validate_trace_alignment(rows, trace_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    resolved = json.loads((Path(args.visualthink_checkpoint_dir) / "resolved_config.json").read_text(encoding="utf-8"))
    channels = tuple(resolved.get("channels", CHANNELS))
    dataset = LearnedGatingDataset(
        args.feature_manifest,
        image_size=64,
        bow_dim=256,
        limit=int(args.limit),
        channels=channels,
    )
    if len(dataset) != len(rows):
        raise RuntimeError(f"dataset/rows mismatch: {len(dataset)} vs {len(rows)}")

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
    adapter, masks = load_adapter(args.visualthink_checkpoint_dir, channels, hidden_size, device)

    shuffle_channels = tuple(ch.strip() for ch in args.shuffle_channels.split(",") if ch.strip())
    eval_rows = []

    for idx, row in enumerate(rows):
        sample = dataset[idx]
        mask = np.asarray(masks[idx], dtype=np.float32)
        donor_idx = (idx + 1) % len(dataset)
        donor_sample = dataset[donor_idx]
        trace_row = trace_rows[idx] if trace_rows else None
        removed_channel = top_channel_from_trace(trace_row, mask, channels, threshold=float(args.mask_threshold))

        gt = np.asarray(row["action"], dtype=np.float32)
        base_action, base_t, base_ok = safe_predict_with_sample_mask(
            vla=vla,
            processor=processor,
            tokenizer=tokenizer,
            adapter=adapter,
            row=row,
            sample=sample,
            channels=channels,
            mask=mask,
            model_path=args.model_path,
            device=device,
            dtype=dtype,
            unnorm_key=args.unnorm_key,
            gt=gt,
            label="visualthink_vla",
            idx=idx,
        )
        if not base_ok:
            continue
        base_l1 = action_l1(base_action, gt)

        remove_mask = mask.copy()
        remove_mask[channels.index(removed_channel)] = 0.0
        remove_action, remove_t, _remove_ok = safe_predict_with_sample_mask(
            vla=vla,
            processor=processor,
            tokenizer=tokenizer,
            adapter=adapter,
            row=row,
            sample=sample,
            channels=channels,
            mask=remove_mask,
            model_path=args.model_path,
            device=device,
            dtype=dtype,
            unnorm_key=args.unnorm_key,
            gt=gt,
            label=f"top_remove:{removed_channel}",
            idx=idx,
        )
        remove_l1 = action_l1(remove_action, gt)

        shuffle_sample = shuffle_channels_in_sample(sample, donor_sample, shuffle_channels=shuffle_channels)
        shuffle_action, shuffle_t, _shuffle_ok = safe_predict_with_sample_mask(
            vla=vla,
            processor=processor,
            tokenizer=tokenizer,
            adapter=adapter,
            row=row,
            sample=shuffle_sample,
            channels=channels,
            mask=mask,
            model_path=args.model_path,
            device=device,
            dtype=dtype,
            unnorm_key=args.unnorm_key,
            gt=gt,
            label="shuffle",
            idx=idx,
        )
        shuffle_l1 = action_l1(shuffle_action, gt)

        route_channels = {ch for i, ch in enumerate(channels) if float(mask[i]) > float(args.mask_threshold)}
        if not route_channels and float(mask.max()) > 0:
            route_channels = {channels[int(mask.argmax())]}
        rationale_channels = parse_rationale_channels(trace_row.get("visual_rationale", "") if trace_row else "", channels)
        utility_rank = trace_utility_rank(trace_row)
        top_utility_channel = utility_item_channel(utility_rank[0]) if utility_rank else ""

        eval_rows.append(
            {
                "idx": idx,
                "stage": str(dataset.records[idx].stage),
                "instruction": row.get("instruction", ""),
                "visualthink_success": base_l1 <= args.success_l1_thresh,
                "top_remove_success": remove_l1 <= args.success_l1_thresh,
                "shuffle_success": shuffle_l1 <= args.success_l1_thresh,
                "visualthink_l1": base_l1,
                "top_remove_l1": remove_l1,
                "shuffle_l1": shuffle_l1,
                "visualthink_step_latency_s": base_t,
                "top_remove_step_latency_s": remove_t,
                "shuffle_step_latency_s": shuffle_t,
                "visualthink_fps": 1.0 / max(1e-6, base_t),
                "top_remove_fps": 1.0 / max(1e-6, remove_t),
                "shuffle_fps": 1.0 / max(1e-6, shuffle_t),
                "selected_channels": float(np.sum(mask > float(args.mask_threshold))),
                "route_channels": sorted(route_channels),
                "removed_channel": removed_channel,
                "rationale_channels": sorted(rationale_channels),
                "route_rationale_jaccard": jaccard(route_channels, rationale_channels) if trace_row else math.nan,
                "top_utility_channel": top_utility_channel,
                "top_utility_mentioned": bool(top_utility_channel and top_utility_channel in rationale_channels),
            }
        )
        if (idx + 1) % 10 == 0 or idx + 1 == len(rows):
            print(f"[progress] {idx + 1}/{len(rows)}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "faithfulness_eval.jsonl").open("w", encoding="utf-8") as f:
        for row in eval_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_markdown_tables(out_dir, eval_rows)
    print(f"[ok] wrote {out_dir / 'summary_table.md'}")
    print(f"[ok] wrote {out_dir / 'stage_summary.md'}")
    print(f"[ok] wrote {out_dir / 'instruction_summary.md'}")


if __name__ == "__main__":
    main()
