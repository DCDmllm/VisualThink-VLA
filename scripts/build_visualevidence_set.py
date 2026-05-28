#!/usr/bin/env python3
"""Build an evidence-grounded trace dataset from existing feature and gate artifacts.

Each output row is aligned with one feature-manifest row and contains a short,
channel-grounded rationale instead of a free-form long CoT.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualthink_vla.models.evidence_router import LearnedGatingDataset

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


def load_utility_map(path: Path) -> dict[tuple[Any, Any, str], dict]:
    out = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("episode_idx"), row.get("step_idx"), row.get("npz_path", ""))
            out[key] = row
    return out


def infer_dataset_name(manifest_path: Path, explicit_name: str = "") -> str:
    if explicit_name:
        return explicit_name
    parts = manifest_path.parts
    if "runs" in parts:
        run_idx = parts.index("runs") + 1
        if run_idx < len(parts):
            run_name = parts[run_idx]
            for dataset_name in (
                "bridge",
                "libero",
                "fractal",
                "roboturk",
                "viola",
                "taco_play",
                "jaco_play",
                "stanford_hydra",
                "nyu_franka_play",
                "kuka",
                "berkeley_cable_routing",
                "berkeley_autolab_ur5",
                "dobbe",
                "language_table",
            ):
                if run_name == dataset_name or run_name.startswith(f"{dataset_name}_"):
                    return dataset_name
            return run_name
    return "unknown"


def ref_path(path_value: str) -> str:
    path = Path(path_value)
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path_value)


def infer_channels(visualthink_checkpoint_dir: Path, default_channels: tuple[str, ...] = CHANNELS) -> tuple[str, ...]:
    resolved_path = visualthink_checkpoint_dir / "resolved_config.json"
    if not resolved_path.exists():
        return default_channels
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    return tuple(resolved.get("channels", default_channels))


def unpack_edge_density(npz_path: str) -> dict[str, float]:
    npz = np.load(npz_path, allow_pickle=True)
    if "edge_packed" not in npz.files or "edge_shape" not in npz.files:
        return {"density": 0.0, "center": 0.0}
    shape = tuple(int(x) for x in np.asarray(npz["edge_shape"]).tolist())
    edge = np.unpackbits(np.asarray(npz["edge_packed"], dtype=np.uint8))[: int(shape[0] * shape[1])]
    edge = edge.reshape(shape).astype(np.float32)
    h, w = edge.shape
    center = edge[h // 4 : (3 * h) // 4, w // 4 : (3 * w) // 4]
    return {
        "density": float(edge.mean()),
        "center": float(center.mean()) if center.size else float(edge.mean()),
    }


def bbox_evidence(row: dict) -> str:
    dets = row.get("detections") or []
    if not dets:
        return "bbox: no confident object box is available from the detector."
    det = dets[0]
    label = str(det.get("label", "object"))
    score = float(det.get("score", 0.0))
    box = det.get("bbox", [0, 0, 0, 0])
    return f"bbox: top detection '{label}' has score {score:.2f} at box {box}."


def edge_evidence(row: dict) -> str:
    stats = unpack_edge_density(str(row["npz_path"]))
    return f"edge: contour density is {stats['density']:.3f}, with center-region density {stats['center']:.3f}."


def motion_evidence(row: dict) -> str:
    m = row.get("motion_stats") or {}
    source = row.get("motion_source", "unknown")
    left = float(m.get("left", 0.0))
    right = float(m.get("right", 0.0))
    vertical = "top" if float(m.get("top", 0.0)) >= float(m.get("bottom", 0.0)) else "bottom"
    horizontal = "right" if right >= left else "left"
    return (
        f"motion: source={source}, mean={float(m.get('mean', 0.0)):.3f}, "
        f"density={float(m.get('density', 0.0)):.3f}, strongest region={vertical}-{horizontal}."
    )


def relation_evidence(row: dict) -> str:
    r = row.get("relation_stats") or {}
    target = str(r.get("target_label") or r.get("target_phrase") or "target object")
    anchor = str(r.get("goal_anchor", "unknown"))
    cx = float(r.get("target_cx", 0.0))
    cy = float(r.get("target_cy", 0.0))
    area = float(r.get("target_area", 0.0))
    goal_dist = float(r.get("goal_dist", 0.0))
    return (
        f"relation: target='{target}' at normalized center ({cx:.2f}, {cy:.2f}), area={area:.3f}, "
        f"goal_anchor={anchor}, goal_dist={goal_dist:.3f}."
    )


def build_channel_evidence(row: dict, channels: tuple[str, ...]) -> dict[str, str]:
    evidence = {}
    for ch in channels:
        if ch == "bbox":
            evidence[ch] = bbox_evidence(row)
        elif ch == "edge":
            evidence[ch] = edge_evidence(row)
        elif ch == "motion":
            evidence[ch] = motion_evidence(row)
        elif ch == "relation":
            evidence[ch] = relation_evidence(row)
        else:
            evidence[ch] = f"{ch}: evidence unavailable."
    return evidence


def rank_utilities(utility_row: dict | None, channels: tuple[str, ...]) -> tuple[list[str], dict[str, float], list[dict]]:
    utilities = {ch: 0.0 for ch in channels}
    if utility_row:
        raw = utility_row.get("utilities_raw") or utility_row.get("utilities") or {}
        utilities = {ch: float(raw.get(ch, 0.0)) for ch in channels}
    ranked = sorted(((ch, utilities.get(ch, 0.0)) for ch in channels), key=lambda x: (-x[1], x[0]))
    utility_rank = [ch for ch, _ in ranked]
    utility_rank_records = [{"channel": ch, "utility": val, "rank": i + 1} for i, (ch, val) in enumerate(ranked)]
    return utility_rank, utilities, utility_rank_records


def select_channels(mask: np.ndarray, channels: tuple[str, ...], utility_rank: list[str], threshold: float) -> tuple[list[str], dict[str, int]]:
    route_mask = {ch: int(float(mask[i]) > threshold) for i, ch in enumerate(channels)}
    selected = [ch for ch in channels if route_mask[ch]]
    if not selected and utility_rank:
        selected = [str(utility_rank[0])]
        route_mask[selected[0]] = 1
    return selected, route_mask


def make_action_intent(instruction: str, stage: str) -> str:
    instr = (instruction or "the task").strip()
    if stage == "approach":
        return f"Move toward task-relevant objects to make progress on '{instr}'."
    if stage == "grasp":
        return f"Refine local contact or grasp configuration needed to execute '{instr}'."
    return f"Transport, place, or finish the manipulation step implied by '{instr}'."


def make_visual_rationale(selected_channels: list[str], channel_evidence: dict[str, str], stage: str) -> str:
    if not selected_channels:
        selected_channels = ["edge"]
    reasons = []
    for ch in selected_channels:
        if ch == "bbox":
            reasons.append(f"bbox localizes the candidate object ({channel_evidence[ch]})")
        elif ch == "edge":
            reasons.append(f"edge highlights shape/contact structure ({channel_evidence[ch]})")
        elif ch == "motion":
            reasons.append(f"motion indicates where the scene is changing ({channel_evidence[ch]})")
        elif ch == "relation":
            reasons.append(f"relation links the instruction to object geometry ({channel_evidence[ch]})")
    joined = " ".join(reasons)
    return f"At the {stage} stage, I consult {', '.join(selected_channels)} because {joined}"


def write_summary(out_path: Path, trace_rows: list[dict], channels: tuple[str, ...]) -> None:
    lines = [
        "| Metric | Value |",
        "|---|---:|",
        f"| num_rows | {len(trace_rows)} |",
    ]
    for ch in channels:
        keep = sum(1 for row in trace_rows if row.get("route_mask", {}).get(ch, False)) / max(1, len(trace_rows))
        mention = sum(1 for row in trace_rows if ch in str(row.get("visual_rationale", "")).lower()) / max(1, len(trace_rows))
        lines.append(f"| {ch}_route_rate | {keep:.4f} |")
        lines.append(f"| {ch}_rationale_mention_rate | {mention:.4f} |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--gate_checkpoint_dir", required=True)
    parser.add_argument("--visualthink_checkpoint_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    args = parser.parse_args()

    manifest_path = Path(args.feature_manifest)
    gate_dir = Path(args.gate_checkpoint_dir)
    vt_dir = Path(args.visualthink_checkpoint_dir)
    out_path = Path(args.output_path)
    channels = infer_channels(vt_dir)
    dataset_name = infer_dataset_name(manifest_path, explicit_name=str(args.dataset_name).strip())

    print(f"[stage] loading feature rows: {manifest_path}", flush=True)
    rows = load_jsonl(manifest_path, limit=int(args.limit))
    print(f"[ok] feature rows={len(rows)}", flush=True)
    utility_map = load_utility_map(gate_dir / "counterfactual_utilities.jsonl")
    masks = np.load(vt_dir / "channel_masks.npy")
    if masks.shape[0] < len(rows):
        raise RuntimeError(f"channel mask rows {masks.shape[0]} < feature rows {len(rows)}")

    dataset = LearnedGatingDataset(
        str(manifest_path),
        image_size=64,
        bow_dim=256,
        limit=int(args.limit),
        channels=channels,
    )
    if len(dataset) != len(rows):
        raise RuntimeError(f"dataset/rows mismatch: {len(dataset)} vs {len(rows)}")

    out_rows = []
    build_start_ts = time.time()
    total_rows = len(rows)
    print(f"[stage] building VisualEvidence-Set rows total={total_rows}", flush=True)
    for idx, row in enumerate(rows):
        key = (row.get("episode_idx"), row.get("step_idx"), row.get("npz_path", ""))
        utility_rank, counterfactual_utility, utility_rank_records = rank_utilities(utility_map.get(key), channels)
        selected_channels, route_mask = select_channels(
            mask=np.asarray(masks[idx], dtype=np.float32),
            channels=channels,
            utility_rank=utility_rank,
            threshold=float(args.mask_threshold),
        )
        channel_evidence = build_channel_evidence(row, channels)
        stage = str(dataset.records[idx].stage)
        out_rows.append(
            {
                "idx": idx,
                "dataset_name": dataset_name,
                "episode_id": row.get("episode_idx"),
                "episode_idx": row.get("episode_idx"),
                "step_id": row.get("step_idx"),
                "step_idx": row.get("step_idx"),
                "feature_ref": ref_path(row.get("npz_path", "")),
                "npz_path": row.get("npz_path", ""),
                "image_ref": ref_path(row.get("image_path", "")),
                "image_path": row.get("image_path", ""),
                "stage": stage,
                "instruction": row.get("instruction", ""),
                "action": row.get("action", []),
                "selected_evidence": selected_channels,
                "selected_channels": selected_channels,
                "route_mask": route_mask,
                "channel_evidence": channel_evidence,
                "visual_rationale": make_visual_rationale(selected_channels, channel_evidence, stage),
                "action_intent": make_action_intent(row.get("instruction", ""), stage),
                "counterfactual_utility": counterfactual_utility,
                "utility_rank": utility_rank,
                "utility_rank_records": utility_rank_records,
            }
        )
        current = idx + 1
        if current == 1 or current % 2000 == 0 or current == total_rows:
            elapsed = max(1e-6, time.time() - build_start_ts)
            speed = current / elapsed
            remain = max(0, total_rows - current)
            eta = int(remain / max(1e-6, speed))
            filled = int(round((current / max(1, total_rows)) * 20))
            bar = "#" * filled + "-" * (20 - filled)
            print(
                f"[progress] VisualEvidence-Set build [{bar}] {current}/{total_rows} "
                f"({current / max(1, total_rows) * 100:.1f}%) speed={speed:.2f}/s eta={eta}s",
                flush=True,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_summary(out_path.with_suffix(".summary.md"), out_rows, channels)
    print(f"[ok] wrote {out_path} rows={len(out_rows)}")
    print(f"[ok] wrote {out_path.with_suffix('.summary.md')}")


if __name__ == "__main__":
    main()
