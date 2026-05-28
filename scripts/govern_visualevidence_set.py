#!/usr/bin/env python3
"""Classify and govern VisualEvidence-Set JSONL rows.

This script adds a deterministic taxonomy and quality-governance layer on top
of an existing VisualEvidence-Set JSONL. It intentionally avoids heavyweight model
dependencies so it can run as a first-pass audit on large trace exports.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

CHANNELS = ("bbox", "edge", "motion", "relation")
GOVERNANCE_VERSION = "visualevidence_set_governance_v1"
DEFAULT_DIFFICULTY_THRESHOLDS = (0.3312, 0.4515, 0.6526, 0.7509)

RELATION_WORDS = {
    "left",
    "right",
    "front",
    "behind",
    "back",
    "rear",
    "near",
    "next",
    "between",
    "under",
    "over",
    "above",
    "below",
    "inside",
    "into",
    "in",
    "on",
    "onto",
    "beside",
    "around",
    "toward",
    "towards",
}

PRIMITIVE_PATTERNS = (
    ("open", r"\b(open|unseal|uncover)\b"),
    ("close", r"\b(close|shut|cover)\b"),
    ("push", r"\b(push|press|nudge)\b"),
    ("pull", r"\b(pull|drag)\b"),
    ("slide", r"\b(slide)\b"),
    ("stack", r"\b(stack|pile)\b"),
    ("fold", r"\b(fold)\b"),
    ("wipe", r"\b(wipe|clean)\b"),
    ("pick", r"\b(pick|grasp|grab|take|lift|hold|grip)\b"),
    ("place", r"\b(place|put|insert|drop|release|deposit)\b"),
    ("rearrange", r"\b(move|relocate|transfer|rearrange|sort)\b"),
    ("inspect", r"\b(inspect|look|find|locate|spot|seek)\b"),
)

APPROVED_LEGAL_SOURCES = {
    "bridge",
    "bridge_orig",
    "libero",
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
    "roboturk",
    "utaustin_mutex",
    "viola",
}


def load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError(f"empty input trace: {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", (text or "").lower())


def source_group(dataset_name: str) -> str:
    name = (dataset_name or "unknown").lower()
    if name.startswith("bridge"):
        return "bridge"
    if name.startswith("libero"):
        return "libero"
    if name.startswith("fractal"):
        return "fractal"
    if name.startswith("roboturk"):
        return "roboturk"
    if name.startswith("utaustin_mutex"):
        return "utaustin_mutex"
    if name.startswith("viola"):
        return "viola"
    return name or "unknown"


def legal_status(dataset_name: str) -> str:
    src = source_group(dataset_name)
    if src in APPROVED_LEGAL_SOURCES:
        return "approved"
    if src == "fractal":
        return "review"
    return "review"


def infer_primitive(instruction: str, stage: str) -> tuple[str, list[str]]:
    text = (instruction or "").lower()
    found: list[str] = []
    for label, pattern in PRIMITIVE_PATTERNS:
        if re.search(pattern, text):
            found.append(label)
    if not found:
        return "other", []
    if stage == "grasp" and "pick" in found:
        primary = "pick"
    elif stage == "place" and "place" in found:
        primary = "place"
    else:
        primary = found[0]
    secondary = [x for x in found if x != primary]
    return primary, secondary


def infer_relation_topology(instruction: str, primitive: str) -> str:
    toks = tokenize(instruction)
    rel_count = sum(1 for t in toks if t in RELATION_WORDS)
    object_like = sum(1 for t in toks if t in {"object", "cup", "bowl", "block", "drawer", "door", "plate", "caddy", "cloth"})
    if primitive in {"open", "close"}:
        return "articulated_object"
    if object_like >= 2 and rel_count >= 2:
        return "sequential_multi_object"
    if any(t in {"region", "area", "bin", "container", "box", "caddy", "drawer"} for t in toks):
        return "object_to_region"
    if rel_count > 0:
        return "relative_spatial"
    if object_like >= 2:
        return "object_to_object"
    return "single_target"


def instruction_complexity(instruction: str) -> tuple[str, float]:
    toks = tokenize(instruction)
    rel_count = sum(1 for t in toks if t in RELATION_WORDS)
    conjunctions = sum(1 for t in toks if t in {"and", "then", "after", "before", "while"})
    if len(toks) < 3:
        return "underspecified", 1.0
    if len(toks) > 40 or conjunctions >= 2:
        return "compositional", min(1.0, 0.75 + 0.02 * conjunctions)
    if rel_count > 0:
        return "relational", min(1.0, 0.55 + 0.08 * rel_count)
    if any(t in {"red", "blue", "green", "yellow", "left", "right", "small", "large"} for t in toks):
        return "attribute_grounded", 0.45
    return "minimal_imperative", 0.25


def route_mask_to_selected(route_mask: dict[str, Any], channels: tuple[str, ...]) -> list[str]:
    selected = []
    for ch in channels:
        try:
            active = float(route_mask.get(ch, 0)) > 0.5
        except Exception:
            active = bool(route_mask.get(ch, False))
        if active:
            selected.append(ch)
    return selected


def utility_values(row: dict[str, Any], channels: tuple[str, ...]) -> dict[str, float]:
    raw = row.get("counterfactual_utility") or {}
    return {ch: float(raw.get(ch, 0.0) or 0.0) for ch in channels}


def utility_rank_channels(row: dict[str, Any]) -> list[str]:
    records = row.get("utility_rank_records")
    if isinstance(records, list) and records:
        out = []
        for item in records:
            if isinstance(item, dict) and "channel" in item:
                out.append(str(item["channel"]))
            elif isinstance(item, str):
                out.append(item)
        return out
    rank = row.get("utility_rank")
    if isinstance(rank, list):
        out = []
        for item in rank:
            if isinstance(item, dict) and "channel" in item:
                out.append(str(item["channel"]))
            elif isinstance(item, str):
                out.append(item)
        return out
    return []


def kendall_tau(order_a: list[str], order_b: list[str], channels: tuple[str, ...]) -> float:
    a = [x for x in order_a if x in channels]
    b = [x for x in order_b if x in channels]
    if len(a) < 2 or len(b) < 2:
        return 1.0
    common = [ch for ch in channels if ch in a and ch in b]
    if len(common) < 2:
        return 1.0
    pos_a = {ch: a.index(ch) for ch in common}
    pos_b = {ch: b.index(ch) for ch in common}
    concordant = 0
    discordant = 0
    for i, ch_i in enumerate(common):
        for ch_j in common[i + 1 :]:
            sign_a = pos_a[ch_i] - pos_a[ch_j]
            sign_b = pos_b[ch_i] - pos_b[ch_j]
            if sign_a * sign_b > 0:
                concordant += 1
            elif sign_a * sign_b < 0:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return 1.0
    return float((concordant - discordant) / denom)


def evidence_dependency(row: dict[str, Any], channels: tuple[str, ...]) -> tuple[str, str, float, float, float]:
    values = utility_values(row, channels)
    positives = {ch: max(0.0, values[ch]) for ch in channels}
    total = sum(positives.values())
    route_count = len(route_mask_to_selected(row.get("route_mask") or {}, channels))
    sparsity = {0: "none", 1: "mono", 2: "bi", 3: "tri"}.get(route_count, "quad")
    if total <= 1e-8:
        return "weak_evidence", sparsity, 0.0, 0.0, 0.0
    ranked = sorted(((ch, val / total) for ch, val in positives.items()), key=lambda x: (-x[1], x[0]))
    top_ch, top_share = ranked[0]
    second_share = ranked[1][1] if len(ranked) > 1 else 0.0
    gap = top_share - second_share
    entropy = -sum(p * math.log(max(p, 1e-12)) for _, p in ranked) / math.log(len(channels))
    if top_share >= 0.55 and gap >= 0.15:
        return f"{top_ch}_dominant", sparsity, float(top_share), float(gap), float(entropy)
    return "balanced", sparsity, float(top_share), float(gap), float(entropy)


def action_norm(row: dict[str, Any]) -> float:
    action = row.get("action")
    if not isinstance(action, list) or not action:
        return 0.0
    try:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
    except Exception:
        return 0.0
    return float(np.linalg.norm(arr))


def episode_stats(rows: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    stats: dict[Any, dict[str, Any]] = {}
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("episode_id", row.get("episode_idx"))].append(row)
    for ep, items in grouped.items():
        stages = {str(item.get("stage", "unknown")) for item in items}
        stats[ep] = {
            "length": len(items),
            "stage_count": len(stages),
        }
    return stats


def temporal_span(length: int, stage_count: int) -> str:
    if stage_count >= 4 and length > 60:
        return "multi_stage_long"
    if length <= 10:
        return "atomic"
    if length <= 25:
        return "short"
    if length <= 60:
        return "medium"
    return "long"


def parse_difficulty_thresholds(raw: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    except Exception as exc:
        raise RuntimeError(f"failed to parse difficulty thresholds: {raw}") from exc
    if len(values) != 4:
        raise RuntimeError(
            f"difficulty thresholds must contain exactly 4 comma-separated floats, got: {raw}"
        )
    if not all(0.0 <= x <= 1.0 for x in values):
        raise RuntimeError(f"difficulty thresholds must be within [0, 1], got: {raw}")
    if not (values[0] < values[1] < values[2] < values[3]):
        raise RuntimeError(f"difficulty thresholds must be strictly increasing, got: {raw}")
    return values


def difficulty(
    row: dict[str, Any],
    ep_stat: dict[str, Any],
    utility_entropy: float,
    thresholds: tuple[float, float, float, float],
) -> tuple[float, str]:
    instruction = str(row.get("instruction", ""))
    toks = tokenize(instruction)
    rel_count = sum(1 for t in toks if t in RELATION_WORDS)
    _, instr_score = instruction_complexity(instruction)
    horizon = min(1.0, float(ep_stat.get("length", 1)) / 80.0)
    stage_count = min(1.0, float(ep_stat.get("stage_count", 1)) / 4.0)
    relation = min(1.0, rel_count / 4.0)
    selected_count = len(route_mask_to_selected(row.get("route_mask") or {}, CHANNELS))
    route_complexity = selected_count / len(CHANNELS)
    score = (
        0.22 * horizon
        + 0.15 * stage_count
        + 0.18 * relation
        + 0.15 * instr_score
        + 0.15 * utility_entropy
        + 0.15 * route_complexity
    )
    if score < thresholds[0]:
        level = "L1"
    elif score < thresholds[1]:
        level = "L2"
    elif score < thresholds[2]:
        level = "L3"
    elif score < thresholds[3]:
        level = "L4"
    else:
        level = "L5"
    return float(score), level


def rationale_mention_recall(row: dict[str, Any], selected: list[str]) -> float:
    if not selected:
        return 1.0
    text = (
        str(row.get("visual_rationale", ""))
        + " "
        + json.dumps(row.get("channel_evidence", {}), ensure_ascii=False).lower()
    ).lower()
    mentioned = sum(1 for ch in selected if ch.lower() in text)
    return float(mentioned / max(1, len(selected)))


def path_ok(path_value: str, root: Path) -> bool:
    if not path_value:
        return False
    path = Path(path_value)
    if path.is_file():
        return True
    candidate = root / path
    return candidate.is_file()


def validate_and_score(
    row: dict[str, Any],
    channels: tuple[str, ...],
    root: Path,
    check_files: bool,
    enforce_legal_review: bool,
) -> tuple[dict[str, Any], float, str, list[str], float]:
    flags: list[str] = []
    required = [
        "dataset_name",
        "episode_id",
        "step_id",
        "stage",
        "instruction",
        "action",
        "selected_evidence",
        "route_mask",
        "counterfactual_utility",
        "utility_rank",
        "visual_rationale",
    ]
    for key in required:
        if key not in row:
            flags.append(f"missing_{key}")

    route_mask = row.get("route_mask") if isinstance(row.get("route_mask"), dict) else {}
    if set(route_mask.keys()) != set(channels):
        flags.append("bad_route_keys")
    utility = row.get("counterfactual_utility") if isinstance(row.get("counterfactual_utility"), dict) else {}
    if set(utility.keys()) != set(channels):
        flags.append("bad_utility_keys")

    action = row.get("action")
    if not isinstance(action, list) or len(action) != 7:
        flags.append("bad_action_dim")

    selected_from_route = sorted(route_mask_to_selected(route_mask, channels))
    selected = sorted([str(x) for x in row.get("selected_evidence", [])]) if isinstance(row.get("selected_evidence"), list) else []
    if selected != selected_from_route:
        flags.append("route_selected_mismatch")

    ranked_by_utility = [ch for ch, _ in sorted(utility_values(row, channels).items(), key=lambda x: (-x[1], x[0]))]
    rank_tau = kendall_tau(utility_rank_channels(row), ranked_by_utility, channels)
    if rank_tau < 0.5:
        flags.append("rank_conflict_fatal")
    elif rank_tau < 0.8:
        flags.append("rank_conflict_repairable")

    mention_recall = rationale_mention_recall(row, selected_from_route or selected)
    if mention_recall < 0.5:
        flags.append("rationale_route_low_recall")

    toks = tokenize(str(row.get("instruction", "")))
    if len(toks) < 3:
        flags.append("underspecified_instruction")
    if len(toks) > 80:
        flags.append("overlong_instruction")

    if action_norm(row) < 0.01:
        flags.append("low_action_magnitude")

    file_ok = True
    if check_files:
        image_ref = str(row.get("image_path") or row.get("image_ref") or "")
        feature_ref = str(row.get("npz_path") or row.get("feature_ref") or "")
        image_ok = path_ok(image_ref, root)
        feature_ok = path_ok(feature_ref, root)
        file_ok = image_ok and feature_ok
        if not image_ok:
            flags.append("missing_image_file")
        if not feature_ok:
            flags.append("missing_feature_file")

    legal = legal_status(str(row.get("dataset_name", "")))
    if legal == "review":
        flags.append("legal_review_source")

    quality = 1.0
    penalties = {
        "bad_route_keys": 0.50,
        "bad_utility_keys": 0.50,
        "bad_action_dim": 0.55,
        "rank_conflict_fatal": 0.45,
        "rank_conflict_repairable": 0.12,
        "route_selected_mismatch": 0.14,
        "rationale_route_low_recall": 0.12,
        "underspecified_instruction": 0.10,
        "overlong_instruction": 0.08,
        "low_action_magnitude": 0.05,
        "missing_image_file": 0.30,
        "missing_feature_file": 0.30,
    }
    for flag in flags:
        if flag.startswith("missing_") and flag not in penalties:
            quality -= 0.40
        else:
            quality -= penalties.get(flag, 0.0)
    quality = float(max(0.0, min(1.0, quality)))

    fatal = any(
        flag.startswith("missing_") and flag not in {"missing_image_file", "missing_feature_file"}
        for flag in flags
    ) or any(flag in flags for flag in {"bad_route_keys", "bad_utility_keys", "bad_action_dim", "rank_conflict_fatal"})
    if check_files and not file_ok:
        fatal = True

    if enforce_legal_review and legal == "review":
        decision = "keep_internal"
    elif fatal or quality < 0.40:
        decision = "drop"
    elif any(flag in flags for flag in {"route_selected_mismatch", "rank_conflict_repairable", "rationale_route_low_recall"}):
        decision = "relabel"
    elif quality < 0.80:
        decision = "keep_weighted"
    else:
        decision = "keep_release"

    details = {
        "rank_tau": rank_tau,
        "rationale_route_recall": mention_recall,
        "file_ok": file_ok,
        "legal_status": legal,
        "selected_from_route": selected_from_route,
    }
    return details, quality, decision, flags, mention_recall


def governance_row(
    row: dict[str, Any],
    ep_stat: dict[str, Any],
    channels: tuple[str, ...],
    difficulty_thresholds: tuple[float, float, float, float],
    root: Path,
    check_files: bool,
    enforce_legal_review: bool,
    repair_selected: bool,
) -> dict[str, Any]:
    dep, sparsity, top_share, utility_gap, utility_entropy = evidence_dependency(row, channels)
    stage = str(row.get("stage", "unknown"))
    primitive, secondary = infer_primitive(str(row.get("instruction", "")), stage)
    instr_complexity, instr_score = instruction_complexity(str(row.get("instruction", "")))
    diff_score, diff_level = difficulty(row, ep_stat, utility_entropy, difficulty_thresholds)
    details, quality, decision, flags, _ = validate_and_score(
        row=row,
        channels=channels,
        root=root,
        check_files=check_files,
        enforce_legal_review=enforce_legal_review,
    )
    if repair_selected and "route_selected_mismatch" in flags and details["selected_from_route"]:
        row = dict(row)
        row["selected_evidence"] = details["selected_from_route"]
        row["selected_channels"] = details["selected_from_route"]

    length = int(ep_stat.get("length", 1))
    stage_count = int(ep_stat.get("stage_count", 1))
    taxonomy = {
        "source_group": source_group(str(row.get("dataset_name", ""))),
        "stage": stage,
        "primary_primitive": primitive,
        "secondary_primitives": secondary,
        "relation_topology": infer_relation_topology(str(row.get("instruction", "")), primitive),
        "dominant_evidence": dep,
        "evidence_sparsity": sparsity,
        "difficulty_score": diff_score,
        "difficulty_level": diff_level,
        "temporal_span": temporal_span(length, stage_count),
        "instruction_complexity": instr_complexity,
        "instruction_complexity_score": instr_score,
    }
    keep_weight = 1.0 if decision == "keep_release" else max(0.1, round(quality, 4)) if decision == "keep_weighted" else 0.0
    governance = {
        "version": GOVERNANCE_VERSION,
        "taxonomy": taxonomy,
        "quality": {
            "quality_score": quality,
            "decision": decision,
            "keep_weight": keep_weight,
            "flags": flags,
            "rank_tau": details["rank_tau"],
            "rationale_route_recall": details["rationale_route_recall"],
            "action_norm": action_norm(row),
            "utility_top_share": top_share,
            "utility_gap": utility_gap,
            "utility_entropy": utility_entropy,
            "file_ok": details["file_ok"],
            "legal_status": details["legal_status"],
            "label_origin": "rule",
            "label_confidence": quality,
        },
    }
    out = dict(row)
    out["governance"] = governance
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decision = Counter(row["governance"]["quality"]["decision"] for row in rows)
    by_source = Counter(row["governance"]["taxonomy"]["source_group"] for row in rows)
    by_stage = Counter(row["governance"]["taxonomy"]["stage"] for row in rows)
    by_primitive = Counter(row["governance"]["taxonomy"]["primary_primitive"] for row in rows)
    by_evidence = Counter(row["governance"]["taxonomy"]["dominant_evidence"] for row in rows)
    by_difficulty = Counter(row["governance"]["taxonomy"]["difficulty_level"] for row in rows)
    quality_scores = [row["governance"]["quality"]["quality_score"] for row in rows]
    return {
        "num_rows": len(rows),
        "decision": dict(sorted(decision.items())),
        "source_group": dict(sorted(by_source.items())),
        "stage": dict(sorted(by_stage.items())),
        "primary_primitive": dict(sorted(by_primitive.items())),
        "dominant_evidence": dict(sorted(by_evidence.items())),
        "difficulty_level": dict(sorted(by_difficulty.items())),
        "quality_mean": float(np.mean(quality_scores)) if quality_scores else 0.0,
        "quality_p10": float(np.quantile(quality_scores, 0.10)) if quality_scores else 0.0,
        "quality_p50": float(np.quantile(quality_scores, 0.50)) if quality_scores else 0.0,
        "quality_p90": float(np.quantile(quality_scores, 0.90)) if quality_scores else 0.0,
    }


def write_summary_markdown(path: Path, summary: dict[str, Any], output_files: dict[str, str]) -> None:
    lines = [
        "# VisualEvidence-Set Governance Summary",
        "",
        "## Outputs",
        "",
    ]
    for name, file_path in output_files.items():
        lines.append(f"- `{name}`: `{file_path}`")
    lines.extend(
        [
            "",
            "## Core Metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| num_rows | {summary['num_rows']} |",
            f"| quality_mean | {summary['quality_mean']:.4f} |",
            f"| quality_p10 | {summary['quality_p10']:.4f} |",
            f"| quality_p50 | {summary['quality_p50']:.4f} |",
            f"| quality_p90 | {summary['quality_p90']:.4f} |",
            "",
        ]
    )
    for section in ["decision", "source_group", "stage", "primary_primitive", "dominant_evidence", "difficulty_level"]:
        lines.extend([f"## {section}", "", "| Label | Count |", "|---|---:|"])
        for key, value in summary[section].items():
            lines.append(f"| `{key}` | {value} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_trace", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--channels", default=",".join(CHANNELS))
    parser.add_argument("--check_files", action="store_true")
    parser.add_argument("--enforce_legal_review", action="store_true")
    parser.add_argument("--repair_selected", action="store_true")
    parser.add_argument("--hq_threshold", type=float, default=0.80)
    parser.add_argument("--gold_threshold", type=float, default=0.90)
    parser.add_argument(
        "--difficulty_thresholds",
        default=",".join(str(x) for x in DEFAULT_DIFFICULTY_THRESHOLDS),
        help="Comma-separated score thresholds for L1/L2/L3/L4 boundaries.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_trace)
    out_dir = Path(args.output_dir)
    root = Path(__file__).resolve().parents[1]
    channels = tuple(ch.strip() for ch in args.channels.split(",") if ch.strip())
    difficulty_thresholds = parse_difficulty_thresholds(args.difficulty_thresholds)
    if not channels:
        raise RuntimeError("--channels resolved to empty list")
    rows = load_jsonl(input_path, limit=int(args.limit))
    ep_stats = episode_stats(rows)

    governed: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        ep_key = row.get("episode_id", row.get("episode_idx"))
        governed.append(
            governance_row(
                row=row,
                ep_stat=ep_stats.get(ep_key, {"length": 1, "stage_count": 1}),
                channels=channels,
                difficulty_thresholds=difficulty_thresholds,
                root=root,
                check_files=bool(args.check_files),
                enforce_legal_review=bool(args.enforce_legal_review),
                repair_selected=bool(args.repair_selected),
            )
        )
        if (idx + 1) == 1 or (idx + 1) % 50000 == 0 or (idx + 1) == len(rows):
            print(f"[progress] governed {idx + 1}/{len(rows)}", flush=True)

    full_clean = [
        row
        for row in governed
        if row["governance"]["quality"]["decision"] in {"keep_release", "keep_weighted"}
    ]
    review_queue = [
        row
        for row in governed
        if row["governance"]["quality"]["decision"] in {"relabel", "keep_internal"}
    ]
    hq_trace = [
        row
        for row in full_clean
        if row["governance"]["quality"]["quality_score"] >= float(args.hq_threshold)
        and row["governance"]["quality"]["rationale_route_recall"] >= 0.5
    ]
    gold = [
        row
        for row in hq_trace
        if row["governance"]["quality"]["quality_score"] >= float(args.gold_threshold)
        and row["governance"]["quality"]["legal_status"] == "approved"
        and not row["governance"]["quality"]["flags"]
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "governed": str(out_dir / "evidence_trace.governed.jsonl"),
        "full_clean": str(out_dir / "release_full_clean.jsonl"),
        "hq_trace": str(out_dir / "release_hq_trace.jsonl"),
        "gold_faithfulness": str(out_dir / "release_gold_faithfulness.jsonl"),
        "review_queue": str(out_dir / "review_queue.jsonl"),
        "label_distribution": str(out_dir / "label_distribution.json"),
        "summary": str(out_dir / "quality_summary.md"),
    }
    write_jsonl(Path(files["governed"]), governed)
    write_jsonl(Path(files["full_clean"]), full_clean)
    write_jsonl(Path(files["hq_trace"]), hq_trace)
    write_jsonl(Path(files["gold_faithfulness"]), gold)
    write_jsonl(Path(files["review_queue"]), review_queue)
    summary = summarize(governed)
    summary["release_counts"] = {
        "full_clean": len(full_clean),
        "hq_trace": len(hq_trace),
        "gold_faithfulness": len(gold),
        "review_queue": len(review_queue),
    }
    Path(files["label_distribution"]).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_markdown(Path(files["summary"]), summary, files)
    print(f"[ok] governed={files['governed']} rows={len(governed)}", flush=True)
    print(f"[ok] full_clean={len(full_clean)} hq_trace={len(hq_trace)} gold={len(gold)} review={len(review_queue)}", flush=True)
    print(f"[ok] summary={files['summary']}", flush=True)


if __name__ == "__main__":
    main()
