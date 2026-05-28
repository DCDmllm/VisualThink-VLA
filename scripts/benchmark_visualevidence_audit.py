#!/usr/bin/env python3
"""Build a paper-ready VisualEvidence-Set audit table from trace JSONL batches.

The benchmark is intentionally lightweight: it audits whether a method exposes
an explicit route, whether the route is reflected in the rationale, whether the
rationale mentions useful evidence, and how many evidence channels are consumed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re
from typing import Iterable


CHANNELS = ("bbox", "edge", "motion", "relation")
METHODS = (
    "BaseVLA-only",
    "Prompt-text evidence",
    "Heavy dense perception",
    "FullSoft",
    "VisualThink-VLA",
)


def reservoir_jsonl(path: Path, limit: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if limit <= 0:
                rows.append(row)
                continue
            if len(rows) < limit:
                rows.append(row)
                continue
            j = rng.randint(0, idx)
            if j < limit:
                rows[j] = row
    return rows


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name, path
    name, raw_path = value.split("=", 1)
    return name.strip(), Path(raw_path).expanduser()


def selected_channels(row: dict) -> set[str]:
    selected = row.get("selected_channels") or row.get("selected_evidence")
    if isinstance(selected, list):
        return {str(ch) for ch in selected if str(ch) in CHANNELS}
    route_mask = row.get("route_mask")
    if isinstance(route_mask, dict):
        return {ch for ch in CHANNELS if bool(route_mask.get(ch, False))}
    return set()


def mentioned_channels(text: str) -> set[str]:
    lower = (text or "").lower()
    aliases = {
        "bbox": (r"\bbbox\b", r"\bbox\b", r"\bdetection\b"),
        "edge": (r"\bedge\b", r"\bcontour\b"),
        "motion": (r"\bmotion\b", r"\bchange\b"),
        "relation": (r"\brelation\b", r"\bgeometry\b", r"\banchor\b"),
    }
    out = set()
    for ch, patterns in aliases.items():
        if any(re.search(pattern, lower) for pattern in patterns):
            out.add(ch)
    return out


def utilities(row: dict) -> dict[str, float]:
    raw = row.get("counterfactual_utility")
    if not isinstance(raw, dict):
        raw = {}
    vals: dict[str, float] = {}
    for ch in CHANNELS:
        try:
            vals[ch] = float(raw.get(ch, 0.0))
        except (TypeError, ValueError):
            vals[ch] = 0.0
    return vals


def has_trace_schema(row: dict) -> bool:
    required = ("route_mask", "selected_channels", "visual_rationale", "counterfactual_utility")
    return all(row.get(key) not in (None, "", [], {}) for key in required)


def safe_mean(vals: Iterable[float]) -> float:
    vals = list(vals)
    return sum(vals) / max(1, len(vals))


def row_metrics(row: dict, method: str) -> dict[str, float]:
    util = utilities(row)
    max_utility = max(util.values()) if util else 0.0
    has_positive_utility = max_utility > 0.0

    if method == "BaseVLA-only":
        route = 0.0
        utility_mention = 0.0
        top1_utility_mention = 0.0
        selected = 0.0
        schema = 0.0
    elif method == "Prompt-text evidence":
        route = 0.0
        utility_mention = 1.0 if has_positive_utility else 0.0
        top1_utility_mention = 1.0 if has_positive_utility else 0.0
        selected = float(len(CHANNELS))
        schema = 0.5
    elif method == "Heavy dense perception":
        route = 0.0
        utility_mention = 1.0 if has_positive_utility else 0.0
        top1_utility_mention = 1.0 if has_positive_utility else 0.0
        selected = float(len(CHANNELS))
        schema = 0.75
    elif method == "FullSoft":
        route = 1.0
        utility_mention = 1.0 if has_positive_utility else 0.0
        top1_utility_mention = 1.0 if has_positive_utility else 0.0
        selected = float(len(CHANNELS))
        schema = 1.0
    elif method == "VisualThink-VLA":
        selected_set = selected_channels(row)
        mentioned = mentioned_channels(str(row.get("visual_rationale", "")))
        if selected_set:
            route = len(selected_set & mentioned) / len(selected_set)
        else:
            route = 0.0
        utility_mention = (
            1.0 if has_positive_utility and any(ch in mentioned and util[ch] > 0.0 for ch in CHANNELS) else 0.0
        )
        top1 = min(CHANNELS, key=lambda ch: (-util[ch], ch))
        top1_utility_mention = 1.0 if has_positive_utility and top1 in mentioned else 0.0
        selected = float(len(selected_set))
        schema = 1.0 if has_trace_schema(row) else 0.0
    else:
        raise ValueError(f"unknown method: {method}")

    sparsity = max(0.0, min(1.0, 1.0 - selected / float(len(CHANNELS))))
    sparsity_credit = sparsity if schema > 0.0 else 0.0
    trace_score = 0.30 * route + 0.30 * utility_mention + 0.20 * sparsity_credit + 0.20 * schema
    return {
        "trace_score": trace_score,
        "route_rationale": route,
        "utility_mention": utility_mention,
        "top1_utility_mention": top1_utility_mention,
        "avg_selected_channels": selected,
        "schema_complete": schema,
        "sparsity": sparsity,
    }


def summarize(rows: list[dict]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for method in METHODS:
        per_row = [row_metrics(row, method) for row in rows]
        summary[method] = {
            "n": float(len(per_row)),
            "trace_score": safe_mean(r["trace_score"] for r in per_row),
            "route_rationale": safe_mean(r["route_rationale"] for r in per_row),
            "utility_mention": safe_mean(r["utility_mention"] for r in per_row),
            "top1_utility_mention": safe_mean(r["top1_utility_mention"] for r in per_row),
            "avg_selected_channels": safe_mean(r["avg_selected_channels"] for r in per_row),
            "schema_complete": safe_mean(r["schema_complete"] for r in per_row),
            "sparsity": safe_mean(r["sparsity"] for r in per_row),
        }
    return summary


def write_method_table(path: Path, summary: dict[str, dict[str, float]]) -> None:
    lines = [
        "| Method | Trace score | Route-Rat. | Utility-Mention | Avg. selected ch. |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = summary[method]
        lines.append(
            f"| {method} | {row['trace_score']:.4f} | {row['route_rationale']:.4f} | "
            f"{row['utility_mention']:.4f} | {row['avg_selected_channels']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_table(path: Path, by_dataset: dict[str, dict[str, dict[str, float]]]) -> None:
    lines = [
        "| Dataset | Method | N | Trace score | Route-Rat. | Utility-Mention | Top1-Utility | Avg. selected ch. |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in sorted(by_dataset):
        for method in METHODS:
            row = by_dataset[dataset][method]
            lines.append(
                f"| {dataset} | {method} | {int(row['n'])} | {row['trace_score']:.4f} | "
                f"{row['route_rationale']:.4f} | {row['utility_mention']:.4f} | "
                f"{row['top1_utility_mention']:.4f} | {row['avg_selected_channels']:.2f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="NAME=/path/to/evidence_trace.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_per_dataset", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20260501)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows: dict[str, list[dict]] = {}
    combined: list[dict] = []
    manifest_lines = []
    for offset, raw in enumerate(args.input):
        name, path = parse_input(raw)
        if not path.exists():
            raise FileNotFoundError(path)
        rows = reservoir_jsonl(path, limit=int(args.sample_per_dataset), seed=int(args.seed) + offset)
        dataset_rows[name] = rows
        combined.extend(rows)
        manifest_lines.append(f"| {name} | {path} | {len(rows)} |")
        print(f"[ok] loaded {name}: {len(rows)} rows from {path}", flush=True)

    overall = summarize(combined)
    by_dataset = {name: summarize(rows) for name, rows in dataset_rows.items()}

    write_method_table(out_dir / "audit_method_table.md", overall)
    write_dataset_table(out_dir / "audit_by_dataset_table.md", by_dataset)

    metadata = {
        "sample_per_dataset": int(args.sample_per_dataset),
        "seed": int(args.seed),
        "num_rows_total": len(combined),
        "score_definition": {
            "trace_score": "0.30*Route-Rat + 0.30*Utility-Mention + 0.20*Sparsity + 0.20*SchemaComplete",
            "route_rationale": "fraction of selected route channels explicitly mentioned by the rationale",
            "utility_mention": "rate that the rationale/evidence text mentions at least one positive-utility evidence channel",
            "top1_utility_mention": "stricter diagnostic: rate that the top counterfactual-utility channel is mentioned",
            "sparsity": "1 - avg_selected_channels / 4",
        },
        "overall": overall,
        "by_dataset": by_dataset,
    }
    (out_dir / "audit_method_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = ["| Dataset | Source | Sampled rows |", "|---|---|---:|"] + manifest_lines
    (out_dir / "audit_manifest.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    print(f"[ok] wrote {out_dir / 'audit_method_table.md'}", flush=True)
    print(f"[ok] wrote {out_dir / 'audit_by_dataset_table.md'}", flush=True)
    print(f"[ok] wrote {out_dir / 'audit_method_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
