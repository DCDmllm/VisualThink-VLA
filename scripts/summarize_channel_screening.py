#!/usr/bin/env python3
"""Summarize evidence-selection and channel-screening evidence for the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CHANNELS = ("bbox", "edge", "motion", "relation", "depth", "caption/query prompt")
SIGNAL_TYPES = {
    "bbox": "localization",
    "edge": "boundary geometry",
    "motion": "temporal change",
    "relation": "language-grounded geometry",
    "depth": "monocular geometry",
    "caption/query prompt": "text-heavy schema",
}
EXTRACTION_COST = {
    "bbox": "medium (OWL-ViT)",
    "edge": "low (Canny)",
    "motion": "low (frame diff)",
    "relation": "low* (derived from det.+instr.)",
    "depth": "high (MiDaS)",
    "caption/query prompt": "high (caption/query VLM)",
}
FINAL_DECISION = {
    "bbox": "keep",
    "edge": "keep",
    "motion": "keep",
    "relation": "keep",
    "depth": "drop",
    "caption/query prompt": "drop",
}


def load_json(path: str) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_feature_markdown(path: str) -> dict[str, float]:
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: dict[str, float] = {}
    header: list[str] | None = None
    for line in lines:
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells:
            continue
        if cells[0] == "Variant":
            header = cells
            continue
        if cells[0].startswith("---"):
            continue
        if header and "Success" in header:
            success_idx = header.index("Success")
            if success_idx < len(cells):
                try:
                    rows[cells[0]] = float(cells[success_idx])
                    continue
                except ValueError:
                    pass
        if len(cells) >= 3:
            for idx in (1, 2):
                try:
                    value = float(cells[idx])
                except ValueError:
                    continue
                if 0.0 <= value <= 1.0:
                    rows[cells[0]] = value
                    break
    return rows


def fmt_pair(a: float | None, b: float | None, scale: float = 100.0, suffix: str = "pts") -> str:
    def one(v: float | None) -> str:
        if v is None:
            return "--"
        return f"{v * scale:+.2f} {suffix}" if suffix else f"{v * scale:.2f}"

    return f"B:{one(a)} / L:{one(b)}"


def fmt_keep(a: float | None, b: float | None) -> str:
    def one(v: float | None) -> str:
        return "--" if v is None else f"{v * 100.0:.1f}%"

    return f"B:{one(a)} / L:{one(b)}"


def index_by_variant(rows: list[dict]) -> dict[str, dict]:
    return {str(row["variant"]): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge_feature_table", required=True)
    parser.add_argument("--libero_feature_table", required=True)
    parser.add_argument("--bridge_recipe_summary", required=True)
    parser.add_argument("--libero_recipe_summary", required=True)
    parser.add_argument("--bridge_prompt_summary", required=True)
    parser.add_argument("--libero_prompt_summary", required=True)
    parser.add_argument("--bridge_depth_summary", required=True)
    parser.add_argument("--libero_depth_summary", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bridge_feature = load_feature_markdown(args.bridge_feature_table)
    libero_feature = load_feature_markdown(args.libero_feature_table)
    bridge_recipe = index_by_variant(load_json(args.bridge_recipe_summary))
    libero_recipe = index_by_variant(load_json(args.libero_recipe_summary))
    bridge_prompt = index_by_variant(load_json(args.bridge_prompt_summary))
    libero_prompt = index_by_variant(load_json(args.libero_prompt_summary))
    bridge_depth = index_by_variant(load_json(args.bridge_depth_summary))
    libero_depth = index_by_variant(load_json(args.libero_depth_summary))

    base_bridge = bridge_recipe.get("blend_0p35", {})
    base_libero = libero_recipe.get("blend_0p35", {})
    depth_bridge = bridge_depth.get("depth5_blend_0p35", {})
    depth_libero = libero_depth.get("depth5_blend_0p35", {})

    feature_drop_keys = {
        "bbox": "without_bbox",
        "edge": "without_edge",
        "motion": "without_motion",
        "relation": "without_relation",
    }

    rows = []
    for channel in CHANNELS:
        bridge_dense = None
        libero_dense = None
        bridge_keep = None
        libero_keep = None
        bridge_comp = None
        libero_comp = None

        prompt_key = "caption_query_only" if channel == "caption/query prompt" else f"{channel}_only"
        if prompt_key in bridge_prompt and "original" in bridge_prompt:
            bridge_dense = bridge_prompt[prompt_key]["success_rate"] - bridge_prompt["original"]["success_rate"]
        if prompt_key in libero_prompt and "original" in libero_prompt:
            libero_dense = libero_prompt[prompt_key]["success_rate"] - libero_prompt["original"]["success_rate"]

        if channel in {"bbox", "edge", "motion", "relation"}:
            bridge_keep = base_bridge.get(f"{channel}_keep_rate")
            libero_keep = base_libero.get(f"{channel}_keep_rate")
            b_full = bridge_feature.get("visualthink_full")
            l_full = libero_feature.get("visualthink_full")
            b_wo = bridge_feature.get(feature_drop_keys[channel])
            l_wo = libero_feature.get(feature_drop_keys[channel])
            bridge_comp = None if b_full is None or b_wo is None else b_full - b_wo
            libero_comp = None if l_full is None or l_wo is None else l_full - l_wo
        elif channel == "depth":
            bridge_keep = depth_bridge.get("depth_keep_rate")
            libero_keep = depth_libero.get("depth_keep_rate")
            bridge_comp = None if not depth_bridge or not base_bridge else depth_bridge["success_rate"] - base_bridge["success_rate"]
            libero_comp = None if not depth_libero or not base_libero else depth_libero["success_rate"] - base_libero["success_rate"]
        else:
            bridge_keep = None
            libero_keep = None
            if "full_schema" in bridge_prompt and "structured_no_text" in bridge_prompt:
                bridge_comp = bridge_prompt["full_schema"]["success_rate"] - bridge_prompt["structured_no_text"]["success_rate"]
            if "full_schema" in libero_prompt and "structured_no_text" in libero_prompt:
                libero_comp = libero_prompt["full_schema"]["success_rate"] - libero_prompt["structured_no_text"]["success_rate"]

        rows.append(
            {
                "channel": channel,
                "signal_type": SIGNAL_TYPES[channel],
                "dense_gain": fmt_pair(bridge_dense, libero_dense),
                "sparse_keep_rate": "prompt-only / always-on" if channel == "caption/query prompt" else fmt_keep(bridge_keep, libero_keep),
                "extraction_cost": EXTRACTION_COST[channel],
                "complementarity": fmt_pair(bridge_comp, libero_comp),
                "final_decision": FINAL_DECISION[channel],
            }
        )

    (out_dir / "channel_screening_summary.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "notes": {
                    "dense_gain": "single-channel prompt evidence success delta against original OpenVLA prompt on fixed subsets",
                    "sparse_keep_rate": "4-channel baseline uses blend_0p35 recipe checkpoints; depth uses new 5-channel blend_0p35 checkpoint",
                    "complementarity": "retained channels use controlled leave-one-out deltas; depth uses add-depth delta vs 4-channel baseline; text uses full-schema vs structured-no-text delta",
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "| Channel | Signal type | Dense gain | Sparse keep rate | Extraction cost | Complementarity | Final decision |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['channel']} | {row['signal_type']} | {row['dense_gain']} | {row['sparse_keep_rate']} | "
            f"{row['extraction_cost']} | {row['complementarity']} | {row['final_decision']} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `B` denotes BridgeData V2; `L` denotes LIBERO-Long.",
            "- `Dense gain` is computed from single-channel prompt evidence on fixed subsets.",
            "- `Complementarity` uses existing controlled feature ablations for retained channels, add-depth delta for depth, and full-schema minus structured-no-text for caption/query prompt.",
        ]
    )
    (out_dir / "channel_screening_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
