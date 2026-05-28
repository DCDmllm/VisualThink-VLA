#!/usr/bin/env python3
"""Normalize offline OpenVLA / FullSoft / VisualThink-VLA outputs into paper-metrics format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict[str, Any]], channels: tuple[str, ...]) -> dict[str, Any]:
    success = [1.0 if row.get("success") else 0.0 for row in rows]
    latencies = [float(row.get("inference_time_s", 0.0)) for row in rows]
    l1s = [float(row.get("l1", 0.0)) for row in rows if "l1" in row]
    selected = [float(row.get("selected_channels", 0.0)) for row in rows]
    out = {
        "n": int(len(rows)),
        "success_rate": float(np.mean(success)) if success else 0.0,
        "avg_completion_time_s": None,
        "timeout_penalized_completion_time_s": None,
        "avg_step_latency_s": float(np.mean(latencies)) if latencies else None,
        "avg_selected_channels": float(np.mean(selected)) if selected else 0.0,
        "avg_l1": float(np.mean(l1s)) if l1s else None,
        "median_l1": float(np.median(l1s)) if l1s else None,
    }
    for ch in channels:
        out[f"{ch}_keep_rate"] = float(
            np.mean([1.0 if row.get("gates", {}).get(ch, False) else 0.0 for row in rows])
        ) if rows else 0.0
    return out


def summarize_episode_proxy(rows: list[dict[str, Any]], channels: tuple[str, ...]) -> dict[str, Any]:
    episodes: dict[int, list[dict[str, Any]]] = {}
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

    completion = []
    success_completion = []
    success_flags = []
    keep_rates = {ch: [] for ch in channels}
    for episode_rows in episodes.values():
        ordered = sorted(episode_rows, key=lambda item: int(item.get("step_idx", item.get("idx", 0))))
        comp = float(sum(float(row.get("inference_time_s", 0.0)) for row in ordered))
        success = all(bool(row.get("success", False)) for row in ordered)
        completion.append(comp)
        success_flags.append(1.0 if success else 0.0)
        if success:
            success_completion.append(comp)
        for ch in channels:
            keep_rates[ch].append(
                float(np.mean([1.0 if row.get("gates", {}).get(ch, False) else 0.0 for row in ordered]))
            )
    out = {
        "n_episodes": int(len(episodes)),
        "success_rate_episode_proxy": float(np.mean(success_flags)),
        "avg_completion_time_s": float(np.mean(success_completion)) if success_completion else None,
        "timeout_penalized_completion_time_s": float(np.mean(completion)) if completion else None,
    }
    for ch in channels:
        out[f"{ch}_keep_rate_episode"] = float(np.mean(keep_rates[ch])) if keep_rates[ch] else 0.0
    return out


def fmt(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--channels", default="bbox,edge,motion,relation")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    channels = tuple(ch.strip() for ch in args.channels.split(",") if ch.strip())

    model_rows = {
        "openvla": load_jsonl(input_dir / "openvla_eval.jsonl"),
        "full_soft": load_jsonl(input_dir / "fullsoft_eval.jsonl"),
        "visualthink_vla": load_jsonl(input_dir / "visualthink_vla_eval.jsonl"),
    }
    summary = {
        "dataset": args.dataset,
        "run_name": args.run_name,
        "benchmark_type": "offline_action_prediction",
        "completion_metrics_available": True,
        "completion_time_definition": "proxy_sum_of_step_inference_latency_per_episode",
        "episode_success_definition": "all_steps_success_under_l1_threshold",
        "models": {
            name: {**summarize(rows, channels), **summarize_episode_proxy(rows, channels)} for name, rows in model_rows.items()
        },
    }
    (output_dir / "paper_metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics = [
        "success_rate",
        "success_rate_episode_proxy",
        "avg_completion_time_s",
        "timeout_penalized_completion_time_s",
        "avg_step_latency_s",
        "avg_selected_channels",
        "avg_l1",
        "median_l1",
    ] + [f"{ch}_keep_rate" for ch in channels]
    lines = [
        "| Metric | OpenVLA | FullSoft | VisualThink-VLA |",
        "|---|---:|---:|---:|",
    ]
    for metric in metrics:
        lines.append(
            f"| {metric} | "
            f"{fmt(summary['models']['openvla'].get(metric))} | "
            f"{fmt(summary['models']['full_soft'].get(metric))} | "
            f"{fmt(summary['models']['visualthink_vla'].get(metric))} |"
        )
    (output_dir / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] summary={output_dir / 'summary_table.md'}", flush=True)


if __name__ == "__main__":
    main()
