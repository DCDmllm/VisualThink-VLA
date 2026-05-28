#!/usr/bin/env python3
"""Build a 2x2 small-multiple Pareto scatter for the selected benchmark subset."""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import io

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



DATASET_ORDER = ["bridge", "fractal", "libero_spatial", "libero_goal"]
DATASET_TITLES = {
    "bridge": "BridgeData V2",
    "fractal": "Fractal",
    "libero_spatial": "LIBERO-Spatial",
    "libero_goal": "LIBERO-Goal",
}

# Per-panel ranges: tighter y windows improve local separation.
AXIS_LIMITS = {
    "bridge": {"x": (0.28, 9.5), "y": (10, 92)},
    "fractal": {"x": (0.28, 9.5), "y": (15, 95)},
    "libero_spatial": {"x": (0.28, 2.0), "y": (82, 98.5)},
    "libero_goal": {"x": (0.28, 2.2), "y": (78, 98.5)},
}

MODEL_STYLES = {
    "openvla": {"label": "BaseVLA", "short": "BV", "color": "#6b7280", "marker": "o", "size": 80, "z": 3},
    "full_soft": {"label": "FullSoft", "short": "FS", "color": "#4c78a8", "marker": "s", "size": 82, "z": 3},
    "visualthink_vla": {"label": "VisualThink-VLA", "short": "EI", "color": "#2a9d55", "marker": "h", "size": 108, "z": 4},
    "ECoT-OXE": {"label": "ECoT-OXE", "short": "EC", "color": "#c17c2a", "marker": "D", "size": 82, "z": 3},
    "DeepThinkVLA-RL": {"label": "DeepThinkVLA-RL", "short": "DT", "color": "#d55e5e", "marker": "^", "size": 88, "z": 3},
    "InternVLA-M1": {"label": "InternVLA-M1", "short": "IV", "color": "#7c69b3", "marker": "P", "size": 92, "z": 3},
    "InternVLA-M1-LIBERO-Spatial": {"label": "InternVLA-M1", "short": "IV", "color": "#7c69b3", "marker": "P", "size": 92, "z": 3},
    "InternVLA-M1-LIBERO-Goal": {"label": "InternVLA-M1", "short": "IV", "color": "#7c69b3", "marker": "P", "size": 92, "z": 3},
}

LABEL_OFFSETS = {
    "BV": (4, -8),
    "FS": (4, 5),
    "EI": (4, 6),
    "EC": (4, 6),
    "DT": (4, 5),
    "IV": (4, -10),
}


def parse_table(md_text: str) -> dict[str, list[dict[str, float | str]]]:
    lines = md_text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| Dataset"))
    raw_rows: list[str] = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        raw_rows.append(line)

    data = "\n".join(raw_rows)
    reader = csv.reader(io.StringIO(data), delimiter="|")
    parsed: dict[str, list[dict[str, float | str]]] = {key: [] for key in DATASET_ORDER}

    for row in reader:
        cells = [cell.strip() for cell in row[1:-1]]
        if len(cells) != 4:
            continue
        dataset, model, success_str, latency_str = cells
        if dataset not in parsed or not success_str or not latency_str:
            continue
        parsed[dataset].append(
            {
                "dataset": dataset,
                "model": model,
                "success": float(success_str) * 100.0,
                "latency": float(latency_str),
            }
        )
    return parsed


def pareto_front(points: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    frontier = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if q["latency"] <= p["latency"] and q["success"] >= p["success"]:
                if q["latency"] < p["latency"] or q["success"] > p["success"]:
                    dominated = True
                    break
        if not dominated:
            frontier.append(p)
    return sorted(frontier, key=lambda item: float(item["latency"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_path", required=True, help="Markdown table with Dataset/Model/Success/Latency columns.")
    parser.add_argument("--out_pdf", required=True)
    parser.add_argument("--out_png", required=True)
    args = parser.parse_args()

    report_path = Path(args.report_path)
    out_pdf = Path(args.out_pdf)
    out_png = Path(args.out_png)
    parsed = parse_table(report_path.read_text(encoding="utf-8"))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.0))
    axes = axes.flatten()

    for idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[idx]
        points = parsed[dataset]
        frontier = pareto_front(points)

        for point in points:
            style = MODEL_STYLES[point["model"]]
            ax.scatter(
                point["latency"],
                point["success"],
                s=style["size"],
                c=style["color"],
                marker=style["marker"],
                edgecolors="black" if style["short"] == "EI" else "white",
                linewidths=1.2 if style["short"] == "EI" else 0.9,
                alpha=0.96,
                zorder=style["z"],
            )
            dx, dy = LABEL_OFFSETS[style["short"]]
            ax.annotate(
                style["short"],
                (point["latency"], point["success"]),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=8,
                color=style["color"],
                weight="bold" if style["short"] == "EI" else "normal",
            )

        if len(frontier) >= 2:
            ax.plot(
                [float(p["latency"]) for p in frontier],
                [float(p["success"]) for p in frontier],
                linestyle=(0, (3, 2)),
                color="#4b5563",
                linewidth=1.4,
                alpha=0.85,
                zorder=2,
            )

        ax.set_title(DATASET_TITLES[dataset], weight="bold", pad=8)
        ax.set_xscale("log")
        ax.set_xlim(*AXIS_LIMITS[dataset]["x"])
        ax.set_ylim(*AXIS_LIMITS[dataset]["y"])
        ax.grid(True, which="major", linestyle="--", linewidth=0.55, alpha=0.35)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Success rate (%)")
    axes[2].set_ylabel("Success rate (%)")
    axes[2].set_xlabel("Avg. step latency (s, log)")
    axes[3].set_xlabel("Avg. step latency (s, log)")

    legend_handles = []
    used = set()
    for model_key in ["openvla", "full_soft", "visualthink_vla", "ECoT-OXE", "DeepThinkVLA-RL", "InternVLA-M1"]:
        style = MODEL_STYLES[model_key]
        if style["label"] in used:
            continue
        used.add(style["label"])
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="none",
                markerfacecolor=style["color"],
                markeredgecolor="black" if style["short"] == "EI" else "white",
                markeredgewidth=1.1 if style["short"] == "EI" else 0.9,
                markersize=8.2,
                label=f'{style["short"]} = {style["label"]}',
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=9,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.12, wspace=0.18, hspace=0.22)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_pdf}")
    print(f"[ok] wrote {out_png}")


if __name__ == "__main__":
    main()
