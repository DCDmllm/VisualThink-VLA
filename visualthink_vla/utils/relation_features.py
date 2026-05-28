from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Sequence

import numpy as np


RELATION_DIM = 18


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def phrase_head(text: str) -> str:
    toks = _tokens(text)
    return toks[-1] if toks else ""


def _goal_anchor(instruction: str) -> tuple[str, tuple[float, float] | None]:
    text = (instruction or "").lower()
    if any(kw in text for kw in ["middle", "center", "centre"]):
        return "center", (0.5, 0.5)
    if any(kw in text for kw in ["far left", "left edge", "left side"]):
        return "left_edge", (0.10, 0.5)
    if any(kw in text for kw in ["far right", "right edge", "right side"]):
        return "right_edge", (0.90, 0.5)
    if any(kw in text for kw in ["far edge", "back edge", "top edge", "upper edge"]):
        return "far_edge", (0.5, 0.10)
    if any(kw in text for kw in ["near edge", "front edge", "bottom edge", "lower edge"]):
        return "near_edge", (0.5, 0.90)
    if any(kw in text for kw in ["top", "upper"]):
        return "top", (0.5, 0.10)
    if any(kw in text for kw in ["bottom", "lower"]):
        return "bottom", (0.5, 0.90)
    if "left" in text:
        return "left", (0.20, 0.5)
    if "right" in text:
        return "right", (0.80, 0.5)
    return "unknown", None


def _match_score(label: str, target_phrase: str) -> float:
    label_l = (label or "").lower().strip()
    target_l = (target_phrase or "").lower().strip()
    if not label_l or not target_l:
        return 0.0
    if label_l == target_l:
        return 3.0
    label_head = phrase_head(label_l)
    target_head = phrase_head(target_l)
    if label_head and label_head == target_head:
        return 2.0
    label_toks = set(_tokens(label_l))
    target_toks = set(_tokens(target_l))
    overlap = len(label_toks & target_toks)
    if overlap > 0:
        return 1.0 + overlap / max(1, len(target_toks))
    return 0.0


def _normalize_bbox(bbox: Sequence[float], width: int, height: int) -> np.ndarray:
    w = max(1.0, float(width))
    h = max(1.0, float(height))
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return np.array([x1 / w, y1 / h, x2 / w, y2 / h], dtype=np.float32)


def _bbox_center(box: np.ndarray) -> tuple[float, float]:
    return float((box[0] + box[2]) * 0.5), float((box[1] + box[3]) * 0.5)


def _bbox_area(box: np.ndarray) -> float:
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def _bearing_deg(src: tuple[float, float], dst: tuple[float, float]) -> float:
    dx = float(dst[0] - src[0])
    dy = float(src[1] - dst[1])
    return float(math.degrees(math.atan2(dy, dx)))


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    return float(inter / max(1e-6, union))


def build_relation_stats(
    instruction: str,
    query_words: Sequence[str],
    detections: Sequence[Dict[str, Any]],
    image_size: tuple[int, int],
) -> Dict[str, Any]:
    width, height = image_size
    target_phrase = query_words[0] if query_words else ""
    goal_anchor, goal_point = _goal_anchor(instruction)
    goal_known = 1.0 if goal_point is not None else 0.0

    if not detections:
        return {
            "source": "bbox_instruction_heuristic",
            "target_phrase": target_phrase,
            "target_label": "",
            "target_bbox": [],
            "goal_anchor": goal_anchor,
            "target_selected": 0.0,
            "target_matched": 0.0,
            "target_count_norm": 0.0,
            "target_score_mean": 0.0,
            "target_cx": 0.0,
            "target_cy": 0.0,
            "target_area": 0.0,
            "target_center_dist": 0.0,
            "nearest_other_dist": 0.0,
            "max_iou": 0.0,
            "left_frac": 0.0,
            "right_frac": 0.0,
            "above_frac": 0.0,
            "below_frac": 0.0,
            "goal_known": goal_known,
            "goal_x": float(goal_point[0]) if goal_point is not None else 0.0,
            "goal_y": float(goal_point[1]) if goal_point is not None else 0.0,
            "goal_angle_deg": 0.0,
            "goal_dx": 0.0,
            "goal_dy": 0.0,
            "goal_dist": 0.0,
            "nearest_other_known": 0.0,
            "nearest_other_label": "",
            "nearest_other_cx": 0.0,
            "nearest_other_cy": 0.0,
            "nearest_other_angle_deg": 0.0,
        }

    match_scores = [_match_score(d.get("label", ""), target_phrase) for d in detections]
    matched_indices = [i for i, s in enumerate(match_scores) if s > 0.0]
    if matched_indices:
        selected_indices = matched_indices
        target_matched = 1.0
    else:
        best_idx = int(np.argmax([float(d.get("score", 0.0)) for d in detections]))
        selected_indices = [best_idx]
        target_matched = 0.0

    sel_boxes = [_normalize_bbox(detections[i]["bbox"], width, height) for i in selected_indices]
    sel_scores = [float(detections[i].get("score", 0.0)) for i in selected_indices]
    target_box = np.mean(np.stack(sel_boxes, axis=0), axis=0)
    target_cx, target_cy = _bbox_center(target_box)
    target_area = _bbox_area(target_box)
    target_center_dist = math.sqrt((target_cx - 0.5) ** 2 + (target_cy - 0.5) ** 2) / math.sqrt(0.5)

    other_boxes = [
        _normalize_bbox(d["bbox"], width, height)
        for i, d in enumerate(detections)
        if i not in selected_indices
    ]
    if other_boxes:
        other_centers = np.array([_bbox_center(b) for b in other_boxes], dtype=np.float32)
        center = np.array([target_cx, target_cy], dtype=np.float32)
        dists = np.linalg.norm(other_centers - center[None, :], axis=1) / math.sqrt(2.0)
        nearest_idx = int(np.argmin(dists))
        nearest_other_dist = float(dists[nearest_idx])
        max_iou = max(_bbox_iou(target_box, b) for b in other_boxes)
        left_frac = float(np.mean(other_centers[:, 0] < target_cx))
        right_frac = float(np.mean(other_centers[:, 0] > target_cx))
        above_frac = float(np.mean(other_centers[:, 1] < target_cy))
        below_frac = float(np.mean(other_centers[:, 1] > target_cy))
        nearest_other_cx = float(other_centers[nearest_idx, 0])
        nearest_other_cy = float(other_centers[nearest_idx, 1])
        remaining = [d for i, d in enumerate(detections) if i not in selected_indices]
        nearest_other_label = str(remaining[nearest_idx].get("label", "other"))
        nearest_other_known = 1.0
        nearest_other_angle_deg = _bearing_deg((target_cx, target_cy), (nearest_other_cx, nearest_other_cy))
    else:
        nearest_other_dist = 1.0
        max_iou = 0.0
        left_frac = 0.0
        right_frac = 0.0
        above_frac = 0.0
        below_frac = 0.0
        nearest_other_cx = 0.0
        nearest_other_cy = 0.0
        nearest_other_label = ""
        nearest_other_known = 0.0
        nearest_other_angle_deg = 0.0

    if goal_point is not None:
        goal_dx = float(target_cx - goal_point[0])
        goal_dy = float(target_cy - goal_point[1])
        goal_dist = math.sqrt(goal_dx**2 + goal_dy**2) / math.sqrt(2.0)
        goal_angle_deg = _bearing_deg((target_cx, target_cy), goal_point)
    else:
        goal_dx = 0.0
        goal_dy = 0.0
        goal_dist = 0.0
        goal_angle_deg = 0.0

    return {
        "source": "bbox_instruction_heuristic",
        "target_phrase": target_phrase,
        "target_label": str(detections[selected_indices[0]].get("label", "")),
        "target_bbox": [float(v) for v in detections[selected_indices[0]].get("bbox", [])],
        "goal_anchor": goal_anchor,
        "target_selected": 1.0,
        "target_matched": target_matched,
        "target_count_norm": min(1.0, len(selected_indices) / 3.0),
        "target_score_mean": float(np.mean(sel_scores)),
        "target_cx": float(target_cx),
        "target_cy": float(target_cy),
        "target_area": float(target_area),
        "target_center_dist": float(target_center_dist),
        "nearest_other_dist": float(nearest_other_dist),
        "max_iou": float(max_iou),
        "left_frac": float(left_frac),
        "right_frac": float(right_frac),
        "above_frac": float(above_frac),
        "below_frac": float(below_frac),
        "goal_known": float(goal_known),
        "goal_x": float(goal_point[0]) if goal_point is not None else 0.0,
        "goal_y": float(goal_point[1]) if goal_point is not None else 0.0,
        "goal_angle_deg": float(goal_angle_deg),
        "goal_dx": float(goal_dx),
        "goal_dy": float(goal_dy),
        "goal_dist": float(goal_dist),
        "nearest_other_known": float(nearest_other_known),
        "nearest_other_label": nearest_other_label,
        "nearest_other_cx": float(nearest_other_cx),
        "nearest_other_cy": float(nearest_other_cy),
        "nearest_other_angle_deg": float(nearest_other_angle_deg),
    }


def relation_vector_from_stats(stats: Dict[str, Any] | None) -> np.ndarray:
    if not stats:
        return np.zeros((RELATION_DIM,), dtype=np.float32)
    return np.array(
        [
            float(stats.get("target_selected", 0.0)),
            float(stats.get("target_matched", 0.0)),
            float(stats.get("target_count_norm", 0.0)),
            float(stats.get("target_score_mean", 0.0)),
            float(stats.get("target_cx", 0.0)),
            float(stats.get("target_cy", 0.0)),
            float(stats.get("target_area", 0.0)),
            float(stats.get("target_center_dist", 0.0)),
            float(stats.get("nearest_other_dist", 0.0)),
            float(stats.get("max_iou", 0.0)),
            float(stats.get("left_frac", 0.0)),
            float(stats.get("right_frac", 0.0)),
            float(stats.get("above_frac", 0.0)),
            float(stats.get("below_frac", 0.0)),
            float(stats.get("goal_known", 0.0)),
            float(stats.get("goal_dx", 0.0)),
            float(stats.get("goal_dy", 0.0)),
            float(stats.get("goal_dist", 0.0)),
        ],
        dtype=np.float32,
    )


def relation_text(stats: Dict[str, Any] | None) -> str:
    if not stats or not float(stats.get("target_selected", 0.0)):
        goal_anchor = (stats or {}).get("goal_anchor", "unknown")
        return f"target=none goal={goal_anchor}"
    return (
        f"target={stats.get('target_label', '') or stats.get('target_phrase', '')} "
        f"matched={int(float(stats.get('target_matched', 0.0)) > 0.5)} "
        f"goal={stats.get('goal_anchor', 'unknown')} "
        f"goal_angle={float(stats.get('goal_angle_deg', 0.0)):.1f} "
        f"goal_dist={float(stats.get('goal_dist', 0.0)):.4f} "
        f"other_angle={float(stats.get('nearest_other_angle_deg', 0.0)):.1f} "
        f"nearest_other={float(stats.get('nearest_other_dist', 0.0)):.4f} "
        f"iou={float(stats.get('max_iou', 0.0)):.4f} "
        f"left={float(stats.get('left_frac', 0.0)):.2f} "
        f"right={float(stats.get('right_frac', 0.0)):.2f} "
        f"above={float(stats.get('above_frac', 0.0)):.2f} "
        f"below={float(stats.get('below_frac', 0.0)):.2f}"
    )
