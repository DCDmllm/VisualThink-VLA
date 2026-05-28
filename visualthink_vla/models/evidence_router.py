#!/usr/bin/env python3
"""Learnable evidence routing components for VisualThink-VLA."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from visualthink_vla.utils.motion_features import motion_vector_from_map
from visualthink_vla.utils.relation_features import RELATION_DIM, relation_vector_from_stats


CHANNELS = ("bbox", "edge", "motion", "relation")
FEATURE_DIMS = {
    "bbox": 10,
    "depth": 8,
    "edge": 5,
    "motion": 8,
    "relation": RELATION_DIM,
}
STAGES = ("approach", "grasp", "place")
INSTRUCTION_META_DIM = 8
INSTANCE_AMBIGUITY_DIM = 12


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:d}h{mins:02d}m{secs:02d}s"
    if mins > 0:
        return f"{mins:d}m{secs:02d}s"
    return f"{secs:d}s"


def _print_progress(prefix: str, current: int, total: int, start_ts: float) -> None:
    total = max(1, int(total))
    current = max(0, min(int(current), total))
    frac = current / total
    filled = int(round(frac * 20))
    bar = "#" * filled + "-" * (20 - filled)
    elapsed = max(1e-6, time.time() - start_ts)
    speed = current / elapsed
    remain = max(0, total - current)
    eta = remain / max(1e-6, speed)
    print(
        f"[progress] {prefix} [{bar}] {current}/{total} "
        f"({frac * 100:.1f}%) speed={speed:.2f}/s eta={_format_eta(eta)}",
        flush=True,
    )


def load_local_tokenizer(tokenizer_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def tokenize_to_arrays(tokenizer, text: str, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_attention_mask=True,
    )
    input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    attention_mask = np.asarray(encoded["attention_mask"], dtype=np.float32)
    return input_ids, attention_mask


def infer_stage(step_ratio: float, boundaries: tuple[float, float] = (0.60, 0.85)) -> str:
    """Map trajectory progress to temporal phases used by the gating policy.

    The gating method needs phase labels that vary within a trajectory. Using the
    instruction text here is too weak because all steps in an episode share the
    same instruction. Temporal phase buckets are a better first-order proxy.
    """
    approach_end, grasp_end = boundaries
    if step_ratio < approach_end:
        return "approach"
    if step_ratio < grasp_end:
        return "grasp"
    return "place"


def infer_event_stage(
    step_ratio: float,
    row: dict | None,
    bbox_vec: np.ndarray,
    edge_vec: np.ndarray,
    motion_vec: np.ndarray,
    relation_vec: np.ndarray,
    cfg: dict | None = None,
) -> str:
    """Infer a coarse event-driven phase from cheap per-step evidence.

    This is a pragmatic proxy, not a semantic ground-truth phase label.
    It uses currently available signals so that phase boundaries can vary
    across tasks and episodes instead of being fixed normalized-time cuts.
    """
    cfg = cfg or {}
    motion_stats = (row or {}).get("motion_stats") or {}
    relation_stats = (row or {}).get("relation_stats") or {}

    motion_strength = float(
        max(
            motion_stats.get("density", float(motion_vec[2]) if motion_vec.size >= 3 else 0.0),
            motion_stats.get("center", float(motion_vec[3]) if motion_vec.size >= 4 else 0.0),
            motion_stats.get("std", float(motion_vec[1]) if motion_vec.size >= 2 else 0.0),
        )
    )
    goal_known = float(relation_stats.get("goal_known", float(relation_vec[14]) if relation_vec.size >= 15 else 0.0))
    goal_dist = float(relation_stats.get("goal_dist", float(relation_vec[17]) if relation_vec.size >= 18 else 0.0))
    target_center_dist = float(
        relation_stats.get("target_center_dist", float(relation_vec[7]) if relation_vec.size >= 8 else 0.0)
    )
    target_area = float(relation_stats.get("target_area", float(relation_vec[6]) if relation_vec.size >= 7 else 0.0))
    nearest_other = float(relation_stats.get("nearest_other_dist", float(relation_vec[8]) if relation_vec.size >= 9 else 1.0))
    max_iou = float(relation_stats.get("max_iou", float(relation_vec[9]) if relation_vec.size >= 10 else 0.0))
    target_matched = float(relation_stats.get("target_matched", float(relation_vec[1]) if relation_vec.size >= 2 else 0.0))
    target_score = float(relation_stats.get("target_score_mean", float(relation_vec[3]) if relation_vec.size >= 4 else 0.0))

    crowding = float(np.clip(max(max_iou, 1.0 - nearest_other), 0.0, 1.0))
    target_conf = float(np.clip(0.5 * target_matched + 0.5 * target_score, 0.0, 1.0))
    area_signal = float(np.clip(target_area * 6.0, 0.0, 1.0))
    goal_closeness = float((1.0 - goal_dist) if goal_known > 0.5 else step_ratio)
    edge_complexity = float(np.clip(edge_vec[0] if edge_vec.size else 0.0, 0.0, 1.0))

    grasp_motion_thresh = float(cfg.get("grasp_motion_thresh", 0.06))
    place_step_thresh = float(cfg.get("place_step_thresh", 0.75))
    place_motion_max = float(cfg.get("place_motion_max", 0.10))
    place_goal_dist_max = float(cfg.get("place_goal_dist_max", 0.18))

    approach_score = (
        0.45 * (1.0 - motion_strength)
        + 0.25 * np.clip(target_center_dist, 0.0, 1.0)
        + 0.20 * (1.0 - step_ratio)
        + 0.10 * (1.0 - crowding)
    )
    grasp_score = (
        0.30 * motion_strength
        + 0.20 * crowding
        + 0.15 * area_signal
        + 0.15 * edge_complexity
        + 0.10 * target_conf
        + 0.10 * max(0.0, 1.0 - abs(step_ratio - 0.55) / 0.55)
    )
    place_score = (
        0.35 * goal_closeness
        + 0.25 * step_ratio
        + 0.20 * (1.0 - motion_strength)
        + 0.20 * target_conf
    )

    if motion_strength >= grasp_motion_thresh:
        grasp_score += 0.35
    if step_ratio >= place_step_thresh and (motion_strength <= place_motion_max or goal_dist <= place_goal_dist_max):
        place_score += 0.30

    scores = {
        "approach": float(approach_score),
        "grasp": float(grasp_score),
        "place": float(place_score),
    }
    return max(scores.items(), key=lambda kv: kv[1])[0]


def _smooth_signal(values: np.ndarray, passes: int = 2) -> np.ndarray:
    if values.size <= 2:
        return values.astype(np.float32)
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    out = values.astype(np.float32)
    for _ in range(max(1, passes)):
        padded = np.pad(out, (1, 1), mode="edge")
        out = np.convolve(padded, kernel, mode="valid")
    return out.astype(np.float32)


def _event_signals_from_record(record: "FeatureRecord") -> dict[str, float]:
    relation_vec = record.relation_vec
    edge_vec = record.edge_vec
    motion_vec = record.motion_vec
    motion_strength = float(
        np.clip(
            max(
                float(motion_vec[2]) if motion_vec.size >= 3 else 0.0,
                float(motion_vec[3]) if motion_vec.size >= 4 else 0.0,
                float(motion_vec[1]) if motion_vec.size >= 2 else 0.0,
            ),
            0.0,
            1.0,
        )
    )
    target_conf = float(
        np.clip(
            0.5 * (float(relation_vec[1]) if relation_vec.size >= 2 else 0.0)
            + 0.5 * (float(relation_vec[3]) if relation_vec.size >= 4 else 0.0),
            0.0,
            1.0,
        )
    )
    target_area = float(np.clip((float(relation_vec[6]) if relation_vec.size >= 7 else 0.0) * 6.0, 0.0, 1.0))
    nearest_other = float(relation_vec[8]) if relation_vec.size >= 9 else 1.0
    max_iou = float(relation_vec[9]) if relation_vec.size >= 10 else 0.0
    crowding = float(np.clip(max(max_iou, 1.0 - nearest_other), 0.0, 1.0))
    target_cx = float(relation_vec[4]) if relation_vec.size >= 5 else 0.5
    target_cy = float(relation_vec[5]) if relation_vec.size >= 6 else 0.5
    goal_known = float(relation_vec[14]) if relation_vec.size >= 15 else 0.0
    goal_dist = float(relation_vec[17]) if relation_vec.size >= 18 else 0.0
    goal_closeness = float((1.0 - goal_dist) if goal_known > 0.5 else 0.0)
    edge_density = float(np.clip(float(edge_vec[0]) if edge_vec.size >= 1 else 0.0, 0.0, 1.0))
    return {
        "motion_strength": motion_strength,
        "target_conf": target_conf,
        "target_area": target_area,
        "crowding": crowding,
        "target_cx": target_cx,
        "target_cy": target_cy,
        "goal_known": goal_known,
        "goal_dist": goal_dist,
        "goal_closeness": goal_closeness,
        "edge_density": edge_density,
        "step_ratio": float(record.step_ratio),
    }


def assign_event_stages_inplace(records: List["FeatureRecord"], cfg: dict | None = None) -> None:
    """Assign event-driven pseudo phases using episode-level contact/transport structure."""
    cfg = cfg or {}
    episodes: Dict[int, List[FeatureRecord]] = {}
    for record in records:
        episodes.setdefault(record.episode_idx, []).append(record)

    contact_peak_frac = float(cfg.get("contact_peak_frac", 0.60))
    place_min_score = float(cfg.get("place_min_score", 0.45))
    place_motion_frac = float(cfg.get("place_motion_frac", 0.70))
    place_progress_thresh = float(cfg.get("place_progress_thresh", 0.08))
    min_grasp_width = int(cfg.get("min_grasp_width", 1))
    min_approach_steps = int(cfg.get("min_approach_steps", 1))
    force_tail_place = bool(cfg.get("force_tail_place", True))

    for episode_records in episodes.values():
        episode_records.sort(key=lambda r: r.step_idx)
        n = len(episode_records)
        if n <= 2:
            for record in episode_records:
                record.stage = infer_event_stage(
                    record.step_ratio,
                    row=None,
                    bbox_vec=record.bbox_vec,
                    edge_vec=record.edge_vec,
                    motion_vec=record.motion_vec,
                    relation_vec=record.relation_vec,
                    cfg=cfg,
                )
            continue

        signals = [_event_signals_from_record(record) for record in episode_records]
        motion = np.asarray([s["motion_strength"] for s in signals], dtype=np.float32)
        edge = np.asarray([s["edge_density"] for s in signals], dtype=np.float32)
        crowding = np.asarray([s["crowding"] for s in signals], dtype=np.float32)
        target_conf = np.asarray([s["target_conf"] for s in signals], dtype=np.float32)
        target_area = np.asarray([s["target_area"] for s in signals], dtype=np.float32)
        goal_closeness = np.asarray([s["goal_closeness"] for s in signals], dtype=np.float32)
        step = np.asarray([s["step_ratio"] for s in signals], dtype=np.float32)
        target_xy = np.asarray([[s["target_cx"], s["target_cy"]] for s in signals], dtype=np.float32)

        anchor_prefix = max(1, min(3, n // 4 if n >= 4 else 1))
        start_xy = target_xy[:anchor_prefix].mean(axis=0)
        shift = np.linalg.norm(target_xy - start_xy[None, :], axis=1) / math.sqrt(2.0)
        shift = np.clip(shift.astype(np.float32), 0.0, 1.0)
        goal_progress = np.maximum(0.0, goal_closeness - float(goal_closeness[:anchor_prefix].mean()))

        contact_signal = _smooth_signal(
            0.38 * motion
            + 0.22 * crowding
            + 0.14 * edge
            + 0.12 * target_conf
            + 0.08 * target_area
            + 0.06 * (1.0 - np.abs(step - 0.55)),
            passes=2,
        )
        anchor_lo = 1 if n >= 3 else 0
        anchor_hi = (n - 2) if n >= 4 else (n - 1)
        grasp_anchor = int(np.argmax(contact_signal[anchor_lo : anchor_hi + 1]) + anchor_lo)
        peak_contact = float(max(contact_signal[grasp_anchor], 1e-6))

        pre_contact = np.where(contact_signal[: grasp_anchor + 1] >= peak_contact * contact_peak_frac)[0]
        if pre_contact.size:
            grasp_start = int(pre_contact[0])
        else:
            grasp_start = max(1, grasp_anchor - max(min_grasp_width, n // 8))
        grasp_start = min(grasp_start, grasp_anchor)
        grasp_start = max(grasp_start, min_approach_steps if n >= 3 else 0)

        place_signal = _smooth_signal(
            0.34 * goal_progress
            + 0.24 * shift
            + 0.20 * (1.0 - motion)
            + 0.12 * target_conf
            + 0.10 * np.clip(step, 0.0, 1.0),
            passes=2,
        )
        post_start = min(n - 1, grasp_anchor + 1)
        peak_motion_after_grasp = float(max(motion[grasp_anchor:].max(), 1e-6))
        place_mask = np.zeros((n,), dtype=bool)
        for idx in range(post_start, n):
            low_motion = motion[idx] <= peak_motion_after_grasp * place_motion_frac
            progressed = max(goal_progress[idx], shift[idx]) >= place_progress_thresh
            strong_place = place_signal[idx] >= place_min_score
            if (low_motion and progressed) or strong_place:
                place_mask[idx] = True
        if place_mask[post_start:].any():
            place_start = int(np.argmax(place_mask[post_start:]) + post_start)
        else:
            tail_best = int(np.argmax(place_signal[post_start:]) + post_start)
            place_start = tail_best if tail_best > grasp_anchor else n

        if place_start <= grasp_start + min_grasp_width:
            place_start = min(n, grasp_anchor + max(min_grasp_width, 1))
        if force_tail_place and n >= 4 and place_start >= n:
            if goal_progress[-1] >= place_progress_thresh or step[-1] >= 0.75:
                place_start = n - 1

        for idx, record in enumerate(episode_records):
            if idx < grasp_start:
                record.stage = "approach"
            elif idx < place_start:
                record.stage = "grasp"
            else:
                record.stage = "place"


def infer_stage_with_proxy(
    step_ratio: float,
    row: dict | None,
    bbox_vec: np.ndarray,
    edge_vec: np.ndarray,
    motion_vec: np.ndarray,
    relation_vec: np.ndarray,
    phase_proxy_cfg: dict | None = None,
) -> str:
    cfg = phase_proxy_cfg or {}
    mode = str(cfg.get("mode", "time")).lower()
    if mode == "event":
        return infer_event_stage(
            step_ratio,
            row=row,
            bbox_vec=bbox_vec,
            edge_vec=edge_vec,
            motion_vec=motion_vec,
            relation_vec=relation_vec,
            cfg=cfg.get("event", {}),
        )
    boundaries = tuple(float(v) for v in cfg.get("time_boundaries", (0.60, 0.85)))
    return infer_stage(step_ratio, boundaries=boundaries)


def tokenize_instruction(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def hashed_bow(text: str, dim: int) -> np.ndarray:
    vec = np.zeros((dim,), dtype=np.float32)
    for token in tokenize_instruction(text):
        h = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(h[:8], 16) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def record_key(episode_idx: int, step_idx: int, npz_path: str) -> str:
    return f"{int(episode_idx)}:{int(step_idx)}:{npz_path}"


def parse_goal_anchor(text: str) -> str:
    lower = text.lower()
    if "left" in lower:
        return "left"
    if "right" in lower:
        return "right"
    if "middle" in lower or "center" in lower or "centre" in lower:
        return "center"
    if "above" in lower or "on top of" in lower:
        return "above"
    if "below" in lower or "under" in lower:
        return "below"
    return "none"


def instruction_meta_vector(instruction: str, query_words: Sequence[str] | None = None) -> np.ndarray:
    queries = [q.strip().lower() for q in (query_words or []) if str(q).strip()]
    query_count = min(len(queries), 8) / 8.0
    uniq_count = min(len(set(queries)), 8) / 8.0
    avg_query_len = 0.0
    if queries:
        avg_query_len = min(np.mean([len(tokenize_instruction(q)) for q in queries]), 8.0) / 8.0
    anchor = parse_goal_anchor(instruction)
    anchor_vec = {
        "left": [1.0, 0.0, 0.0, 0.0, 0.0],
        "right": [0.0, 1.0, 0.0, 0.0, 0.0],
        "center": [0.0, 0.0, 1.0, 0.0, 0.0],
        "above": [0.0, 0.0, 0.0, 1.0, 0.0],
        "below": [0.0, 0.0, 0.0, 0.0, 1.0],
        "none": [0.0, 0.0, 0.0, 0.0, 0.0],
    }[anchor]
    return np.array([query_count, uniq_count, avg_query_len] + anchor_vec, dtype=np.float32)


def instance_ambiguity_vector(row: dict) -> np.ndarray:
    detections = row.get("detections") or []
    scores = sorted([float(d.get("score", 0.0)) for d in detections], reverse=True)
    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) > 1 else 0.0
    score_gap = max(0.0, top1 - top2)
    score_std = float(np.std(scores)) if scores else 0.0
    det_count_norm = min(1.0, len(detections) / 8.0)

    relation = row.get("relation_stats") or {}
    motion = row.get("motion_stats") or {}

    target_matched = float(relation.get("target_matched", 0.0))
    target_count_norm = float(relation.get("target_count_norm", 0.0))
    target_score_mean = float(relation.get("target_score_mean", 0.0))
    nearest_other_dist = float(relation.get("nearest_other_dist", 0.0))
    max_iou = float(relation.get("max_iou", 0.0))
    goal_dist = float(relation.get("goal_dist", 0.0))

    motion_density = float(motion.get("density", 0.0))
    motion_center = float(motion.get("center", 0.0))
    motion_std = float(motion.get("std", 0.0))

    return np.array(
        [
            det_count_norm,
            top1,
            top2,
            score_gap,
            score_std,
            1.0 - target_matched,
            target_count_norm,
            target_score_mean,
            1.0 - nearest_other_dist,
            max_iou,
            goal_dist,
            max(motion_density, motion_center, motion_std),
        ],
        dtype=np.float32,
    )


def unpack_edge(edge_packed: np.ndarray, edge_shape: np.ndarray) -> np.ndarray:
    if edge_shape.size == 0 or int(edge_shape[0]) == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    h, w = int(edge_shape[0]), int(edge_shape[1])
    bits = np.unpackbits(edge_packed)[: h * w]
    return (bits.reshape(h, w).astype(np.uint8) * 255)


def bbox_vector(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    bboxes = npz["bboxes"].astype(np.float32)
    scores = npz["scores"].astype(np.float32)
    if bboxes.size == 0:
        return np.zeros((10,), dtype=np.float32)
    widths = np.clip(bboxes[:, 2] - bboxes[:, 0], 0.0, 1.0)
    heights = np.clip(bboxes[:, 3] - bboxes[:, 1], 0.0, 1.0)
    areas = widths * heights
    cx = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
    cy = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
    top = int(np.argmax(scores)) if scores.size else 0
    return np.array(
        [
            float(len(bboxes)),
            float(scores.mean()) if scores.size else 0.0,
            float(scores.max()) if scores.size else 0.0,
            float(cx.mean()),
            float(cy.mean()),
            float(widths.mean()),
            float(heights.mean()),
            float(areas.mean()),
            float(cx[top]),
            float(cy[top]),
        ],
        dtype=np.float32,
    )


def depth_vector(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    depth = npz["depth_u16"].astype(np.float32) / 65535.0
    hist, _ = np.histogram(depth, bins=4, range=(0.0, 1.0))
    hist = hist.astype(np.float32)
    hist /= max(1.0, hist.sum())
    return np.concatenate(
        [
            np.array(
                [
                    float(depth.mean()),
                    float(depth.std()),
                    float(depth.min()),
                    float(depth.max()),
                ],
                dtype=np.float32,
            ),
            hist,
        ],
        axis=0,
    ).astype(np.float32)


def edge_vector(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    edge = unpack_edge(npz["edge_packed"], npz["edge_shape"])
    if edge.size == 0:
        return np.zeros((5,), dtype=np.float32)
    edge_bin = (edge > 0).astype(np.float32)
    h, w = edge_bin.shape
    top = edge_bin[: h // 2].mean() if h > 1 else edge_bin.mean()
    bottom = edge_bin[h // 2 :].mean() if h > 1 else edge_bin.mean()
    left = edge_bin[:, : w // 2].mean() if w > 1 else edge_bin.mean()
    right = edge_bin[:, w // 2 :].mean() if w > 1 else edge_bin.mean()
    return np.array(
        [
            float(edge_bin.mean()),
            float(top),
            float(bottom),
            float(left),
            float(right),
        ],
        dtype=np.float32,
    )


def motion_vector(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    if "motion_u8" not in npz.files:
        return np.zeros((FEATURE_DIMS["motion"],), dtype=np.float32)
    return motion_vector_from_map(npz["motion_u8"].astype(np.uint8))


def relation_vector(npz: np.lib.npyio.NpzFile, row: dict | None = None) -> np.ndarray:
    if "relation_vec" in npz.files:
        return np.asarray(npz["relation_vec"], dtype=np.float32)
    if row is not None and row.get("relation_stats") is not None:
        return relation_vector_from_stats(row["relation_stats"])
    return np.zeros((FEATURE_DIMS["relation"],), dtype=np.float32)


def load_image_tensor(image_path: str, image_size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr.tolist(), dtype=torch.float32)


def tensor_from_array(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr.tolist(), dtype=torch.float32)


def stage_to_one_hot(stage: str) -> np.ndarray:
    vec = np.zeros((len(STAGES),), dtype=np.float32)
    vec[STAGES.index(stage)] = 1.0
    return vec


def load_stage_keep_priors(policy_yaml: str | None) -> Dict[str, Dict[str, float]]:
    if policy_yaml is None:
        return {
            "approach": {"bbox": 0.65, "edge": 0.70, "motion": 0.15, "relation": 0.55},
            "grasp": {"bbox": 0.30, "edge": 0.65, "motion": 0.50, "relation": 0.75},
            "place": {"bbox": 0.50, "edge": 0.65, "motion": 0.15, "relation": 0.50},
        }
    import yaml

    cfg = yaml.safe_load(Path(policy_yaml).read_text(encoding="utf-8"))
    if "stage_keep_priors" in cfg:
        return cfg["stage_keep_priors"]
    if all(stage in cfg for stage in STAGES):
        priors = {}
        for stage in STAGES:
            priors[stage] = {ch: 1.0 - float(cfg[stage].get(ch, 0.0)) for ch in CHANNELS}
        return priors
    raise RuntimeError(f"could not read stage priors from: {policy_yaml}")


@dataclass
class FeatureRecord:
    episode_idx: int
    step_idx: int
    image_path: str
    npz_path: str
    instruction: str
    action: np.ndarray
    stage: str
    step_ratio: float
    bow: np.ndarray
    query_bow: np.ndarray
    instruction_meta: np.ndarray
    ambiguity_vec: np.ndarray
    bbox_vec: np.ndarray
    depth_vec: np.ndarray
    edge_vec: np.ndarray
    motion_vec: np.ndarray
    relation_vec: np.ndarray
    utility_targets: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    utility_raw_targets: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    utility_budget_target: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    utility_full_loss: float = 0.0
    utility_available: bool = False
    utility_weight: float = 1.0
    instruction_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))
    instruction_mask: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    query_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))
    query_mask: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))


class LearnedGatingDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        image_size: int = 64,
        bow_dim: int = 256,
        query_bow_dim: int = 64,
        limit: int = 0,
        channels: Sequence[str] | None = None,
        text_encoder_type: str = "bow",
        tokenizer_path: str | None = None,
        instruction_max_len: int = 24,
        query_max_len: int = 12,
        phase_proxy_cfg: dict | None = None,
    ):
        load_start_ts = time.time()
        print(f"[stage] loading manifest: {manifest_path}", flush=True)
        rows = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
                if len(rows) == 1 or len(rows) % 50000 == 0:
                    print(f"[progress] manifest rows loaded={len(rows)}", flush=True)
                if limit > 0 and len(rows) >= limit:
                    break
        if not rows:
            raise RuntimeError(f"empty manifest: {manifest_path}")
        print(f"[ok] manifest rows={len(rows)} loaded in {_format_eta(time.time() - load_start_ts)}", flush=True)
        self.action_dim = len(rows[0]["action"])

        self.image_size = image_size
        self.bow_dim = bow_dim
        self.query_bow_dim = query_bow_dim
        self.channels = tuple(channels or CHANNELS)
        self.text_encoder_type = text_encoder_type
        self.instruction_max_len = instruction_max_len
        self.query_max_len = query_max_len
        self.phase_proxy_cfg = dict(phase_proxy_cfg or {})
        phase_mode = str(self.phase_proxy_cfg.get("mode", "time")).lower()
        self.tokenizer = None
        self.text_vocab_size = 0
        if self.text_encoder_type == "sequence":
            if not tokenizer_path:
                raise RuntimeError("text_encoder_type=sequence requires tokenizer_path")
            self.tokenizer = load_local_tokenizer(tokenizer_path)
            self.text_vocab_size = int(len(self.tokenizer))
        step_max = {}
        for row in rows:
            step_max[row["episode_idx"]] = max(step_max.get(row["episode_idx"], 0), int(row["step_idx"]))

        self.records: List[FeatureRecord] = []
        build_start_ts = time.time()
        total_rows = len(rows)
        print(f"[stage] building gating dataset records total={total_rows}", flush=True)
        for row in rows:
            if len(row["action"]) != self.action_dim:
                raise RuntimeError(
                    f"inconsistent action dimension in manifest: expected {self.action_dim}, got {len(row['action'])}"
                )
            denom = max(1, step_max[row["episode_idx"]])
            step_ratio = float(row["step_idx"]) / float(denom)
            with np.load(row["npz_path"], allow_pickle=True) as npz:
                bbox = bbox_vector(npz)
                depth = depth_vector(npz)
                edge = edge_vector(npz)
                motion = motion_vector(npz)
                relation = relation_vector(npz, row=row)
            if phase_mode == "event":
                stage = infer_stage(step_ratio)
            else:
                stage = infer_stage_with_proxy(
                    step_ratio,
                    row=row,
                    bbox_vec=bbox,
                    edge_vec=edge,
                    motion_vec=motion,
                    relation_vec=relation,
                    phase_proxy_cfg=self.phase_proxy_cfg,
                )
            if self.text_encoder_type == "sequence":
                instruction_ids, instruction_mask = tokenize_to_arrays(
                    self.tokenizer, row["instruction"], self.instruction_max_len
                )
                query_ids, query_mask = tokenize_to_arrays(
                    self.tokenizer, " ".join(row.get("query_words") or []), self.query_max_len
                )
            else:
                instruction_ids = np.zeros((self.instruction_max_len,), dtype=np.int64)
                instruction_mask = np.zeros((self.instruction_max_len,), dtype=np.float32)
                query_ids = np.zeros((self.query_max_len,), dtype=np.int64)
                query_mask = np.zeros((self.query_max_len,), dtype=np.float32)
            self.records.append(
                FeatureRecord(
                    episode_idx=int(row["episode_idx"]),
                    step_idx=int(row["step_idx"]),
                    image_path=row["image_path"],
                    npz_path=row["npz_path"],
                    instruction=row["instruction"],
                    action=np.asarray(row["action"], dtype=np.float32),
                    stage=stage,
                    step_ratio=step_ratio,
                    bow=hashed_bow(row["instruction"], bow_dim),
                    query_bow=hashed_bow(" ".join(row.get("query_words") or []), query_bow_dim),
                    instruction_meta=instruction_meta_vector(row["instruction"], row.get("query_words")),
                    ambiguity_vec=instance_ambiguity_vector(row),
                    bbox_vec=bbox,
                    depth_vec=depth,
                    edge_vec=edge,
                    motion_vec=motion,
                    relation_vec=relation,
                    instruction_ids=instruction_ids,
                    instruction_mask=instruction_mask,
                    query_ids=query_ids,
                    query_mask=query_mask,
                )
            )
            current = len(self.records)
            if current == 1 or current % 2000 == 0 or current == total_rows:
                _print_progress("dataset build", current, total_rows, build_start_ts)

        if phase_mode == "event":
            print(f"[stage] assigning event-driven stages total={len(self.records)}", flush=True)
            stage_start_ts = time.time()
            assign_event_stages_inplace(self.records, cfg=self.phase_proxy_cfg.get("event", {}))
            _print_progress("event stage assign", len(self.records), len(self.records), stage_start_ts)

    def attach_counterfactual_utilities(self, utility_map: Dict[str, dict], channels: Sequence[str], budget_values: Sequence[int]) -> None:
        for row in self.records:
            key = record_key(row.episode_idx, row.step_idx, row.npz_path)
            utility = utility_map.get(key)
            if utility is None:
                row.utility_targets = np.zeros((len(channels),), dtype=np.float32)
                row.utility_raw_targets = np.zeros((len(channels),), dtype=np.float32)
                row.utility_budget_target = np.zeros((len(budget_values),), dtype=np.float32)
                row.utility_available = False
                row.utility_full_loss = 0.0
                row.utility_weight = 1.0
                continue
            row.utility_targets = np.asarray([float(utility["utilities"].get(ch, 0.0)) for ch in channels], dtype=np.float32)
            row.utility_raw_targets = np.asarray([float(utility["utilities_raw"].get(ch, 0.0)) for ch in channels], dtype=np.float32)
            row.utility_budget_target = np.asarray(utility["budget_onehot"], dtype=np.float32)
            row.utility_available = True
            row.utility_full_loss = float(utility.get("full_loss", 0.0))
            row.utility_weight = float(utility.get("sample_weight", 1.0))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.records[idx]
        utility_targets = row.utility_targets
        if utility_targets.size == 0:
            utility_targets = np.zeros((len(self.channels),), dtype=np.float32)
        utility_raw_targets = row.utility_raw_targets
        if utility_raw_targets.size == 0:
            utility_raw_targets = np.zeros((len(self.channels),), dtype=np.float32)
        utility_budget = row.utility_budget_target
        if utility_budget.size == 0:
            utility_budget = np.zeros((len(self.channels),), dtype=np.float32)
        return {
            "episode_idx": torch.tensor(row.episode_idx, dtype=torch.long),
            "step_idx": torch.tensor(row.step_idx, dtype=torch.long),
            "image": load_image_tensor(row.image_path, self.image_size),
            "bow": tensor_from_array(row.bow),
            "query_bow": tensor_from_array(row.query_bow),
            "instruction_meta": tensor_from_array(row.instruction_meta),
            "ambiguity_vec": tensor_from_array(row.ambiguity_vec),
            "stage_one_hot": tensor_from_array(stage_to_one_hot(row.stage)),
            "step_ratio": torch.tensor([row.step_ratio], dtype=torch.float32),
            "instruction_ids": torch.tensor(row.instruction_ids.tolist(), dtype=torch.long),
            "instruction_mask": tensor_from_array(row.instruction_mask),
            "query_ids": torch.tensor(row.query_ids.tolist(), dtype=torch.long),
            "query_mask": tensor_from_array(row.query_mask),
            "bbox": tensor_from_array(row.bbox_vec),
            "depth": tensor_from_array(row.depth_vec),
            "edge": tensor_from_array(row.edge_vec),
            "motion": tensor_from_array(row.motion_vec),
            "relation": tensor_from_array(row.relation_vec),
            "action": tensor_from_array(row.action),
            "utility_targets": tensor_from_array(utility_targets),
            "utility_raw_targets": tensor_from_array(utility_raw_targets),
            "utility_budget_target": tensor_from_array(utility_budget),
            "utility_mask": torch.tensor([1.0 if row.utility_available else 0.0], dtype=torch.float32),
            "utility_weight": torch.tensor([row.utility_weight], dtype=torch.float32),
        }


class TextSequenceEncoder(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, embed_dim: int = 96, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0).expand_as(token_ids)
        x = self.token_embed(token_ids) + self.pos_embed(pos)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        mask = attention_mask.unsqueeze(-1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.proj(pooled)


class CheapContextEncoder(nn.Module):
    def __init__(self, bow_dim: int = 256, ctx_dim: int = 128, text_input_dim: int = 64, use_text_sequence: bool = False):
        super().__init__()
        self.use_text_sequence = use_text_sequence
        self.image_backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        text_in_dim = text_input_dim if use_text_sequence else bow_dim
        self.text_proj = nn.Sequential(nn.Linear(text_in_dim, 64), nn.GELU(), nn.Linear(64, 64))
        self.scalar_proj = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 32))
        self.fuse = nn.Sequential(nn.Linear(64 + 64 + 32, ctx_dim), nn.GELU(), nn.Linear(ctx_dim, ctx_dim))

    def forward(
        self,
        image: torch.Tensor,
        bow: torch.Tensor,
        step_ratio: torch.Tensor,
        text_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        img = self.image_backbone(image).flatten(1)
        text_input = text_embedding if self.use_text_sequence else bow
        txt = self.text_proj(text_input)
        scl = self.scalar_proj(step_ratio)
        return self.fuse(torch.cat([img, txt, scl], dim=1))


class EvidenceEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StageConditionedBudgetAwareGateNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_stages: int = len(STAGES),
        num_channels: int = len(CHANNELS),
        num_budgets: int = 3,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.stage_scale = nn.Linear(num_stages, hidden_dim)
        self.stage_shift = nn.Linear(num_stages, hidden_dim)
        self.channel_residual_head = nn.Linear(hidden_dim, num_channels)
        self.utility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_channels),
        )
        self.budget_head = nn.Linear(hidden_dim + num_channels, num_budgets)
        self.channel_stage_bias = nn.Linear(num_stages, num_channels, bias=False)
        self.budget_stage_bias = nn.Linear(num_stages, num_budgets, bias=False)

    def forward(self, x: torch.Tensor, stage_one_hot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        scale = torch.tanh(self.stage_scale(stage_one_hot))
        shift = self.stage_shift(stage_one_hot)
        h_stage = h * (1.0 + scale) + shift
        utility_pred = F.softplus(self.utility_head(h))
        residual = self.channel_residual_head(h_stage)
        channel_logits = residual + self.channel_stage_bias(stage_one_hot) + torch.log1p(utility_pred)
        budget_logits = self.budget_head(torch.cat([h_stage, utility_pred], dim=1)) + self.budget_stage_bias(stage_one_hot)
        return channel_logits, budget_logits, utility_pred


class StageConditionedRouteMixtureGateNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_stages: int = len(STAGES),
        num_channels: int = len(CHANNELS),
        num_routes: int = 4,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.stage_scale = nn.Linear(num_stages, hidden_dim)
        self.stage_shift = nn.Linear(num_stages, hidden_dim)
        self.channel_residual_head = nn.Linear(hidden_dim, num_channels)
        self.utility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_channels),
        )
        self.route_head = nn.Linear(hidden_dim + num_channels, num_routes)
        self.route_stage_bias = nn.Linear(num_stages, num_routes, bias=False)

    def forward(self, x: torch.Tensor, stage_one_hot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        scale = torch.tanh(self.stage_scale(stage_one_hot))
        shift = self.stage_shift(stage_one_hot)
        h_stage = h * (1.0 + scale) + shift
        utility_pred = F.softplus(self.utility_head(h))
        channel_logits = self.channel_residual_head(h_stage) + torch.log1p(utility_pred)
        route_logits = self.route_head(torch.cat([h_stage, utility_pred], dim=1)) + self.route_stage_bias(stage_one_hot)
        return route_logits, channel_logits, utility_pred


class LatentPhaseAdaptiveGateNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_phase_slots: int = 8,
        num_channels: int = len(CHANNELS),
        num_budgets: int = 3,
    ):
        super().__init__()
        self.num_phase_slots = num_phase_slots
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.phase_head = nn.Linear(hidden_dim, num_phase_slots)
        self.phase_scale = nn.Parameter(torch.zeros(num_phase_slots, hidden_dim))
        self.phase_shift = nn.Parameter(torch.zeros(num_phase_slots, hidden_dim))
        self.channel_phase_bias = nn.Parameter(torch.zeros(num_phase_slots, num_channels))
        self.budget_phase_bias = nn.Parameter(torch.zeros(num_phase_slots, num_budgets))
        self.channel_residual_head = nn.Linear(hidden_dim, num_channels)
        self.utility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_channels),
        )
        self.budget_head = nn.Linear(hidden_dim + num_channels, num_budgets)

    def forward(
        self,
        x: torch.Tensor,
        *,
        temperature: float = 1.0,
        hard_assignment: bool = False,
        training: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        phase_logits = self.phase_head(h)
        phase_probs = torch.softmax(phase_logits, dim=1)
        use_training = self.training if training is None else training
        if hard_assignment:
            if use_training:
                phase_assign = F.gumbel_softmax(phase_logits, tau=max(1e-4, float(temperature)), hard=True, dim=1)
            else:
                phase_assign = torch.zeros_like(phase_probs)
                phase_assign.scatter_(1, torch.argmax(phase_probs, dim=1, keepdim=True), 1.0)
        else:
            phase_assign = phase_probs
        scale = torch.matmul(phase_assign, self.phase_scale)
        shift = torch.matmul(phase_assign, self.phase_shift)
        h_phase = h * (1.0 + torch.tanh(scale)) + shift
        utility_pred = F.softplus(self.utility_head(h))
        phase_channel_bias = torch.matmul(phase_assign, self.channel_phase_bias)
        phase_budget_bias = torch.matmul(phase_assign, self.budget_phase_bias)
        channel_logits = self.channel_residual_head(h_phase) + phase_channel_bias + torch.log1p(utility_pred)
        budget_logits = self.budget_head(torch.cat([h_phase, utility_pred], dim=1)) + phase_budget_bias
        return channel_logits, budget_logits, utility_pred, phase_probs, phase_assign


class TeacherPolicy(nn.Module):
    def __init__(
        self,
        ctx_dim: int,
        channel_dim: int,
        channels: Sequence[str],
        hidden_dim: int = 512,
        out_dim: int = 7,
    ):
        super().__init__()
        self.channels = tuple(channels)
        in_dim = ctx_dim + len(self.channels) * channel_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, ctx: torch.Tensor, channel_embeds: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([ctx] + [channel_embeds[ch] for ch in self.channels], dim=1)
        return self.net(x)


class StudentPolicy(nn.Module):
    def __init__(
        self,
        ctx_dim: int,
        channel_dim: int,
        channels: Sequence[str],
        hidden_dim: int = 256,
        out_dim: int = 7,
    ):
        super().__init__()
        self.channels = tuple(channels)
        in_dim = ctx_dim + len(self.channels) * channel_dim + len(self.channels) * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        ctx: torch.Tensor,
        channel_embeds: Dict[str, torch.Tensor],
        hard_gates: torch.Tensor,
        gate_probs: torch.Tensor,
    ) -> torch.Tensor:
        gated = []
        for idx, ch in enumerate(self.channels):
            gated.append(channel_embeds[ch] * hard_gates[:, idx : idx + 1])
        x = torch.cat([ctx] + gated + [hard_gates, gate_probs], dim=1)
        return self.net(x)


class LearnedEvidencePolicy(nn.Module):
    def __init__(
        self,
        bow_dim: int = 256,
        query_bow_dim: int = 64,
        ctx_dim: int = 128,
        channel_dim: int = 64,
        teacher_hidden: int = 512,
        student_hidden: int = 256,
        action_dim: int = 7,
        channels: Sequence[str] | None = None,
        budget_values: Sequence[int] | None = None,
        text_encoder_type: str = "bow",
        text_vocab_size: int = 0,
        instruction_max_len: int = 24,
        query_max_len: int = 12,
        text_embed_dim: int = 96,
        text_hidden_dim: int = 128,
        gate_type: str = "stage_conditioned",
        latent_phase_slots: int = 8,
        route_bank: Sequence[Sequence[str]] | None = None,
    ):
        super().__init__()
        self.channels = tuple(channels or CHANNELS)
        self.budget_values = tuple(int(v) for v in (budget_values or range(1, len(self.channels) + 1)))
        self.text_encoder_type = text_encoder_type
        self.use_text_sequence = text_encoder_type == "sequence"
        self.gate_type = gate_type
        self.route_bank = tuple(tuple(route) for route in (route_bank or []))
        self.query_repr_dim = query_bow_dim
        if self.use_text_sequence:
            if text_vocab_size <= 0:
                raise RuntimeError("sequence text encoder requires text_vocab_size > 0")
            self.instruction_text_encoder = TextSequenceEncoder(
                vocab_size=text_vocab_size,
                max_len=instruction_max_len,
                embed_dim=text_embed_dim,
                hidden_dim=text_hidden_dim,
                out_dim=64,
            )
            self.query_text_encoder = TextSequenceEncoder(
                vocab_size=text_vocab_size,
                max_len=query_max_len,
                embed_dim=text_embed_dim,
                hidden_dim=text_hidden_dim,
                out_dim=query_bow_dim,
            )
        else:
            self.instruction_text_encoder = None
            self.query_text_encoder = None
        self.context = CheapContextEncoder(
            bow_dim=bow_dim,
            ctx_dim=ctx_dim,
            text_input_dim=64,
            use_text_sequence=self.use_text_sequence,
        )
        self.bbox_encoder = EvidenceEncoder(10, channel_dim)
        self.depth_encoder = EvidenceEncoder(8, channel_dim)
        self.edge_encoder = EvidenceEncoder(5, channel_dim)
        self.motion_encoder = EvidenceEncoder(8, channel_dim)
        self.relation_encoder = EvidenceEncoder(RELATION_DIM, channel_dim)
        self.encoder_map = {
            "bbox": self.bbox_encoder,
            "depth": self.depth_encoder,
            "edge": self.edge_encoder,
            "motion": self.motion_encoder,
            "relation": self.relation_encoder,
        }
        gate_in_dim = ctx_dim + 1 + self.query_repr_dim + INSTRUCTION_META_DIM + INSTANCE_AMBIGUITY_DIM
        if self.gate_type == "latent_phase":
            self.gate = LatentPhaseAdaptiveGateNet(
                in_dim=gate_in_dim,
                hidden_dim=ctx_dim,
                num_phase_slots=latent_phase_slots,
                num_channels=len(self.channels),
                num_budgets=len(self.budget_values),
            )
        elif self.gate_type == "route_mixture":
            if not self.route_bank:
                raise RuntimeError("gate_type=route_mixture requires a non-empty route_bank")
            self.gate = StageConditionedRouteMixtureGateNet(
                in_dim=gate_in_dim,
                hidden_dim=ctx_dim,
                num_channels=len(self.channels),
                num_routes=len(self.route_bank),
            )
        else:
            self.gate = StageConditionedBudgetAwareGateNet(
                in_dim=gate_in_dim,
                hidden_dim=ctx_dim,
                num_channels=len(self.channels),
                num_budgets=len(self.budget_values),
            )
        self.teacher = TeacherPolicy(
            ctx_dim=ctx_dim,
            channel_dim=channel_dim,
            channels=self.channels,
            hidden_dim=teacher_hidden,
            out_dim=action_dim,
        )
        self.student = StudentPolicy(
            ctx_dim=ctx_dim,
            channel_dim=channel_dim,
            channels=self.channels,
            hidden_dim=student_hidden,
            out_dim=action_dim,
        )

    def encode_channels(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {ch: self.encoder_map[ch](batch[ch]) for ch in self.channels}

    def encode_context(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        text_embedding = None
        if self.use_text_sequence:
            text_embedding = self.instruction_text_encoder(batch["instruction_ids"], batch["instruction_mask"])
        return self.context(batch["image"], batch["bow"], batch["step_ratio"], text_embedding=text_embedding)

    def gate_inputs(self, ctx: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        query_repr = batch["query_bow"]
        if self.use_text_sequence:
            query_repr = self.query_text_encoder(batch["query_ids"], batch["query_mask"])
        return torch.cat([ctx, batch["step_ratio"], query_repr, batch["instruction_meta"], batch["ambiguity_vec"]], dim=1)

    def forward_gate(
        self,
        ctx: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        stage_one_hot: torch.Tensor,
        *,
        phase_temperature: float = 1.0,
        hard_phase: bool = False,
        training: bool | None = None,
    ) -> Dict[str, torch.Tensor]:
        gate_in = self.gate_inputs(ctx, batch)
        if self.gate_type == "latent_phase":
            channel_logits, budget_logits, utility_pred, phase_probs, phase_assign = self.gate(
                gate_in,
                temperature=phase_temperature,
                hard_assignment=hard_phase,
                training=training,
            )
            return {
                "channel_logits": channel_logits,
                "budget_logits": budget_logits,
                "utility_pred": utility_pred,
                "phase_probs": phase_probs,
                "phase_assign": phase_assign,
            }
        if self.gate_type == "route_mixture":
            route_logits, channel_logits, utility_pred = self.gate(gate_in, stage_one_hot)
            return {
                "route_logits": route_logits,
                "channel_logits": channel_logits,
                "budget_logits": None,
                "utility_pred": utility_pred,
                "phase_probs": None,
                "phase_assign": None,
            }
        channel_logits, budget_logits, utility_pred = self.gate(gate_in, stage_one_hot)
        return {
            "channel_logits": channel_logits,
            "budget_logits": budget_logits,
            "utility_pred": utility_pred,
            "phase_probs": None,
            "phase_assign": None,
        }


def sample_budget_onehot(logits: torch.Tensor, temperature: float, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=1)
    if not training:
        hard = torch.zeros_like(probs)
        hard.scatter_(1, torch.argmax(probs, dim=1, keepdim=True), 1.0)
        return hard, probs
    soft = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=1)
    hard = torch.zeros_like(soft)
    hard.scatter_(1, torch.argmax(soft, dim=1, keepdim=True), 1.0)
    hard_st = hard.detach() - soft.detach() + soft
    return hard_st, probs


def budget_from_onehot(onehot: torch.Tensor, budget_values: Sequence[int], device: torch.device) -> torch.Tensor:
    values = torch.tensor(list(budget_values), dtype=torch.float32, device=device).unsqueeze(0)
    return torch.sum(onehot * values, dim=1)


def topk_hard_mask(scores: torch.Tensor, k_values: torch.Tensor) -> torch.Tensor:
    batch, channels = scores.shape
    mask = torch.zeros_like(scores)
    for i in range(batch):
        k = int(max(1, min(channels, round(float(k_values[i].item())))))
        top_idx = torch.topk(scores[i], k=k, dim=0).indices
        mask[i, top_idx] = 1.0
    return mask


def sample_budget_topk_gates(
    channel_logits: torch.Tensor,
    budget_logits: torch.Tensor,
    budget_values: Sequence[int],
    temperature: float,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    channel_probs = torch.sigmoid(channel_logits)
    hard_budget, budget_probs = sample_budget_onehot(budget_logits, temperature=temperature, training=training)
    hard_budget_values = budget_from_onehot(hard_budget, budget_values=budget_values, device=channel_logits.device)
    soft_budget_values = budget_from_onehot(budget_probs, budget_values=budget_values, device=channel_logits.device)
    hard_mask = topk_hard_mask(channel_probs, hard_budget_values)
    denom = torch.clamp(channel_probs.sum(dim=1, keepdim=True), min=1e-6)
    soft_mask = torch.clamp(channel_probs * (soft_budget_values.unsqueeze(1) / denom), 0.0, 1.0)
    hard_mask_st = hard_mask.detach() - soft_mask.detach() + soft_mask
    return hard_mask_st, channel_probs, hard_budget, budget_probs


def resolve_route_bank(cfg: dict, channels: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    routes = cfg.get("route_bank", [])
    parsed = []
    for entry in routes:
        if isinstance(entry, dict):
            route_channels = entry.get("channels", [])
        else:
            route_channels = entry
        route = tuple(ch for ch in route_channels if ch in channels)
        if route:
            parsed.append(route)
    return tuple(parsed)


def route_bank_tensor(route_bank: Sequence[Sequence[str]], channels: Sequence[str], device: torch.device) -> torch.Tensor:
    masks = []
    for route in route_bank:
        route_set = set(route)
        masks.append([1.0 if ch in route_set else 0.0 for ch in channels])
    if not masks:
        raise RuntimeError("route_bank must contain at least one route")
    return torch.tensor(masks, dtype=torch.float32, device=device)


def route_budget_probs(route_probs: torch.Tensor, route_masks: torch.Tensor, budget_values: Sequence[int]) -> torch.Tensor:
    route_sizes = route_masks.sum(dim=1).to(torch.long)
    cols = []
    for budget in budget_values:
        cols.append((route_probs * (route_sizes == int(budget)).to(route_probs.dtype).unsqueeze(0)).sum(dim=1, keepdim=True))
    probs = torch.cat(cols, dim=1)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return probs


def sample_route_mixture_gates(
    route_logits: torch.Tensor,
    route_masks: torch.Tensor,
    budget_values: Sequence[int],
    temperature: float,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not training:
        route_probs = torch.softmax(route_logits, dim=1)
        hard_route = torch.zeros_like(route_probs)
        hard_route.scatter_(1, torch.argmax(route_probs, dim=1, keepdim=True), 1.0)
    else:
        soft = F.gumbel_softmax(route_logits, tau=max(1e-4, float(temperature)), hard=False, dim=1)
        hard = torch.zeros_like(soft)
        hard.scatter_(1, torch.argmax(soft, dim=1, keepdim=True), 1.0)
        hard_route = hard.detach() - soft.detach() + soft
        route_probs = torch.softmax(route_logits, dim=1)
    hard_gates = hard_route @ route_masks
    gate_probs = route_probs @ route_masks
    budget_probs = route_budget_probs(route_probs, route_masks, budget_values)
    hard_budget = route_budget_probs(hard_route, route_masks, budget_values)
    hard_gates = torch.clamp(hard_gates, 0.0, 1.0)
    gate_probs = torch.clamp(gate_probs, 0.0, 1.0)
    return hard_gates, gate_probs, hard_route, route_probs, hard_budget, budget_probs


def route_targets_from_utility(
    utility_targets: torch.Tensor,
    route_masks: torch.Tensor,
    channel_costs: torch.Tensor,
    route_cost_weight: float,
) -> torch.Tensor:
    route_scores = utility_targets @ route_masks.T - float(route_cost_weight) * (route_masks @ channel_costs)
    target_idx = torch.argmax(route_scores, dim=1)
    target = torch.zeros_like(route_scores)
    target.scatter_(1, target_idx.unsqueeze(1), 1.0)
    return target


def gate_entropy(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp(1e-6, 1.0 - 1e-6)
    ent = -(probs * probs.log() + (1.0 - probs) * (1.0 - probs).log())
    return ent.mean(dim=1)


def budget_entropy(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp(1e-6, 1.0 - 1e-6)
    return -(probs * probs.log()).sum(dim=1)


def phase_sample_entropy(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp(1e-6, 1.0)
    return -(probs * probs.log()).sum(dim=1)


def phase_balance_loss(probs: torch.Tensor) -> torch.Tensor:
    usage = probs.mean(dim=0).clamp(1e-6, 1.0)
    return torch.sum(usage * usage.log())


def effective_phase_count(probs: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    usage = probs.mean(dim=0)
    return torch.sum((usage >= threshold).to(torch.float32))


def temporal_phase_smoothness(
    phase_repr: torch.Tensor,
    episode_idx: torch.Tensor,
    step_idx: torch.Tensor,
) -> torch.Tensor:
    if phase_repr.shape[0] <= 1:
        return torch.zeros((), device=phase_repr.device)
    order = torch.argsort(episode_idx * 100000 + step_idx)
    ep = episode_idx[order]
    st = step_idx[order]
    ph = phase_repr[order]
    same_ep = ep[1:] == ep[:-1]
    forward_gap = (st[1:] - st[:-1]).to(torch.float32).clamp(min=1.0)
    if not bool(same_ep.any().item()):
        return torch.zeros((), device=phase_repr.device)
    diffs = (ph[1:] - ph[:-1]).pow(2).mean(dim=1)
    weights = (1.0 / forward_gap) * same_ep.to(torch.float32)
    return (diffs * weights).sum() / weights.sum().clamp(min=1.0)


def stage_prior_targets(
    stage_one_hot: torch.Tensor,
    priors: Dict[str, Dict[str, float]],
    channels: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    targets = []
    for idx in torch.argmax(stage_one_hot, dim=1).tolist():
        stage = STAGES[idx]
        targets.append([float(priors[stage][ch]) for ch in channels])
    return torch.tensor(targets, dtype=torch.float32, device=device)


def stage_budget_targets(stage_one_hot: torch.Tensor, targets: Dict[str, List[float]], device: torch.device) -> torch.Tensor:
    rows = []
    for idx in torch.argmax(stage_one_hot, dim=1).tolist():
        stage = STAGES[idx]
        rows.append([float(x) for x in targets[stage]])
    return torch.tensor(rows, dtype=torch.float32, device=device)


def mean_action_l1(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - gt))


def channel_cost_tensor(costs: Dict[str, float], channels: Sequence[str], device: torch.device) -> torch.Tensor:
    return torch.tensor([float(costs[ch]) for ch in channels], dtype=torch.float32, device=device)


def stage_channel_targets(
    stage_one_hot: torch.Tensor,
    targets: Dict[str, Dict[str, float]],
    channels: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    rows = []
    for idx in torch.argmax(stage_one_hot, dim=1).tolist():
        stage = STAGES[idx]
        rows.append([float(targets[stage][ch]) for ch in channels])
    return torch.tensor(rows, dtype=torch.float32, device=device)


def coverage_underuse_penalty(probs: torch.Tensor, min_targets: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.relu(min_targets - probs) ** 2)


def temperature_schedule(start: float, end: float, progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return float(start * ((end / start) ** progress)) if start > 0 and end > 0 else end


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def apply_stage_dropout(stage_one_hot: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if (not training) or drop_prob <= 0.0:
        return stage_one_hot
    keep = (torch.rand((stage_one_hot.shape[0], 1), device=stage_one_hot.device) >= drop_prob).to(stage_one_hot.dtype)
    return stage_one_hot * keep


def load_yaml(path: str) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def resolve_channels(cfg: dict) -> tuple[str, ...]:
    if "channel_order" in cfg:
        return tuple(cfg["channel_order"])
    if "channels" in cfg:
        return tuple(cfg["channels"].keys())
    return CHANNELS


def resolve_budget_values(cfg: dict, channels: Sequence[str]) -> tuple[int, ...]:
    if "budget_values" in cfg:
        return tuple(int(v) for v in cfg["budget_values"])
    return tuple(range(1, len(tuple(channels)) + 1))


def utility_budget_onehot(
    utilities: np.ndarray,
    channel_costs: Sequence[float],
    budget_values: Sequence[int],
    relative_keep_thresh: float,
    absolute_score_thresh: float,
) -> tuple[np.ndarray, int]:
    scores = utilities / np.maximum(np.asarray(channel_costs, dtype=np.float32), 1e-6)
    best = float(scores.max()) if scores.size else 0.0
    keep = scores >= max(absolute_score_thresh, best * relative_keep_thresh)
    recommended = int(np.clip(np.count_nonzero(keep), 1, max(budget_values)))
    if best <= absolute_score_thresh:
        recommended = 1
    onehot = np.zeros((len(budget_values),), dtype=np.float32)
    idx = list(budget_values).index(recommended)
    onehot[idx] = 1.0
    return onehot, recommended


def build_counterfactual_utility_map(
    model: LearnedEvidencePolicy,
    dataset: LearnedGatingDataset,
    device: torch.device,
    channels: Sequence[str],
    channel_costs: Sequence[float],
    budget_values: Sequence[int],
    batch_size: int = 64,
    relative_keep_thresh: float = 0.35,
    absolute_score_thresh: float = 1e-4,
    hard_relation_thresh_raw: float = 0.0,
    hard_relation_weight: float = 1.0,
) -> Dict[str, dict]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    utility_map: Dict[str, dict] = {}
    staged_rows: List[dict] = []
    rec_idx = 0
    total_rows = len(dataset)
    start_ts = time.time()
    print(f"[stage] computing counterfactual utilities total={total_rows} batch_size={batch_size}", flush=True)
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            ctx = model.encode_context(batch)
            embeds = model.encode_channels(batch)
            gt = batch["action"]
            full_pred = model.teacher(ctx, embeds)
            full_loss = torch.mean((full_pred - gt) ** 2, dim=1)
            util_cols = []
            for ch in channels:
                ablated = {name: emb if name != ch else torch.zeros_like(emb) for name, emb in embeds.items()}
                drop_pred = model.teacher(ctx, ablated)
                drop_loss = torch.mean((drop_pred - gt) ** 2, dim=1)
                util_cols.append(torch.clamp(drop_loss - full_loss, min=0.0))
            util_mat = np.asarray(torch.stack(util_cols, dim=1).detach().cpu().tolist(), dtype=np.float32)
            full_loss_np = np.asarray(full_loss.detach().cpu().tolist(), dtype=np.float32)
            for row_i in range(util_mat.shape[0]):
                rec = dataset.records[rec_idx]
                staged_rows.append(
                    {
                        "key": record_key(rec.episode_idx, rec.step_idx, rec.npz_path),
                        "episode_idx": rec.episode_idx,
                        "step_idx": rec.step_idx,
                        "npz_path": rec.npz_path,
                        "stage": rec.stage,
                        "full_loss": float(full_loss_np[row_i]),
                        "raw_utilities": util_mat[row_i].copy(),
                    }
                )
                rec_idx += 1
            if rec_idx == 1 or rec_idx % max(batch_size * 10, 2000) == 0 or rec_idx == total_rows:
                _print_progress("counterfactual utility", rec_idx, total_rows, start_ts)

    stage_means = {
        stage: np.mean([row["raw_utilities"] for row in staged_rows if row["stage"] == stage], axis=0).astype(np.float32)
        if any(row["stage"] == stage for row in staged_rows)
        else np.zeros((len(channels),), dtype=np.float32)
        for stage in STAGES
    }
    for row in staged_rows:
        centered = row["raw_utilities"] - stage_means[row["stage"]]
        positive_centered = np.clip(centered, a_min=0.0, a_max=None).astype(np.float32)
        onehot, budget = utility_budget_onehot(
            row["raw_utilities"],
            channel_costs=channel_costs,
            budget_values=budget_values,
            relative_keep_thresh=relative_keep_thresh,
            absolute_score_thresh=absolute_score_thresh,
        )
        relation_raw = 0.0
        if "relation" in channels:
            relation_raw = float(row["raw_utilities"][list(channels).index("relation")])
        sample_weight = float(hard_relation_weight) if relation_raw >= hard_relation_thresh_raw > 0.0 else 1.0
        utility_map[row["key"]] = {
            "episode_idx": row["episode_idx"],
            "step_idx": row["step_idx"],
            "npz_path": row["npz_path"],
            "stage": row["stage"],
            "full_loss": row["full_loss"],
            "utilities": {ch: float(positive_centered[idx]) for idx, ch in enumerate(channels)},
            "utilities_raw": {ch: float(row["raw_utilities"][idx]) for idx, ch in enumerate(channels)},
            "stage_mean_utilities": {ch: float(stage_means[row["stage"]][idx]) for idx, ch in enumerate(channels)},
            "utilities_centered": {ch: float(centered[idx]) for idx, ch in enumerate(channels)},
            "budget_onehot": onehot.tolist(),
            "recommended_budget": int(budget),
            "sample_weight": sample_weight,
        }
    _print_progress("counterfactual utility", total_rows, total_rows, start_ts)
    return utility_map
