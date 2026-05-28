from __future__ import annotations

from typing import Dict

import numpy as np
from PIL import Image


DEFAULT_MOTION_SIZE = 128
MOTION_THRESHOLD = 0.08


def image_to_gray_array(image: Image.Image, size: int = DEFAULT_MOTION_SIZE) -> np.ndarray:
    gray = image.convert("L").resize((size, size))
    return np.asarray(gray, dtype=np.float32) / 255.0


def compute_motion_map_from_gray(prev_gray: np.ndarray | None, curr_gray: np.ndarray) -> np.ndarray:
    if prev_gray is None or prev_gray.shape != curr_gray.shape:
        return np.zeros_like(curr_gray, dtype=np.uint8)
    diff = np.abs(curr_gray - prev_gray)
    return np.clip(diff * 255.0, 0.0, 255.0).astype(np.uint8)


def compute_motion_map(
    prev_image: Image.Image | None,
    curr_image: Image.Image,
    size: int = DEFAULT_MOTION_SIZE,
) -> np.ndarray:
    prev_gray = image_to_gray_array(prev_image, size=size) if prev_image is not None else None
    curr_gray = image_to_gray_array(curr_image, size=size)
    return compute_motion_map_from_gray(prev_gray, curr_gray)


def motion_stats(motion_u8: np.ndarray) -> Dict[str, float]:
    motion = motion_u8.astype(np.float32) / 255.0
    if motion.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "density": 0.0,
            "center": 0.0,
            "top": 0.0,
            "bottom": 0.0,
            "left": 0.0,
            "right": 0.0,
        }
    h, w = motion.shape
    center = motion[h // 4 : (3 * h) // 4, w // 4 : (3 * w) // 4]
    return {
        "mean": float(motion.mean()),
        "std": float(motion.std()),
        "density": float((motion > MOTION_THRESHOLD).mean()),
        "center": float(center.mean()) if center.size else 0.0,
        "top": float(motion[: h // 2].mean()) if h > 1 else float(motion.mean()),
        "bottom": float(motion[h // 2 :].mean()) if h > 1 else float(motion.mean()),
        "left": float(motion[:, : w // 2].mean()) if w > 1 else float(motion.mean()),
        "right": float(motion[:, w // 2 :].mean()) if w > 1 else float(motion.mean()),
    }


def motion_vector_from_map(motion_u8: np.ndarray) -> np.ndarray:
    stats = motion_stats(motion_u8)
    return np.array(
        [
            stats["mean"],
            stats["std"],
            stats["density"],
            stats["center"],
            stats["top"],
            stats["bottom"],
            stats["left"],
            stats["right"],
        ],
        dtype=np.float32,
    )
