from __future__ import annotations

import math
from typing import Any, Dict, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PANEL_BG = (255, 255, 255)
TEXT = (30, 30, 30)
MUTED = (110, 110, 110)
TARGET = (64, 170, 72)
GOAL = (235, 140, 52)
OTHER = (126, 87, 194)
BOX = (49, 130, 189)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, PANEL_BG)
    img = image.copy()
    img.thumbnail(size, Image.Resampling.LANCZOS)
    ox = (size[0] - img.width) // 2
    oy = (size[1] - img.height) // 2
    canvas.paste(img, (ox, oy))
    return canvas


def _draw_panel_title(draw: ImageDraw.ImageDraw, x: int, y: int, title: str) -> None:
    draw.text((x, y), title, fill=TEXT, font=_font(20))


def _heatmap_rgb(motion_u8: np.ndarray, size: tuple[int, int]) -> Image.Image:
    if motion_u8.size == 0:
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")
    heat = cv2.applyColorMap(motion_u8, cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return Image.fromarray(heat, mode="RGB").resize(size, Image.Resampling.BILINEAR)


def render_motion_panel(
    prev_image: Image.Image | None,
    curr_image: Image.Image,
    motion_u8: np.ndarray,
    motion_stats: Dict[str, float] | None = None,
) -> Image.Image:
    panel_w = 360
    panel_h = 260
    gap = 24
    top = 44
    bottom = 64
    canvas = Image.new("RGB", (panel_w * 3 + gap * 4, top + panel_h + bottom), PANEL_BG)
    draw = ImageDraw.Draw(canvas)

    x_positions = [gap, gap * 2 + panel_w, gap * 3 + panel_w * 2]
    titles = ["Prev frame", "Current frame", "Motion heatmap"]
    images = [
        _fit(prev_image, (panel_w, panel_h)) if prev_image is not None else Image.new("RGB", (panel_w, panel_h), (245, 245, 245)),
        _fit(curr_image, (panel_w, panel_h)),
        _fit(_heatmap_rgb(motion_u8, (panel_w, panel_h)), (panel_w, panel_h)),
    ]

    for x, title, image in zip(x_positions, titles, images):
        _draw_panel_title(draw, x, 12, title)
        canvas.paste(image, (x, top))
        draw.rectangle((x, top, x + panel_w, top + panel_h), outline=(215, 215, 215), width=2)

    if prev_image is None:
        draw.text((x_positions[0] + 18, top + panel_h - 28), "episode start", fill=MUTED, font=_font(18))

    stats = motion_stats or {}
    summary = (
        f"mean={float(stats.get('mean', 0.0)):.3f}  "
        f"density={float(stats.get('density', 0.0)):.3f}  "
        f"center={float(stats.get('center', 0.0)):.3f}"
    )
    draw.text((gap, top + panel_h + 18), summary, fill=MUTED, font=_font(18))
    return canvas


def _point_from_norm(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    return float(x) * width, float(y) * height


def _draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], color: tuple[int, int, int], width: int = 4) -> None:
    draw.line((start[0], start[1], end[0], end[1]), fill=color, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux = dx / length
    uy = dy / length
    head = 12.0
    angle = math.radians(25.0)
    rx1 = math.cos(angle) * ux - math.sin(angle) * uy
    ry1 = math.sin(angle) * ux + math.cos(angle) * uy
    rx2 = math.cos(-angle) * ux - math.sin(-angle) * uy
    ry2 = math.sin(-angle) * ux + math.cos(-angle) * uy
    p1 = (end[0] - head * rx1, end[1] - head * ry1)
    p2 = (end[0] - head * rx2, end[1] - head * ry2)
    draw.polygon([end, p1, p2], fill=color)


def render_relation_panel(
    image: Image.Image,
    detections: Sequence[Dict[str, Any]],
    relation_stats: Dict[str, Any] | None,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    stats = relation_stats or {}

    for det in detections:
        x1, y1, x2, y2 = [float(v) for v in det.get("bbox", [0, 0, 0, 0])]
        draw.rectangle((x1, y1, x2, y2), outline=BOX, width=2)
        draw.text((x1 + 4, max(4, y1 - 22)), str(det.get("label", "object")), fill=BOX, font=_font(16))

    target_selected = float(stats.get("target_selected", 0.0)) > 0.5
    target_bbox = stats.get("target_bbox") or []
    if target_selected and len(target_bbox) == 4:
        x1, y1, x2, y2 = [float(v) for v in target_bbox]
        draw.rectangle((x1, y1, x2, y2), outline=TARGET, width=4)

    target_xy = _point_from_norm(float(stats.get("target_cx", 0.0)), float(stats.get("target_cy", 0.0)), width, height)
    goal_known = float(stats.get("goal_known", 0.0)) > 0.5
    if target_selected and goal_known:
        goal_xy = _point_from_norm(float(stats.get("goal_x", 0.0)), float(stats.get("goal_y", 0.0)), width, height)
        r = 7
        draw.ellipse((goal_xy[0] - r, goal_xy[1] - r, goal_xy[0] + r, goal_xy[1] + r), fill=GOAL, outline=GOAL)
        _draw_arrow(draw, target_xy, goal_xy, GOAL)
        draw.text(
            (goal_xy[0] + 10, goal_xy[1] + 6),
            f"goal {stats.get('goal_anchor', 'unknown')} ({float(stats.get('goal_angle_deg', 0.0)):.1f}°)",
            fill=GOAL,
            font=_font(17),
        )

    if target_selected and float(stats.get("nearest_other_known", 0.0)) > 0.5:
        other_xy = _point_from_norm(
            float(stats.get("nearest_other_cx", 0.0)),
            float(stats.get("nearest_other_cy", 0.0)),
            width,
            height,
        )
        r = 6
        draw.ellipse((other_xy[0] - r, other_xy[1] - r, other_xy[0] + r, other_xy[1] + r), fill=OTHER, outline=OTHER)
        _draw_arrow(draw, target_xy, other_xy, OTHER, width=3)
        other_label = str(stats.get("nearest_other_label", "other"))
        draw.text(
            (other_xy[0] + 10, other_xy[1] - 18),
            f"{other_label} ({float(stats.get('nearest_other_angle_deg', 0.0)):.1f}°)",
            fill=OTHER,
            font=_font(17),
        )

    if target_selected:
        cx, cy = target_xy
        draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=TARGET, outline=TARGET)
        label = str(stats.get("target_label") or stats.get("target_phrase") or "target")
        draw.text((cx + 10, cy + 10), label, fill=TARGET, font=_font(18))

    legend_x = 14
    legend_y = 14
    legend_w = min(width - 28, 260)
    legend_h = 112
    draw.rounded_rectangle((legend_x, legend_y, legend_x + legend_w, legend_y + legend_h), radius=12, fill=(255, 255, 255), outline=(220, 220, 220), width=2)
    draw.text((legend_x + 12, legend_y + 10), "Relation legend", fill=TEXT, font=_font(18))
    draw.text((legend_x + 12, legend_y + 36), "green: target", fill=MUTED, font=_font(14))
    draw.text((legend_x + 12, legend_y + 54), "orange: target->goal angle", fill=MUTED, font=_font(14))
    draw.text((legend_x + 12, legend_y + 72), "purple: target->nearest angle", fill=MUTED, font=_font(14))
    draw.text((legend_x + 12, legend_y + 92), f"goal_dist={float(stats.get('goal_dist', 0.0)):.3f}  nearest={float(stats.get('nearest_other_dist', 0.0)):.3f}", fill=MUTED, font=_font(14))
    return canvas
