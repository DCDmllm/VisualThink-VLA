#!/usr/bin/env python3
"""Run unified visual evidence pipeline on one image + instruction."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualthink_vla.pipeline.visual_evidence import VisualEvidencePipeline, VisualPipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--prev_image_path", default=None)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--query_api_url", default=None)
    parser.add_argument("--query_api_key", default=None)
    parser.add_argument("--qwen-model-id", default=None)
    parser.add_argument("--qwen-image-edit-model-id", default=None)
    parser.add_argument("--qwen-image-edit-api-url", default=None)
    parser.add_argument("--qwen-image-edit-api-key", default=None)
    parser.add_argument("--owl-model-id", default=None)
    parser.add_argument("--disable_qwen", action="store_true")
    parser.add_argument("--disable_qwen_image_edit", action="store_true")
    parser.add_argument("--disable_owl", action="store_true")
    parser.add_argument("--disable_sam2", action="store_true")
    parser.add_argument("--disable_midas", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image_path).convert("RGB")
    prev_image = Image.open(args.prev_image_path).convert("RGB") if args.prev_image_path else None

    cfg = VisualPipelineConfig(
        qwen_model_id=args.qwen_model_id or VisualPipelineConfig.qwen_model_id,
        qwen_image_edit_model_id=args.qwen_image_edit_model_id or VisualPipelineConfig.qwen_image_edit_model_id,
        qwen_image_edit_api_url=args.qwen_image_edit_api_url,
        qwen_image_edit_api_key=args.qwen_image_edit_api_key,
        owl_model_id=args.owl_model_id or VisualPipelineConfig.owl_model_id,
        query_api_url=args.query_api_url,
        query_api_key=args.query_api_key,
        use_qwen=not args.disable_qwen,
        use_qwen_image_edit=not args.disable_qwen_image_edit,
        use_owl=not args.disable_owl,
        use_sam2=not args.disable_sam2,
        use_midas=not args.disable_midas,
    )
    pipe = VisualEvidencePipeline(cfg)
    result = pipe.run(image=image, instruction=args.instruction, prev_image=prev_image)

    # Save depth / edge / masks
    depth = result["depth"]
    depth_u8 = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "depth.png"), depth_u8)

    edge = result["edge"]
    cv2.imwrite(str(out_dir / "edge.png"), edge)
    motion = result["motion"]
    cv2.imwrite(str(out_dir / "motion.png"), motion)
    result["motion_visual"].save(out_dir / "motion_adjacent.png")
    (out_dir / "motion_prompt.txt").write_text(result.get("motion_visual_prompt", ""), encoding="utf-8")
    if result.get("motion_qwen_visual") is not None:
        result["motion_qwen_visual"].save(out_dir / "motion_qwen_edit.png")

    for i, m in enumerate(result["masks"]):
        cv2.imwrite(str(out_dir / f"mask_{i:02d}.png"), (m.astype(np.uint8) * 255))
    result["relation_visual"].save(out_dir / "relation_overlay.png")
    (out_dir / "relation_prompt.txt").write_text(result.get("relation_visual_prompt", ""), encoding="utf-8")
    if result.get("relation_qwen_visual") is not None:
        result["relation_qwen_visual"].save(out_dir / "relation_qwen_edit.png")

    relation_stats = result.get("relation_stats", {})
    relation_vec = np.asarray(
        [
            relation_stats.get("target_selected", 0.0),
            relation_stats.get("target_matched", 0.0),
            relation_stats.get("target_count_norm", 0.0),
            relation_stats.get("target_score_mean", 0.0),
            relation_stats.get("target_cx", 0.0),
            relation_stats.get("target_cy", 0.0),
            relation_stats.get("target_area", 0.0),
            relation_stats.get("target_center_dist", 0.0),
            relation_stats.get("nearest_other_dist", 0.0),
            relation_stats.get("max_iou", 0.0),
            relation_stats.get("left_frac", 0.0),
            relation_stats.get("right_frac", 0.0),
            relation_stats.get("above_frac", 0.0),
            relation_stats.get("below_frac", 0.0),
            relation_stats.get("goal_known", 0.0),
            relation_stats.get("goal_dx", 0.0),
            relation_stats.get("goal_dy", 0.0),
            relation_stats.get("goal_dist", 0.0),
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        out_dir / "features.npz",
        depth=depth.astype(np.float32),
        edge=edge,
        motion=motion,
        relation=relation_vec,
    )
    (out_dir / "schema.txt").write_text(result["schema_text"], encoding="utf-8")

    serializable = {
        "instruction": result["instruction"],
        "caption": result["caption"],
        "caption_source": result.get("caption_source", "unknown"),
        "caption_error": result.get("caption_error", ""),
        "query_words": result["query_words"],
        "query_source": result.get("query_source", "unknown"),
        "detections": result["detections"],
        "detection_source": result.get("detection_source", "unknown"),
        "sam2_backend": result.get("sam2_backend", "unknown"),
        "depth_source": result.get("depth_source", "unknown"),
        "motion_source": result.get("motion_source", "unknown"),
        "motion_stats": result.get("motion_stats", {}),
        "motion_visual_prompt": result.get("motion_visual_prompt", ""),
        "motion_qwen_visual_source": result.get("motion_qwen_visual_source", "unknown"),
        "motion_qwen_visual_error": result.get("motion_qwen_visual_error", ""),
        "relation_source": result.get("relation_source", "unknown"),
        "relation_stats": result.get("relation_stats", {}),
        "relation_visual_prompt": result.get("relation_visual_prompt", ""),
        "relation_qwen_visual_source": result.get("relation_qwen_visual_source", "unknown"),
        "relation_qwen_visual_error": result.get("relation_qwen_visual_error", ""),
        "schema_text": result["schema_text"],
    }
    (out_dir / "result.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    if prev_image is not None:
        prev_image.close()
    image.close()
    print(f"[ok] saved to {out_dir}")


if __name__ == "__main__":
    main()
