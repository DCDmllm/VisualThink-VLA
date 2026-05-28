#!/usr/bin/env python3
"""Offline batch feature generation for extracted frame manifest."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualthink_vla.pipeline.visual_evidence import VisualEvidencePipeline, VisualEvidenceSchema, VisualPipelineConfig
from visualthink_vla.utils.evidence_visualization import render_motion_panel, render_relation_panel
from visualthink_vla.utils.motion_features import compute_motion_map_from_gray, image_to_gray_array, motion_stats
from visualthink_vla.utils.relation_features import build_relation_stats, relation_vector_from_stats


def pack_masks(masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if len(masks) == 0:
        return np.zeros((0,), dtype=np.uint8), np.array([0, 0, 0], dtype=np.int32)
    h, w = masks[0].shape[:2]
    arr = np.stack([m.astype(np.uint8) for m in masks], axis=0)
    packed = np.packbits(arr.reshape(arr.shape[0], -1), axis=1)
    shape = np.array([arr.shape[0], h, w], dtype=np.int32)
    return packed, shape


def quantize_depth(depth: np.ndarray) -> np.ndarray:
    return (np.clip(depth, 0, 1) * 65535).astype(np.uint16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSONL from extract_rlds_bridge_raw.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=20)
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
    parser.add_argument("--caption_batch_size", type=int, default=4)
    parser.add_argument("--save_debug_visuals", action="store_true")
    parser.add_argument("--debug_visual_limit", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    npz_dir = out_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = out_dir / "feature_manifest.jsonl"

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
    debug_dir = out_dir / "debug_visuals"
    if args.save_debug_visuals:
        debug_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    manifest_count = sum(1 for line in open(args.manifest, "r", encoding="utf-8") if line.strip())
    total = manifest_count if args.limit <= 0 else min(args.limit, manifest_count)
    start_ts = time.time()
    print(f"[info] total_samples={total} log_every={args.log_every}", flush=True)
    batch_size = max(1, int(args.caption_batch_size))
    last_gray_by_episode: dict[int, np.ndarray] = {}
    last_image_path_by_episode: dict[int, str] = {}
    debug_written = 0
    with open(args.manifest, "r", encoding="utf-8") as fin, out_manifest.open("w", encoding="utf-8") as fout:
        pending: list[dict] = []

        def flush_batch(rows: list[dict]) -> int:
            nonlocal debug_written
            if not rows:
                return 0
            images = [Image.open(row["image_path"]).convert("RGB") for row in rows]
            try:
                instructions = [row["instruction"] for row in rows]
                captions, caption_sources, caption_errors = pipe.captioner.batch_generate(images, instructions)
                written = 0
                for rec, image, caption, caption_source, caption_error in zip(
                    rows, images, captions, caption_sources, caption_errors
                ):
                    image_path = rec["image_path"]
                    instruction = rec["instruction"]
                    action = rec["action"]
                    episode_idx = rec["episode_idx"]
                    step_idx = rec["step_idx"]

                    query_words = pipe.query_extractor(caption, instruction)
                    dets = pipe.detector(image, query_words)
                    masks = pipe.segmenter(image, dets)
                    depth = pipe.depth_estimator(image)
                    edge = pipe.edge_extractor(image)
                    curr_gray = image_to_gray_array(image)
                    prev_gray = last_gray_by_episode.get(episode_idx)
                    motion_u8 = compute_motion_map_from_gray(prev_gray, curr_gray)
                    motion_source = "frame_diff" if prev_gray is not None else "episode_start_zero"
                    motion_summary = motion_stats(motion_u8)
                    relation_stats = build_relation_stats(
                        instruction=instruction,
                        query_words=query_words,
                        detections=dets,
                        image_size=image.size,
                    )
                    relation_vec = relation_vector_from_stats(relation_stats)
                    schema_text = VisualEvidenceSchema.build(
                        instruction=instruction,
                        caption=caption,
                        query_words=query_words,
                        detections=dets,
                        depth=depth,
                        edge=edge,
                        motion=motion_u8,
                        relation=relation_stats,
                    )

                    bboxes = np.array([d["bbox"] for d in dets], dtype=np.float32) if dets else np.zeros((0, 4), dtype=np.float32)
                    scores = np.array([d["score"] for d in dets], dtype=np.float32) if dets else np.zeros((0,), dtype=np.float32)
                    labels = np.array([d["label"] for d in dets], dtype=object) if dets else np.array([], dtype=object)

                    packed_masks, mask_shape = pack_masks(masks)
                    depth_u16 = quantize_depth(depth)
                    edge_packed = np.packbits((edge > 0).astype(np.uint8).reshape(-1))
                    edge_shape = np.array(edge.shape, dtype=np.int32)
                    stem = f"ep{episode_idx:06d}_s{step_idx:04d}"
                    npz_path = npz_dir / f"{stem}.npz"
                    np.savez_compressed(
                        npz_path,
                        bboxes=bboxes,
                        scores=scores,
                        labels=labels,
                        packed_masks=packed_masks,
                        mask_shape=mask_shape,
                        depth_u16=depth_u16,
                        edge_packed=edge_packed,
                        edge_shape=edge_shape,
                        motion_u8=motion_u8,
                        relation_vec=relation_vec,
                    )
                    debug_enabled = args.save_debug_visuals and (args.debug_visual_limit <= 0 or debug_written < args.debug_visual_limit)
                    if debug_enabled:
                        prev_path = last_image_path_by_episode.get(episode_idx)
                        prev_pil = Image.open(prev_path).convert("RGB") if prev_path else None
                        sample_dir = debug_dir / stem
                        sample_dir.mkdir(parents=True, exist_ok=True)
                        motion_vis = render_motion_panel(prev_image=prev_pil, curr_image=image, motion_u8=motion_u8, motion_stats=motion_summary)
                        motion_vis.save(sample_dir / "motion_adjacent.png")
                        relation_vis = render_relation_panel(image=image, detections=dets, relation_stats=relation_stats)
                        relation_vis.save(sample_dir / "relation_overlay.png")
                        if prev_pil is not None:
                            prev_pil.close()
                        debug_written += 1

                    out = {
                        "episode_idx": episode_idx,
                        "step_idx": step_idx,
                        "image_path": image_path,
                        "instruction": instruction,
                        "action": action,
                        "caption": caption,
                        "caption_source": caption_source,
                        "caption_error": caption_error,
                        "query_words": query_words,
                        "query_source": pipe.query_extractor.last_source,
                        "detections": dets,
                        "detection_source": pipe.detector.last_source,
                        "sam2_backend": pipe.segmenter.backend,
                        "depth_source": pipe.depth_estimator.last_source,
                        "motion_source": motion_source,
                        "motion_stats": motion_summary,
                        "relation_source": relation_stats.get("source", "unknown"),
                        "relation_stats": relation_stats,
                        "motion_debug_path": str((debug_dir / stem / "motion_adjacent.png")) if debug_enabled else "",
                        "relation_debug_path": str((debug_dir / stem / "relation_overlay.png")) if debug_enabled else "",
                        "schema_text": schema_text,
                        "npz_path": str(npz_path),
                    }
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    last_gray_by_episode[episode_idx] = curr_gray
                    last_image_path_by_episode[episode_idx] = image_path
                    written += 1
                return written
            finally:
                for image in images:
                    image.close()

        for line in fin:
            if not line.strip():
                continue
            pending.append(json.loads(line))
            if len(pending) < batch_size and (args.limit <= 0 or (count + len(pending)) < args.limit):
                continue
            written = flush_batch(pending)
            count += written
            pending = []
            if count % args.log_every == 0 or count == total:
                elapsed = max(1e-6, time.time() - start_ts)
                speed = count / elapsed
                remain = max(0, total - count)
                eta_sec = int(remain / max(1e-6, speed))
                print(
                    f"[progress] {count}/{total} ({count * 100.0 / total:.1f}%) "
                    f"speed={speed:.2f} samples/s eta={eta_sec}s",
                    flush=True,
                )
            if args.limit > 0 and count >= args.limit:
                break

        if pending and (args.limit <= 0 or count < args.limit):
            written = flush_batch(pending)
            count += written
            if count % args.log_every == 0 or count == total:
                elapsed = max(1e-6, time.time() - start_ts)
                speed = count / elapsed
                remain = max(0, total - count)
                eta_sec = int(remain / max(1e-6, speed))
                print(
                    f"[progress] {count}/{total} ({count * 100.0 / total:.1f}%) "
                    f"speed={speed:.2f} samples/s eta={eta_sec}s",
                    flush=True,
                )

    print(f"[ok] feature samples: {count}")
    print(f"[ok] manifest: {out_manifest}")


if __name__ == "__main__":
    main()
