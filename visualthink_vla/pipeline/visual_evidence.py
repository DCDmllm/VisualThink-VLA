#!/usr/bin/env python3
"""Unified visual evidence pipeline for caption, query extraction, detection, depth, edge, motion, and relation features."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, pipeline as hf_pipeline

from visualthink_vla.utils.motion_features import compute_motion_map, motion_stats
from visualthink_vla.utils.evidence_visualization import render_motion_panel, render_relation_panel
from visualthink_vla.utils.qwen_image_edit import QwenImageEditClient, QwenImageEditConfig
from visualthink_vla.utils.relation_features import build_relation_stats, relation_text


STOPWORDS = {
    "a",
    "an",
    "the",
    "to",
    "in",
    "on",
    "and",
    "or",
    "of",
    "for",
    "with",
    "is",
    "are",
    "be",
    "robot",
    "what",
    "should",
    "take",
    "action",
    "image",
    "shows",
    "show",
    "scene",
    "background",
    "features",
    "feature",
    "task",
    "involves",
    "involving",
    "require",
    "requires",
    "would",
    "current",
    "location",
    "towards",
    "from",
    "near",
    "center",
    "middle",
    "left",
    "right",
    "side",
    "part",
    "table",
    "tabletop",
    "wall",
    "various",
    "arranged",
    "positioned",
    "moving",
    "move",
    "possibly",
}

COLORS = {
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "white",
    "black",
    "gray",
    "grey",
    "brown",
    "pink",
    "purple",
}

DESCRIPTORS = {
    "metal",
    "metallic",
    "wood",
    "wooden",
    "plastic",
    "rubber",
    "ceramic",
    "glass",
    "silver",
    "gold",
    "textured",
    "small",
    "large",
    "big",
}

OBJECT_NOUN_HINTS = {
    "object",
    "pot",
    "spatula",
    "cup",
    "bowl",
    "plate",
    "bottle",
    "block",
    "toy",
    "tool",
    "container",
    "lid",
    "box",
    "can",
    "mug",
    "spoon",
    "fork",
    "knife",
    "pan",
}

NON_OBJECT_WORDS = {
    "corner",
    "middle",
    "center",
    "side",
    "background",
    "layout",
    "tabletop",
    "table",
    "wall",
    "further",
    "towards",
    "off-center",
    "visible",
    "partially",
    "current",
    "location",
    "item",
}


@dataclass
class VisualPipelineConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    qwen_model_id: str = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
    qwen_image_edit_model_id: str = os.environ.get("QWEN_IMAGE_EDIT_MODEL_ID", "")
    qwen_image_edit_api_url: str | None = os.environ.get("QWEN_IMAGE_EDIT_API_URL")
    qwen_image_edit_api_key: str | None = os.environ.get("QWEN_IMAGE_EDIT_API_KEY")
    qwen_image_edit_timeout_s: int = int(os.environ.get("QWEN_IMAGE_EDIT_TIMEOUT_S", "120"))
    owl_model_id: str = os.environ.get("OWL_MODEL_ID", "google/owlv2-base-patch16-ensemble")
    query_api_url: str | None = None
    query_api_key: str | None = None
    max_query_words: int = 8
    owl_score_thresh: float = 0.1
    owl_nms_iou_thresh: float = 0.5
    owl_cross_label_iou_thresh: float = 0.8
    owl_max_per_label: int = 2
    owl_max_total: int = 8
    use_qwen: bool = True
    use_qwen_image_edit: bool = os.environ.get("ENABLE_QWEN_IMAGE_EDIT", "0") == "1"
    use_owl: bool = True
    use_sam2: bool = True
    use_midas: bool = True


class QwenCaptioner:
    def __init__(self, cfg: VisualPipelineConfig):
        self.cfg = cfg
        self.pipe = None
        self.model = None
        self.processor = None
        self.last_source = "uninitialized"
        self.last_error = ""

    def _lazy_init(self) -> None:
        if self.model is not None or self.pipe is not None:
            return
        try:
            qwen_id = self.cfg.qwen_model_id.lower()
            if "qwen2.5-vl" in qwen_id or "qwen2_5_vl" in qwen_id:
                dtype = torch.bfloat16 if self.cfg.device == "cuda" else torch.float32
                self.processor = AutoProcessor.from_pretrained(self.cfg.qwen_model_id, trust_remote_code=True)
                tokenizer = getattr(self.processor, "tokenizer", None)
                if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
                    tokenizer.padding_side = "left"
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.cfg.qwen_model_id,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                ).to(self.cfg.device)
                self.model.eval()
            else:
                device_idx = 0 if self.cfg.device == "cuda" else -1
                self.pipe = hf_pipeline("image-to-text", model=self.cfg.qwen_model_id, device=device_idx)
            self.last_error = ""
        except Exception as exc:
            self.pipe = None
            self.model = None
            self.processor = None
            self.last_error = f"pipeline_init_failed:{type(exc).__name__}"

    @staticmethod
    def _prompt_text(instruction: str) -> str:
        return (
            "Describe this robot manipulation scene in one concise paragraph. "
            "Focus on visible objects, colors, spatial relations, tabletop layout, "
            f"and the object relevant to this instruction: {instruction}"
        )

    def _build_messages(self, image: Image.Image, instruction: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self._prompt_text(instruction)},
                ],
            }
        ]

    def _run_qwen25vl(self, image: Image.Image, instruction: str) -> str:
        messages = self._build_messages(image, instruction)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: (v.to(self.cfg.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        generated_ids = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
        trimmed_ids = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        text = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        if not text:
            raise RuntimeError("empty_caption")
        return text

    def _run_qwen25vl_batch(self, images: list[Image.Image], instructions: list[str]) -> list[str]:
        conversations = [self._build_messages(image, instruction) for image, instruction in zip(images, instructions)]
        prompts = [
            self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in conversations
        ]
        image_inputs, video_inputs = process_vision_info(conversations)
        inputs = self.processor(
            text=prompts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: (v.to(self.cfg.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        generated_ids = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
        trimmed_ids = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        texts = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        outputs = [str(text).strip() for text in texts]
        if any(not text for text in outputs):
            raise RuntimeError("empty_caption")
        return outputs

    def batch_generate(
        self,
        images: list[Image.Image],
        instructions: list[str],
    ) -> tuple[list[str], list[str], list[str]]:
        if len(images) != len(instructions):
            raise ValueError("images and instructions must have the same length")
        if not images:
            return [], [], []
        if not self.cfg.use_qwen:
            captions = [f"A robot scene related to instruction: {instruction}" for instruction in instructions]
            sources = ["disabled_fallback"] * len(captions)
            errors = [""] * len(captions)
            self.last_source = sources[-1]
            self.last_error = errors[-1]
            return captions, sources, errors

        self._lazy_init()
        if self.model is None and self.pipe is None:
            captions = [f"A robot manipulation scene. Instruction: {instruction}" for instruction in instructions]
            sources = ["fallback"] * len(captions)
            errors = [self.last_error] * len(captions)
            self.last_source = sources[-1]
            return captions, sources, errors

        if self.model is not None and self.processor is not None:
            try:
                captions = self._run_qwen25vl_batch(images, instructions)
                self.last_source = "qwen"
                self.last_error = ""
                return captions, ["qwen"] * len(captions), [""] * len(captions)
            except Exception as exc:
                batch_error = f"caption_batch_failed:{type(exc).__name__}"
                captions: list[str] = []
                sources: list[str] = []
                errors: list[str] = []
                for image, instruction in zip(images, instructions):
                    try:
                        text = self._run_qwen25vl(image, instruction)
                        captions.append(text)
                        sources.append("qwen")
                        errors.append("")
                    except Exception as inner_exc:
                        captions.append(f"A robot manipulation scene. Instruction: {instruction}")
                        sources.append("fallback")
                        errors.append(f"{batch_error}|caption_inference_failed:{type(inner_exc).__name__}")
                self.last_source = sources[-1]
                self.last_error = errors[-1]
                return captions, sources, errors

        if self.pipe is not None:
            captions = []
            sources = []
            errors = []
            for image, instruction in zip(images, instructions):
                try:
                    out = self.pipe(image, max_new_tokens=96)
                    text = out[0].get("generated_text", "") if isinstance(out, list) and out else ""
                    if isinstance(text, list):
                        text = json.dumps(text, ensure_ascii=False)
                    text = str(text).strip()
                    if text:
                        captions.append(text)
                        sources.append("qwen")
                        errors.append("")
                    else:
                        captions.append(f"A robot manipulation scene. Instruction: {instruction}")
                        sources.append("fallback")
                        errors.append("empty_caption")
                except Exception as exc:
                    captions.append(f"A robot manipulation scene. Instruction: {instruction}")
                    sources.append("fallback")
                    errors.append(f"caption_inference_failed:{type(exc).__name__}")
            self.last_source = sources[-1]
            self.last_error = errors[-1]
            return captions, sources, errors

        captions = []
        sources = []
        errors = []
        for image, instruction in zip(images, instructions):
            captions.append(f"A robot manipulation scene. Instruction: {instruction}")
            sources.append("fallback")
            errors.append(self.last_error)
        return captions, sources, errors

    def __call__(self, image: Image.Image, instruction: str) -> str:
        captions, sources, errors = self.batch_generate([image], [instruction])
        self.last_source = sources[0] if sources else "fallback"
        self.last_error = errors[0] if errors else ""
        if captions:
            return captions[0]
        self.last_source = "fallback"
        return f"A robot manipulation scene. Instruction: {instruction}"


class QueryWordExtractor:
    def __init__(self, cfg: VisualPipelineConfig):
        self.cfg = cfg
        self.last_source = "fallback"

    @staticmethod
    def _normalize_phrase(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _is_object_like(self, token: str) -> bool:
        if token in STOPWORDS or len(token) < 3:
            return False
        if token in NON_OBJECT_WORDS:
            return False
        if token in OBJECT_NOUN_HINTS:
            return True
        if token.endswith(("er", "or", "ula", "cup", "pot", "pan", "lid", "box", "can")):
            return True
        return False

    def _collect_phrases(self, text: str, *, prefer_instruction: bool) -> List[str]:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text.lower())
        phrases: List[str] = []
        seen = set()

        def add_phrase(raw: str) -> None:
            phrase = self._normalize_phrase(raw)
            if not phrase:
                return
            if phrase in seen:
                return
            parts = phrase.split()
            if any(part in NON_OBJECT_WORDS for part in parts):
                return
            head = phrase.split()[-1]
            if head in STOPWORDS:
                return
            if head in NON_OBJECT_WORDS:
                return
            if head == "object" and not any(word in COLORS or word in DESCRIPTORS for word in phrase.split()[:-1]):
                return
            seen.add(phrase)
            phrases.append(phrase)

        for idx, token in enumerate(tokens):
            if token in STOPWORDS:
                continue
            prev1 = tokens[idx - 1] if idx - 1 >= 0 else ""
            prev2 = tokens[idx - 2] if idx - 2 >= 0 else ""

            if prev1 == "or" or prev2 == "possibly":
                continue

            if self._is_object_like(token):
                mods: List[str] = []
                if prev2 in COLORS | DESCRIPTORS and prev1 in COLORS | DESCRIPTORS:
                    mods.extend([prev2, prev1])
                elif prev1 in COLORS | DESCRIPTORS:
                    mods.append(prev1)
                if mods:
                    add_phrase(" ".join(mods + [token]))
                add_phrase(token)

            if prefer_instruction and token in COLORS and idx + 1 < len(tokens):
                nxt = tokens[idx + 1]
                if self._is_object_like(nxt):
                    add_phrase(f"{token} {nxt}")

        return phrases

    def _fallback_extract(self, caption: str, instruction: str) -> List[str]:
        ordered: List[str] = []
        seen = set()

        for phrase in self._collect_phrases(instruction, prefer_instruction=True):
            if phrase not in seen:
                seen.add(phrase)
                ordered.append(phrase)

        for phrase in self._collect_phrases(caption, prefer_instruction=False):
            if phrase not in seen:
                seen.add(phrase)
                ordered.append(phrase)

        if not ordered:
            tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", f"{instruction.lower()} {caption.lower()}")
            for token in tokens:
                if token in STOPWORDS or len(token) < 3:
                    continue
                if token not in seen:
                    seen.add(token)
                    ordered.append(token)
                if len(ordered) >= self.cfg.max_query_words:
                    break

        if "object" in ordered and any(" object" in phrase for phrase in ordered):
            ordered = [phrase for phrase in ordered if phrase != "object"]

        cleaned = [phrase for phrase in ordered if all(part not in NON_OBJECT_WORDS for part in phrase.split())]
        return cleaned[: self.cfg.max_query_words] or ["object"]

    def __call__(self, caption: str, instruction: str) -> List[str]:
        if not self.cfg.query_api_url:
            self.last_source = "heuristic"
            return self._fallback_extract(caption, instruction)
        try:
            headers = {"Content-Type": "application/json"}
            if self.cfg.query_api_key:
                headers["Authorization"] = f"Bearer {self.cfg.query_api_key}"
            payload = {"caption": caption, "instruction": instruction, "max_words": self.cfg.max_query_words}
            resp = requests.post(self.cfg.query_api_url, json=payload, headers=headers, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            words = data.get("query_words", [])
            words = [str(w).strip() for w in words if str(w).strip()]
            if words:
                self.last_source = "api"
                return words[: self.cfg.max_query_words]
        except Exception:
            pass
        self.last_source = "heuristic"
        return self._fallback_extract(caption, instruction)


class OwlV2Detector:
    def __init__(self, cfg: VisualPipelineConfig):
        self.cfg = cfg
        self.pipe = None
        self.last_source = "uninitialized"

    def _lazy_init(self) -> None:
        if self.pipe is not None or not self.cfg.use_owl:
            return
        device_idx = 0 if self.cfg.device == "cuda" else -1
        try:
            model_dtype = torch.float16 if self.cfg.device == "cuda" else torch.float32
            self.pipe = hf_pipeline(
                task="zero-shot-object-detection",
                model=self.cfg.owl_model_id,
                device=device_idx,
                torch_dtype=model_dtype,
            )
            self.last_source = "owlv2"
        except Exception:
            self.pipe = None
            self.last_source = "disabled_or_failed"

    @staticmethod
    def _iou(box_a: List[float], box_b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    @staticmethod
    def _label_head(label: str) -> str:
        parts = label.strip().lower().split()
        return parts[-1] if parts else label.strip().lower()

    def _sort_key(self, det: Dict[str, Any]) -> tuple[float, int]:
        return (float(det["score"]), len(str(det["label"]).split()))

    def _nms(self, detections: List[Dict[str, Any]], iou_thresh: float) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        for det in sorted(detections, key=self._sort_key, reverse=True):
            if all(self._iou(det["bbox"], prev["bbox"]) < iou_thresh for prev in kept):
                kept.append(det)
        return kept

    def _postprocess(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered = [d for d in detections if float(d["score"]) >= self.cfg.owl_score_thresh]
        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for det in filtered:
            by_label.setdefault(str(det["label"]), []).append(det)

        per_label: List[Dict[str, Any]] = []
        for dets in by_label.values():
            kept = self._nms(dets, self.cfg.owl_nms_iou_thresh)
            per_label.extend(kept[: self.cfg.owl_max_per_label])

        final: List[Dict[str, Any]] = []
        for det in sorted(per_label, key=self._sort_key, reverse=True):
            head = self._label_head(str(det["label"]))
            duplicate = False
            for prev in final:
                if self._label_head(str(prev["label"])) != head:
                    continue
                if self._iou(det["bbox"], prev["bbox"]) >= self.cfg.owl_cross_label_iou_thresh:
                    duplicate = True
                    break
            if not duplicate:
                final.append(det)

        return final[: self.cfg.owl_max_total]

    def __call__(self, image: Image.Image, query_words: List[str]) -> List[Dict[str, Any]]:
        self._lazy_init()
        if self.pipe is None or len(query_words) == 0:
            return []
        try:
            outputs = self.pipe(image, candidate_labels=query_words)
            dets = []
            for o in outputs:
                box = o.get("box", {})
                dets.append(
                    {
                        "label": str(o.get("label", "object")),
                        "score": float(o.get("score", 0.0)),
                        "bbox": [
                            float(box.get("xmin", 0)),
                            float(box.get("ymin", 0)),
                            float(box.get("xmax", 0)),
                            float(box.get("ymax", 0)),
                        ],
                    }
                )
            return self._postprocess(dets)
        except Exception:
            self.last_source = "disabled_or_failed"
            return []


class SAM2Segmenter:
    """SAM2 wrapper with rectangle-mask fallback."""

    def __init__(self, cfg: VisualPipelineConfig):
        self.cfg = cfg
        self.predictor = None
        self._tried_init = False
        self.backend = "bbox_fallback"

    def _lazy_init(self) -> None:
        if self._tried_init or not self.cfg.use_sam2:
            return
        self._tried_init = True
        ckpt = os.environ.get("SAM2_CHECKPOINT")
        model_cfg = os.environ.get("SAM2_MODEL_CFG")
        if not ckpt or not model_cfg:
            return
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model = build_sam2(model_cfg, ckpt, device=self.cfg.device)
            self.predictor = SAM2ImagePredictor(model)
            self.backend = "sam2"
        except Exception:
            self.predictor = None
            self.backend = "bbox_fallback"

    def __call__(self, image: Image.Image, detections: List[Dict[str, Any]]) -> List[np.ndarray]:
        self._lazy_init()
        h, w = image.height, image.width
        masks: List[np.ndarray] = []

        if self.predictor is not None and len(detections) > 0:
            try:
                np_img = np.array(image.convert("RGB"))
                self.predictor.set_image(np_img)
                for d in detections:
                    x1, y1, x2, y2 = d["bbox"]
                    box = np.array([x1, y1, x2, y2], dtype=np.float32)
                    pred_masks, _, _ = self.predictor.predict(box=box[None, :], multimask_output=False)
                    m = pred_masks[0].astype(np.uint8)
                    masks.append(m)
                self.backend = "sam2"
                return masks
            except Exception:
                masks = []
                self.backend = "bbox_fallback"

        # Fallback: rectangle masks from bbox.
        self.backend = "bbox_fallback"
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            m = np.zeros((h, w), dtype=np.uint8)
            x1i, y1i = max(0, int(x1)), max(0, int(y1))
            x2i, y2i = min(w - 1, int(x2)), min(h - 1, int(y2))
            if x2i > x1i and y2i > y1i:
                m[y1i : y2i + 1, x1i : x2i + 1] = 1
            masks.append(m)
        return masks


class MiDaSDepthEstimator:
    def __init__(self, cfg: VisualPipelineConfig):
        self.cfg = cfg
        self.model = None
        self.transform = None
        self.last_source = "uninitialized"

    def _lazy_init(self) -> None:
        if self.model is not None or not self.cfg.use_midas:
            return
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.model.to(self.cfg.device).eval()
            tx = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = tx.small_transform
            self.last_source = "midas"
        except Exception:
            self.model = None
            self.transform = None
            self.last_source = "grayscale_fallback"

    def __call__(self, image: Image.Image) -> np.ndarray:
        self._lazy_init()
        rgb = np.array(image.convert("RGB"))
        if self.model is None or self.transform is None:
            self.last_source = "grayscale_fallback"
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            return gray

        inp = self.transform(rgb).to(self.cfg.device)
        with torch.no_grad():
            pred = self.model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            depth = pred.detach().cpu().numpy().astype(np.float32)
        dmin, dmax = float(depth.min()), float(depth.max())
        if dmax > dmin:
            depth = (depth - dmin) / (dmax - dmin)
        self.last_source = "midas"
        return depth


class EdgeExtractor:
    def __call__(self, image: Image.Image) -> np.ndarray:
        gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        return cv2.Canny(gray, threshold1=60, threshold2=160).astype(np.uint8)


class VisualEvidenceSchema:
    @staticmethod
    def build(
        instruction: str,
        caption: str,
        query_words: List[str],
        detections: List[Dict[str, Any]],
        depth: np.ndarray | None,
        edge: np.ndarray | None,
        motion: np.ndarray | None = None,
        relation: Dict[str, Any] | None = None,
    ) -> str:
        det_text = []
        for i, d in enumerate(detections):
            x1, y1, x2, y2 = d["bbox"]
            det_text.append(
                f"obj{i}:{d['label']} score={d['score']:.3f} bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
            )
        depth_text = "none"
        if depth is not None:
            depth_text = f"mean={float(depth.mean()):.4f},std={float(depth.std()):.4f}"
        edge_text = "none"
        if edge is not None:
            edge_text = f"density={float((edge > 0).mean()):.4f}"
        motion_text = "none"
        if motion is not None:
            stats = motion_stats(motion)
            motion_text = (
                f"mean={stats['mean']:.4f},std={stats['std']:.4f},density={stats['density']:.4f}"
            )
        relation_line = relation_text(relation)

        return (
            "<visual_evidence>\n"
            f"instruction: {instruction}\n"
            f"caption: {caption}\n"
            f"queries: {', '.join(query_words)}\n"
            f"detections: {'; '.join(det_text) if det_text else 'none'}\n"
            f"depth: {depth_text}\n"
            f"edge: {edge_text}\n"
            f"motion: {motion_text}\n"
            f"relation: {relation_line}\n"
            "</visual_evidence>"
        )


class VisualEvidencePipeline:
    def __init__(self, cfg: VisualPipelineConfig | None = None):
        self.cfg = cfg or VisualPipelineConfig()
        self.captioner = QwenCaptioner(self.cfg)
        self.query_extractor = QueryWordExtractor(self.cfg)
        self.detector = OwlV2Detector(self.cfg)
        self.segmenter = SAM2Segmenter(self.cfg)
        self.depth_estimator = MiDaSDepthEstimator(self.cfg)
        self.edge_extractor = EdgeExtractor()
        self.image_editor = QwenImageEditClient(
            QwenImageEditConfig(
                enabled=self.cfg.use_qwen_image_edit,
                model_id=self.cfg.qwen_image_edit_model_id,
                api_url=self.cfg.qwen_image_edit_api_url or "",
                api_key=self.cfg.qwen_image_edit_api_key or "",
                timeout_s=self.cfg.qwen_image_edit_timeout_s,
            )
        )

    @staticmethod
    def _motion_edit_prompt(motion_stats: Dict[str, float]) -> str:
        return (
            "Refine this robotics analysis canvas into a crisp publication-style motion figure. "
            "Preserve the three-panel layout exactly: previous frame, current frame, and motion heatmap. "
            "Do not change object geometry or panel order. Keep labels readable and emphasize adjacent-frame motion."
            f" Motion mean={float(motion_stats.get('mean', 0.0)):.3f}, density={float(motion_stats.get('density', 0.0)):.3f}."
        )

    @staticmethod
    def _relation_edit_prompt(relation_stats: Dict[str, Any]) -> str:
        return (
            "Refine this relation-analysis overlay into a clean scientific annotation. "
            "Preserve all bounding boxes, target marker, goal anchor, nearest-object marker, and angle labels exactly. "
            "Do not move geometry; only improve readability."
            f" Goal angle={float(relation_stats.get('goal_angle_deg', 0.0)):.1f} degrees, "
            f"nearest-object angle={float(relation_stats.get('nearest_other_angle_deg', 0.0)):.1f} degrees."
        )

    def run(self, image: Image.Image, instruction: str, prev_image: Image.Image | None = None) -> Dict[str, Any]:
        caption = self.captioner(image, instruction)
        query_words = self.query_extractor(caption, instruction)
        detections = self.detector(image, query_words)
        masks = self.segmenter(image, detections)
        depth = self.depth_estimator(image)
        edge = self.edge_extractor(image)
        motion = compute_motion_map(prev_image, image) if prev_image is not None else compute_motion_map(None, image)
        motion_source = "frame_diff" if prev_image is not None else "episode_start_zero"
        relation = build_relation_stats(
            instruction=instruction,
            query_words=query_words,
            detections=detections,
            image_size=image.size,
        )
        motion_vis = render_motion_panel(prev_image=prev_image, curr_image=image, motion_u8=motion, motion_stats=motion_stats(motion))
        relation_vis = render_relation_panel(image=image, detections=detections, relation_stats=relation)
        motion_prompt = self._motion_edit_prompt(motion_stats(motion))
        relation_prompt = self._relation_edit_prompt(relation)
        motion_qwen_vis, motion_qwen_source, motion_qwen_error = self.image_editor.edit(
            motion_vis,
            prompt=motion_prompt,
            source_prompt="robot manipulation adjacent-frame motion analysis",
        )
        relation_qwen_vis, relation_qwen_source, relation_qwen_error = self.image_editor.edit(
            relation_vis,
            prompt=relation_prompt,
            source_prompt="robot manipulation spatial relation analysis",
        )

        schema = VisualEvidenceSchema.build(
            instruction=instruction,
            caption=caption,
            query_words=query_words,
            detections=detections,
            depth=depth,
            edge=edge,
            motion=motion,
            relation=relation,
        )
        return {
            "instruction": instruction,
            "caption": caption,
            "caption_source": self.captioner.last_source,
            "caption_error": self.captioner.last_error,
            "query_words": query_words,
            "query_source": self.query_extractor.last_source,
            "detections": detections,
            "detection_source": self.detector.last_source,
            "masks": masks,
            "sam2_backend": self.segmenter.backend,
            "depth": depth,
            "depth_source": self.depth_estimator.last_source,
            "edge": edge,
            "motion": motion,
            "motion_source": motion_source,
            "motion_stats": motion_stats(motion),
            "motion_visual": motion_vis,
            "motion_visual_prompt": motion_prompt,
            "motion_qwen_visual": motion_qwen_vis,
            "motion_qwen_visual_source": motion_qwen_source,
            "motion_qwen_visual_error": motion_qwen_error,
            "relation_source": relation.get("source", "unknown"),
            "relation_stats": relation,
            "relation_visual": relation_vis,
            "relation_visual_prompt": relation_prompt,
            "relation_qwen_visual": relation_qwen_vis,
            "relation_qwen_visual_source": relation_qwen_source,
            "relation_qwen_visual_error": relation_qwen_error,
            "schema_text": schema,
        }
