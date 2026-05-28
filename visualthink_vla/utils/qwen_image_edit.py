from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from PIL import Image


@dataclass
class QwenImageEditConfig:
    enabled: bool = False
    model_id: str = ""
    api_url: str = ""
    api_key: str = ""
    timeout_s: int = 120


class QwenImageEditClient:
    def __init__(self, cfg: QwenImageEditConfig):
        self.cfg = cfg
        self.backend = "disabled"
        self.last_error = ""
        self.pipe = None
        self._tried_init = False

    def _lazy_init(self) -> None:
        if self._tried_init or not self.cfg.enabled:
            return
        self._tried_init = True
        if self.cfg.api_url:
            self.backend = "api"
            return
        model_id = (self.cfg.model_id or "").strip()
        if not model_id:
            self.backend = "unavailable"
            self.last_error = "missing_model_or_api"
            return
        if not Path(model_id).exists() and "/" not in model_id:
            self.backend = "unavailable"
            self.last_error = "missing_model_or_api"
            return
        try:
            from modelscope.pipelines import pipeline as ms_pipeline
            from modelscope.utils.constant import Tasks

            self.pipe = ms_pipeline(Tasks.image_editing, model=model_id)
            self.backend = "modelscope"
            self.last_error = ""
        except Exception as exc:
            self.pipe = None
            self.backend = "unavailable"
            self.last_error = f"modelscope_init_failed:{type(exc).__name__}"

    @staticmethod
    def _decode_json_image(data: dict[str, Any]) -> Image.Image:
        b64 = (
            data.get("image_base64")
            or data.get("b64_json")
            or data.get("output", {}).get("image_base64")
            or ((data.get("data") or [{}])[0].get("b64_json") if isinstance(data.get("data"), list) else None)
        )
        if b64:
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        url = data.get("image_url") or ((data.get("data") or [{}])[0].get("url") if isinstance(data.get("data"), list) else None)
        if url:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        raise ValueError("image_not_found_in_response")

    def edit(self, image: Image.Image, prompt: str, source_prompt: str = "") -> tuple[Image.Image | None, str, str]:
        self._lazy_init()
        if not self.cfg.enabled:
            return None, "disabled", ""
        if self.backend == "api":
            try:
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                headers = {}
                if self.cfg.api_key:
                    headers["Authorization"] = f"Bearer {self.cfg.api_key}"
                resp = requests.post(
                    self.cfg.api_url,
                    data={"model": self.cfg.model_id, "prompt": prompt, "source_prompt": source_prompt},
                    files={"image": ("image.png", buf.getvalue(), "image/png")},
                    headers=headers,
                    timeout=self.cfg.timeout_s,
                )
                resp.raise_for_status()
                out = self._decode_json_image(resp.json())
                self.last_error = ""
                return out, "api", ""
            except Exception as exc:
                self.last_error = f"api_edit_failed:{type(exc).__name__}"
                return None, "api_failed", self.last_error
        if self.backend == "modelscope" and self.pipe is not None:
            try:
                output = self.pipe({"img": image, "prompts": [source_prompt or "", prompt]})
                arr = output.get("output_img")
                if arr is None:
                    raise RuntimeError("missing_output_img")
                out = Image.fromarray(arr[:, :, ::-1]).convert("RGB")
                self.last_error = ""
                return out, "modelscope", ""
            except Exception as exc:
                self.last_error = f"modelscope_edit_failed:{type(exc).__name__}"
                return None, "modelscope_failed", self.last_error
        return None, self.backend or "unavailable", self.last_error or "missing_model_or_api"
