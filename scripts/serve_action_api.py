#!/usr/bin/env python3
"""Serve OpenVLA/VisualThink-VLA style action prediction over HTTP.

The server intentionally uses only Python's standard-library HTTP stack so it
can run in the existing OpenVLA environments without installing FastAPI.

Request:
  POST /predict_action
  {
    "instruction": "pick up the cup",
    "image_base64": "<base64 encoded RGB/JPEG/PNG image>",
    "unnorm_key": "bridge_orig",              # optional
    "request_id": "robot-step-0001"           # optional
  }

Response:
  {
    "action": [dx, dy, dz, droll, dpitch, dyaw, gripper],
    "action_dict": {...},
    "openvla_action_dict": {...}
  }

Important: the 7D action is an end-effector delta action, not a seven-joint
command. The robot-side controller should map it to the target robot.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.robot.openvla_utils import get_processor  # noqa: E402
from experiments.robot.robot_utils import get_action, get_model  # noqa: E402


@dataclass
class ServerConfig:
    model_family: str
    pretrained_checkpoint: str
    unnorm_key: str
    center_crop: bool
    load_in_8bit: bool
    load_in_4bit: bool
    api_key: str | None
    postprocess_gripper: str


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _decode_image_base64(value: str) -> np.ndarray:
    if "," in value and value.strip().lower().startswith("data:"):
        value = value.split(",", 1)[1]
    raw = base64.b64decode(value, validate=False)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _as_action_dict(action: np.ndarray) -> dict[str, Any]:
    values = [float(x) for x in np.asarray(action, dtype=np.float32).reshape(-1).tolist()]
    if len(values) != 7:
        raise ValueError(f"expected 7D action, got {len(values)} values")
    dx, dy, dz, droll, dpitch, dyaw, gripper = values
    return {
        "delta_position": {"x": dx, "y": dy, "z": dz},
        "delta_rotation": {"roll": droll, "pitch": dpitch, "yaw": dyaw},
        "gripper": gripper,
        "raw_7d": values,
    }


def _as_openvla_action_dict(action: np.ndarray) -> dict[str, Any]:
    values = [float(x) for x in np.asarray(action, dtype=np.float32).reshape(-1).tolist()]
    return {
        "world_vector": values[:3],
        "rotation_delta": values[3:6],
        "open_gripper": values[6],
    }


def _postprocess_gripper(action: np.ndarray, mode: str) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).copy()
    if mode == "none":
        return out
    if mode == "libero":
        # Match the official LIBERO eval path: [0, 1] -> [-1, +1], then invert.
        out[-1] = np.sign(2.0 * out[-1] - 1.0)
        out[-1] *= -1.0
        return out
    raise ValueError(f"unknown gripper postprocess mode: {mode}")


class ActionServer:
    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        self.lock = threading.Lock()
        self.model = get_model(cfg)
        if cfg.unnorm_key not in getattr(self.model, "norm_stats", {}):
            alt = f"{cfg.unnorm_key}_no_noops"
            if alt in getattr(self.model, "norm_stats", {}):
                self.cfg.unnorm_key = alt
            else:
                keys = sorted(getattr(self.model, "norm_stats", {}).keys())
                raise KeyError(f"unnorm_key={cfg.unnorm_key!r} not found in model norm_stats; available={keys}")
        self.processor = get_processor(cfg)

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        instruction = str(payload.get("instruction", "")).strip()
        if not instruction:
            raise ValueError("field `instruction` is required")
        image_b64 = payload.get("image_base64")
        if not image_b64:
            raise ValueError("field `image_base64` is required")

        image = _decode_image_base64(str(image_b64))
        unnorm_key = str(payload.get("unnorm_key") or self.cfg.unnorm_key)
        request_id = payload.get("request_id")
        observation = {"full_image": image, "state": np.zeros(7, dtype=np.float32)}

        start = time.time()
        with self.lock:
            old_key = self.cfg.unnorm_key
            self.cfg.unnorm_key = unnorm_key
            action = get_action(self.cfg, self.model, observation, instruction, processor=self.processor)
            self.cfg.unnorm_key = old_key
        action = _postprocess_gripper(np.asarray(action, dtype=np.float32), self.cfg.postprocess_gripper)
        latency_s = time.time() - start

        return {
            "ok": True,
            "request_id": request_id,
            "model_family": self.cfg.model_family,
            "checkpoint": self.cfg.pretrained_checkpoint,
            "unnorm_key": unnorm_key,
            "postprocess_gripper": self.cfg.postprocess_gripper,
            "latency_s": latency_s,
            "action": [float(x) for x in action.tolist()],
            "action_dict": _as_action_dict(action),
            "openvla_action_dict": _as_openvla_action_dict(action),
            "note": "7D end-effector delta action: dx, dy, dz, droll, dpitch, dyaw, gripper; not seven joint angles.",
        }


def make_handler(server_state: ActionServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), fmt % args))

        def _check_auth(self) -> bool:
            if not server_state.cfg.api_key:
                return True
            auth = self.headers.get("Authorization", "")
            return auth == f"Bearer {server_state.cfg.api_key}"

        def do_GET(self) -> None:  # noqa: N802
            if self.path not in {"/", "/health", "/metadata"}:
                _json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
                return
            payload = {
                "ok": True,
                "service": "openvla-action-api",
                "model_family": server_state.cfg.model_family,
                "checkpoint": server_state.cfg.pretrained_checkpoint,
                "default_unnorm_key": server_state.cfg.unnorm_key,
                "endpoints": {"predict_action": "POST /predict_action"},
            }
            _json_response(self, HTTPStatus.OK, payload)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/predict_action":
                _json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
                return
            if not self._check_auth():
                _json_response(self, HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                payload = json.loads(body.decode("utf-8"))
                result = server_state.predict(payload)
                _json_response(self, HTTPStatus.OK, result)
            except Exception as exc:  # Keep robot client failures debuggable.
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--model_family", default="openvla")
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--unnorm_key", default="bridge_orig")
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--postprocess_gripper", choices=["none", "libero"], default="none")
    parser.add_argument("--api_key", default=os.environ.get("OPENVLA_ACTION_API_KEY"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ServerConfig(
        model_family=args.model_family,
        pretrained_checkpoint=args.pretrained_checkpoint,
        unnorm_key=args.unnorm_key,
        center_crop=args.center_crop,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        api_key=args.api_key,
        postprocess_gripper=args.postprocess_gripper,
    )
    action_server = ActionServer(cfg)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(action_server))
    print(f"[ready] OpenVLA action API listening on http://{args.host}:{args.port}")
    print("[ready] POST /predict_action with JSON fields: instruction, image_base64, optional unnorm_key")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
