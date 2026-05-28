#!/usr/bin/env python3
"""Evidence-token side-channel for real OpenVLA inference/training.

This module avoids prompt-text evidence injection. Instead, it maps structured
evidence vectors to a small set of learned soft tokens that are inserted between
the visual patch tokens and the task prompt tokens before dispatching to the
frozen OpenVLA language model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import LlamaTokenizerFast


STAGES = ("approach", "grasp", "place")


def make_openvla_prompt(instruction: str, model_path: str) -> str:
    if "v01" in model_path:
        sys_prompt = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )
        return f"{sys_prompt} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def ensure_action_start_token(
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    empty_token_id: int = 29871,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(tokenizer, LlamaTokenizerFast):
        if not torch.all(input_ids[:, -1] == empty_token_id):
            suffix = torch.full((input_ids.shape[0], 1), empty_token_id, dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([input_ids, suffix], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)],
                dim=1,
            )
    return input_ids, attention_mask


def preprocess_image(processor, image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    pixel_values = processor.image_processor(image, return_tensors="pt")["pixel_values"]
    if not isinstance(pixel_values, torch.Tensor):
        raise ValueError(f"Unsupported pixel_values type: {type(pixel_values)}")
    return pixel_values.to(device=device, dtype=dtype)


def decode_action_token_ids(vla, token_ids: np.ndarray, unnorm_key: str) -> np.ndarray:
    predicted_action_token_ids = np.asarray(token_ids, dtype=np.int64)
    discretized_actions = vla.vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=vla.bin_centers.shape[0] - 1)
    normalized_actions = vla.bin_centers[discretized_actions]

    action_norm_stats = vla.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high = np.array(action_norm_stats["q99"])
    action_low = np.array(action_norm_stats["q01"])
    return np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )


def actions_to_token_ids(vla, actions: np.ndarray, unnorm_key: str, device: torch.device) -> torch.Tensor:
    actions = np.asarray(actions, dtype=np.float32)
    action_norm_stats = vla.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    q_high = np.asarray(action_norm_stats["q99"], dtype=np.float32)
    q_low = np.asarray(action_norm_stats["q01"], dtype=np.float32)
    denom = np.maximum(q_high - q_low, 1e-6)
    normalized = np.where(mask, 2.0 * (actions - q_low) / denom - 1.0, actions)
    normalized = np.clip(normalized, -1.0, 1.0)
    distances = np.abs(vla.bin_centers[None, :] - normalized[:, None])
    discretized = np.argmin(distances, axis=1).astype(np.int64)
    token_ids = vla.vocab_size - (discretized + 1)
    return torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)


def build_multimodal_inputs(
    vla,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    evidence_tokens: torch.Tensor,
    fusion_mode: str = "insert",
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        patch_features = vla.vision_backbone(pixel_values)
        projected_patch_embeddings = vla.projector(patch_features)
        input_embeddings = vla.get_input_embeddings()(input_ids)

    fusion_mode = str(fusion_mode).lower()
    if fusion_mode == "visual_residual":
        evidence_residual = evidence_tokens.mean(dim=1, keepdim=True)
        projected_patch_embeddings = projected_patch_embeddings + evidence_residual
        multimodal_embeddings = torch.cat(
            [
                input_embeddings[:, :1, :],
                projected_patch_embeddings,
                input_embeddings[:, 1:, :],
            ],
            dim=1,
        )
        multimodal_attention_mask = torch.cat(
            [
                attention_mask[:, :1],
                torch.ones(
                    (attention_mask.shape[0], projected_patch_embeddings.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask[:, 1:],
            ],
            dim=1,
        )
        return multimodal_embeddings, multimodal_attention_mask

    if fusion_mode != "insert":
        raise ValueError(f"Unsupported evidence fusion_mode={fusion_mode!r}")

    multimodal_embeddings = torch.cat(
        [
            input_embeddings[:, :1, :],
            projected_patch_embeddings,
            evidence_tokens,
            input_embeddings[:, 1:, :],
        ],
        dim=1,
    )
    multimodal_attention_mask = torch.cat(
        [
            attention_mask[:, :1],
            torch.ones(
                (attention_mask.shape[0], projected_patch_embeddings.shape[1] + evidence_tokens.shape[1]),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            ),
            attention_mask[:, 1:],
        ],
        dim=1,
    )
    return multimodal_embeddings, multimodal_attention_mask


def build_training_labels(
    prompt_ids: torch.Tensor,
    action_token_ids: torch.Tensor,
    patch_len: int,
    evidence_len: int,
    device: torch.device,
) -> torch.Tensor:
    text_ids = torch.cat([prompt_ids, action_token_ids], dim=1)
    text_labels = torch.full_like(text_ids, fill_value=-100)
    text_labels[:, prompt_ids.shape[1] :] = action_token_ids
    ignore_prefix = torch.full((text_labels.shape[0], patch_len + evidence_len), -100, dtype=text_labels.dtype, device=device)
    return torch.cat([text_labels[:, :1], ignore_prefix, text_labels[:, 1:]], dim=1)


@dataclass
class EvidenceTokenBatch:
    channel_features: Dict[str, torch.Tensor]
    channel_mask: torch.Tensor
    stage_one_hot: torch.Tensor
    step_ratio: torch.Tensor


class EvidenceTokenAdapter(nn.Module):
    """Project structured evidence into a small set of soft prompt tokens."""

    def __init__(
        self,
        channel_dims: Dict[str, int],
        channels: Sequence[str],
        hidden_size: int,
        num_global_tokens: int = 2,
        proj_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.channels = tuple(channels)
        self.hidden_size = int(hidden_size)
        self.num_global_tokens = int(num_global_tokens)
        self.proj_dim = int(proj_dim)

        self.channel_projs = nn.ModuleDict(
            {
                ch: nn.Sequential(
                    nn.Linear(int(channel_dims[ch]), self.proj_dim),
                    nn.GELU(),
                    nn.Linear(self.proj_dim, self.hidden_size),
                )
                for ch in self.channels
            }
        )
        self.channel_type = nn.Parameter(torch.randn(len(self.channels), self.hidden_size) * 0.02)
        self.channel_absent = nn.Parameter(torch.randn(len(self.channels), self.hidden_size) * 0.02)
        self.global_mlp = nn.Sequential(
            nn.Linear(self.hidden_size + len(self.channels) + len(STAGES) + 1, self.proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.proj_dim, self.num_global_tokens * self.hidden_size),
        )
        self.global_pos = nn.Parameter(torch.randn(self.num_global_tokens, self.hidden_size) * 0.02)
        self.trace_head = nn.Sequential(
            nn.Linear(self.hidden_size + len(self.channels) + len(STAGES) + 1, self.proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.proj_dim, len(self.channels)),
        )
        self.norm = nn.LayerNorm(self.hidden_size)

    def _encode_tokens_and_state(
        self,
        channel_features: Dict[str, torch.Tensor],
        channel_mask: torch.Tensor,
        stage_one_hot: torch.Tensor,
        step_ratio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = channel_mask.shape[0]
        channel_tokens = []
        encoded = []
        for idx, ch in enumerate(self.channels):
            feat = self.channel_projs[ch](channel_features[ch])
            encoded.append(feat)
            present = channel_mask[:, idx : idx + 1]
            token = feat * present + self.channel_absent[idx].unsqueeze(0) * (1.0 - present)
            token = token + self.channel_type[idx].unsqueeze(0)
            channel_tokens.append(token.unsqueeze(1))

        encoded_stack = torch.stack(encoded, dim=1)
        denom = torch.clamp(channel_mask.sum(dim=1, keepdim=True), min=1.0)
        pooled = torch.sum(encoded_stack * channel_mask.unsqueeze(-1), dim=1) / denom
        aux_state = torch.cat([pooled, channel_mask, stage_one_hot, step_ratio], dim=1)
        global_tokens = self.global_mlp(aux_state).view(batch, self.num_global_tokens, self.hidden_size)
        global_tokens = global_tokens + self.global_pos.unsqueeze(0)
        out = torch.cat([global_tokens] + channel_tokens, dim=1)
        return self.norm(out), aux_state

    def forward(
        self,
        channel_features: Dict[str, torch.Tensor],
        channel_mask: torch.Tensor,
        stage_one_hot: torch.Tensor,
        step_ratio: torch.Tensor,
    ) -> torch.Tensor:
        tokens, _ = self._encode_tokens_and_state(
            channel_features=channel_features,
            channel_mask=channel_mask,
            stage_one_hot=stage_one_hot,
            step_ratio=step_ratio,
        )
        return tokens

    def predict_trace_logits(
        self,
        channel_features: Dict[str, torch.Tensor],
        channel_mask: torch.Tensor,
        stage_one_hot: torch.Tensor,
        step_ratio: torch.Tensor,
    ) -> torch.Tensor:
        _, aux_state = self._encode_tokens_and_state(
            channel_features=channel_features,
            channel_mask=channel_mask,
            stage_one_hot=stage_one_hot,
            step_ratio=step_ratio,
        )
        return self.trace_head(aux_state)


@torch.inference_mode()
def predict_action_with_soft_evidence(
    vla,
    processor,
    tokenizer,
    adapter: EvidenceTokenAdapter,
    image: Image.Image,
    instruction: str,
    evidence_batch: EvidenceTokenBatch,
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    unnorm_key: str,
    evidence_scale: float = 1.0,
    fusion_mode: str = "insert",
) -> np.ndarray:
    prompt = make_openvla_prompt(instruction, model_path)
    tok = tokenizer(prompt, truncation=True, return_tensors="pt")
    input_ids = tok.input_ids.to(device)
    attention_mask = tok.attention_mask.to(device)
    input_ids, attention_mask = ensure_action_start_token(tokenizer, input_ids, attention_mask)
    pixel_values = preprocess_image(processor, image, device=device, dtype=dtype)
    evidence_tokens = adapter(
        channel_features={k: v.to(device) for k, v in evidence_batch.channel_features.items()},
        channel_mask=evidence_batch.channel_mask.to(device),
        stage_one_hot=evidence_batch.stage_one_hot.to(device),
        step_ratio=evidence_batch.step_ratio.to(device),
    ).to(dtype=dtype)
    evidence_tokens = evidence_tokens * float(evidence_scale)
    mm_embeds, mm_attn = build_multimodal_inputs(
        vla,
        input_ids,
        attention_mask,
        pixel_values,
        evidence_tokens,
        fusion_mode=fusion_mode,
    )
    action_dim = vla.get_action_dim(unnorm_key)
    generated_ids = vla.language_model.generate(
        inputs_embeds=mm_embeds,
        attention_mask=mm_attn,
        max_length=mm_attn.shape[-1] + action_dim,
        do_sample=False,
    )
    return decode_action_token_ids(vla, generated_ids[0, -action_dim:].cpu().numpy(), unnorm_key)
