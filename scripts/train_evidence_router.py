#!/usr/bin/env python3
"""Train a learnable evidence gating policy on full visual evidence manifests."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualthink_vla.models.evidence_router import (
    LearnedEvidencePolicy,
    LearnedGatingDataset,
    apply_stage_dropout,
    build_counterfactual_utility_map,
    budget_entropy,
    channel_cost_tensor,
    coverage_underuse_penalty,
    effective_phase_count,
    gate_entropy,
    load_yaml,
    mean_action_l1,
    move_batch_to_device,
    phase_balance_loss,
    phase_sample_entropy,
    resolve_budget_values,
    resolve_channels,
    resolve_route_bank,
    route_bank_tensor,
    route_targets_from_utility,
    sample_route_mixture_gates,
    sample_budget_topk_gates,
    stage_budget_targets,
    stage_channel_targets,
    stage_prior_targets,
    temperature_schedule,
    temporal_phase_smoothness,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    val_n = max(1, int(round(n * val_ratio))) if n > 1 else 0
    val_idx = idx[:val_n]
    train_idx = idx[val_n:] if val_n < n else idx[:]
    if not train_idx:
        train_idx = idx[:]
        val_idx = idx[: min(1, len(idx))]
    return train_idx, val_idx


def build_loaders(
    dataset: LearnedGatingDataset,
    batch_size: int,
    val_ratio: float,
    seed: int,
    sequential_order: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_idx, val_idx = split_indices(len(dataset), val_ratio=val_ratio, seed=seed)
    if sequential_order:
        order_key = lambda i: (dataset.records[i].episode_idx, dataset.records[i].step_idx)
        train_idx = sorted(train_idx, key=order_key)
        val_idx = sorted(val_idx, key=order_key)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=not sequential_order, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def teacher_forward(model: LearnedEvidencePolicy, batch: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
    ctx = model.encode_context(batch)
    channel_embeds = model.encode_channels(batch)
    pred = model.teacher(ctx, channel_embeds)
    return pred, ctx, channel_embeds


def apply_teacher_channel_dropout(
    channel_embeds: dict[str, torch.Tensor],
    channels: tuple[str, ...],
    dropout_prob: float,
    min_drop: int,
    max_drop: int,
    channel_weights: dict[str, float] | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if dropout_prob <= 0.0 or max_drop <= 0:
        batch_size = next(iter(channel_embeds.values())).shape[0]
        keep_mask = torch.ones((batch_size, len(channels)), device=next(iter(channel_embeds.values())).device)
        return channel_embeds, keep_mask

    device = next(iter(channel_embeds.values())).device
    batch_size = next(iter(channel_embeds.values())).shape[0]
    weights = torch.tensor(
        [float((channel_weights or {}).get(ch, 1.0)) for ch in channels],
        dtype=torch.float32,
        device=device,
    )
    weights = torch.clamp(weights, min=1e-6)
    keep_mask = torch.ones((batch_size, len(channels)), dtype=torch.float32, device=device)
    max_drop = min(max_drop, max(0, len(channels) - 1))
    min_drop = min(min_drop, max_drop)
    if max_drop <= 0:
        return channel_embeds, keep_mask

    for row_idx in range(batch_size):
        if float(torch.rand(1, device=device).item()) >= dropout_prob:
            continue
        num_drop = min_drop
        if max_drop > min_drop:
            num_drop += int(torch.randint(0, max_drop - min_drop + 1, (1,), device=device).item())
        if num_drop <= 0:
            continue
        drop_idx = torch.multinomial(weights, num_samples=num_drop, replacement=False)
        keep_mask[row_idx, drop_idx] = 0.0

    dropped = {
        ch: emb * keep_mask[:, idx : idx + 1]
        for idx, (ch, emb) in enumerate(channel_embeds.items())
    }
    return dropped, keep_mask


def evaluate_teacher(model: LearnedEvidencePolicy, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_mse = 0.0
    total_l1 = 0.0
    n = 0
    action_dim = None
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            pred, _, _ = teacher_forward(model, batch)
            gt = batch["action"]
            total_mse += float(F.mse_loss(pred, gt, reduction="sum").item())
            total_l1 += float(torch.abs(pred - gt).sum().item())
            n += gt.shape[0]
            action_dim = gt.shape[1]
    if action_dim is None:
        action_dim = int(getattr(model.teacher, "out_dim", 1)) if hasattr(model.teacher, "out_dim") else 1
    return {
        "mse": total_mse / max(1, n),
        "l1": total_l1 / max(1, n * action_dim),
    }


def evaluate_joint(
    model: LearnedEvidencePolicy,
    loader: DataLoader,
    device: torch.device,
    channels: tuple[str, ...],
    priors: dict,
    budget_priors: dict,
    coverage_cfg: dict,
    channel_costs: torch.Tensor,
    budget_values: tuple[int, ...],
    gate_cfg: dict,
    loss_cfg: dict,
    utility_cfg: dict,
    usage_band_cfg: dict,
    stage_diversity_cfg: dict,
    stage_conditioning_enabled: bool,
    latent_phase_cfg: dict,
    route_mixture_cfg: dict,
    route_bank_masks: torch.Tensor | None,
    load_balance_cfg: dict,
    ranking_cfg: dict,
) -> dict:
    model.eval()
    total_action = 0.0
    total_distill = 0.0
    total_cost = 0.0
    total_prior = 0.0
    total_budget = 0.0
    total_coverage = 0.0
    total_entropy = 0.0
    total_utility = 0.0
    total_utility_budget = 0.0
    total_utility_ranking = 0.0
    total_phase_sample_entropy = 0.0
    total_phase_balance = 0.0
    total_effective_phases = 0.0
    total_phase_temporal = 0.0
    total_usage_band = 0.0
    total_stage_diversity = 0.0
    total_route = 0.0
    total_load_balance = 0.0
    total_selected = torch.zeros((len(channels),), dtype=torch.float64)
    total_budget_level = 0.0
    n = 0
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            gt = batch["action"]
            t_pred, ctx, channel_embeds = teacher_forward(model, batch)
            gate_stage = resolved_stage_input(batch["stage_one_hot"], stage_conditioning_enabled)
            gate_out = model.forward_gate(
                ctx,
                batch,
                gate_stage,
                phase_temperature=float(latent_phase_cfg.get("temperature_end", gate_cfg["temperature_end"])),
                hard_phase=bool(latent_phase_cfg.get("hard_assignment", False)),
                training=False,
            )
            utility_pred = gate_out["utility_pred"]
            phase_probs = gate_out["phase_probs"]
            phase_assign = gate_out["phase_assign"]
            if route_bank_masks is not None:
                hard_gates, gate_probs, hard_route, route_probs, hard_budget, budget_probs = sample_route_mixture_gates(
                    gate_out["route_logits"],
                    route_bank_masks,
                    budget_values=budget_values,
                    temperature=float(gate_cfg["temperature_end"]),
                    training=False,
                )
                route_targets = route_targets_from_utility(
                    batch["utility_targets"],
                    route_bank_masks,
                    channel_costs,
                    route_cost_weight=float(route_mixture_cfg.get("target_cost_weight", 0.05)),
                )
                route_loss = torch.mean(
                    -(route_targets * torch.log(route_probs.clamp(1e-6, 1.0))).sum(dim=1)
                )
            else:
                hard_gates, gate_probs, hard_budget, budget_probs = sample_budget_topk_gates(
                    gate_out["channel_logits"],
                    gate_out["budget_logits"],
                    budget_values=budget_values,
                    temperature=float(gate_cfg["temperature_end"]),
                    training=False,
                )
                route_loss = torch.zeros((), device=device)
            s_pred = model.student(ctx, channel_embeds, hard_gates, gate_probs)

            if stage_conditioning_enabled:
                prior_targets = stage_prior_targets(batch["stage_one_hot"], priors, channels, device)
                budget_targets = stage_budget_targets(batch["stage_one_hot"], budget_priors, device)
            else:
                prior_targets = torch.zeros_like(gate_probs)
                budget_targets = torch.zeros_like(budget_probs)
            total_action += float(F.mse_loss(s_pred, gt, reduction="sum").item())
            total_distill += float(F.mse_loss(s_pred, t_pred, reduction="sum").item())
            total_cost += float((hard_gates * channel_costs.unsqueeze(0)).sum().item())
            if stage_conditioning_enabled:
                total_prior += float(F.binary_cross_entropy(gate_probs, prior_targets, reduction="sum").item())
                total_budget += float((-(budget_targets * torch.log(budget_probs.clamp(1e-6, 1.0))).sum(dim=1)).sum().item())
                coverage_targets = stage_channel_targets(batch["stage_one_hot"], coverage_cfg, channels, device)
                total_coverage += float((coverage_underuse_penalty(gate_probs, coverage_targets) * gt.shape[0]).item())
            total_entropy += float((gate_entropy(gate_probs) + budget_entropy(budget_probs)).sum().item())
            if float(usage_band_cfg.get("weight", 0.0)) > 0.0:
                total_usage_band += float(
                    (usage_band_penalty(
                        gate_probs,
                        float(usage_band_cfg.get("low", 0.1)),
                        float(usage_band_cfg.get("high", 0.9)),
                    ) * gt.shape[0]).item()
                )
            if float(stage_diversity_cfg.get("weight", 0.0)) > 0.0:
                total_stage_diversity += float(
                    (stage_route_diversity_reward(
                        gate_probs,
                        batch["stage_one_hot"],
                        min_stage_samples=int(stage_diversity_cfg.get("min_stage_samples", 4)),
                    ) * gt.shape[0]).item()
                )
            utility_mask = batch["utility_mask"].view(-1, 1)
            utility_weight = batch["utility_weight"].view(-1, 1)
            utility_sample_weight = utility_mask * utility_weight
            utility_loss = compute_utility_loss(
                utility_pred,
                batch["utility_targets"],
                utility_sample_weight,
                utility_cfg,
            )
            utility_ranking_loss = compute_utility_ranking_loss(
                utility_pred,
                batch["utility_raw_targets"],
                utility_sample_weight,
                ranking_cfg,
            )
            utility_budget_loss = (
                (-(batch["utility_budget_target"] * torch.log(budget_probs.clamp(1e-6, 1.0))).sum(dim=1) * utility_sample_weight.view(-1)).sum()
                / utility_sample_weight.sum().clamp(min=1.0)
            )
            total_utility += float(utility_loss.item() * gt.shape[0])
            total_utility_ranking += float(utility_ranking_loss.item() * gt.shape[0])
            total_utility_budget += float(utility_budget_loss.item() * gt.shape[0])
            total_route += float(route_loss.item() * gt.shape[0])
            if float(load_balance_cfg.get("weight", 0.0)) > 0.0:
                total_load_balance += float((marginal_usage_entropy_reward(gate_probs) * gt.shape[0]).item())
            if phase_probs is not None:
                total_phase_sample_entropy += float(phase_sample_entropy(phase_probs).sum().item())
                total_phase_balance += float((phase_balance_loss(phase_probs) * gt.shape[0]).item())
                total_effective_phases += float((effective_phase_count(phase_probs, float(latent_phase_cfg.get("active_threshold", 0.05))) * gt.shape[0]).item())
                phase_repr = phase_assign if phase_assign is not None else phase_probs
                total_phase_temporal += float(
                    temporal_phase_smoothness(phase_repr, batch["episode_idx"], batch["step_idx"]).item() * gt.shape[0]
                )
            total_selected += hard_gates.sum(dim=0).detach().cpu().to(torch.float64)
            total_budget_level += float(hard_gates.sum().item())
            n += gt.shape[0]
    return {
        "action_mse": total_action / max(1, n),
        "distill_mse": total_distill / max(1, n),
        "cost": total_cost / max(1, n),
        "prior_bce": total_prior / max(1, n),
        "budget_ce": total_budget / max(1, n),
        "coverage": total_coverage / max(1, n),
        "entropy": total_entropy / max(1, n),
        "utility": total_utility / max(1, n),
        "utility_ranking": total_utility_ranking / max(1, n),
        "utility_budget": total_utility_budget / max(1, n),
        "route": total_route / max(1, n),
        "phase_sample_entropy": total_phase_sample_entropy / max(1, n),
        "phase_balance": total_phase_balance / max(1, n),
        "effective_phases": total_effective_phases / max(1, n),
        "phase_temporal": total_phase_temporal / max(1, n),
        "usage_band": total_usage_band / max(1, n),
        "stage_diversity": total_stage_diversity / max(1, n),
        "load_balance": total_load_balance / max(1, n),
        "avg_budget": total_budget_level / max(1, n),
        "selected_rate": {ch: float(total_selected[i] / max(1, n)) for i, ch in enumerate(channels)},
    }


def anneal_scale(progress: float, warmup_fraction: float) -> float:
    if warmup_fraction <= 0.0:
        return 1.0
    progress = min(max(progress, 0.0), 1.0)
    return min(1.0, progress / warmup_fraction)


def resolved_stage_input(stage_one_hot: torch.Tensor, enabled: bool) -> torch.Tensor:
    return stage_one_hot if enabled else torch.zeros_like(stage_one_hot)


def usage_band_penalty(gate_probs: torch.Tensor, low: float, high: float) -> torch.Tensor:
    usage = gate_probs.mean(dim=0)
    low_pen = torch.relu(torch.full_like(usage, float(low)) - usage)
    high_pen = torch.relu(usage - torch.full_like(usage, float(high)))
    return (low_pen + high_pen).mean()


def marginal_usage_entropy_reward(gate_probs: torch.Tensor) -> torch.Tensor:
    usage = gate_probs.mean(dim=0).clamp(1e-6, 1.0 - 1e-6)
    return -(usage * usage.log() + (1.0 - usage) * (1.0 - usage).log()).mean()


def compute_utility_loss(
    utility_pred: torch.Tensor,
    utility_target: torch.Tensor,
    utility_sample_weight: torch.Tensor,
    utility_cfg: dict,
) -> torch.Tensor:
    mode = str(utility_cfg.get("mode", "absolute"))
    if mode == "hybrid":
        temperature = float(utility_cfg.get("relative_temperature", 0.25))
        absolute_weight = float(utility_cfg.get("absolute_weight", 0.7))
        relative_weight = float(utility_cfg.get("relative_weight", 0.3))
        abs_loss = F.smooth_l1_loss(utility_pred, utility_target, reduction="none")
        abs_loss = (abs_loss * utility_sample_weight).sum() / utility_sample_weight.sum().clamp(min=1.0)
        target_probs = torch.softmax(utility_target / max(1e-6, temperature), dim=1)
        pred_log_probs = torch.log_softmax(utility_pred / max(1e-6, temperature), dim=1)
        rel_per_sample = -(target_probs * pred_log_probs).sum(dim=1, keepdim=True)
        rel_loss = (rel_per_sample * utility_sample_weight).sum() / utility_sample_weight.sum().clamp(min=1.0)
        return absolute_weight * abs_loss + relative_weight * rel_loss
    if mode == "relative":
        temperature = float(utility_cfg.get("relative_temperature", 0.25))
        target_probs = torch.softmax(utility_target / max(1e-6, temperature), dim=1)
        pred_log_probs = torch.log_softmax(utility_pred / max(1e-6, temperature), dim=1)
        per_sample = -(target_probs * pred_log_probs).sum(dim=1, keepdim=True)
        return (per_sample * utility_sample_weight).sum() / utility_sample_weight.sum().clamp(min=1.0)
    utility_loss = F.smooth_l1_loss(utility_pred, utility_target, reduction="none")
    return (utility_loss * utility_sample_weight).sum() / utility_sample_weight.sum().clamp(min=1.0)


def compute_utility_ranking_loss(
    utility_pred: torch.Tensor,
    utility_target_raw: torch.Tensor,
    utility_sample_weight: torch.Tensor,
    ranking_cfg: dict,
) -> torch.Tensor:
    if float(ranking_cfg.get("weight", 0.0)) <= 0.0:
        return torch.zeros((), device=utility_pred.device)
    margin = float(ranking_cfg.get("margin", 0.05))
    min_diff = float(ranking_cfg.get("min_diff", 1.0e-4))
    batch_weights = utility_sample_weight.view(-1)
    losses = []
    weights = []
    num_channels = utility_pred.shape[1]
    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            target_diff = utility_target_raw[:, i] - utility_target_raw[:, j]
            valid = torch.abs(target_diff) > min_diff
            if not bool(valid.any()):
                continue
            sign = torch.sign(target_diff[valid])
            pred_diff = utility_pred[:, i][valid] - utility_pred[:, j][valid]
            pair_loss = torch.relu(margin - sign * pred_diff)
            pair_weight = batch_weights[valid]
            losses.append((pair_loss * pair_weight).sum())
            weights.append(pair_weight.sum())
    if not losses:
        return torch.zeros((), device=utility_pred.device)
    return torch.stack(losses).sum() / torch.stack(weights).sum().clamp(min=1.0)


def apply_student_channel_dropout(
    hard_gates: torch.Tensor,
    drop_prob: float,
    min_keep: int = 1,
) -> torch.Tensor:
    if drop_prob <= 0.0:
        return hard_gates
    dropped = hard_gates.clone()
    batch, channels = dropped.shape
    device = dropped.device
    rand = torch.rand_like(dropped)
    for i in range(batch):
        selected = torch.nonzero(dropped[i] > 0.5, as_tuple=False).flatten()
        if selected.numel() <= min_keep:
            continue
        drop_mask = rand[i, selected] < drop_prob
        if int(drop_mask.sum().item()) <= 0:
            continue
        keep_after = int(selected.numel() - drop_mask.sum().item())
        if keep_after < min_keep:
            max_drop = max(0, int(selected.numel()) - int(min_keep))
            if max_drop <= 0:
                continue
            drop_indices = torch.nonzero(drop_mask, as_tuple=False).flatten()[:max_drop]
            drop_mask = torch.zeros_like(drop_mask, dtype=torch.bool)
            drop_mask[drop_indices] = True
        dropped[i, selected[drop_mask]] = 0.0
    return dropped


def apply_student_epsilon_greedy(
    hard_gates: torch.Tensor,
    epsilon: float,
    min_keep: int = 1,
    max_keep: int | None = None,
) -> torch.Tensor:
    if epsilon <= 0.0:
        return hard_gates
    explored = hard_gates.clone()
    batch, channels = explored.shape
    device = explored.device
    max_keep = channels if max_keep is None else max(min_keep, min(channels, int(max_keep)))
    for i in range(batch):
        if float(torch.rand(1, device=device).item()) >= epsilon:
            continue
        k = int(torch.randint(min_keep, max_keep + 1, (1,), device=device).item())
        chosen = torch.randperm(channels, device=device)[:k]
        explored[i].zero_()
        explored[i, chosen] = 1.0
    return explored


def stage_route_diversity_reward(
    gate_probs: torch.Tensor,
    stage_one_hot: torch.Tensor,
    min_stage_samples: int = 4,
) -> torch.Tensor:
    stage_ids = torch.argmax(stage_one_hot, dim=1)
    rewards = []
    for stage in stage_ids.unique(sorted=True):
        mask = stage_ids == stage
        if int(mask.sum().item()) < min_stage_samples:
            continue
        probs = gate_probs[mask]
        rewards.append(probs.var(dim=0, unbiased=False).mean())
    if not rewards:
        return torch.zeros((), device=gate_probs.device)
    return torch.stack(rewards).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "evidence_router.yaml"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_yaml(args.config)
    model_cfg = cfg["model"]
    text_cfg = model_cfg.get("text_encoder", {})
    text_encoder_type = str(text_cfg.get("type", "bow"))
    phase_proxy_cfg = cfg.get("phase_proxy", {})
    stage_conditioning_cfg = cfg.get("stage_conditioning", {})
    stage_conditioning_enabled = bool(stage_conditioning_cfg.get("enabled", True))
    latent_phase_cfg = cfg.get("latent_phase", {})
    latent_phase_enabled = bool(latent_phase_cfg.get("enabled", False))
    route_mixture_cfg = cfg.get("route_mixture", {})
    route_mixture_enabled = bool(route_mixture_cfg.get("enabled", False))
    train_cfg = cfg["training"]
    gate_cfg = cfg["gating"]
    loss_cfg = cfg["loss_weights"]
    utility_cfg = cfg.get("utility", {})
    teacher_dropout_cfg = cfg.get("teacher_dropout", {})
    usage_band_cfg = cfg.get("usage_band", {})
    student_dropout_cfg = cfg.get("student_dropout", {})
    student_exploration_cfg = cfg.get("student_exploration", {})
    stage_diversity_cfg = cfg.get("stage_diversity", {})
    load_balance_cfg = cfg.get("load_balance", {})
    ranking_cfg = cfg.get("ranking", {})
    channels = resolve_channels(cfg)
    budget_values = resolve_budget_values(cfg, channels)
    route_bank = resolve_route_bank(cfg, channels)
    priors = cfg["stage_keep_priors"]
    budget_priors = cfg["stage_budget_priors"]
    coverage_cfg = cfg["coverage"]
    channel_costs = channel_cost_tensor({ch: cfg["channels"][ch]["cost"] for ch in channels}, channels, device)
    route_bank_masks = route_bank_tensor(route_bank, channels, device) if route_mixture_enabled else None

    dataset = LearnedGatingDataset(
        args.feature_manifest,
        image_size=int(model_cfg["image_size"]),
        bow_dim=int(model_cfg["bow_dim"]),
        query_bow_dim=int(model_cfg.get("query_bow_dim", 64)),
        limit=int(args.limit),
        channels=channels,
        text_encoder_type=text_encoder_type,
        tokenizer_path=text_cfg.get("tokenizer_path"),
        instruction_max_len=int(text_cfg.get("instruction_max_len", 24)),
        query_max_len=int(text_cfg.get("query_max_len", 12)),
        phase_proxy_cfg=phase_proxy_cfg,
    )
    train_loader, val_loader = build_loaders(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        val_ratio=float(train_cfg["val_ratio"]),
        seed=args.seed,
        sequential_order=bool(latent_phase_cfg.get("sequential_batches", False)),
    )

    model = LearnedEvidencePolicy(
        bow_dim=int(model_cfg["bow_dim"]),
        query_bow_dim=int(model_cfg.get("query_bow_dim", 64)),
        ctx_dim=int(model_cfg["ctx_dim"]),
        channel_dim=int(model_cfg["channel_dim"]),
        teacher_hidden=int(model_cfg["teacher_hidden"]),
        student_hidden=int(model_cfg["student_hidden"]),
        action_dim=int(dataset.action_dim),
        channels=channels,
        budget_values=budget_values,
        text_encoder_type=text_encoder_type,
        text_vocab_size=int(getattr(dataset, "text_vocab_size", 0)),
        instruction_max_len=int(text_cfg.get("instruction_max_len", 24)),
        query_max_len=int(text_cfg.get("query_max_len", 12)),
        text_embed_dim=int(text_cfg.get("embed_dim", 96)),
        text_hidden_dim=int(text_cfg.get("hidden_dim", 128)),
        gate_type="route_mixture" if route_mixture_enabled else ("latent_phase" if latent_phase_enabled else "stage_conditioned"),
        latent_phase_slots=int(latent_phase_cfg.get("max_phase_slots", 8)),
        route_bank=route_bank,
    ).to(device)

    teacher_params = list(model.context.parameters())
    for ch in channels:
        teacher_params += list(getattr(model, f"{ch}_encoder").parameters())
    teacher_params += list(model.teacher.parameters())
    teacher_opt = torch.optim.AdamW(
        teacher_params,
        lr=float(train_cfg["lr_teacher"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    gate_opt = torch.optim.AdamW(
        model.gate.parameters(),
        lr=float(train_cfg["lr_gate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    student_opt = torch.optim.AdamW(
        list(model.student.parameters()) + list(model.gate.parameters()),
        lr=float(train_cfg["lr_student"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "train_metrics.jsonl"
    resolved_cfg = {
        "config_path": args.config,
        "seed": args.seed,
        "limit": args.limit,
        "dataset_size": len(dataset),
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "channels": list(channels),
        "budget_values": list(budget_values),
        "phase_proxy": phase_proxy_cfg,
        "config": cfg,
    }
    resolved_cfg["config"].setdefault("model", {})
    resolved_cfg["config"]["model"].setdefault("text_encoder", {})
    if text_encoder_type == "sequence":
        resolved_cfg["config"]["model"]["text_encoder"]["vocab_size"] = int(getattr(dataset, "text_vocab_size", 0))
    (out_dir / "resolved_config.json").write_text(json.dumps(resolved_cfg, indent=2), encoding="utf-8")

    best_teacher = float("inf")
    best_student = float("inf")
    teacher_epochs = int(train_cfg["teacher_epochs"])
    warmup_epochs = int(train_cfg["gate_warmup_epochs"]) if (stage_conditioning_enabled and not latent_phase_enabled and not route_mixture_enabled) else 0
    joint_epochs = int(train_cfg["joint_epochs"])

    with metrics_path.open("w", encoding="utf-8") as mf:
        # Phase 1: teacher pretraining on full evidence.
        for epoch in range(1, teacher_epochs + 1):
            model.train()
            sum_mse = 0.0
            sum_teacher_keep = 0.0
            steps = 0
            start_ts = time.time()
            total_steps = len(train_loader)
            print(f"[teacher] epoch={epoch}/{teacher_epochs} steps={total_steps}", flush=True)
            for batch in train_loader:
                batch = move_batch_to_device(batch, device)
                gt = batch["action"]
                ctx = model.encode_context(batch)
                channel_embeds = model.encode_channels(batch)
                dropped_embeds, keep_mask = apply_teacher_channel_dropout(
                    channel_embeds,
                    channels=channels,
                    dropout_prob=float(teacher_dropout_cfg.get("prob", 0.0)),
                    min_drop=int(teacher_dropout_cfg.get("min_drop", 0)),
                    max_drop=int(teacher_dropout_cfg.get("max_drop", 0)),
                    channel_weights=teacher_dropout_cfg.get("channel_weights", {}),
                )
                pred = model.teacher(ctx, dropped_embeds)
                loss = F.mse_loss(pred, gt)
                teacher_opt.zero_grad()
                loss.backward()
                teacher_opt.step()
                sum_mse += float(loss.item())
                sum_teacher_keep += float(keep_mask.mean().item())
                steps += 1
                if steps % args.log_every == 0 or steps == total_steps:
                    elapsed = max(1e-6, time.time() - start_ts)
                    speed = steps / elapsed
                    eta = int(max(0, total_steps - steps) / max(1e-6, speed))
                    print(
                        f"[progress] phase=teacher epoch={epoch}/{teacher_epochs} step={steps}/{total_steps} "
                        f"loss={loss.item():.6f} keep={keep_mask.mean().item():.4f} "
                        f"speed={speed:.2f} steps/s eta={eta}s",
                        flush=True,
                    )
            val_metrics = evaluate_teacher(model, val_loader, device)
            rec = {
                "phase": "teacher",
                "epoch": epoch,
                "train_mse": sum_mse / max(1, steps),
                "train_teacher_keep_rate": sum_teacher_keep / max(1, steps),
                "val_mse": val_metrics["mse"],
                "val_l1": val_metrics["l1"],
            }
            mf.write(json.dumps(rec) + "\n")
            print(rec, flush=True)
            if val_metrics["mse"] < best_teacher:
                best_teacher = val_metrics["mse"]
                torch.save(model.context.state_dict(), out_dir / "context.pt")
                if getattr(model, "instruction_text_encoder", None) is not None:
                    torch.save(model.instruction_text_encoder.state_dict(), out_dir / "instruction_text_encoder.pt")
                if getattr(model, "query_text_encoder", None) is not None:
                    torch.save(model.query_text_encoder.state_dict(), out_dir / "query_text_encoder.pt")
                for ch in channels:
                    torch.save(getattr(model, f"{ch}_encoder").state_dict(), out_dir / f"{ch}_encoder.pt")
                torch.save(model.teacher.state_dict(), out_dir / "teacher.pt")

        # Reload best teacher backbone before later phases.
        model.context.load_state_dict(torch.load(out_dir / "context.pt", map_location=device))
        if getattr(model, "instruction_text_encoder", None) is not None and (out_dir / "instruction_text_encoder.pt").exists():
            model.instruction_text_encoder.load_state_dict(
                torch.load(out_dir / "instruction_text_encoder.pt", map_location=device)
            )
        if getattr(model, "query_text_encoder", None) is not None and (out_dir / "query_text_encoder.pt").exists():
            model.query_text_encoder.load_state_dict(
                torch.load(out_dir / "query_text_encoder.pt", map_location=device)
            )
        for ch in channels:
            getattr(model, f"{ch}_encoder").load_state_dict(torch.load(out_dir / f"{ch}_encoder.pt", map_location=device))
        model.teacher.load_state_dict(torch.load(out_dir / "teacher.pt", map_location=device))

        for param in model.teacher.parameters():
            param.requires_grad = False

        utility_map = build_counterfactual_utility_map(
            model,
            dataset,
            device=device,
            channels=channels,
            channel_costs=[cfg["channels"][ch]["cost"] for ch in channels],
            budget_values=budget_values,
            batch_size=int(utility_cfg.get("batch_size", max(32, int(train_cfg["batch_size"])))),
            relative_keep_thresh=float(utility_cfg.get("relative_keep_thresh", 0.35)),
            absolute_score_thresh=float(utility_cfg.get("absolute_score_thresh", 1e-4)),
            hard_relation_thresh_raw=float(utility_cfg.get("hard_relation_thresh_raw", 0.0)),
            hard_relation_weight=float(utility_cfg.get("hard_relation_weight", 1.0)),
        )
        dataset.attach_counterfactual_utilities(utility_map, channels=channels, budget_values=budget_values)
        with (out_dir / "counterfactual_utilities.jsonl").open("w", encoding="utf-8") as uf:
            for rec in dataset.records:
                key = f"{rec.episode_idx}:{rec.step_idx}:{rec.npz_path}"
                if key in utility_map:
                    uf.write(json.dumps(utility_map[key]) + "\n")

        # Phase 2: gate warmup from stage priors.
        for epoch in range(1, warmup_epochs + 1):
            model.train()
            sum_bce = 0.0
            sum_budget_bce = 0.0
            steps = 0
            start_ts = time.time()
            total_steps = len(train_loader)
            print(f"[gate_warmup] epoch={epoch}/{warmup_epochs} steps={total_steps}", flush=True)
            for batch in train_loader:
                batch = move_batch_to_device(batch, device)
                ctx = model.encode_context(batch)
                gate_stage = resolved_stage_input(batch["stage_one_hot"], stage_conditioning_enabled)
                channel_logits, budget_logits, utility_pred = model.gate(model.gate_inputs(ctx, batch), gate_stage)
                probs = torch.sigmoid(channel_logits)
                budget_probs = torch.softmax(budget_logits, dim=1)
                prior_targets = stage_prior_targets(batch["stage_one_hot"], priors, channels, device)
                budget_targets = stage_budget_targets(batch["stage_one_hot"], budget_priors, device)
                prior_loss = F.binary_cross_entropy(probs, prior_targets)
                budget_loss = torch.mean(
                    -(budget_targets * torch.log(budget_probs.clamp(1e-6, 1.0))).sum(dim=1)
                )
                loss = prior_loss + float(loss_cfg.get("budget", 0.0)) * budget_loss
                gate_opt.zero_grad()
                loss.backward()
                gate_opt.step()
                sum_bce += float(prior_loss.item())
                sum_budget_bce += float(budget_loss.item())
                steps += 1
                if steps % args.log_every == 0 or steps == total_steps:
                    elapsed = max(1e-6, time.time() - start_ts)
                    speed = steps / elapsed
                    eta = int(max(0, total_steps - steps) / max(1e-6, speed))
                    print(
                        f"[progress] phase=gate_warmup epoch={epoch}/{warmup_epochs} step={steps}/{total_steps} "
                        f"prior_bce={prior_loss.item():.6f} budget_ce={budget_loss.item():.6f} "
                        f"speed={speed:.2f} steps/s eta={eta}s",
                        flush=True,
                    )
            val_joint = evaluate_joint(model, val_loader, device, channels, priors, budget_priors, coverage_cfg, channel_costs, budget_values, gate_cfg, loss_cfg, utility_cfg, usage_band_cfg, stage_diversity_cfg, stage_conditioning_enabled, latent_phase_cfg, route_mixture_cfg, route_bank_masks, load_balance_cfg, ranking_cfg)
            rec = {
                "phase": "gate_warmup",
                "epoch": epoch,
                "train_prior_bce": sum_bce / max(1, steps),
                "train_budget_ce": sum_budget_bce / max(1, steps),
                "val_prior_bce": val_joint["prior_bce"],
                "val_budget_ce": val_joint["budget_ce"],
                "val_coverage": val_joint["coverage"],
                "val_utility": val_joint["utility"],
                "val_utility_ranking": val_joint["utility_ranking"],
                "val_utility_budget": val_joint["utility_budget"],
                "val_load_balance": val_joint["load_balance"],
                "val_route": val_joint["route"],
                "val_avg_budget": val_joint["avg_budget"],
                "val_selected_rate": val_joint["selected_rate"],
            }
            mf.write(json.dumps(rec) + "\n")
            print(rec, flush=True)

        # Phase 3: joint training of student + learned gate.
        total_joint_steps = max(1, joint_epochs * len(train_loader))
        seen_steps = 0
        anneal_fraction = float(train_cfg.get("cost_budget_anneal_fraction", 0.0))
        for epoch in range(1, joint_epochs + 1):
            model.train()
            sum_loss = 0.0
            sum_action = 0.0
            sum_distill = 0.0
            sum_cost = 0.0
            sum_prior = 0.0
            sum_budget = 0.0
            sum_coverage = 0.0
            sum_utility = 0.0
            sum_utility_ranking = 0.0
            sum_utility_budget = 0.0
            sum_entropy = 0.0
            sum_phase_sample_entropy = 0.0
            sum_phase_balance = 0.0
            sum_effective_phases = 0.0
            sum_phase_temporal = 0.0
            sum_usage_band = 0.0
            sum_stage_diversity = 0.0
            sum_route = 0.0
            sum_load_balance = 0.0
            total_steps = len(train_loader)
            start_ts = time.time()
            print(f"[joint] epoch={epoch}/{joint_epochs} steps={total_steps}", flush=True)
            for step, batch in enumerate(train_loader, start=1):
                batch = move_batch_to_device(batch, device)
                gt = batch["action"]
                with torch.no_grad():
                    t_pred, _, _ = teacher_forward(model, batch)

                ctx = model.encode_context(batch)
                channel_embeds = model.encode_channels(batch)
                progress = seen_steps / max(1, total_joint_steps - 1)
                temperature = temperature_schedule(
                    float(gate_cfg["temperature_start"]),
                    float(gate_cfg["temperature_end"]),
                    progress,
                )
                anneal = anneal_scale(progress, anneal_fraction)
                cost_weight = float(loss_cfg["cost"]) * anneal
                budget_weight = float(loss_cfg.get("budget", 0.0)) * anneal
                dropped_stage = apply_stage_dropout(
                    resolved_stage_input(batch["stage_one_hot"], stage_conditioning_enabled),
                    drop_prob=float(train_cfg.get("stage_dropout_prob", 0.0)),
                    training=True,
                )
                gate_out = model.forward_gate(
                    ctx,
                    batch,
                    dropped_stage,
                    phase_temperature=temperature,
                    hard_phase=bool(latent_phase_cfg.get("hard_assignment", False)),
                    training=True,
                )
                utility_pred = gate_out["utility_pred"]
                phase_probs = gate_out["phase_probs"]
                phase_assign = gate_out["phase_assign"]
                if route_bank_masks is not None:
                    hard_gates, gate_probs, hard_route, route_probs, hard_budget, budget_probs = sample_route_mixture_gates(
                        gate_out["route_logits"],
                        route_bank_masks,
                        budget_values=budget_values,
                        temperature=temperature,
                        training=True,
                    )
                    route_targets = route_targets_from_utility(
                        batch["utility_targets"],
                        route_bank_masks,
                        channel_costs,
                        route_cost_weight=float(route_mixture_cfg.get("target_cost_weight", 0.05)),
                    )
                    route_loss = torch.mean(
                        -(route_targets * torch.log(route_probs.clamp(1e-6, 1.0))).sum(dim=1)
                    )
                else:
                    hard_gates, gate_probs, hard_budget, budget_probs = sample_budget_topk_gates(
                        gate_out["channel_logits"],
                        gate_out["budget_logits"],
                        budget_values=budget_values,
                        temperature=temperature,
                        training=True,
                    )
                    route_loss = torch.zeros((), device=device)
                student_gates = apply_student_channel_dropout(
                    hard_gates,
                    drop_prob=float(student_dropout_cfg.get("prob", 0.0)),
                    min_keep=int(student_dropout_cfg.get("min_keep", 1)),
                )
                student_gates = apply_student_epsilon_greedy(
                    student_gates,
                    epsilon=float(student_exploration_cfg.get("epsilon", 0.0)),
                    min_keep=int(student_exploration_cfg.get("min_keep", 1)),
                    max_keep=student_exploration_cfg.get("max_keep"),
                )
                s_pred = model.student(ctx, channel_embeds, student_gates, gate_probs)

                action_loss = F.mse_loss(s_pred, gt)
                distill_loss = F.mse_loss(s_pred, t_pred)
                cost_loss = torch.mean(torch.sum(hard_gates * channel_costs.unsqueeze(0), dim=1))
                if stage_conditioning_enabled:
                    prior_targets = stage_prior_targets(batch["stage_one_hot"], priors, channels, device)
                    prior_loss = F.binary_cross_entropy(gate_probs, prior_targets)
                    budget_targets = stage_budget_targets(batch["stage_one_hot"], budget_priors, device)
                    budget_loss = torch.mean(
                        -(budget_targets * torch.log(budget_probs.clamp(1e-6, 1.0))).sum(dim=1)
                    )
                    coverage_targets = stage_channel_targets(batch["stage_one_hot"], coverage_cfg, channels, device)
                    coverage_loss = coverage_underuse_penalty(gate_probs, coverage_targets)
                else:
                    prior_loss = torch.zeros((), device=device)
                    budget_loss = torch.zeros((), device=device)
                    coverage_loss = torch.zeros((), device=device)
                entropy_loss = (gate_entropy(gate_probs) + budget_entropy(budget_probs)).mean()
                phase_sample_loss = torch.zeros((), device=device)
                phase_balance_reg = torch.zeros((), device=device)
                eff_phases = torch.zeros((), device=device)
                if phase_probs is not None:
                    phase_sample_loss = phase_sample_entropy(phase_probs).mean()
                    phase_balance_reg = phase_balance_loss(phase_probs)
                    eff_phases = effective_phase_count(phase_probs, float(latent_phase_cfg.get("active_threshold", 0.05)))
                phase_temporal_loss = torch.zeros((), device=device)
                if phase_probs is not None:
                    phase_repr = phase_assign if phase_assign is not None else phase_probs
                    phase_temporal_loss = temporal_phase_smoothness(
                        phase_repr,
                        batch["episode_idx"],
                        batch["step_idx"],
                    )
                usage_band_loss = torch.zeros((), device=device)
                usage_band_weight = 0.0
                if float(usage_band_cfg.get("weight", 0.0)) > 0.0:
                    usage_band_loss = usage_band_penalty(
                        gate_probs,
                        float(usage_band_cfg.get("low", 0.1)),
                        float(usage_band_cfg.get("high", 0.9)),
                    )
                    usage_anneal_fraction = float(usage_band_cfg.get("anneal_fraction", 0.5))
                    if usage_anneal_fraction > 0:
                        usage_band_weight = float(usage_band_cfg.get("weight", 0.0)) * max(0.0, 1.0 - min(1.0, progress / usage_anneal_fraction))
                    else:
                        usage_band_weight = float(usage_band_cfg.get("weight", 0.0))
                stage_diversity_reward = torch.zeros((), device=device)
                stage_diversity_weight = 0.0
                if float(stage_diversity_cfg.get("weight", 0.0)) > 0.0:
                    stage_diversity_reward = stage_route_diversity_reward(
                        gate_probs,
                        batch["stage_one_hot"],
                        min_stage_samples=int(stage_diversity_cfg.get("min_stage_samples", 4)),
                    )
                    warmup_fraction = float(stage_diversity_cfg.get("warmup_fraction", 0.35))
                    if warmup_fraction > 0:
                        stage_diversity_weight = float(stage_diversity_cfg.get("weight", 0.0)) * min(1.0, progress / warmup_fraction)
                    else:
                        stage_diversity_weight = float(stage_diversity_cfg.get("weight", 0.0))
                load_balance_reward = torch.zeros((), device=device)
                load_balance_weight = 0.0
                if float(load_balance_cfg.get("weight", 0.0)) > 0.0:
                    load_balance_reward = marginal_usage_entropy_reward(gate_probs)
                    schedule = str(load_balance_cfg.get("schedule", "warmup"))
                    if schedule == "early_decay":
                        active_fraction = float(load_balance_cfg.get("active_fraction", 0.35))
                        if active_fraction > 0:
                            load_balance_weight = float(load_balance_cfg.get("weight", 0.0)) * max(0.0, 1.0 - min(1.0, progress / active_fraction))
                        else:
                            load_balance_weight = float(load_balance_cfg.get("weight", 0.0))
                    else:
                        warmup_fraction = float(load_balance_cfg.get("warmup_fraction", 0.35))
                        if warmup_fraction > 0:
                            load_balance_weight = float(load_balance_cfg.get("weight", 0.0)) * min(1.0, progress / warmup_fraction)
                        else:
                            load_balance_weight = float(load_balance_cfg.get("weight", 0.0))
                utility_mask = batch["utility_mask"].view(-1, 1)
                utility_weight = batch["utility_weight"].view(-1, 1)
                utility_sample_weight = utility_mask * utility_weight
                utility_loss = compute_utility_loss(
                    utility_pred,
                    batch["utility_targets"],
                    utility_sample_weight,
                    utility_cfg,
                )
                utility_ranking_loss = compute_utility_ranking_loss(
                    utility_pred,
                    batch["utility_raw_targets"],
                    utility_sample_weight,
                    ranking_cfg,
                )
                utility_budget_loss = (
                    (-(batch["utility_budget_target"] * torch.log(budget_probs.clamp(1e-6, 1.0))).sum(dim=1) * utility_sample_weight.view(-1)).sum()
                    / utility_sample_weight.sum().clamp(min=1.0)
                )
                total_loss = (
                    float(loss_cfg["action"]) * action_loss
                    + float(loss_cfg["distill"]) * distill_loss
                    + cost_weight * cost_loss
                    + float(loss_cfg["prior"]) * prior_loss
                    + budget_weight * budget_loss
                    + float(loss_cfg.get("coverage", 0.0)) * coverage_loss
                    + float(loss_cfg.get("utility", 0.0)) * utility_loss
                    + float(ranking_cfg.get("weight", 0.0)) * utility_ranking_loss
                    + float(loss_cfg.get("utility_budget", 0.0)) * utility_budget_loss
                    + float(route_mixture_cfg.get("loss_weight", 0.0)) * route_loss
                    + float(loss_cfg["entropy"]) * entropy_loss
                    + usage_band_weight * usage_band_loss
                    - stage_diversity_weight * stage_diversity_reward
                    - load_balance_weight * load_balance_reward
                    + float(latent_phase_cfg.get("sample_entropy_weight", 0.0)) * phase_sample_loss
                    + float(latent_phase_cfg.get("balance_weight", 0.0)) * phase_balance_reg
                    + float(latent_phase_cfg.get("temporal_smoothness_weight", 0.0)) * phase_temporal_loss
                )

                student_opt.zero_grad()
                total_loss.backward()
                student_opt.step()

                sum_loss += float(total_loss.item())
                sum_action += float(action_loss.item())
                sum_distill += float(distill_loss.item())
                sum_cost += float(cost_loss.item())
                sum_prior += float(prior_loss.item())
                sum_budget += float(budget_loss.item())
                sum_coverage += float(coverage_loss.item())
                sum_utility += float(utility_loss.item())
                sum_utility_ranking += float(utility_ranking_loss.item())
                sum_utility_budget += float(utility_budget_loss.item())
                sum_entropy += float(entropy_loss.item())
                sum_phase_sample_entropy += float(phase_sample_loss.item())
                sum_phase_balance += float(phase_balance_reg.item())
                sum_effective_phases += float(eff_phases.item())
                sum_phase_temporal += float(phase_temporal_loss.item())
                sum_usage_band += float(usage_band_loss.item())
                sum_stage_diversity += float(stage_diversity_reward.item())
                sum_route += float(route_loss.item())
                sum_load_balance += float(load_balance_reward.item())
                seen_steps += 1
                if step % args.log_every == 0 or step == total_steps:
                    elapsed = max(1e-6, time.time() - start_ts)
                    speed = step / elapsed
                    eta = int(max(0, total_steps - step) / max(1e-6, speed))
                    selected_means = hard_gates.detach().mean(dim=0).cpu().tolist()
                    selected = {ch: float(selected_means[i]) for i, ch in enumerate(channels)}
                    print(
                        f"[progress] phase=joint epoch={epoch}/{joint_epochs} step={step}/{total_steps} "
                        f"loss={total_loss.item():.6f} action={action_loss.item():.6f} "
                        f"distill={distill_loss.item():.6f} cost={cost_loss.item():.6f} "
                        f"budget={budget_loss.item():.6f} coverage={coverage_loss.item():.6f} "
                        f"utility={utility_loss.item():.6f} utility_rank={utility_ranking_loss.item():.6f} utility_budget={utility_budget_loss.item():.6f} route={route_loss.item():.6f} "
                        f"phase_ent={phase_sample_loss.item():.6f} phase_bal={phase_balance_reg.item():.6f} "
                        f"phase_tmp={phase_temporal_loss.item():.6f} usage={usage_band_loss.item():.6f} "
                        f"stage_div={stage_diversity_reward.item():.6f} load_bal={load_balance_reward.item():.6f} "
                        f"temp={temperature:.3f} anneal={anneal:.3f} "
                        f"w_cost={cost_weight:.5f} w_budget={budget_weight:.5f} w_usage={usage_band_weight:.5f} w_sdiv={stage_diversity_weight:.5f} w_lbal={load_balance_weight:.5f} "
                        f"gates={selected} budget_probs={budget_probs.mean(dim=0).detach().cpu().tolist()} "
                        f"speed={speed:.2f} steps/s eta={eta}s",
                        flush=True,
                    )

            val_joint = evaluate_joint(model, val_loader, device, channels, priors, budget_priors, coverage_cfg, channel_costs, budget_values, gate_cfg, loss_cfg, utility_cfg, usage_band_cfg, stage_diversity_cfg, stage_conditioning_enabled, latent_phase_cfg, route_mixture_cfg, route_bank_masks, load_balance_cfg, ranking_cfg)
            rec = {
                "phase": "joint",
                "epoch": epoch,
                "train_total_loss": sum_loss / max(1, total_steps),
                "train_action_mse": sum_action / max(1, total_steps),
                "train_distill_mse": sum_distill / max(1, total_steps),
                "train_cost": sum_cost / max(1, total_steps),
                "train_prior_bce": sum_prior / max(1, total_steps),
                "train_budget_ce": sum_budget / max(1, total_steps),
                "train_coverage": sum_coverage / max(1, total_steps),
                "train_utility": sum_utility / max(1, total_steps),
                "train_utility_ranking": sum_utility_ranking / max(1, total_steps),
                "train_utility_budget": sum_utility_budget / max(1, total_steps),
                "train_entropy": sum_entropy / max(1, total_steps),
                "train_phase_sample_entropy": sum_phase_sample_entropy / max(1, total_steps),
                "train_phase_balance": sum_phase_balance / max(1, total_steps),
                "train_effective_phases": sum_effective_phases / max(1, total_steps),
                "train_phase_temporal": sum_phase_temporal / max(1, total_steps),
                "train_usage_band": sum_usage_band / max(1, total_steps),
                "train_stage_diversity": sum_stage_diversity / max(1, total_steps),
                "train_route": sum_route / max(1, total_steps),
                "train_load_balance": sum_load_balance / max(1, total_steps),
                "val_action_mse": val_joint["action_mse"],
                "val_distill_mse": val_joint["distill_mse"],
                "val_cost": val_joint["cost"],
                "val_prior_bce": val_joint["prior_bce"],
                "val_budget_ce": val_joint["budget_ce"],
                "val_coverage": val_joint["coverage"],
                "val_utility": val_joint["utility"],
                "val_utility_ranking": val_joint["utility_ranking"],
                "val_utility_budget": val_joint["utility_budget"],
                "val_entropy": val_joint["entropy"],
                "val_phase_sample_entropy": val_joint["phase_sample_entropy"],
                "val_phase_balance": val_joint["phase_balance"],
                "val_effective_phases": val_joint["effective_phases"],
                "val_phase_temporal": val_joint["phase_temporal"],
                "val_usage_band": val_joint["usage_band"],
                "val_stage_diversity": val_joint["stage_diversity"],
                "val_load_balance": val_joint["load_balance"],
                "val_route": val_joint["route"],
                "val_avg_budget": val_joint["avg_budget"],
                "val_selected_rate": val_joint["selected_rate"],
            }
            mf.write(json.dumps(rec) + "\n")
            print(rec, flush=True)
            val_objective = (
                float(loss_cfg["action"]) * val_joint["action_mse"]
                + float(loss_cfg["distill"]) * val_joint["distill_mse"]
                + float(loss_cfg["cost"]) * val_joint["cost"]
                + float(loss_cfg["prior"]) * val_joint["prior_bce"]
                + float(loss_cfg.get("budget", 0.0)) * val_joint["budget_ce"]
                + float(loss_cfg.get("coverage", 0.0)) * val_joint["coverage"]
                + float(loss_cfg.get("utility", 0.0)) * val_joint["utility"]
                + float(ranking_cfg.get("weight", 0.0)) * val_joint["utility_ranking"]
                + float(loss_cfg.get("utility_budget", 0.0)) * val_joint["utility_budget"]
                + float(route_mixture_cfg.get("loss_weight", 0.0)) * val_joint["route"]
                + float(latent_phase_cfg.get("sample_entropy_weight", 0.0)) * val_joint["phase_sample_entropy"]
                + float(latent_phase_cfg.get("balance_weight", 0.0)) * val_joint["phase_balance"]
                + float(latent_phase_cfg.get("temporal_smoothness_weight", 0.0)) * val_joint["phase_temporal"]
                - float(stage_diversity_cfg.get("weight", 0.0)) * val_joint["stage_diversity"]
                - float(load_balance_cfg.get("weight", 0.0)) * val_joint["load_balance"]
            )
            if val_objective < best_student:
                best_student = val_objective
                torch.save(model.context.state_dict(), out_dir / "context.pt")
                if getattr(model, "instruction_text_encoder", None) is not None:
                    torch.save(model.instruction_text_encoder.state_dict(), out_dir / "instruction_text_encoder.pt")
                if getattr(model, "query_text_encoder", None) is not None:
                    torch.save(model.query_text_encoder.state_dict(), out_dir / "query_text_encoder.pt")
                for ch in channels:
                    torch.save(getattr(model, f"{ch}_encoder").state_dict(), out_dir / f"{ch}_encoder.pt")
                torch.save(model.gate.state_dict(), out_dir / "gate.pt")
                torch.save(model.student.state_dict(), out_dir / "student.pt")

    print(f"[ok] saved: {out_dir}")


if __name__ == "__main__":
    main()
