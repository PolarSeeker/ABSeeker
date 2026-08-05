"""
A2 multi-turn GRPO advantage estimator.

Handles per-turn token-level rewards (anchor-clue turn scores) with separate
outcome vs intermediate-turn normalization and turn-level discounting.
Registers as "grpo_abseeker" via the advantage estimator registry.
"""

from typing import Optional

import numpy as np
import torch

from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import register_adv_est


def _compute_turn_level_advantage(
    normalized_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    bsz: int,
    seq_len: int,
    device: torch.device,
    turn_boundary_mask: torch.Tensor = None,
    adaptive_gamma_min_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    Turn-level discounted accumulation + broadcast.

    When adaptive_gamma_min_weight > 0, gamma is dynamically adjusted per sample
    so that gamma^num_turns = adaptive_gamma_min_weight, ensuring early-turn
    signals retain a minimum proportion of their original magnitude regardless
    of trajectory length.

    Returns (discounted_returns, stats) where stats contains per-sample
    effective_gamma and num_turns for diagnostic logging.
    """
    discounted_returns = torch.zeros(bsz, seq_len, device=device, dtype=normalized_rewards.dtype)
    per_sample_gammas = []
    per_sample_num_turns = []

    for sample_idx in range(bsz):
        sample_rewards = normalized_rewards[sample_idx]
        sample_mask = response_mask[sample_idx]

        if turn_boundary_mask is not None:
            reward_positions = turn_boundary_mask[sample_idx].nonzero(as_tuple=True)[0].tolist()
        else:
            reward_positions = (sample_rewards != 0).nonzero(as_tuple=True)[0].tolist()

        if len(reward_positions) == 0:
            continue

        num_turns = len(reward_positions)
        effective_gamma = gamma
        if adaptive_gamma_min_weight > 0 and num_turns > 1:
            effective_gamma = adaptive_gamma_min_weight ** (1.0 / num_turns)

        per_sample_gammas.append(effective_gamma)
        per_sample_num_turns.append(num_turns)

        turn_data = []
        next_turn_adv = 0.0

        for pos in reversed(reward_positions):
            turn_reward = sample_rewards[pos].item()
            turn_adv = turn_reward + effective_gamma * next_turn_adv
            turn_data.append((pos, turn_adv))
            next_turn_adv = turn_adv

        turn_data.reverse()

        prev_end = 0
        for _, (reward_pos, adv) in enumerate(turn_data):
            discounted_returns[sample_idx, prev_end:reward_pos + 1] = adv * sample_mask[prev_end:reward_pos + 1]
            prev_end = reward_pos + 1

    stats = {"gammas": per_sample_gammas, "num_turns": per_sample_num_turns}
    return discounted_returns, stats


@register_adv_est("grpo_abseeker")
def compute_grpo_abseeker_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GRPO advantage for A2 multi-turn rewards: separate outcome vs turn-score
    normalization, optional curriculum weights, turn-level discount.
    """
    _VALID_NORM_MODES = {"joint", "separate", "scaled_separate", "raw_turn"}
    gamma = config.gamma if config else 1.0
    abseeker_norm_mode = config.abseeker_norm_mode if config else "joint"
    if abseeker_norm_mode not in _VALID_NORM_MODES:
        raise ValueError(
            f"abseeker_norm_mode={abseeker_norm_mode!r} is invalid. "
            f"Must be one of {_VALID_NORM_MODES}"
        )
    adaptive_gamma_min_weight = float(getattr(config, "adaptive_gamma_min_weight", 0.0)) if config else 0.0
    if config:
        norm_adv_by_std_in_grpo = getattr(config, "norm_adv_by_std_in_grpo", True)
    turn_weight = float(getattr(config, "abseeker_turn_weight", 1.0)) if config else 1.0
    curriculum_outcome_weight = 1.0
    curriculum_turn_weight = 1.0
    if config and getattr(config, "use_curriculum", False):
        curriculum_outcome_weight = getattr(config, "_curriculum_f1_weight", 1.0)
        curriculum_turn_weight = getattr(config, "_curriculum_turn_weight", 1.0)

    bsz, seq_len = token_level_rewards.shape
    device = token_level_rewards.device

    with torch.no_grad():
        last_valid_pos = (seq_len - 1) - response_mask.flip(dims=[1]).to(torch.long).argmax(dim=1)

        position_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        outcome_mask = (position_indices == last_valid_pos.unsqueeze(1)) & (response_mask == 1)
        turn_mask = (response_mask == 1) & (~outcome_mask) & (token_level_rewards != 0)

    unique_indices, inverse_indices = np.unique(index, return_inverse=True)
    group_ids = torch.tensor(inverse_indices, device=device, dtype=torch.long)
    num_groups = len(unique_indices)

    group_ids_expanded = group_ids.unsqueeze(1).expand(-1, seq_len)

    def compute_group_stats(mask):
        flat_mask = mask.view(-1)
        flat_rewards = token_level_rewards.view(-1)
        flat_group_ids = group_ids_expanded.reshape(-1)

        valid_idx = flat_mask.nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            return torch.zeros(num_groups, device=device), torch.ones(num_groups, device=device)

        valid_rewards = flat_rewards[valid_idx]
        valid_groups = flat_group_ids[valid_idx]

        group_sum = torch.zeros(num_groups, device=device).scatter_add_(0, valid_groups, valid_rewards)
        group_count = torch.zeros(num_groups, device=device).scatter_add_(0, valid_groups, torch.ones_like(valid_rewards))

        group_mean = group_sum / group_count.clamp(min=1.0)

        expanded_mean = group_mean[valid_groups]
        sq_diff = (valid_rewards - expanded_mean) ** 2
        group_sq_sum = torch.zeros(num_groups, device=device).scatter_add_(0, valid_groups, sq_diff)
        group_var = group_sq_sum / group_count.clamp(min=1.0)
        group_std = torch.sqrt(group_var + 1e-8)

        group_std = torch.where(group_count <= 1, torch.ones_like(group_std), group_std)

        return group_mean, group_std

    normalized_rewards = torch.zeros_like(token_level_rewards)

    _diag_turn_scale = None
    _diag_outcome_abs_mean = None
    _diag_turn_abs_mean = None
    _diag_outcome_std = None

    if abseeker_norm_mode == "separate":
        outcome_mean, outcome_std = compute_group_stats(outcome_mask)
        _diag_outcome_std = outcome_std
        outcome_mean_map = outcome_mean[group_ids_expanded]
        outcome_std_map = outcome_std[group_ids_expanded]

        norm_outcome = (token_level_rewards - outcome_mean_map)
        if norm_adv_by_std_in_grpo:
            norm_outcome = norm_outcome / (outcome_std_map + epsilon)
        normalized_rewards = torch.where(outcome_mask, norm_outcome, normalized_rewards)

        turn_mean, turn_std = compute_group_stats(turn_mask)
        turn_mean_map = turn_mean[group_ids_expanded]
        turn_std_map = turn_std[group_ids_expanded]

        norm_turn = (token_level_rewards - turn_mean_map)
        if norm_adv_by_std_in_grpo:
            norm_turn = norm_turn / (turn_std_map + epsilon)
        normalized_rewards = torch.where(turn_mask, norm_turn, normalized_rewards)

    elif abseeker_norm_mode == "scaled_separate":
        outcome_mean, outcome_std = compute_group_stats(outcome_mask)
        _diag_outcome_std = outcome_std
        outcome_mean_map = outcome_mean[group_ids_expanded]
        outcome_std_map = outcome_std[group_ids_expanded]

        norm_outcome = (token_level_rewards - outcome_mean_map)
        if norm_adv_by_std_in_grpo:
            norm_outcome = norm_outcome / (outcome_std_map + epsilon)
        normalized_rewards = torch.where(outcome_mask, norm_outcome, normalized_rewards)

        turn_mean, turn_std = compute_group_stats(turn_mask)
        turn_mean_map = turn_mean[group_ids_expanded]
        turn_std_map = turn_std[group_ids_expanded]

        norm_turn = (token_level_rewards - turn_mean_map)
        if norm_adv_by_std_in_grpo:
            norm_turn = norm_turn / (turn_std_map + epsilon)

        outcome_abs_mean = normalized_rewards[outcome_mask].abs().mean() if outcome_mask.any() else torch.tensor(1.0, device=device)
        outcome_abs_mean = torch.clamp(outcome_abs_mean, min=0.3)
        turn_abs_mean = norm_turn[turn_mask].abs().mean() if turn_mask.any() else torch.tensor(1.0, device=device)
        turn_scale = outcome_abs_mean / (turn_abs_mean + epsilon)
        turn_scale = turn_scale.clamp(max=10.0)
        _diag_turn_scale = turn_scale.item()
        _diag_outcome_abs_mean = outcome_abs_mean.item()
        _diag_turn_abs_mean = turn_abs_mean.item()
        norm_turn = norm_turn * turn_scale
        normalized_rewards = torch.where(turn_mask, norm_turn, normalized_rewards)

    elif abseeker_norm_mode == "raw_turn":
        outcome_mean, outcome_std = compute_group_stats(outcome_mask)
        _diag_outcome_std = outcome_std
        outcome_mean_map = outcome_mean[group_ids_expanded]
        outcome_std_map = outcome_std[group_ids_expanded]

        norm_outcome = (token_level_rewards - outcome_mean_map)
        if norm_adv_by_std_in_grpo:
            norm_outcome = norm_outcome / (outcome_std_map + epsilon)
        normalized_rewards = torch.where(outcome_mask, norm_outcome, normalized_rewards)

        _turn_mean, turn_std = compute_group_stats(turn_mask)
        turn_std_map = turn_std[group_ids_expanded]

        norm_turn = token_level_rewards / (turn_std_map + epsilon)

        if turn_weight != 1.0:
            norm_turn = norm_turn * turn_weight
        normalized_rewards = torch.where(turn_mask, norm_turn, normalized_rewards)

        outcome_per_sample = (normalized_rewards * outcome_mask.float()).sum(dim=1, keepdim=True)
        normalized_rewards = normalized_rewards + outcome_per_sample * turn_mask.float()

        gamma = 0.0
        adaptive_gamma_min_weight = 0.0

    else:  # joint
        _diag_outcome_std = compute_group_stats(outcome_mask)[1]
        joint_mask = outcome_mask | turn_mask
        g_mean, g_std = compute_group_stats(joint_mask)
        mean_map = g_mean[group_ids_expanded]
        std_map = g_std[group_ids_expanded]

        norm_val = (token_level_rewards - mean_map)
        if norm_adv_by_std_in_grpo:
            norm_val = norm_val / (std_map + epsilon)
        normalized_rewards = torch.where(joint_mask, norm_val, normalized_rewards)

    if abseeker_norm_mode != "raw_turn" and turn_weight != 1.0:
        normalized_rewards = torch.where(
            turn_mask, normalized_rewards * turn_weight, normalized_rewards)

    if curriculum_outcome_weight != 1.0 or curriculum_turn_weight != 1.0:
        normalized_rewards = torch.where(
            outcome_mask, normalized_rewards * curriculum_outcome_weight, normalized_rewards)
        normalized_rewards = torch.where(
            turn_mask, normalized_rewards * curriculum_turn_weight, normalized_rewards)

    turn_boundary_mask = outcome_mask | turn_mask

    turn_active_samples = (turn_mask.any(dim=1)).sum().item()
    turn_total_positions = turn_mask.sum().item()
    nonzero_reward_positions = ((token_level_rewards != 0) & (~outcome_mask)).sum().item()
    _warmup_gate_off = bool(getattr(config, "_abseeker_warmup_gate_off", False)) if config else False
    if turn_active_samples == 0:
        if _warmup_gate_off:
            pass
        else:
            msg = (f"[abseeker_grpo-CRITICAL] turn_mask is entirely empty — "
                   f"grpo_abseeker has DEGRADED to standard GRPO for this batch "
                   f"(bsz={bsz}, outcome_positions={outcome_mask.sum().item()}, "
                   f"nonzero_non_outcome_rewards={nonzero_reward_positions})")
            if nonzero_reward_positions > 0:
                raise RuntimeError(
                    f"{msg}. Turn rewards exist at {nonzero_reward_positions} positions "
                    f"but response_mask is 0 there — likely response_mask was not "
                    f"set to all-1s before compute_advantage (see ray_trainer.py "
                    f"compute_response_mask).")
            print(msg)

    discounted_returns, turn_stats = _compute_turn_level_advantage(
        normalized_rewards=normalized_rewards,
        response_mask=response_mask,
        gamma=gamma,
        bsz=bsz,
        seq_len=seq_len,
        device=device,
        turn_boundary_mask=turn_boundary_mask,
        adaptive_gamma_min_weight=adaptive_gamma_min_weight,
    )

    adv_metrics = {
        "abseeker_grpo/turn_active_samples": turn_active_samples,
        "abseeker_grpo/turn_active_ratio": turn_active_samples / max(bsz, 1),
        "abseeker_grpo/turn_total_positions": turn_total_positions,
        "abseeker_grpo/outcome_total_positions": outcome_mask.sum().item(),
    }

    if turn_mask.any():
        raw_turn = token_level_rewards[turn_mask]
        adv_metrics["abseeker_grpo/raw_turn_mean"] = raw_turn.mean().item()
        adv_metrics["abseeker_grpo/raw_turn_std"] = raw_turn.std().item() if raw_turn.numel() > 1 else 0.0
        adv_metrics["abseeker_grpo/raw_turn_positive_ratio"] = (raw_turn > 0).float().mean().item()
        adv_metrics["abseeker_grpo/raw_turn_negative_count"] = int((raw_turn < 0).sum().item())
        adv_metrics["abseeker_grpo/raw_turn_abs_mean"] = raw_turn.abs().mean().item()

        if abseeker_norm_mode == "raw_turn":
            adv_metrics["abseeker_grpo/turn_mean_over_std"] = (
                adv_metrics["abseeker_grpo/raw_turn_mean"] / max(adv_metrics["abseeker_grpo/raw_turn_std"], 1e-8)
            )

    if outcome_mask.any():
        raw_outcome = token_level_rewards[outcome_mask]
        adv_metrics["abseeker_grpo/outcome_reward_mean"] = raw_outcome.mean().item()

    if _diag_turn_scale is not None:
        adv_metrics["abseeker_grpo/turn_scale"] = _diag_turn_scale
        adv_metrics["abseeker_grpo/outcome_norm_abs_mean"] = _diag_outcome_abs_mean
        adv_metrics["abseeker_grpo/turn_norm_abs_mean"] = _diag_turn_abs_mean

    if _diag_outcome_std is not None:
        outcome_count = torch.zeros(num_groups, device=device)
        if outcome_mask.any():
            flat_outcome_mask = outcome_mask.view(-1)
            flat_group_ids = group_ids_expanded.reshape(-1)
            valid_outcome_idx = flat_outcome_mask.nonzero(as_tuple=True)[0]
            if valid_outcome_idx.numel() > 0:
                outcome_count.scatter_add_(0, flat_group_ids[valid_outcome_idx],
                                           torch.ones(valid_outcome_idx.numel(), device=device))
        groups_with_outcome = (outcome_count > 1).sum().item()
        all_same_groups = ((outcome_count > 1) & (_diag_outcome_std < 1e-6)).sum().item()
        adv_metrics["abseeker_grpo/all_same_outcome_ratio"] = all_same_groups / max(groups_with_outcome, 1)
        adv_metrics["abseeker_grpo/num_groups"] = num_groups

    gammas = turn_stats["gammas"]
    num_turns_list = turn_stats["num_turns"]
    if gammas:
        adv_metrics["abseeker_grpo/effective_gamma_mean"] = sum(gammas) / len(gammas)
        adv_metrics["abseeker_grpo/effective_gamma_min"] = min(gammas)
        adv_metrics["abseeker_grpo/effective_gamma_max"] = max(gammas)
    if num_turns_list:
        adv_metrics["abseeker_grpo/num_turns_mean"] = sum(num_turns_list) / len(num_turns_list)
        adv_metrics["abseeker_grpo/num_turns_min"] = min(num_turns_list)
        adv_metrics["abseeker_grpo/num_turns_max"] = max(num_turns_list)

    active_adv = discounted_returns[response_mask == 1]
    if active_adv.numel() > 0:
        adv_metrics["abseeker_grpo/adv_abs_mean"] = active_adv.abs().mean().item()
        adv_metrics["abseeker_grpo/adv_std"] = active_adv.std().item() if active_adv.numel() > 1 else 0.0
        adv_metrics["abseeker_grpo/adv_positive_ratio"] = (active_adv > 0).float().mean().item()

    if turn_mask.any():
        turn_adv = discounted_returns[turn_mask]
        adv_metrics["abseeker_grpo/adv_turn_positions_mean"] = turn_adv.mean().item()
    if outcome_mask.any():
        outcome_adv = discounted_returns[outcome_mask]
        adv_metrics["abseeker_grpo/adv_outcome_positions_mean"] = outcome_adv.mean().item()

    return discounted_returns, discounted_returns, adv_metrics
