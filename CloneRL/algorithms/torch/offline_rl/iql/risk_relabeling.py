from __future__ import annotations

import math

import torch


def clearance_cost(
    clearance: torch.Tensor,
    *,
    d_ref: float = 5.0,
    clearance_exponent: float = 2.5,
) -> torch.Tensor:
    """Map nonnegative obstacle clearance in meters to a bounded risk cost."""
    if d_ref <= 0:
        raise ValueError("d_ref must be positive.")
    if clearance_exponent <= 0:
        raise ValueError("clearance_exponent must be positive.")
    if torch.any(clearance < 0):
        raise ValueError("clearance must be nonnegative.")

    clearance = clearance.float()
    scaled_clearance = clearance / float(d_ref)
    return torch.exp(-math.log(100.0) * scaled_clearance.pow(float(clearance_exponent)))


class RiskRewardRelabeler:
    """Sample sequence-level risk preferences and relabel task rewards."""

    def __init__(
        self,
        *,
        lambda_risk: float = 0.02,
        d_ref: float = 5.0,
        clearance_exponent: float = 2.5,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
    ) -> None:
        if lambda_risk < 0:
            raise ValueError("lambda_risk must be nonnegative.")
        if not 0.0 <= alpha_min <= alpha_max <= 1.0:
            raise ValueError("Expected 0 <= alpha_min <= alpha_max <= 1.")

        self.lambda_risk = float(lambda_risk)
        self.d_ref = float(d_ref)
        self.clearance_exponent = float(clearance_exponent)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

    def sample_alpha(self, reference: torch.Tensor) -> torch.Tensor:
        """Return one alpha per batch element with shape [B, 1, 1]."""
        if reference.dim() < 2:
            raise ValueError(f"Expected batched sequence tensor, got shape {tuple(reference.shape)}.")
        shape = (reference.shape[0], 1, 1)
        alpha = torch.rand(shape, device=reference.device, dtype=torch.float32)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * alpha

    def relabel(
        self,
        task_reward: torch.Tensor,
        clearance: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return relabeled reward, bounded clearance cost, and risk penalty."""
        task_reward = task_reward.float()
        clearance = clearance.to(device=task_reward.device, dtype=torch.float32)
        alpha = alpha.to(device=task_reward.device, dtype=torch.float32)

        if task_reward.shape != clearance.shape:
            raise ValueError(
                f"Task reward and clearance must have the same shape, got "
                f"{tuple(task_reward.shape)} and {tuple(clearance.shape)}."
            )
        if alpha.shape != (task_reward.shape[0], 1, 1):
            raise ValueError(
                f"Expected alpha [B, 1, 1] for B={task_reward.shape[0]}, got {tuple(alpha.shape)}."
            )

        cost = clearance_cost(
            clearance,
            d_ref=self.d_ref,
            clearance_exponent=self.clearance_exponent,
        )
        penalty = self.lambda_risk * alpha * cost
        return task_reward - penalty, cost, penalty

