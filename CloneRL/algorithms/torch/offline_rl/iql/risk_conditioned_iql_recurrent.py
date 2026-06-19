from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import wandb

from CloneRL.algorithms.torch.offline_rl.iql.iql_recurrent import IQLRecurrent
from CloneRL.algorithms.torch.offline_rl.iql.risk_relabeling import RiskRewardRelabeler
from CloneRL.dataloader.hdf.rlroverlab_risk_dino_da import RISK_CLEARANCE_STATE_KEY
from CloneRL.models.torch.risk_conditioned_dino_da_recurrent import RISK_PREFERENCE_KEY


class RiskConditionedIQLRecurrent(IQLRecurrent):
    """Recurrent IQL with sequence-level risk conditioning and reward relabeling."""

    def __init__(
        self,
        actor_policy: nn.Module,
        value_policy: nn.Module,
        critic_policy: nn.Module,
        cfg: Dict,
        *,
        lambda_risk: float = 0.02,
        d_ref: float = 5.0,
        clearance_exponent: float = 2.5,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        **kwargs,
    ) -> None:
        self.risk_relabeler = RiskRewardRelabeler(
            lambda_risk=lambda_risk,
            d_ref=d_ref,
            clearance_exponent=clearance_exponent,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
        super().__init__(
            actor_policy=actor_policy,
            value_policy=value_policy,
            critic_policy=critic_policy,
            cfg=cfg,
            **kwargs,
        )

    def train(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: Dict[str, torch.Tensor],
        done: torch.Tensor,
        weights: torch.Tensor,
        step: int,
        masks: torch.Tensor,
        hidden_reset: Optional[torch.Tensor] = None,
    ) -> float:
        del weights, hidden_reset
        reward, done, masks = self._ensure_loss_shapes(reward, done, masks)
        conditioned_state, conditioned_next_state, risk_reward, alpha, cost, penalty = self._condition_batch(
            state,
            next_state,
            reward,
        )

        self.update_count += 1
        self._update_value_network(conditioned_state, action, masks, step)
        actor_loss = self._update_actor_network(conditioned_state, action, masks, step)
        self._update_critic_network(
            conditioned_state,
            action,
            risk_reward,
            conditioned_next_state,
            done,
            masks,
            step,
        )

        if self.update_count % self.target_update_freq == 0:
            self._update_target_network()

        self._log_risk_metrics(reward, risk_reward, alpha, cost, penalty, masks, step)
        return actor_loss

    def _update_actor_network(
        self,
        states: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        masks: torch.Tensor,
        step: int,
    ) -> float:
        """Fit the actor using advantages from the clipped target critics."""
        with torch.no_grad():
            value, _ = self.value(states)
            q1, q2, _, _ = self.critic_target(states, actions)
            q = torch.min(q1, q2)
            advantage = q - value
            exp_weights = torch.exp(advantage * self.temperature).clamp(max=100.0)

        distribution, _ = self.actor(states)
        log_probs = distribution.log_prob(actions).sum(dim=-1, keepdim=True)
        actor_loss = -self._masked_mean(exp_weights * log_probs, masks)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        wandb.log(
            {
                "train/actor_loss": actor_loss.item(),
                "train/advantage_mean": self._masked_mean(advantage, masks).item(),
                "train/advantage_std": (
                    advantage[masks.bool().expand_as(advantage)].std().item()
                    if masks.sum() > 0
                    else 0.0
                ),
                "train/exp_weights_mean": self._masked_mean(exp_weights, masks).item(),
                "train/log_probs_mean": self._masked_mean(log_probs, masks).item(),
            },
            step=step,
        )
        return actor_loss.item()

    def validate(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: Dict[str, torch.Tensor],
        done: torch.Tensor,
        weights: torch.Tensor,
        step: int,
        masks: torch.Tensor,
        hidden_reset: Optional[torch.Tensor] = None,
    ) -> float:
        del done, weights, hidden_reset
        if reward.dim() == 2:
            reward = reward.unsqueeze(-1)
        if masks.dim() == 2:
            masks = masks.unsqueeze(-1)
        conditioned_state, _, _, _, _, _ = self._condition_batch(state, next_state, reward)

        with torch.no_grad():
            value, _ = self.value(conditioned_state)
            q1, q2, _, _ = self.critic_target(conditioned_state, action)
            advantage = torch.min(q1, q2) - value
            exp_weights = torch.exp(advantage * self.temperature).clamp(max=100.0)
            distribution, _ = self.actor(conditioned_state)
            log_probs = distribution.log_prob(action).sum(dim=-1, keepdim=True)
            return -self._masked_mean(exp_weights * log_probs, masks).item()

    @staticmethod
    def _ensure_loss_shapes(
        reward: torch.Tensor,
        done: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if reward.dim() == 2:
            reward = reward.unsqueeze(-1)
        if done.dim() == 2:
            done = done.unsqueeze(-1)
        if masks.dim() == 2:
            masks = masks.unsqueeze(-1)
        return reward, done, masks

    def _condition_batch(
        self,
        state: Dict[str, torch.Tensor],
        next_state: Dict[str, torch.Tensor],
        task_reward: torch.Tensor,
    ) -> tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if RISK_CLEARANCE_STATE_KEY not in state:
            raise KeyError(
                f"RC-RIQL training state is missing privileged clearance key "
                f"{RISK_CLEARANCE_STATE_KEY!r}."
            )

        clearance = state[RISK_CLEARANCE_STATE_KEY]
        if clearance.dim() == 2:
            clearance = clearance.unsqueeze(-1)
        clearance = clearance.to(device=task_reward.device, dtype=torch.float32)

        alpha = self.risk_relabeler.sample_alpha(task_reward)
        risk_reward, cost, penalty = self.risk_relabeler.relabel(
            task_reward,
            clearance,
            alpha,
        )

        conditioned_state = {
            key: value
            for key, value in state.items()
            if key not in (RISK_CLEARANCE_STATE_KEY, RISK_PREFERENCE_KEY)
        }
        conditioned_next_state = {
            key: value
            for key, value in next_state.items()
            if key not in (RISK_CLEARANCE_STATE_KEY, RISK_PREFERENCE_KEY)
        }
        if not conditioned_next_state:
            raise ValueError("RC-RIQL requires next-state observations.")

        conditioned_state[RISK_PREFERENCE_KEY] = alpha.expand(-1, task_reward.shape[1], -1)
        next_sequence_length = self._state_sequence_length(conditioned_next_state)
        conditioned_next_state[RISK_PREFERENCE_KEY] = alpha.expand(-1, next_sequence_length, -1)
        return conditioned_state, conditioned_next_state, risk_reward, alpha, cost, penalty

    @staticmethod
    def _state_sequence_length(state: Dict[str, torch.Tensor]) -> int:
        tensor = next(iter(state.values()))
        if tensor.dim() < 2:
            raise ValueError(f"Expected sequential next state, got tensor shape {tuple(tensor.shape)}.")
        return int(tensor.shape[1])

    def _log_risk_metrics(
        self,
        task_reward: torch.Tensor,
        risk_reward: torch.Tensor,
        alpha: torch.Tensor,
        cost: torch.Tensor,
        penalty: torch.Tensor,
        masks: torch.Tensor,
        step: int,
    ) -> None:
        wandb.log(
            {
                "risk/alpha_mean": alpha.mean().item(),
                "risk/clearance_cost_mean": self._masked_mean(cost, masks).item(),
                "risk/penalty_mean": self._masked_mean(penalty, masks).item(),
                "risk/task_reward_mean": self._masked_mean(task_reward, masks).item(),
                "risk/relabelled_reward_mean": self._masked_mean(risk_reward, masks).item(),
            },
            step=step,
        )
