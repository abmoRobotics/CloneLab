from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch
import torch.nn as nn
import wandb

from CloneRL.algorithms.torch.offline_rl.iql.iql_recurrent import IQLRecurrent
from CloneRL.algorithms.torch.offline_rl.iql.risk_conditioned_iql_recurrent import (
    RiskConditionedIQLRecurrent,
)
from CloneRL.algorithms.torch.offline_rl.iql.risk_relabeling import (
    RiskRewardRelabeler,
    clearance_cost,
)
from CloneRL.dataloader.hdf.rlroverlab_dino_da import SCHEMA_NAME
from CloneRL.dataloader.hdf.rlroverlab_risk_dino_da import (
    DEFAULT_CLEARANCE_PATH,
    RISK_CLEARANCE_STATE_KEY,
    RLRoverLabRiskDinoDARandomSequenceDataset,
)
from CloneRL.models.torch.risk_conditioned_dino_da_recurrent import (
    RISK_PREFERENCE_KEY,
    RiskConditionedDinoDARecurrentActorGaussian,
    RiskConditionedDinoDARecurrentTwinQ,
    RiskConditionedDinoDARecurrentValue,
)


class RiskRelabelingTests(unittest.TestCase):
    def test_clearance_cost_contract(self):
        clearance = torch.tensor([0.0, 1.0, 5.0, 7.0])
        cost = clearance_cost(clearance, d_ref=5.0, clearance_exponent=2.5)
        self.assertAlmostEqual(cost[0].item(), 1.0, places=6)
        self.assertAlmostEqual(cost[2].item(), 0.01, places=6)
        self.assertTrue(torch.all(cost[:-1] > cost[1:]))

    def test_lambda_and_alpha_relabeling(self):
        relabeler = RiskRewardRelabeler(lambda_risk=0.02)
        task_reward = torch.ones(2, 3, 1)
        clearance = torch.zeros_like(task_reward)

        no_risk, _, no_penalty = relabeler.relabel(
            task_reward,
            clearance,
            torch.zeros(2, 1, 1),
        )
        full_risk, _, full_penalty = relabeler.relabel(
            task_reward,
            clearance,
            torch.ones(2, 1, 1),
        )

        torch.testing.assert_close(no_risk, task_reward)
        torch.testing.assert_close(no_penalty, torch.zeros_like(task_reward))
        torch.testing.assert_close(full_penalty, torch.full_like(task_reward, 0.02))
        torch.testing.assert_close(full_risk, torch.full_like(task_reward, 0.98))

    def test_alpha_is_sampled_once_per_sequence(self):
        relabeler = RiskRewardRelabeler()
        alpha = relabeler.sample_alpha(torch.zeros(4, 8, 1))
        self.assertEqual(alpha.shape, (4, 1, 1))
        expanded = alpha.expand(-1, 8, -1)
        torch.testing.assert_close(expanded, expanded[:, :1].expand_as(expanded))


class RiskDatasetTests(unittest.TestCase):
    def test_loader_exposes_transition_clearance_as_private_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "risk.hdf5"
            self._write_dataset(path)
            dataset = RLRoverLabRiskDinoDARandomSequenceDataset(
                str(path),
                sequence_length=2,
                total_samples=1,
            )

            obs, actions, rewards, next_obs, dones, weights, masks = dataset[0]
            self.assertEqual(obs[RISK_CLEARANCE_STATE_KEY].shape, (2, 1))
            torch.testing.assert_close(
                obs[RISK_CLEARANCE_STATE_KEY],
                torch.tensor([[0.5], [2.0]]),
            )
            self.assertNotIn(RISK_CLEARANCE_STATE_KEY, next_obs)
            self.assertEqual(actions.shape, (2, 2))
            self.assertEqual(rewards.shape, dones.shape)
            self.assertEqual(weights.shape, masks.shape)

    @staticmethod
    def _write_dataset(path: Path) -> None:
        with h5py.File(path, "w") as file:
            file.attrs["schema_name"] = SCHEMA_NAME
            file.attrs["writer_status"] = "complete"
            file.attrs["total_transitions"] = 2
            file.attrs["total_observations"] = 3

            observations = file.create_group("observations")
            observations.create_dataset(
                "dino_tokens_t",
                data=np.zeros((3, 577, 384), dtype=np.float16),
            )
            observations.create_dataset(
                "da_depth_t",
                data=np.zeros((3, 1, 72, 128), dtype=np.float16),
            )
            state = observations.create_group("state")
            for key in ("distance", "heading", "angle_diff"):
                state.create_dataset(key, data=np.zeros((3,), dtype=np.float32))

            transitions = file.create_group("transitions")
            transitions.create_dataset("actions", data=np.zeros((2, 2), dtype=np.float32))
            transitions.create_dataset("rewards", data=np.ones((2,), dtype=np.float32))
            transitions.create_dataset("dones", data=np.array([False, True]))
            risk = transitions.create_group("extra").create_group("risk")
            risk.create_dataset("min_distance_to_rock", data=np.array([0.5, 2.0], dtype=np.float32))

            index = file.create_group("index")
            index.create_dataset("episode_lengths", data=np.array([2], dtype=np.int64))
            index.create_dataset("obs_offsets", data=np.array([0], dtype=np.int64))
            index.create_dataset("transition_offsets", data=np.array([0], dtype=np.int64))


class RiskModelTests(unittest.TestCase):
    def test_conditioned_models_have_independent_encoders_and_expected_shapes(self):
        config = {
            "dino_token_count": 577,
            "dino_dim": 384,
            "dino_proj_dim": 16,
            "depth_channels": 1,
            "depth_height": 72,
            "depth_width": 128,
            "depth_token_dim": 8,
            "visual_token_dim": 24,
            "visual_dim": 32,
            "proprioception_channels": 3,
            "action_dim": 2,
            "gru_hidden_size": 16,
            "gru_num_layers": 2,
            "num_queries": 2,
            "num_heads": 4,
            "num_pool_blocks": 1,
            "dropout": 0.0,
            "device": "cpu",
        }
        actor = RiskConditionedDinoDARecurrentActorGaussian(**config)
        value = RiskConditionedDinoDARecurrentValue(**config)
        critic = RiskConditionedDinoDARecurrentTwinQ(**config)

        self.assertEqual(actor.gru.input_size, 36)
        self.assertEqual(value.gru.input_size, 36)
        self.assertEqual(critic.q1.gru.input_size, 38)
        self.assertIsNot(actor.visual_encoder, value.visual_encoder)
        self.assertIsNot(critic.q1.visual_encoder, critic.q2.visual_encoder)
        self.assertTrue(
            set(map(id, critic.q1.parameters())).isdisjoint(set(map(id, critic.q2.parameters())))
        )

        state = {
            "dino_tokens": torch.zeros(1, 2, 577, 384),
            "da_depth": torch.zeros(1, 2, 1, 72, 128),
            "proprioceptive": torch.zeros(1, 2, 3),
            RISK_PREFERENCE_KEY: torch.full((1, 2, 1), 0.75),
        }
        action = torch.zeros(1, 2, 2)
        actor_dist, _ = actor(state)
        value_output, _ = value(state)
        q1, q2, _, _ = critic(state, action)

        self.assertEqual(actor_dist.mean.shape, (1, 2, 2))
        self.assertEqual(value_output.shape, (1, 2, 1))
        self.assertEqual(q1.shape, (1, 2, 1))
        self.assertEqual(q2.shape, (1, 2, 1))


class RiskAgentTests(unittest.TestCase):
    def test_complete_optimization_step_strips_clearance_and_uses_target_critic(self):
        actor = _TinyActor()
        value = _TinyValue()
        critic = _TinyTwinQ()
        config = {
            "actions_lr": 1e-3,
            "value_lr": 1e-3,
            "critic_lr": 1e-3,
            "discount": 0.99,
            "tau": 0.005,
            "expectile": 0.7,
            "temperature": 1.0,
            "target_update_freq": 1,
            "grad_clip": 1.0,
            "reset_hidden_on_done": True,
            "lambda_risk": 0.02,
            "d_ref": 5.0,
            "clearance_exponent": 2.5,
            "alpha_min": 0.0,
            "alpha_max": 1.0,
        }
        with mock.patch.object(IQLRecurrent, "initialize", return_value=None):
            agent = RiskConditionedIQLRecurrent(
                actor_policy=actor,
                value_policy=value,
                critic_policy=critic,
                cfg=config,
                device="cpu",
                **config,
            )

        state = {
            "features": torch.randn(2, 4, 3),
            RISK_CLEARANCE_STATE_KEY: torch.rand(2, 4, 1) * 5.0,
        }
        next_state = {"features": torch.randn(2, 4, 3)}
        action = torch.tanh(torch.randn(2, 4, 2))
        reward = torch.randn(2, 4, 1) * 0.01
        done = torch.zeros(2, 4, 1)
        weights = torch.ones(2, 4, 1)
        masks = torch.ones(2, 4, 1)

        with mock.patch.object(wandb, "log"):
            loss = agent.train(
                state,
                action,
                reward,
                next_state,
                done,
                weights,
                step=0,
                masks=masks,
            )

        self.assertTrue(np.isfinite(loss))
        self.assertGreaterEqual(agent.critic_target.forward_calls, 2)
        self.assertTrue(all(not parameter.requires_grad for parameter in agent.critic_target.parameters()))
        self.assertNotIn(RISK_PREFERENCE_KEY, state)
        self.assertIn(RISK_CLEARANCE_STATE_KEY, state)
        self.assertFalse(actor.saw_clearance)
        self.assertFalse(value.saw_clearance)
        self.assertFalse(critic.saw_clearance)


class _TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Linear(4, 2)
        self.log_std = nn.Parameter(torch.zeros(2))
        self.saw_clearance = False

    def forward(self, state, hidden=None):
        self.saw_clearance |= RISK_CLEARANCE_STATE_KEY in state
        features = torch.cat([state["features"], state[RISK_PREFERENCE_KEY]], dim=-1)
        mean = torch.tanh(self.mean(features))
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std), hidden

    def reset_hidden(self, batch_size=1):
        return None


class _TinyValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Linear(4, 1)
        self.saw_clearance = False

    def forward(self, state, hidden=None):
        self.saw_clearance |= RISK_CLEARANCE_STATE_KEY in state
        features = torch.cat([state["features"], state[RISK_PREFERENCE_KEY]], dim=-1)
        return self.value(features), hidden

    def reset_hidden(self, batch_size=1):
        return None


class _TinyQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(6, 1)

    def forward(self, state, action, hidden=None):
        features = torch.cat(
            [state["features"], action, state[RISK_PREFERENCE_KEY]],
            dim=-1,
        )
        return self.q(features), hidden


class _TinyTwinQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = _TinyQ()
        self.q2 = _TinyQ()
        self.forward_calls = 0
        self.saw_clearance = False

    def forward(self, state, action, hidden1=None, hidden2=None):
        self.forward_calls += 1
        self.saw_clearance |= RISK_CLEARANCE_STATE_KEY in state
        q1, hidden1 = self.q1(state, action, hidden1)
        q2, hidden2 = self.q2(state, action, hidden2)
        return q1, q2, hidden1, hidden2

    def reset_hidden(self, batch_size=1):
        return None


if __name__ == "__main__":
    unittest.main()

