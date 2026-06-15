#!/usr/bin/env python3
"""Collect DAgger labels for a DINO/DA CloneLab student in RLRoverLab.

The student drives the simulator from front-facing RGB observations. For each
state visited by the student, the privileged RLRoverLab teacher is queried and
its action is stored as the supervised target. The output uses the same cached
DINO/DA HDF5 schema as ``precompute_dino_da3_dataset.py`` so it can be used by
the existing recurrent BC trainer.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from isaaclab.app import AppLauncher


DEFAULT_TASK = "AAURoverEnvRGBDRawHD720-v0"
SCHEMA_NAME = "rlroverlab.offline_dino_da3_v1"
DINO_TOKENS = 577
DINO_DIM = 384


parser = argparse.ArgumentParser("Collect state-to-visual DAgger data for DINO/DA CloneLab policies.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--steps", type=int, default=5000)
parser.add_argument("--warmup_steps", type=int, default=1)
parser.add_argument("--student_checkpoint", type=str, required=True)
parser.add_argument("--student_checkpoint_name", type=str, default="best_model.pt")
parser.add_argument("--student_policy_factory", type=str, default=None)
parser.add_argument("--student_policy_config", type=str, default=None)
parser.add_argument("--teacher_agent", type=str, default="PPO")
parser.add_argument("--teacher_checkpoint", type=str, default=None)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--metrics_out", type=str, default=None)
parser.add_argument("--torch_device", type=str, default=None)
parser.add_argument("--store_dtype", choices=("float16", "float32"), default="float16")
parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
parser.add_argument("--gzip_level", type=int, default=4)
parser.add_argument("--clip_student_actions", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("--terrain", type=str, default=None)
parser.add_argument("--terrain-seed", type=int, default=None)
parser.add_argument("--keep-terrain", action="store_true", default=False)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from skrl.utils import set_seed  # noqa: E402
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

import rover_envs  # noqa: E402, F401
import rover_envs.envs.navigation.robots  # noqa: E402, F401
from rover_envs.integrations.clonelab import (  # noqa: E402
    CloneLabActorPolicy,
    CloneLabObservationConfig,
    RoverToCloneLabObservation,
    load_policy_config,
)
from rover_envs.learning.agents import create_agent  # noqa: E402
from rover_envs.utils.config import parse_skrl_cfg  # noqa: E402
from rover_envs.utils.terrain_utils import cleanup_terrain, handle_terrain_config  # noqa: E402

simulation_app = app_launcher.app


@dataclass
class EpisodeBuffer:
    dino_tokens: list[np.ndarray] = field(default_factory=list)
    da_depth: list[np.ndarray] = field(default_factory=list)
    state: dict[str, list[float]] = field(default_factory=lambda: {"distance": [], "heading": [], "angle_diff": []})
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    terminals: list[bool] = field(default_factory=list)
    timeouts: list[bool] = field(default_factory=list)

    @property
    def transition_count(self) -> int:
        return len(self.actions)

    @property
    def observation_count(self) -> int:
        return len(self.dino_tokens)

    def append_transition(
        self,
        *,
        dino_tokens: np.ndarray,
        da_depth: np.ndarray,
        state_values: dict[str, float],
        teacher_action: np.ndarray,
        reward: float,
        terminal: bool,
        timeout: bool,
    ) -> None:
        self.dino_tokens.append(dino_tokens)
        self.da_depth.append(da_depth)
        for key in self.state:
            self.state[key].append(float(state_values[key]))
        self.actions.append(teacher_action.astype(np.float32, copy=False))
        self.rewards.append(float(reward))
        self.terminals.append(bool(terminal))
        self.timeouts.append(bool(timeout))

    def append_final_observation(
        self,
        *,
        dino_tokens: np.ndarray,
        da_depth: np.ndarray,
        state_values: dict[str, float],
    ) -> None:
        self.dino_tokens.append(dino_tokens)
        self.da_depth.append(da_depth)
        for key in self.state:
            self.state[key].append(float(state_values[key]))

    def clear(self) -> None:
        self.dino_tokens.clear()
        self.da_depth.clear()
        for values in self.state.values():
            values.clear()
        self.actions.clear()
        self.rewards.clear()
        self.terminals.clear()
        self.timeouts.clear()


class DinoDADAggerWriter:
    def __init__(
        self,
        path: str | Path,
        *,
        store_dtype: str,
        compression: str,
        gzip_level: int,
        metadata: dict[str, Any],
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store_dtype = np.float16 if store_dtype == "float16" else np.float32
        self.compression_kwargs = self._compression_kwargs(compression, gzip_level)
        self.file = h5py.File(self.path, "w")
        self.total_observations = 0
        self.total_transitions = 0
        self.total_episodes = 0
        self._create_file(metadata)

    @staticmethod
    def _compression_kwargs(compression: str, gzip_level: int) -> dict[str, Any]:
        if compression == "none":
            return {}
        if compression == "gzip":
            return {"compression": "gzip", "compression_opts": int(gzip_level), "shuffle": True}
        return {"compression": "lzf", "shuffle": True}

    def _create_file(self, metadata: dict[str, Any]) -> None:
        self.file.attrs["schema_name"] = SCHEMA_NAME
        self.file.attrs["source_schema_name"] = "rlroverlab.dagger_online_dino_da3_v1"
        self.file.attrs["visual_profile"] = "dino_v3_vits16_tokens_da3_depth"
        self.file.attrs["visual_storage_contract"] = "precomputed_float_features"
        self.file.attrs["dino_tokens_shape"] = json.dumps([DINO_TOKENS, DINO_DIM])
        self.file.attrs["da_depth_shape"] = json.dumps([1, 72, 128])
        self.file.attrs["feature_store_dtype"] = "float16" if self.store_dtype == np.float16 else "float32"
        self.file.attrs["writer_status"] = "in_progress"
        for key, value in metadata.items():
            if value is not None:
                self.file.attrs[key] = value

        observations = self.file.create_group("observations")
        state = observations.create_group("state")
        transitions = self.file.create_group("transitions")
        index = self.file.create_group("index")

        observations.create_dataset(
            "dino_tokens_t",
            shape=(0, DINO_TOKENS, DINO_DIM),
            maxshape=(None, DINO_TOKENS, DINO_DIM),
            dtype=self.store_dtype,
            chunks=(1, DINO_TOKENS, DINO_DIM),
            **self.compression_kwargs,
        )
        observations.create_dataset(
            "da_depth_t",
            shape=(0, 1, 72, 128),
            maxshape=(None, 1, 72, 128),
            dtype=self.store_dtype,
            chunks=(16, 1, 72, 128),
            **self.compression_kwargs,
        )
        for key in ("distance", "heading", "angle_diff"):
            state.create_dataset(key, shape=(0,), maxshape=(None,), dtype=np.float32, chunks=(1024,))

        transitions.create_dataset("actions", shape=(0, 2), maxshape=(None, 2), dtype=np.float32, chunks=(1024, 2))
        transitions.create_dataset("rewards", shape=(0, 1), maxshape=(None, 1), dtype=np.float32, chunks=(1024, 1))
        transitions.create_dataset("terminals", shape=(0, 1), maxshape=(None, 1), dtype=np.bool_, chunks=(1024, 1))
        transitions.create_dataset("timeouts", shape=(0, 1), maxshape=(None, 1), dtype=np.bool_, chunks=(1024, 1))
        transitions.create_dataset("dones", shape=(0, 1), maxshape=(None, 1), dtype=np.bool_, chunks=(1024, 1))

        for key in ("obs_index", "next_obs_index", "episode_id", "episode_transition_index"):
            index.create_dataset(key, shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(1024,))
        for key in ("episode_lengths", "obs_offsets", "transition_offsets"):
            index.create_dataset(key, shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(128,))

    def append_episode(self, episode: EpisodeBuffer) -> None:
        length = episode.transition_count
        if length <= 0:
            return
        if episode.observation_count != length + 1:
            raise ValueError(
                f"Episode must contain transitions + 1 observations; got {length} transitions and "
                f"{episode.observation_count} observations."
            )

        obs_start = self.total_observations
        transition_start = self.total_transitions
        episode_id = self.total_episodes
        obs_stop = obs_start + episode.observation_count
        transition_stop = transition_start + length

        self._resize_observations(obs_stop)
        self._resize_transitions(transition_stop)
        self._resize_episodes(episode_id + 1)

        self.file["observations/dino_tokens_t"][obs_start:obs_stop] = np.asarray(
            episode.dino_tokens, dtype=self.store_dtype
        )
        self.file["observations/da_depth_t"][obs_start:obs_stop] = np.asarray(episode.da_depth, dtype=self.store_dtype)
        for key, values in episode.state.items():
            self.file[f"observations/state/{key}"][obs_start:obs_stop] = np.asarray(values, dtype=np.float32)

        self.file["transitions/actions"][transition_start:transition_stop] = np.asarray(
            episode.actions, dtype=np.float32
        )
        self.file["transitions/rewards"][transition_start:transition_stop] = np.asarray(
            episode.rewards, dtype=np.float32
        ).reshape(-1, 1)
        terminals = np.asarray(episode.terminals, dtype=np.bool_).reshape(-1, 1)
        timeouts = np.asarray(episode.timeouts, dtype=np.bool_).reshape(-1, 1)
        self.file["transitions/terminals"][transition_start:transition_stop] = terminals
        self.file["transitions/timeouts"][transition_start:transition_stop] = timeouts
        self.file["transitions/dones"][transition_start:transition_stop] = np.logical_or(terminals, timeouts)

        local = np.arange(length, dtype=np.int64)
        self.file["index/obs_index"][transition_start:transition_stop] = obs_start + local
        self.file["index/next_obs_index"][transition_start:transition_stop] = obs_start + local + 1
        self.file["index/episode_id"][transition_start:transition_stop] = episode_id
        self.file["index/episode_transition_index"][transition_start:transition_stop] = local
        self.file["index/episode_lengths"][episode_id] = length
        self.file["index/obs_offsets"][episode_id] = obs_start
        self.file["index/transition_offsets"][episode_id] = transition_start

        self.total_observations = obs_stop
        self.total_transitions = transition_stop
        self.total_episodes += 1
        self._write_counts()

    def _resize_observations(self, size: int) -> None:
        self.file["observations/dino_tokens_t"].resize((size, DINO_TOKENS, DINO_DIM))
        self.file["observations/da_depth_t"].resize((size, 1, 72, 128))
        for key in ("distance", "heading", "angle_diff"):
            self.file[f"observations/state/{key}"].resize((size,))

    def _resize_transitions(self, size: int) -> None:
        self.file["transitions/actions"].resize((size, 2))
        for key in ("rewards", "terminals", "timeouts", "dones"):
            self.file[f"transitions/{key}"].resize((size, 1))
        for key in ("obs_index", "next_obs_index", "episode_id", "episode_transition_index"):
            self.file[f"index/{key}"].resize((size,))

    def _resize_episodes(self, size: int) -> None:
        for key in ("episode_lengths", "obs_offsets", "transition_offsets"):
            self.file[f"index/{key}"].resize((size,))

    def _write_counts(self) -> None:
        self.file.attrs["total_observations"] = int(self.total_observations)
        self.file.attrs["total_transitions"] = int(self.total_transitions)
        self.file.attrs["total_episodes"] = int(self.total_episodes)
        self.file.attrs["feature_observation_count_written"] = int(self.total_observations)

    def close(self, *, complete: bool) -> None:
        self.file.attrs["writer_status"] = "complete" if complete else "failed"
        self._write_counts()
        self.file.flush()
        self.file.close()


def _as_done_tensor(terminated, truncated, device: str | torch.device) -> torch.Tensor:
    terminated = torch.as_tensor(terminated, device=device).bool()
    truncated = torch.as_tensor(truncated, device=device).bool()
    return torch.logical_or(terminated, truncated).reshape(-1)


def _get_termination_terms(env) -> tuple[str, ...]:
    termination_manager = getattr(getattr(env, "unwrapped", env), "termination_manager", None)
    if termination_manager is None:
        return ()
    return tuple(getattr(termination_manager, "active_terms", ()))


def _get_termination_term(env, name: str, device: str | torch.device) -> torch.Tensor | None:
    termination_manager = getattr(getattr(env, "unwrapped", env), "termination_manager", None)
    if termination_manager is None or name not in getattr(termination_manager, "active_terms", ()):
        return None
    return torch.as_tensor(termination_manager.get_term(name), device=device).bool().reshape(-1)


def _student_action(policy: CloneLabActorPolicy, feature_state: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
        if hasattr(policy.actor, "get_action"):
            actions = policy.actor.get_action(feature_state, deterministic=True)
        else:
            output = policy.actor(feature_state)
            actions = CloneLabActorPolicy._actions_from_output(output, deterministic=True)
    return actions.detach()


def _teacher_action(agent, observations: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        preprocessed = agent._observation_preprocessor(observations)
        actions, _ = agent.policy.compute({"observations": preprocessed, "states": None}, role="policy")
    return actions.detach()


def _teacher_observations(teacher_env, observations: dict[str, Any]) -> torch.Tensor:
    return flatten_tensorized_space(tensorize_space(teacher_env.observation_space, observations["policy"]))


def _state_values(feature_state: dict[str, torch.Tensor], env_index: int) -> dict[str, float]:
    proprio = feature_state["proprioceptive"][env_index].detach().float().cpu().numpy()
    return {
        "distance": float(proprio[0]),
        "heading": float(proprio[1]),
        "angle_diff": float(proprio[2]),
    }


def _append_final_observation(
    buffer: EpisodeBuffer,
    feature_state: dict[str, torch.Tensor],
    env_index: int,
) -> None:
    buffer.append_final_observation(
        dino_tokens=feature_state["dino_tokens"][env_index].detach().cpu().numpy(),
        da_depth=feature_state["da_depth"][env_index].detach().cpu().numpy(),
        state_values=_state_values(feature_state, env_index),
    )


def _load_student(device: str | torch.device) -> CloneLabActorPolicy:
    model_config = load_policy_config(args_cli.student_policy_config) if args_cli.student_policy_config else None
    policy = CloneLabActorPolicy.from_checkpoint(
        factory_spec=args_cli.student_policy_factory,
        checkpoint=args_cli.student_checkpoint,
        checkpoint_name=args_cli.student_checkpoint_name,
        model_config=model_config,
        device=device,
    )
    if policy.online_adapter is None:
        raise RuntimeError("The student checkpoint is not a DINO/DA policy, so no online feature adapter was created.")
    return policy


def _load_teacher(env, device: str | torch.device):
    teacher_env = SkrlVecEnvWrapper(env, ml_framework="torch")
    experiment_cfg_file = gym.spec(args_cli.task).kwargs.get("skrl_cfgs")[args_cli.teacher_agent.upper()]
    experiment_cfg = parse_skrl_cfg(experiment_cfg_file)
    experiment_cfg["agent"]["rollouts"] = 1
    experiment_cfg.setdefault("agent", {}).setdefault("experiment", {})["wandb"] = False
    agent = create_agent(args_cli.teacher_agent, teacher_env, experiment_cfg)
    checkpoint = args_cli.teacher_checkpoint or gym.spec(args_cli.task).kwargs.get("best_model_path")
    agent.load(checkpoint)
    if hasattr(agent, "set_running_mode"):
        agent.set_running_mode("eval")
    if hasattr(agent, "policy"):
        agent.policy.eval()
    return teacher_env, agent, checkpoint


def main() -> None:
    if args_cli.steps <= args_cli.warmup_steps:
        raise ValueError("--steps must be greater than --warmup_steps so at least one labeled transition is written.")

    device = args_cli.torch_device or ("cpu" if args_cli.cpu else "cuda:0")
    seed = args_cli.seed if args_cli.seed is not None else random.randint(0, 100000000)
    set_seed(seed)

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    terrain_name, terrain_cleanup_path = handle_terrain_config(
        terrain_arg=args_cli.terrain,
        terrain_seed=args_cli.terrain_seed,
        keep_terrain=args_cli.keep_terrain,
    )
    if terrain_name is not None:
        env_cfg.scene.set_terrain(terrain_name)

    env = None
    writer = None
    collection_complete = False
    try:
        env = gym.make(args_cli.task, cfg=env_cfg)
        student = _load_student(device)
        teacher_env, teacher, teacher_checkpoint = _load_teacher(env, device)

        obs_adapter = RoverToCloneLabObservation(
            CloneLabObservationConfig(
                device=device,
                proprioceptive_keys=tuple(getattr(student, "proprioceptive_keys", ("distance", "heading", "angle_diff"))),
            )
        )
        obs, _ = env.reset(seed=seed)
        num_envs = obs_adapter.num_envs(obs)
        student.reset(num_envs)

        writer = DinoDADAggerWriter(
            args_cli.output,
            store_dtype=args_cli.store_dtype,
            compression=args_cli.compression,
            gzip_level=args_cli.gzip_level,
            metadata={
                "task": args_cli.task,
                "seed": int(seed),
                "num_envs": int(num_envs),
                "student_checkpoint": args_cli.student_checkpoint,
                "teacher_checkpoint": teacher_checkpoint,
                "dino_model": getattr(student.online_adapter, "dino_model_id", ""),
                "dino_input_width": int(student.online_adapter.dino_size[0]),
                "dino_input_height": int(student.online_adapter.dino_size[1]),
                "da3_model": getattr(student.online_adapter, "da3_model_id", ""),
                "dagger_steps_requested": int(args_cli.steps),
            },
        )

        buffers = [EpisodeBuffer() for _ in range(num_envs)]
        done_for_student = torch.zeros(num_envs, dtype=torch.bool, device=device)
        actions = torch.zeros((num_envs, 2), device=device)
        termination_terms = _get_termination_terms(env)
        termination_counts = {name: 0 for name in termination_terms}
        completed_episodes = 0
        total_reward = 0.0
        total_env_steps = 0

        progress = tqdm(
            range(args_cli.steps),
            total=args_cli.steps,
            desc="Collecting DAgger",
            unit="step",
            dynamic_ncols=True,
            disable=args_cli.log_interval == 0,
        )
        for step in progress:
            if step >= args_cli.warmup_steps:
                student.reset_done(done_for_student)
                clone_state = obs_adapter.to_state(obs)
                feature_state = student.online_adapter.to_state(clone_state)
                actions = _student_action(student, feature_state)
                if args_cli.clip_student_actions:
                    actions = torch.clamp(actions, -1.0, 1.0)
                teacher_obs = _teacher_observations(teacher_env, obs)
                teacher_actions = _teacher_action(teacher, teacher_obs)
            else:
                feature_state = None
                teacher_actions = None

            next_obs, rewards, terminated, truncated, _ = env.step(actions)
            rewards = torch.as_tensor(rewards, device=device, dtype=torch.float32).reshape(-1)
            terminated_tensor = torch.as_tensor(terminated, device=device).bool().reshape(-1)
            truncated_tensor = torch.as_tensor(truncated, device=device).bool().reshape(-1)
            done = _as_done_tensor(terminated, truncated, device)

            if step >= args_cli.warmup_steps and feature_state is not None and teacher_actions is not None:
                teacher_actions_cpu = teacher_actions.detach().cpu().numpy()
                for env_index, buffer in enumerate(buffers):
                    buffer.append_transition(
                        dino_tokens=feature_state["dino_tokens"][env_index].detach().cpu().numpy(),
                        da_depth=feature_state["da_depth"][env_index].detach().cpu().numpy(),
                        state_values=_state_values(feature_state, env_index),
                        teacher_action=teacher_actions_cpu[env_index],
                        reward=float(rewards[env_index].item()),
                        terminal=bool(terminated_tensor[env_index].item()),
                        timeout=bool(truncated_tensor[env_index].item()),
                    )

            if done.any():
                next_clone_state = obs_adapter.to_state(next_obs)
                next_feature_state = student.online_adapter.to_state(next_clone_state)
                done_indices = done.nonzero(as_tuple=False).reshape(-1).detach().cpu().tolist()
                for env_index in done_indices:
                    if buffers[env_index].transition_count > 0:
                        _append_final_observation(buffers[env_index], next_feature_state, env_index)
                        writer.append_episode(buffers[env_index])
                        buffers[env_index].clear()
                completed_episodes += len(done_indices)

            for term_name in termination_terms:
                term_done = _get_termination_term(env, term_name, device)
                if term_done is not None:
                    termination_counts[term_name] += int(term_done.sum().item())

            total_reward += float(rewards.sum().item())
            total_env_steps += int(rewards.numel())
            done_for_student = done
            obs = next_obs

            should_log = args_cli.log_interval > 0 and (
                (step + 1) % args_cli.log_interval == 0 or (step + 1) == args_cli.steps
            )
            if should_log:
                progress.set_postfix(
                    episodes=writer.total_episodes,
                    transitions=writer.total_transitions,
                    success_rate=(
                        "n/a"
                        if completed_episodes == 0
                        else f"{termination_counts.get('is_success', 0) / completed_episodes:.3f}"
                    ),
                    mean_reward=f"{total_reward / max(total_env_steps, 1):.4f}",
                    refresh=True,
                )

        final_clone_state = obs_adapter.to_state(obs)
        final_feature_state = student.online_adapter.to_state(final_clone_state)
        for env_index, buffer in enumerate(buffers):
            if buffer.transition_count > 0:
                _append_final_observation(buffer, final_feature_state, env_index)
                writer.append_episode(buffer)
                buffer.clear()

        if writer.total_transitions <= 0:
            raise RuntimeError("DAgger collection produced zero labeled transitions.")

        metrics = {
            "task": args_cli.task,
            "steps": args_cli.steps,
            "num_envs": num_envs,
            "seed": seed,
            "episodes": writer.total_episodes,
            "transitions": writer.total_transitions,
            "observations": writer.total_observations,
            "successes": termination_counts.get("is_success", 0),
            "success_rate": (
                termination_counts.get("is_success", 0) / completed_episodes if completed_episodes else None
            ),
            "termination_counts": termination_counts,
            "mean_step_reward": total_reward / max(total_env_steps, 1),
            "output": str(args_cli.output),
        }
        print(f"[INFO] DAgger collection metrics: {metrics}", flush=True)
        if args_cli.metrics_out:
            metrics_path = Path(args_cli.metrics_out)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            print(f"[INFO] Wrote collection metrics: {metrics_path}", flush=True)
        collection_complete = True

    finally:
        if writer is not None:
            writer.close(complete=collection_complete)
        if env is not None:
            env.close()
        cleanup_terrain(terrain_cleanup_path if "terrain_cleanup_path" in locals() else None)
        simulation_app.close()


if __name__ == "__main__":
    main()
