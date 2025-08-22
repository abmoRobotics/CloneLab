import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch

import wandb


class BaseAgent:
    def __init__(
            self,
            cfg: Optional[Dict[str, Any]],
            policy: torch.nn.Module,
            device: Union[str, torch.device] = "cuda:0" if torch.cuda.is_available() else "cpu"):

        self.device = device
        self.policy = policy
        self.cfg = cfg
        # Episode trackers (lazy-init on first record_transitions call)
        self._ep_returns: Optional[torch.Tensor] = None
        self._ep_lengths: Optional[torch.Tensor] = None
        # Run-level success tracking
        self._total_episodes: int = 0
        self._total_successes: int = 0

    def initialize(self):
        import time

        import wandb
        run_name = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        wandb.init(name=run_name, id=run_name)

    def train(self):
        raise NotImplementedError

    def save_model(self, name):
        path = f"runs/{wandb.run.project}/{wandb.run.name}/checkpoints/"
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.policy.state_dict(), path + name)

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path))
    

    def record_transitions(
            self,
            rewards: torch.Tensor,
            terminated: torch.Tensor,
            truncated: torch.Tensor,
            infos: Dict[str, Any],
            step: Optional[int] = None) -> None:
        """
        Record and log reward-related metrics to Weights & Biases.

        Logs:
        - Reward / Instantaneous reward (mean)
        - Reward / Total reward (mean)  [on episode end]
        - Episode / Total timesteps (mean)  [on episode end]
        - All rewards from infos["episode"] averaged over finished envs  [on episode end]

        Args:
            rewards: Tensor of shape (N,) or (N, 1) with per-env instantaneous rewards.
            terminated: Bool tensor of shape (N,) or (N, 1) with termination flags.
            truncated: Bool tensor of shape (N,) or (N, 1) with truncation flags.
            infos: Env info dict; may contain an "episode" dict with reward components.
            step: Optional global step for wandb logging.
        """
        if rewards is None:
            return

        # Ensure 1D tensors on CPU for aggregation
        if torch.is_tensor(rewards):
            r = rewards.detach()
        else:
            r = torch.as_tensor(rewards)
        if r.ndim > 1:
            r = r.squeeze(-1)
        r = r.to("cpu")

        def _to_bool_1d(x: Union[torch.Tensor, np.ndarray, List[bool]]):
            t = x if torch.is_tensor(x) else torch.as_tensor(x)
            if t.ndim > 1:
                t = t.squeeze(-1)
            return t.to(torch.bool).to("cpu")

        term = _to_bool_1d(terminated) if terminated is not None else torch.zeros_like(r, dtype=torch.bool)
        trunc = _to_bool_1d(truncated) if truncated is not None else torch.zeros_like(r, dtype=torch.bool)
        done = term | trunc

        # Lazy init trackers
        if self._ep_returns is None or self._ep_returns.numel() != r.numel():
            self._ep_returns = torch.zeros_like(r, dtype=torch.float32)
            self._ep_lengths = torch.zeros_like(r, dtype=torch.int64)

        # Update trackers
        self._ep_returns += r.to(torch.float32)
        self._ep_lengths += 1

        # Log instantaneous reward mean
        try:
            wandb.log({"Reward / Instantaneous reward (mean)": float(r.float().mean().item())}, step=step)
        except Exception:
            pass

        # If any episodes finished, log episode metrics
        if done.any():
            finished_mask = done
            # Ensure CPU mask for indexing CPU tensors
            finished_mask_cpu = finished_mask.detach().to("cpu")
            ep_returns_finished = self._ep_returns[finished_mask]
            ep_lengths_finished = self._ep_lengths[finished_mask]

            if ep_returns_finished.numel() > 0:
                try:
                    wandb.log({
                        "Reward / Total reward (mean)": float(ep_returns_finished.mean().item()),
                        "Episode / Total timesteps (mean)": float(ep_lengths_finished.float().mean().item()),
                    }, step=step)
                except Exception:
                    pass

            # Log all reward components from infos["episode"] if available
            ep_info = infos.get("episode", None) if isinstance(infos, dict) else None
            if isinstance(ep_info, dict):
                for k, v in ep_info.items():
                    try:
                        # Convert to tensor on CPU
                        if torch.is_tensor(v):
                            tv = v.detach().to("cpu")
                        else:
                            tv = torch.as_tensor(v)
                        # Try to select finished envs if shapes match
                        if tv.numel() == finished_mask_cpu.numel():
                            tv = tv.view_as(finished_mask_cpu)[finished_mask_cpu]
                        # Reduce to mean if there are multiple values; else cast to float
                        val: float
                        if tv.numel() > 1:
                            val = float(tv.float().mean().item())
                        else:
                            val = float(tv.float().item())
                        wandb.log({f"Task / {k} (mean)": val}, step=step)
                    except Exception:
                        # Best-effort logging; ignore malformed entries
                        continue

                # Try to extract success signal and update run-level summary
                try:
                    # Prefer keys containing 'is_success', else any containing 'success'
                    keys_lower = {k: k.lower() for k in ep_info.keys()}
                    candidates = [k for k, kl in keys_lower.items() if "is_success" in kl]
                    if not candidates:
                        candidates = [k for k, kl in keys_lower.items() if "success" in kl]

                    for success_key in candidates:
                        v = ep_info[success_key]
                        sv = v.detach().to("cpu") if torch.is_tensor(v) else torch.as_tensor(v)
                        if sv.ndim > 1:
                            sv = sv.squeeze(-1)
                        num_finished = int(finished_mask.sum().item())
                        # Align shapes if possible
                        if sv.numel() == finished_mask_cpu.numel():
                            sv = sv.view_as(finished_mask_cpu)[finished_mask_cpu]
                        elif sv.numel() == num_finished:
                            # Already per-finished-episode values
                            sv = sv
                        elif sv.numel() == 1 and num_finished > 0:
                            sv = sv.expand(num_finished)
                        else:
                            # Shape mismatch; skip this candidate
                            continue

                        # Convert to boolean successes
                        success_tensor = sv if sv.dtype == torch.bool else (sv.float() > 0.5)
                        successes = int(success_tensor.sum().item()) if success_tensor.numel() > 0 else 0
                        episodes = num_finished
                        self._total_successes += successes
                        self._total_episodes += episodes

                        if self._total_episodes > 0 and wandb.run is not None:
                            rate = self._total_successes / self._total_episodes
                            summary_payload = {
                                # Pretty keys (for display)
                                "Task / Success rate": rate,
                                "Task / Successes": int(self._total_successes),
                                "Task / Episodes finished": int(self._total_episodes),
                                # Simple keys (robust visibility)
                                "task_success_rate": rate,
                                "task_successes": int(self._total_successes),
                                "task_episodes_finished": int(self._total_episodes),
                            }
                            try:
                                wandb.run.summary.update(summary_payload)
                            except Exception:
                                pass
                            try:
                                wandb.summary.update(summary_payload)
                            except Exception:
                                pass
                            try:
                                print(f"[W&B] Summary updated | success_rate={rate:.4f} | successes={int(self._total_successes)} | episodes={int(self._total_episodes)}")
                            except Exception:
                                pass
                        # Use the first valid candidate
                        break
                except Exception:
                    pass

            # Reset trackers for finished envs
            self._ep_returns[finished_mask] = 0.0
            self._ep_lengths[finished_mask] = 0

    def finalize_summary(self) -> None:
        """Write run-level summary metrics (success rate, counts) to W&B summary.

        Call this at the end of evaluation or simulation to ensure the latest values
        are visible in the run's Summary tab, even if no more episodes finish.
        """
        if wandb.run is None:
            return
        try:
            rate = (self._total_successes / self._total_episodes) if self._total_episodes > 0 else 0.0
            summary_payload = {
                # Pretty keys (for display)
                "Task / Success rate": rate,
                "Task / Successes": int(self._total_successes),
                "Task / Episodes finished": int(self._total_episodes),
                # Simple keys (robust visibility)
                "task_success_rate": rate,
                "task_successes": int(self._total_successes),
                "task_episodes_finished": int(self._total_episodes),
            }
            try:
                wandb.run.summary.update(summary_payload)
            except Exception:
                pass
            try:
                wandb.summary.update(summary_payload)
            except Exception:
                pass
        except Exception:
            pass
