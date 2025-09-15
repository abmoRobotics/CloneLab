import gymnasium as gym
import torch
import numpy as np
from collections import deque

class SuccessTrackerWrapper(gym.Wrapper):
    """
    A Gymnasium wrapper that tracks the success rate of episodes.

    This wrapper inspects the `info` dictionary at the end of each episode
    to find a success signal, similar to the logic in CloneRL's BaseAgent.
    It populates an `episode_stats` dictionary with a deque of success
    booleans, which can be used to calculate a running success rate.

    Attributes:
        episode_stats (dict): A dictionary containing 'success', a deque
                              of booleans for recent episodes.
    """
    def __init__(self, env: gym.Env, success_buffer_size: int = 100):
        super().__init__(env)
        self.episode_stats = {
            "success": deque(maxlen=success_buffer_size)
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Determine if vectorized and which episodes are done
        is_vectorized = isinstance(terminated, (np.ndarray, torch.Tensor))
        if is_vectorized:
            term = torch.as_tensor(terminated, device='cpu', dtype=torch.bool)
            trunc = torch.as_tensor(truncated, device='cpu', dtype=torch.bool)
            if term.ndim > 1: term = term.squeeze(-1)
            if trunc.ndim > 1: trunc = trunc.squeeze(-1)
            done = term | trunc
        else:
            done = torch.tensor([terminated or truncated], device='cpu', dtype=torch.bool)

        # If any episodes finished, try to log success
        if done.any():
            finished_mask_cpu = done
            num_finished = int(finished_mask_cpu.sum().item())
            
            successes_found = []
            
            try:
                ep_info = info.get("episode", None) if isinstance(info, dict) else None
                if isinstance(ep_info, dict):
                    keys_lower = {k: k.lower() for k in ep_info.keys()}
                    candidates = [k for k, kl in keys_lower.items() if "is_success" in kl]
                    if not candidates:
                        candidates = [k for k, kl in keys_lower.items() if "success" in kl]

                    for success_key in candidates:
                        v = ep_info[success_key]
                        sv = v.detach().to("cpu") if torch.is_tensor(v) else torch.as_tensor(v)
                        if sv.ndim > 1:
                            sv = sv.squeeze(-1)
                        
                        aligned_sv = None
                        if sv.numel() == finished_mask_cpu.numel():
                            aligned_sv = sv.view_as(finished_mask_cpu)[finished_mask_cpu]
                        elif sv.numel() == num_finished:
                            aligned_sv = sv
                        elif sv.numel() == 1 and num_finished > 0:
                            aligned_sv = sv.expand(num_finished)
                        
                        if aligned_sv is not None:
                            success_tensor = aligned_sv if aligned_sv.dtype == torch.bool else (aligned_sv.float() > 0.5)
                            if success_tensor.ndim == 0:
                                successes_found.append(bool(success_tensor.item()))
                            else:
                                for success_val in success_tensor:
                                    successes_found.append(bool(success_val.item()))
                            break
            except Exception:
                pass

            # Log successes, or failures if no success info was found
            if len(successes_found) == num_finished:
                for success in successes_found:
                    self.episode_stats["success"].append(success)
            else:
                for _ in range(num_finished):
                    self.episode_stats["success"].append(False)

        return obs, reward, terminated, truncated, info
