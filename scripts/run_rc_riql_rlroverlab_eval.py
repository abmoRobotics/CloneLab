#!/usr/bin/env python3
"""Run paired RLRoverLab risk-preference sweeps for an RC-RIQL checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shlex
import subprocess
import sys
from pathlib import Path
from statistics import mean


SUMMARY_METRICS = (
    "success_rate",
    "collision_rate",
    "mean_episode_minimum_clearance_m",
    "lower_tail_clearance_p05_m",
    "mean_risk_exposure",
    "mean_risk_exposure_per_second",
    "mean_successful_path_efficiency",
    "mean_successful_time_to_goal_s",
)


def alpha_slug(alpha: float) -> str:
    return f"alpha_{alpha:.3f}".replace(".", "p")


def bootstrap_mean_interval(
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1 or samples <= 0:
        value = float(values[0])
        return [value, value]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(mean(draw))
    means.sort()
    lower = means[int(0.025 * (len(means) - 1))]
    upper = means[int(0.975 * (len(means) - 1))]
    return [lower, upper]


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def container_path(path: Path, host_root: Path, container_root: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(host_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path must be inside the CloneLab checkout {host_root}: {resolved}") from exc
    return str(container_root / relative)


def terrain_seed_for(args: argparse.Namespace, index: int, seed: int) -> int:
    if not args.terrain_seeds:
        return seed
    if len(args.terrain_seeds) == 1:
        return args.terrain_seeds[0]
    if len(args.terrain_seeds) != len(args.seeds):
        raise ValueError("--terrain-seeds must contain one value or one value per --seeds entry.")
    return args.terrain_seeds[index]


def run_evaluation(
    args: argparse.Namespace,
    *,
    alpha: float,
    seed: int,
    terrain_seed: int,
    metrics_path: Path,
    episodes_path: Path,
    log_path: Path,
) -> None:
    host_clone_root = Path(args.host_clone_root).resolve()
    container_clone_root = Path(args.container_clone_root)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = host_clone_root / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Actor checkpoint does not exist: {checkpoint}")

    metrics_path.unlink(missing_ok=True)
    episodes_path.unlink(missing_ok=True)

    evaluator_args = [
        args.python,
        "examples/04_clonelab/eval_risk_conditioned_policy.py",
        "--headless",
        "--enable_cameras",
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
        "--seed",
        str(seed),
        "--terrain-seed",
        str(terrain_seed),
        "--target_episodes",
        str(args.target_episodes),
        "--max_steps",
        str(args.max_steps),
        "--risk_preference",
        str(alpha),
        "--checkpoint",
        container_path(checkpoint, host_clone_root, container_clone_root),
        "--metrics_out",
        container_path(metrics_path, host_clone_root, container_clone_root),
        "--episodes_out",
        container_path(episodes_path, host_clone_root, container_clone_root),
        "--log_interval",
        str(args.log_interval),
    ]
    if args.terrain:
        evaluator_args.extend(["--terrain", args.terrain])
    if args.stochastic:
        evaluator_args.append("--stochastic")

    shell_command = (
        f"cd {shlex.quote(args.container_rlroverlab_root)} && "
        f"PYTHONPATH={shlex.quote(args.container_clone_root)}:$PYTHONPATH "
        f"{shlex.join(evaluator_args)}"
    )
    command = ["docker", "exec", args.container, "bash", "-lc", shell_command]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if process.stdout is None:
            raise RuntimeError("Failed to capture evaluator output.")
        while chunk := process.stdout.read1(4096):
            log.write(chunk)
            log.flush()
            if args.live_output:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Evaluation failed for alpha={alpha} seed={seed}; see {log_path}."
        )
    if not metrics_path.exists() or not episodes_path.exists():
        raise RuntimeError(
            f"Evaluation did not produce its required outputs for alpha={alpha} seed={seed}; "
            f"see {log_path}."
        )
    metrics = load_json(metrics_path)
    if not metrics.get("target_episodes_reached"):
        raise RuntimeError(
            f"Evaluation stopped before reaching {args.target_episodes} episodes for "
            f"alpha={alpha} seed={seed}; see {log_path}."
        )


def write_reports(
    root: Path,
    alphas: list[float],
    seeds: list[int],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> None:
    per_run_rows: list[dict] = []
    episode_rows: list[dict] = []
    runs_by_alpha: dict[float, list[dict]] = {alpha: [] for alpha in alphas}

    for alpha in alphas:
        slug = alpha_slug(alpha)
        for seed in seeds:
            metrics_path = root / slug / f"seed_{seed}_metrics.json"
            episodes_path = root / slug / f"seed_{seed}_episodes.jsonl"
            if not metrics_path.exists():
                continue
            metrics = load_json(metrics_path)
            runs_by_alpha[alpha].append(metrics)
            per_run_rows.append(
                {
                    "alpha": alpha,
                    "seed": seed,
                    **{key: metrics.get(key) for key in SUMMARY_METRICS},
                    "completed_episodes": metrics.get("completed_episodes"),
                    "successes": metrics.get("successes"),
                    "collisions": metrics.get("collisions"),
                    "target_episodes_reached": metrics.get("target_episodes_reached"),
                }
            )
            for episode in load_jsonl(episodes_path):
                episode_rows.append({"alpha": alpha, "seed": seed, **episode})

    root.mkdir(parents=True, exist_ok=True)
    _write_csv(root / "per_run.csv", per_run_rows)
    _write_csv(root / "episodes.csv", episode_rows)

    summary: dict[str, object] = {
        "alphas": {},
        "paired_against_first_alpha": {},
        "bootstrap_samples": bootstrap_samples,
    }
    for alpha_index, alpha in enumerate(alphas):
        runs = runs_by_alpha[alpha]
        episodes = [row for row in episode_rows if float(row["alpha"]) == alpha]
        alpha_summary: dict[str, object] = {
            "runs_completed": len(runs),
            "completed_episodes": len(episodes),
            "successes": sum(bool(row["success"]) for row in episodes),
            "collisions": sum(bool(row["collision"]) for row in episodes),
        }
        if episodes:
            success_values = [float(bool(row["success"])) for row in episodes]
            collision_values = [float(bool(row["collision"])) for row in episodes]
            min_clearances = [float(row["minimum_clearance_m"]) for row in episodes]
            alpha_summary.update(
                {
                    "pooled_success_rate": mean(success_values),
                    "pooled_success_rate_bootstrap_95_ci": bootstrap_mean_interval(
                        success_values,
                        samples=bootstrap_samples,
                        seed=bootstrap_seed + alpha_index * 100,
                    ),
                    "pooled_collision_rate": mean(collision_values),
                    "pooled_collision_rate_bootstrap_95_ci": bootstrap_mean_interval(
                        collision_values,
                        samples=bootstrap_samples,
                        seed=bootstrap_seed + alpha_index * 100 + 1,
                    ),
                    "pooled_lower_tail_clearance_p05_m": percentile(min_clearances, 0.05),
                }
            )
            episode_metric_keys = (
                "minimum_clearance_m",
                "risk_exposure",
                "risk_exposure_per_second",
                "traveled_path_length_m",
            )
            for metric_index, key in enumerate(episode_metric_keys, start=2):
                values = [float(row[key]) for row in episodes if row.get(key) is not None]
                alpha_summary[f"mean_{key}"] = mean(values) if values else None
                alpha_summary[f"mean_{key}_bootstrap_95_ci"] = bootstrap_mean_interval(
                    values,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + alpha_index * 100 + metric_index,
                )
            successful = [row for row in episodes if row["success"]]
            for metric_index, key in enumerate(("path_efficiency", "time_to_goal_s"), start=10):
                values = [float(row[key]) for row in successful if row.get(key) is not None]
                alpha_summary[f"mean_successful_{key}"] = mean(values) if values else None
                alpha_summary[f"mean_successful_{key}_bootstrap_95_ci"] = bootstrap_mean_interval(
                    values,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + alpha_index * 100 + metric_index,
                )
        summary["alphas"][str(alpha)] = alpha_summary

    if alphas:
        baseline_alpha = alphas[0]
        baseline_by_seed = {
            int(run["seed"]): run
            for run in runs_by_alpha[baseline_alpha]
        }
        for alpha_index, alpha in enumerate(alphas[1:], start=1):
            comparison_by_seed = {
                int(run["seed"]): run
                for run in runs_by_alpha[alpha]
            }
            common_seeds = sorted(set(baseline_by_seed) & set(comparison_by_seed))
            paired_metrics: dict[str, object] = {
                "baseline_alpha": baseline_alpha,
                "comparison_alpha": alpha,
                "common_seeds": common_seeds,
                "differences": {},
            }
            for metric_index, key in enumerate(SUMMARY_METRICS):
                differences = []
                for seed in common_seeds:
                    baseline = baseline_by_seed[seed].get(key)
                    comparison = comparison_by_seed[seed].get(key)
                    if baseline is not None and comparison is not None:
                        differences.append(float(comparison) - float(baseline))
                paired_metrics["differences"][key] = {
                    "values": differences,
                    "mean": mean(differences) if differences else None,
                    "bootstrap_95_ci": bootstrap_mean_interval(
                        differences,
                        samples=bootstrap_samples,
                        seed=bootstrap_seed + alpha_index * 1000 + metric_index,
                    ),
                }
            summary["paired_against_first_alpha"][str(alpha)] = paired_metrics

    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    clone_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path relative to CloneLab or absolute.")
    parser.add_argument("--output-root", default="logs/rc_riql_rlroverlab_eval")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(12345, 12355)))
    parser.add_argument("--terrain-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--terrain", default=None)
    parser.add_argument("--task", default="AAURoverEnvRGBDRawHD720-v0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--target-episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument(
        "--live-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream evaluator progress while preserving each per-run log.",
    )
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260614)
    parser.add_argument("--container", default="rover-lab-base")
    parser.add_argument("--host-clone-root", default=str(clone_root))
    parser.add_argument("--container-clone-root", default="/workspace/clonelab")
    parser.add_argument("--container-rlroverlab-root", default="/workspace/rlroverlab")
    parser.add_argument("--python", default="/isaac-sim/python.sh")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if any(not 0.0 <= alpha <= 1.0 for alpha in args.alphas):
        raise ValueError("All --alphas values must lie in [0, 1].")

    host_clone_root = Path(args.host_clone_root).resolve()
    root = Path(args.output_root)
    if not root.is_absolute():
        root = host_clone_root / root
    root.mkdir(parents=True, exist_ok=True)

    total = len(args.alphas) * len(args.seeds)
    completed = 0
    for seed_index, seed in enumerate(args.seeds):
        terrain_seed = terrain_seed_for(args, seed_index, seed)
        for alpha in args.alphas:
            slug = alpha_slug(alpha)
            metrics_path = root / slug / f"seed_{seed}_metrics.json"
            episodes_path = root / slug / f"seed_{seed}_episodes.jsonl"
            log_path = root / slug / f"seed_{seed}.log"
            completed += 1
            if metrics_path.exists() and not args.force:
                existing = load_json(metrics_path)
                if existing.get("target_episodes_reached"):
                    print(f"[SKIP] {completed}/{total} alpha={alpha:.3f} seed={seed}", flush=True)
                    continue

            print(f"[RUN ] {completed}/{total} alpha={alpha:.3f} seed={seed}", flush=True)
            run_evaluation(
                args,
                alpha=alpha,
                seed=seed,
                terrain_seed=terrain_seed,
                metrics_path=metrics_path,
                episodes_path=episodes_path,
                log_path=log_path,
            )
            metrics = load_json(metrics_path)
            print(
                f"[DONE] alpha={alpha:.3f} seed={seed} "
                f"success={metrics['success_rate']:.4f} "
                f"collision={metrics['collision_rate']:.4f}",
                flush=True,
            )
            write_reports(
                root,
                args.alphas,
                args.seeds,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            )

    write_reports(
        root,
        args.alphas,
        args.seeds,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(f"[INFO] Wrote {root / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
