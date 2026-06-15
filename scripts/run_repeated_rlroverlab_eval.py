#!/usr/bin/env python3
"""Run repeated RLRoverLab evaluations and summarize success-rate uncertainty."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: str


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return (math.nan, math.nan)
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def paired_t_interval(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return (math.nan, math.nan, math.nan, math.nan)
    avg = mean(values)
    if len(values) == 1:
        return (avg, math.nan, math.nan, math.nan)
    sd = stdev(values)
    se = sd / math.sqrt(len(values))
    # Two-sided 95% t critical values for df 1..30. Falls back to normal.
    t_crit = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }.get(len(values) - 1, 1.96)
    margin = t_crit * se
    return (avg, sd, avg - margin, avg + margin)


def load_metrics(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_summary(root: Path, models: list[ModelSpec], seeds: list[int]) -> None:
    rows: list[dict[str, object]] = []
    by_model: dict[str, list[dict]] = {}
    for model in models:
        records = []
        for seed in seeds:
            path = root / model.name / f"seed_{seed}.json"
            if not path.exists():
                continue
            metrics = load_metrics(path)
            records.append({"seed": seed, **metrics})
            rows.append(
                {
                    "model": model.name,
                    "seed": seed,
                    "successes": metrics.get("successes"),
                    "episodes": metrics.get("completed_episodes"),
                    "success_rate": metrics.get("success_rate"),
                    "collisions": metrics.get("termination_counts", {}).get("collision"),
                    "mean_completed_return": metrics.get("mean_completed_return"),
                    "mean_completed_length": metrics.get("mean_completed_length"),
                }
            )
        by_model[model.name] = records

    with (root / "per_run.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "model",
            "seed",
            "successes",
            "episodes",
            "success_rate",
            "collisions",
            "mean_completed_return",
            "mean_completed_length",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {"models": {}, "paired": {}}
    for model in models:
        records = by_model[model.name]
        successes = sum(int(record.get("successes", 0)) for record in records)
        episodes = sum(int(record.get("completed_episodes", 0)) for record in records)
        rates = [float(record["success_rate"]) for record in records if record.get("success_rate") is not None]
        lo, hi = wilson_interval(successes, episodes)
        run_mean, run_sd, run_lo, run_hi = paired_t_interval(rates)
        summary["models"][model.name] = {
            "runs_completed": len(records),
            "successes": successes,
            "episodes": episodes,
            "pooled_success_rate": successes / episodes if episodes else None,
            "pooled_wilson_95_ci": [lo, hi],
            "per_run_mean_success_rate": run_mean,
            "per_run_sd_success_rate": run_sd,
            "per_run_t_95_ci": [run_lo, run_hi],
        }

    if len(models) >= 2:
        base = by_model[models[0].name]
        compare = by_model[models[1].name]
        base_by_seed = {int(record["seed"]): record for record in base}
        compare_by_seed = {int(record["seed"]): record for record in compare}
        common = sorted(set(base_by_seed) & set(compare_by_seed))
        diffs = [
            float(compare_by_seed[seed]["success_rate"]) - float(base_by_seed[seed]["success_rate"])
            for seed in common
        ]
        avg, sd, lo, hi = paired_t_interval(diffs)
        summary["paired"] = {
            "baseline": models[0].name,
            "comparison": models[1].name,
            "common_runs": len(common),
            "success_rate_differences": diffs,
            "mean_difference": avg,
            "sd_difference": sd,
            "paired_t_95_ci": [lo, hi],
        }

    with (root / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")


def run_eval(args: argparse.Namespace, model: ModelSpec, seed: int, metrics_path: Path, log_path: Path) -> None:
    container_checkpoint = str(Path(args.container_clone_root) / model.checkpoint)
    container_metrics = str(Path(args.container_clone_root) / metrics_path)
    command = [
        "docker",
        "exec",
        args.container,
        "bash",
        "-lc",
        " ".join(
            [
                f"cd {args.container_rlroverlab_root}",
                "&&",
                f"PYTHONPATH={args.container_clone_root}:$PYTHONPATH",
                args.python,
                "examples/04_clonelab/eval_policy.py",
                "--headless",
                "--enable_cameras",
                "--task",
                args.task,
                "--num_envs",
                str(args.num_envs),
                "--seed",
                str(seed),
                "--steps",
                str(args.steps),
                "--recurrent",
                "--checkpoint",
                container_checkpoint,
                "--metrics_out",
                container_metrics,
                "--log_interval",
                str(args.log_interval),
            ]
        ),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Evaluation failed for {model.name} seed {seed}; see {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="logs/eval_repeats_h256")
    parser.add_argument("--container", default="rover-lab-base")
    parser.add_argument("--container-clone-root", default="/workspace/clonelab")
    parser.add_argument("--container-rlroverlab-root", default="/workspace/rlroverlab")
    parser.add_argument("--python", default="/isaac-sim/python.sh")
    parser.add_argument("--task", default="AAURoverEnvRGBDRawHD720-v0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(12345, 12355)))
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    models = [
        ModelSpec(
            name="bc_h256",
            checkpoint="runs/dino-da-bc-rnn-h256/2026-05-23_18-20-44/checkpoints/actor/best_model_epoch_4.pt",
        ),
        ModelSpec(
            name="iql_h256",
            checkpoint="runs/dino-da-iql-rnn-h256/2026-05-23_19-24-26/checkpoints/actor/best_model_epoch_4.pt",
        ),
    ]

    tasks = [
        (model, seed, root / model.name / f"seed_{seed}.json", root / model.name / f"seed_{seed}.log")
        for seed in args.seeds
        for model in models
    ]

    total = len(tasks)
    completed = 0
    pending = []
    for model, seed, metrics_path, log_path in tasks:
        if metrics_path.exists() and not args.force:
            completed += 1
            print(f"[SKIP] {completed}/{total} {model.name} seed={seed}", flush=True)
            continue
        pending.append((model, seed, metrics_path, log_path))

    write_summary(root, models, args.seeds)
    if not pending:
        print(f"[INFO] Wrote {root / 'summary.json'}", flush=True)
        return 0

    workers = max(1, args.workers)
    if workers == 1:
        for model, seed, metrics_path, log_path in pending:
            print(f"[RUN ] {completed + 1}/{total} {model.name} seed={seed}", flush=True)
            run_eval(args, model, seed, metrics_path, log_path)
            completed += 1
            write_summary(root, models, args.seeds)
            metrics = load_metrics(metrics_path)
            print(
                "[DONE] "
                f"{model.name} seed={seed} "
                f"success={metrics['successes']}/{metrics['completed_episodes']} "
                f"sr={metrics['success_rate']:.4f}",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for index, (model, seed, metrics_path, log_path) in enumerate(pending, start=completed + 1):
                print(f"[RUN ] {index}/{total} {model.name} seed={seed}", flush=True)
                future = executor.submit(run_eval, args, model, seed, metrics_path, log_path)
                futures[future] = (model, seed, metrics_path)

            for future in as_completed(futures):
                model, seed, metrics_path = futures[future]
                future.result()
                completed += 1
                write_summary(root, models, args.seeds)
                metrics = load_metrics(metrics_path)
                print(
                    "[DONE] "
                    f"{model.name} seed={seed} "
                    f"success={metrics['successes']}/{metrics['completed_episodes']} "
                    f"sr={metrics['success_rate']:.4f}",
                    flush=True,
                )

    write_summary(root, models, args.seeds)
    print(f"[INFO] Wrote {root / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
