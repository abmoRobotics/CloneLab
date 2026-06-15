#!/usr/bin/env python3
"""Run DINO/DA state-to-visual DAgger iterations.

Pipeline:
1. Train an initial recurrent BC student on the existing teacher dataset, unless
   --initial-checkpoint is supplied.
2. Let the current student drive RLRoverLab while the privileged teacher labels
   each student-visited state.
3. Add the new DAgger shard to the aggregated dataset spec.
4. Retrain recurrent BC on base + all DAgger shards.
5. Evaluate the new checkpoint and repeat for --rounds.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from tqdm.auto import tqdm


DEFAULT_BASE_DATASET = "datasets/rover_dino_da3_512x288_400k.hdf5"
DEFAULT_CONFIG = "Examples/rlroverlab/configs/bc_dino_da_recurrent_h256_5ep.yaml"
DEFAULT_TASK = "AAURoverEnvRGBDRawHD720-v0"
DINODA_SCHEMA_NAME = "rlroverlab.offline_dino_da3_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run DINO/DA DAgger collection, retraining, and evaluation.")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=Path("runs/dagger_dino_da_h256"))
    parser.add_argument("--base-dataset", type=Path, default=Path(DEFAULT_BASE_DATASET))
    parser.add_argument("--base-min-idx", type=int, default=100)
    parser.add_argument("--val-dataset", type=Path, default=None)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help="Optional pretrained student checkpoint file. If omitted, an initial BC student is trained first.",
    )
    parser.add_argument("--train-config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--wandb-project", default="dino-da-dagger-bc-h256")
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--train-epochs", type=int, default=5)
    parser.add_argument("--train-total-samples", type=int, default=240000)
    parser.add_argument("--train-val-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--collect-steps", type=int, default=5000)
    parser.add_argument("--collect-num-envs", type=int, default=4)
    parser.add_argument("--collect-seed-base", type=int, default=22000)
    parser.add_argument("--collect-warmup-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=5000)
    parser.add_argument("--eval-num-envs", type=int, default=4)
    parser.add_argument("--eval-seed-base", type=int, default=32000)
    parser.add_argument("--eval-initial", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--teacher-agent", default="PPO")
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--student-policy-factory", default=None)
    parser.add_argument("--student-policy-config", type=Path, default=None)
    parser.add_argument("--terrain", default=None)
    parser.add_argument("--terrain-seed", type=int, default=None)
    parser.add_argument("--keep-terrain", action="store_true", default=False)
    parser.add_argument(
        "--run-context",
        choices=("host", "container"),
        default="host",
        help="Use 'host' to invoke Isaac commands through docker exec, or 'container' when running inside RLRoverLab.",
    )
    parser.add_argument("--container", default="rover-lab-base")
    parser.add_argument("--container-clone-root", default="/workspace/clonelab")
    parser.add_argument("--container-rlroverlab-root", default="/workspace/rlroverlab")
    parser.add_argument(
        "--python",
        default=None,
        help=(
            "Python executable for CloneLab training. Defaults to 'python' on the host and to the "
            "current interpreter in --run-context container."
        ),
    )
    parser.add_argument("--isaac-python", default="/isaac-sim/python.sh")
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--live-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()


def container_clone_path(path: Path, args: argparse.Namespace) -> str:
    path = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        rel = path.relative_to(cwd)
    except ValueError:
        return str(path)
    return str(Path(args.container_clone_root) / rel)


def runtime_clone_path(path: Path, args: argparse.Namespace) -> str:
    if args.run_context == "container":
        path = Path(path)
        if path.is_absolute():
            return str(path)
        return str(Path(args.container_clone_root) / path)
    return container_clone_path(path, args)


def training_python(args: argparse.Namespace) -> str:
    if args.python is not None:
        return args.python
    if args.run_context == "container":
        return sys.executable
    return "python"


def run_command(command: list[str], log_path: Path, *, dry_run: bool = False, live_output: bool = True) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = shlex.join(command)
    print(f"[RUN] {printable}", flush=True)
    if dry_run:
        log_path.write_text(printable + "\n", encoding="utf-8")
        return

    if live_output:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"$ {printable}\n")
            log.flush()
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=0,
            )
            if proc.stdout is None:
                raise RuntimeError("Subprocess stdout pipe was not created.")
            while True:
                chunk = proc.stdout.read(1)
                if chunk:
                    log.write(chunk)
                    sys.stdout.write(chunk)
                    if chunk in {"\n", "\r"}:
                        log.flush()
                        sys.stdout.flush()
                elif proc.poll() is not None:
                    break
            log.flush()
            sys.stdout.flush()
            returncode = proc.wait()
        if returncode != 0:
            raise RuntimeError(f"Command failed with exit code {returncode}; see {log_path}")
        return

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {printable}\n")
        log.flush()
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}; see {log_path}")


def docker_bash(args: argparse.Namespace, inner: list[str]) -> list[str]:
    return [
        "docker",
        "exec",
        args.container,
        "bash",
        "-lc",
        " ".join(inner),
    ]


def runtime_bash(args: argparse.Namespace, inner: list[str]) -> list[str]:
    if args.run_context == "container":
        return ["bash", "-lc", " ".join(inner)]
    return docker_bash(args, inner)


def append_option(command: list[str], name: str, value: object | None) -> None:
    if value is None:
        return
    command.extend([name, shlex.quote(str(value))])


def collect_round(round_id: int, checkpoint: Path, shard_path: Path, metrics_path: Path, args: argparse.Namespace) -> None:
    complete, status = _hdf5_completion_status(shard_path)
    if args.resume and complete:
        print(f"[SKIP] Round {round_id} collection exists: {shard_path}", flush=True)
        return
    if shard_path.exists():
        archived = _archive_file(shard_path)
        print(f"[WARN] Archived incomplete round {round_id} shard {shard_path} -> {archived}: {status}", flush=True)
        if metrics_path.exists():
            archived_metrics = _archive_file(metrics_path)
            print(f"[WARN] Archived stale round {round_id} metrics {metrics_path} -> {archived_metrics}", flush=True)

    tmp_shard_path = _temp_output_path(shard_path)
    tmp_metrics_path = _temp_output_path(metrics_path)
    for temp_path in (tmp_shard_path, tmp_metrics_path):
        if temp_path.exists():
            archived = _archive_file(temp_path)
            print(f"[WARN] Archived stale temporary file {temp_path} -> {archived}", flush=True)

    inner = [
        f"cd {args.container_rlroverlab_root}",
        "&&",
        f"PYTHONPATH={args.container_clone_root}:$PYTHONPATH",
        args.isaac_python,
        f"{args.container_clone_root}/scripts/collect_dagger_dino_da.py",
        "--headless",
        "--enable_cameras",
        "--task",
        shlex.quote(args.task),
        "--num_envs",
        str(args.collect_num_envs),
        "--steps",
        str(args.collect_steps),
        "--warmup_steps",
        str(args.collect_warmup_steps),
        "--seed",
        str(args.collect_seed_base + round_id),
        "--student_checkpoint",
        shlex.quote(runtime_clone_path(checkpoint, args)),
        "--teacher_agent",
        shlex.quote(args.teacher_agent),
        "--output",
        shlex.quote(runtime_clone_path(tmp_shard_path, args)),
        "--metrics_out",
        shlex.quote(runtime_clone_path(tmp_metrics_path, args)),
        "--log_interval",
        str(args.log_interval),
    ]
    append_option(
        inner,
        "--teacher_checkpoint",
        runtime_clone_path(args.teacher_checkpoint, args) if args.teacher_checkpoint else None,
    )
    append_option(inner, "--student_policy_factory", args.student_policy_factory)
    append_option(
        inner,
        "--student_policy_config",
        runtime_clone_path(args.student_policy_config, args) if args.student_policy_config else None,
    )
    append_option(inner, "--terrain", args.terrain)
    append_option(inner, "--terrain-seed", args.terrain_seed)
    if args.keep_terrain:
        inner.append("--keep-terrain")

    command = runtime_bash(args, inner)
    run_command(
        command,
        args.output_root / f"round_{round_id}" / "collect.log",
        dry_run=args.dry_run,
        live_output=args.live_output,
    )
    if args.dry_run:
        return

    complete, status = _hdf5_completion_status(tmp_shard_path)
    if not complete:
        raise RuntimeError(
            f"Round {round_id} collection did not produce a complete DINO/DA shard at {tmp_shard_path}: {status}"
        )
    tmp_shard_path.replace(shard_path)
    if tmp_metrics_path.exists():
        tmp_metrics_path.replace(metrics_path)
    print(f"[INFO] Round {round_id}: committed complete DAgger shard {shard_path}", flush=True)


def train_student(log_dir: Path, dataset_spec: str, args: argparse.Namespace) -> Path:
    started = time.time()
    command = [
        training_python(args),
        "Examples/rlroverlab/train_bc_recurrent.py",
        "--config",
        str(args.train_config),
        "--dataset",
        dataset_spec,
        "--val_dataset",
        str(args.val_dataset or args.base_dataset),
        "--dataset_format",
        "dino_da",
        "--epochs",
        str(args.train_epochs),
        "--total_samples",
        str(args.train_total_samples),
        "--val_samples",
        str(args.train_val_samples),
        "--batch_size",
        str(args.batch_size),
        "--sequence_length",
        str(args.sequence_length),
        "--num_workers",
        str(args.num_workers),
        "--wandb_project",
        args.wandb_project,
        "--wandb_mode",
        args.wandb_mode,
    ]
    run_command(command, log_dir / "train.log", dry_run=args.dry_run, live_output=args.live_output)
    if args.dry_run:
        return Path("dry_run_checkpoint.pt")
    return latest_actor_checkpoint(args.wandb_project, newer_than=started)


def train_round(round_id: int, dataset_spec: str, args: argparse.Namespace) -> Path:
    return train_student(args.output_root / f"round_{round_id}", dataset_spec, args)


def eval_round(round_id: int, checkpoint: Path, metrics_path: Path, args: argparse.Namespace) -> None:
    if args.resume and metrics_path.exists():
        print(f"[SKIP] Round {round_id} eval exists: {metrics_path}", flush=True)
        return
    command = runtime_bash(
        args,
        [
            f"cd {args.container_rlroverlab_root}",
            "&&",
            f"PYTHONPATH={args.container_clone_root}:$PYTHONPATH",
            args.isaac_python,
            "examples/04_clonelab/eval_policy.py",
            "--headless",
            "--enable_cameras",
            "--task",
            shlex.quote(args.task),
            "--num_envs",
            str(args.eval_num_envs),
            "--steps",
            str(args.eval_steps),
            "--seed",
            str(args.eval_seed_base + round_id),
            "--recurrent",
            "--checkpoint",
            shlex.quote(runtime_clone_path(checkpoint, args)),
            "--metrics_out",
            shlex.quote(runtime_clone_path(metrics_path, args)),
            "--log_interval",
            str(args.log_interval),
        ],
    )
    run_command(
        command,
        args.output_root / f"round_{round_id}" / "eval.log",
        dry_run=args.dry_run,
        live_output=args.live_output,
    )


def latest_actor_checkpoint(project: str, *, newer_than: float) -> Path:
    actor_dir = Path("runs") / project
    candidates = [
        path
        for path in actor_dir.glob("*/checkpoints/actor/best_model_epoch_*.pt")
        if path.stat().st_mtime >= newer_than
    ]
    if not candidates:
        candidates = [
            path
            for path in actor_dir.glob("*/checkpoints/actor/final_model.pt")
            if path.stat().st_mtime >= newer_than
        ]
    if not candidates:
        raise FileNotFoundError(f"Could not find a new actor checkpoint in {actor_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def dataset_spec(base_dataset: Path, base_min_idx: int, shards: list[Path]) -> str:
    specs = [f"{base_dataset}@{base_min_idx}:"]
    specs.extend(f"{shard}@0:" for shard in shards)
    return ",".join(specs)


def _is_complete_hdf5(path: Path) -> bool:
    complete, _ = _hdf5_completion_status(path)
    return complete


def _hdf5_completion_status(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        import h5py

        with h5py.File(path, "r") as file:
            schema = file.attrs.get("schema_name")
            writer_status = file.attrs.get("writer_status")
            transitions = int(file.attrs.get("total_transitions", 0))
            if schema != DINODA_SCHEMA_NAME:
                return False, f"schema_name={schema!r}, expected {DINODA_SCHEMA_NAME!r}"
            if writer_status != "complete":
                return False, f"writer_status={writer_status!r}"
            if transitions <= 0:
                return False, f"total_transitions={transitions}"
            for required_path in (
                "observations/dino_tokens_t",
                "observations/da_depth_t",
                "transitions/actions",
                "index/episode_lengths",
            ):
                if required_path not in file:
                    return False, f"missing required dataset {required_path!r}"
            return True, f"complete with {transitions} transitions"
    except BlockingIOError as exc:
        return False, f"locked by another process: {exc}"
    except OSError as exc:
        return False, f"not readable as HDF5: {exc}"


def _temp_output_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp.{os.getpid()}{path.suffix}")


def _archive_file(path: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    archive = path.with_name(f"{path.stem}.failed_{stamp}{path.suffix}")
    counter = 1
    while archive.exists():
        archive = path.with_name(f"{path.stem}.failed_{stamp}_{counter}{path.suffix}")
        counter += 1
    path.replace(archive)
    return archive


@contextmanager
def pipeline_lock(output_root: Path):
    lock_path = output_root / ".dagger_pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another DAgger pipeline is already using {output_root}. "
                f"Stop it first or use a different --output-root. Lock file: {lock_path}"
            ) from exc
        lock_file.write(f"pid={os.getpid()}\n")
        lock_file.flush()
        yield


def write_summary(records: list[dict], args: argparse.Namespace) -> None:
    path = args.output_root / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rounds": records}, indent=2) + "\n", encoding="utf-8")
    txt = args.output_root / "summary.txt"
    lines = ["DINO/DA DAgger summary", "======================", ""]
    for record in records:
        metrics = record.get("eval_metrics") or {}
        success_rate = metrics.get("success_rate")
        success_text = "n/a" if success_rate is None else f"{100.0 * success_rate:.2f}%"
        if record.get("stage") == "initial_bc":
            label = "Initial BC"
        elif record.get("stage") == "initial_checkpoint":
            label = "Initial checkpoint"
        else:
            label = f"Round {record['round']}"
        lines.append(label)
        if record.get("dagger_shard"):
            lines.append(f"  shard: {record['dagger_shard']}")
        lines.extend(
            [
                f"  checkpoint: {record['checkpoint']}",
                f"  dataset: {record.get('dataset_spec', 'n/a')}",
                f"  eval success rate: {success_text}",
                f"  eval successes: {metrics.get('successes')} / {metrics.get('completed_episodes')}",
                "",
            ]
        )
    txt.write_text("\n".join(lines), encoding="utf-8")


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_resume_records(args: argparse.Namespace) -> tuple[list[dict], list[Path], Path | None, int]:
    if not args.resume:
        return [], [], None, 1

    summary_path = args.output_root / "summary.json"
    if not summary_path.exists():
        return [], [], None, 1

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    records = payload.get("rounds", [])
    if not records:
        return [], [], None, 1

    shards = [Path(record["dagger_shard"]) for record in records if record.get("dagger_shard")]
    checkpoint = Path(records[-1]["checkpoint"]) if records[-1].get("checkpoint") else None
    completed_rounds = [int(record["round"]) for record in records if int(record.get("round", 0)) > 0]
    next_round = max(completed_rounds, default=0) + 1
    print(f"[INFO] Resuming from {summary_path}; next round is {next_round}", flush=True)
    return records, shards, checkpoint, next_round


def pipeline_stage_count(
    args: argparse.Namespace,
    *,
    records: list[dict],
    resumed_checkpoint: Path | None,
    start_round: int,
) -> int:
    stages = 0
    if resumed_checkpoint is None and args.initial_checkpoint is None:
        stages += 1
    if not records and args.eval_initial:
        stages += 1
    stages += max(0, args.rounds - start_round + 1) * 3
    return stages


def set_progress_stage(progress, stage: str) -> None:
    progress.set_postfix_str(stage)
    progress.refresh()


def main() -> int:
    args = parse_args()
    if args.rounds < 0:
        raise ValueError("--rounds must be non-negative.")
    if args.collect_steps <= args.collect_warmup_steps:
        raise ValueError("--collect-steps must be greater than --collect-warmup-steps.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    with pipeline_lock(args.output_root):
        records, shards, resumed_checkpoint, start_round = load_resume_records(args)
        progress = tqdm(
            total=pipeline_stage_count(
                args, records=records, resumed_checkpoint=resumed_checkpoint, start_round=start_round
            ),
            desc="DAgger pipeline",
            unit="stage",
            disable=not args.progress,
        )
        try:
            if resumed_checkpoint is not None:
                current_checkpoint = resumed_checkpoint
            elif args.initial_checkpoint is not None:
                current_checkpoint = args.initial_checkpoint
            else:
                initial_dir = args.output_root / "initial_bc"
                initial_dir.mkdir(parents=True, exist_ok=True)
                initial_spec = dataset_spec(args.base_dataset, args.base_min_idx, [])
                (initial_dir / "dataset_spec.txt").write_text(initial_spec + "\n", encoding="utf-8")
                print(f"[INFO] Training initial BC student on {initial_spec}", flush=True)
                set_progress_stage(progress, "initial BC train")
                current_checkpoint = train_student(initial_dir, initial_spec, args)
                progress.update(1)

            if not records and args.eval_initial:
                initial_dir = args.output_root / "initial_bc"
                initial_dir.mkdir(parents=True, exist_ok=True)
                initial_metrics = initial_dir / "eval_metrics.json"
                print(f"[INFO] Evaluating initial student {current_checkpoint}", flush=True)
                set_progress_stage(progress, "initial eval")
                eval_round(0, current_checkpoint, initial_metrics, args)
                progress.update(1)
                records.append(
                    {
                        "round": 0,
                        "stage": "initial_checkpoint" if args.initial_checkpoint is not None else "initial_bc",
                        "dataset_spec": dataset_spec(args.base_dataset, args.base_min_idx, []),
                        "checkpoint": str(current_checkpoint),
                        "eval_metrics": load_json_if_exists(initial_metrics),
                    }
                )
                write_summary(records, args)

            for round_id in range(start_round, args.rounds + 1):
                round_dir = args.output_root / f"round_{round_id}"
                round_dir.mkdir(parents=True, exist_ok=True)
                shard_path = round_dir / f"dagger_round_{round_id}.hdf5"
                collect_metrics = round_dir / "collect_metrics.json"
                eval_metrics = round_dir / "eval_metrics.json"

                print(f"[INFO] Round {round_id}: collecting with {current_checkpoint}", flush=True)
                set_progress_stage(progress, f"round {round_id} collect")
                collect_round(round_id, current_checkpoint, shard_path, collect_metrics, args)
                progress.update(1)
                shards.append(shard_path)

                spec = dataset_spec(args.base_dataset, args.base_min_idx, shards)
                (round_dir / "dataset_spec.txt").write_text(spec + "\n", encoding="utf-8")
                print(f"[INFO] Round {round_id}: training on {spec}", flush=True)
                set_progress_stage(progress, f"round {round_id} train")
                current_checkpoint = train_round(round_id, spec, args)
                progress.update(1)

                print(f"[INFO] Round {round_id}: evaluating {current_checkpoint}", flush=True)
                set_progress_stage(progress, f"round {round_id} eval")
                eval_round(round_id, current_checkpoint, eval_metrics, args)
                progress.update(1)

                records.append(
                    {
                        "round": round_id,
                        "dagger_shard": str(shard_path),
                        "dataset_spec": spec,
                        "checkpoint": str(current_checkpoint),
                        "collect_metrics": load_json_if_exists(collect_metrics),
                        "eval_metrics": load_json_if_exists(eval_metrics),
                    }
                )
                write_summary(records, args)

            if start_round > args.rounds:
                print(f"[INFO] Nothing to do; summary already has {args.rounds} DAgger rounds.", flush=True)
                write_summary(records, args)
        finally:
            progress.close()

    print(f"[INFO] Wrote {args.output_root / 'summary.json'}", flush=True)
    print(f"[INFO] Wrote {args.output_root / 'summary.txt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
