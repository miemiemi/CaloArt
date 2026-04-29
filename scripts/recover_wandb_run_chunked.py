#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import wandb
from tensorboard.backend.event_processing import event_accumulator
from wandb.errors import CommError


REPO_DIR = Path(__file__).resolve().parents[1]
RECOVER_SCRIPT = REPO_DIR / "scripts" / "recover_wandb_run_from_tfevents.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover one W&B run from TensorBoard in multiple online chunks."
    )
    parser.add_argument("source", help="TensorBoard event file or experiment directory.")
    parser.add_argument("--entity", required=True, help="W&B entity/team.")
    parser.add_argument("--project", required=True, help="W&B project.")
    parser.add_argument("--run-id", required=True, help="Target W&B run id.")
    parser.add_argument("--run-name", default=None, help="Display name for the run.")
    parser.add_argument("--config-yaml", default=None, help="Optional W&B config.yaml.")
    parser.add_argument("--summary-json", default=None, help="Optional wandb-summary.json.")
    parser.add_argument("--job-type", default="tb_run_recovered_chunked", help="W&B job_type.")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Steps per upload chunk.")
    parser.add_argument("--progress-every", type=int, default=5000, help="Log frequency per chunk.")
    parser.add_argument("--min-step", type=int, default=0, help="Inclusive lower bound.")
    parser.add_argument("--max-step", type=int, default=None, help="Inclusive upper bound.")
    parser.add_argument("--poll-seconds", type=int, default=20, help="Remote polling interval.")
    parser.add_argument(
        "--poll-timeout-seconds",
        type=int,
        default=1800,
        help="How long to wait for each chunk to appear remotely.",
    )
    return parser.parse_args()


def resolve_event_file(source: str) -> Path:
    path = Path(source).expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Source does not exist: {path}")
    event_files = sorted(path.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {path}")
    return max(event_files, key=lambda candidate: candidate.stat().st_size)


def infer_final_step(event_file: Path) -> int:
    accumulator = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    max_step = -1
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        if events:
            max_step = max(max_step, int(events[-1].step))
    if max_step < 0:
        raise ValueError(f"No scalar steps found in {event_file}")
    return max_step


def get_remote_run(entity: str, project: str, run_id: str):
    api = wandb.Api(timeout=120)
    try:
        return api.run(f"{entity}/{project}/{run_id}")
    except CommError:
        return None


def get_remote_step(entity: str, project: str, run_id: str) -> tuple[int, str | None]:
    run = get_remote_run(entity, project, run_id)
    if run is None:
        return -1, None
    values = [value for value in (run.summary.get("global_step"), run.lastHistoryStep) if value is not None]
    if not values:
        return -1, run.state
    return int(max(values)), run.state


def wait_for_remote_step(
    entity: str,
    project: str,
    run_id: str,
    expected_step: int,
    poll_seconds: int,
    timeout_seconds: int,
) -> tuple[int, str | None]:
    start = time.time()
    latest_step = -1
    latest_state = None
    while time.time() - start <= timeout_seconds:
        latest_step, latest_state = get_remote_step(entity, project, run_id)
        print(f"  remote step={latest_step}, state={latest_state}", flush=True)
        if latest_step >= expected_step:
            return latest_step, latest_state
        time.sleep(poll_seconds)
    return latest_step, latest_state


def run_chunk(args: argparse.Namespace, event_file: Path, chunk_start: int, chunk_end: int, final_step: int) -> None:
    cmd = [
        sys.executable,
        str(RECOVER_SCRIPT),
        str(event_file),
        "--entity",
        args.entity,
        "--project",
        args.project,
        "--run-id",
        args.run_id,
        "--job-type",
        args.job_type,
        "--min-step",
        str(chunk_start),
        "--max-step",
        str(chunk_end),
        "--progress-every",
        str(args.progress_every),
    ]
    if args.run_name:
        cmd += ["--run-name", args.run_name]
    if args.config_yaml and Path(args.config_yaml).exists():
        cmd += ["--config-yaml", args.config_yaml]
    if args.summary_json and chunk_end >= final_step and Path(args.summary_json).exists():
        cmd += ["--summary-json", args.summary_json]

    print()
    print(f"Uploading chunk {chunk_start}..{chunk_end}", flush=True)
    print("Command:", " ".join(subprocess.list2cmdline([part]) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_DIR, check=True)


def main() -> None:
    args = parse_args()
    event_file = resolve_event_file(args.source)
    final_step = infer_final_step(event_file)
    if args.max_step is not None:
        final_step = min(final_step, args.max_step)

    print(f"Event file: {event_file}", flush=True)
    print(f"Final step target: {final_step}", flush=True)

    remote_step, remote_state = get_remote_step(args.entity, args.project, args.run_id)
    print(f"Initial remote step={remote_step}, state={remote_state}", flush=True)

    next_step = max(args.min_step, remote_step + 1 if remote_step >= 0 else args.min_step)
    while next_step <= final_step:
        chunk_end = min(next_step + args.chunk_size - 1, final_step)
        run_chunk(args, event_file, next_step, chunk_end, final_step)
        remote_step, remote_state = wait_for_remote_step(
            args.entity,
            args.project,
            args.run_id,
            chunk_end,
            args.poll_seconds,
            args.poll_timeout_seconds,
        )
        if remote_step < chunk_end:
            raise RuntimeError(
                f"Remote run did not advance to {chunk_end}. "
                f"Latest remote step={remote_step}, state={remote_state}"
            )
        next_step = chunk_end + 1

    print(f"Finished chunked recovery for {args.run_id}", flush=True)


if __name__ == "__main__":
    main()
