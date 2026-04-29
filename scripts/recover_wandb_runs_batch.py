#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import wandb
import yaml
from tensorboard.backend.event_processing import event_accumulator
from wandb.errors import CommError


REPO_DIR = Path(__file__).resolve().parents[1]
RECOVER_SCRIPT = REPO_DIR / "scripts" / "recover_wandb_run_from_tfevents.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover multiple W&B runs from experiment directories."
    )
    parser.add_argument("experiment_dirs", nargs="+", help="Experiment directories to recover.")
    parser.add_argument("--entity", required=True, help="W&B entity/team.")
    parser.add_argument("--project", required=True, help="W&B project.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Maximum number of global steps uploaded per recovery chunk.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=20,
        help="Seconds between polling W&B for updated summary step.",
    )
    parser.add_argument(
        "--poll-timeout-seconds",
        type=int,
        default=1800,
        help="How long to wait after each chunk for W&B summary to advance.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Progress interval forwarded to the recovery script.",
    )
    return parser.parse_args()


def find_largest_event_file(experiment_dir: Path) -> Path:
    event_files = sorted(experiment_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {experiment_dir}")
    return max(event_files, key=lambda path: path.stat().st_size)


def find_offline_run_dir(experiment_dir: Path) -> Path:
    run_dirs = sorted((experiment_dir / "wandb").glob("offline-run-*"))
    if not run_dirs:
        raise FileNotFoundError(f"No offline W&B run found in {experiment_dir / 'wandb'}")
    return run_dirs[0]


def load_wandb_config(config_yaml: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_yaml.read_text())
    config: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "value" in value:
            config[key] = value["value"]
        else:
            config[key] = value
    return config


def load_summary(summary_json: Path) -> dict[str, Any]:
    if not summary_json.exists():
        return {}
    return json.loads(summary_json.read_text())


def infer_final_step_from_event_file(event_file: Path) -> int:
    accumulator = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    max_step = -1
    for tag in tags:
        events = accumulator.Scalars(tag)
        if events:
            max_step = max(max_step, int(events[-1].step))
    return max_step


def infer_final_step_from_output_log(output_log: Path) -> int:
    if not output_log.exists():
        return -1
    pattern = re.compile(r"step (\d+)/(\d+)")
    max_step = -1
    with output_log.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                max_step = max(max_step, int(match.group(1)))
    return max_step


def resolve_run_metadata(experiment_dir: Path) -> dict[str, Any]:
    offline_run_dir = find_offline_run_dir(experiment_dir)
    run_id = offline_run_dir.name.rsplit("-", 1)[-1]
    config_yaml = offline_run_dir / "files" / "config.yaml"
    summary_json = offline_run_dir / "files" / "wandb-summary.json"
    output_log = offline_run_dir / "files" / "output.log"
    config = load_wandb_config(config_yaml)
    summary = load_summary(summary_json)
    run_name = config.get("experiment.run_name", experiment_dir.name)
    event_file = find_largest_event_file(experiment_dir)
    summary_step = int(summary.get("global_step", -1))
    event_step = infer_final_step_from_event_file(event_file)
    output_log_step = infer_final_step_from_output_log(output_log)
    final_step = max(summary_step, event_step, output_log_step)
    return {
        "experiment_dir": experiment_dir,
        "event_file": event_file,
        "offline_run_dir": offline_run_dir,
        "run_id": run_id,
        "run_name": run_name,
        "config_yaml": config_yaml,
        "summary_json": summary_json,
        "summary_step": summary_step,
        "event_step": event_step,
        "output_log_step": output_log_step,
        "final_step": final_step,
    }


def get_remote_run(api: wandb.Api, entity: str, project: str, run_id: str):
    try:
        return api.run(f"{entity}/{project}/{run_id}")
    except CommError:
        return None


def get_remote_step(api: wandb.Api, entity: str, project: str, run_id: str) -> tuple[int, str | None]:
    run = get_remote_run(api, entity, project, run_id)
    if run is None:
        return -1, None
    summary_step = run.summary.get("global_step")
    history_step = run.lastHistoryStep
    if summary_step is None and history_step is None:
        return -1, run.state
    candidates = [value for value in (summary_step, history_step) if value is not None]
    return int(max(candidates)), run.state


def run_chunk(metadata: dict[str, Any], args: argparse.Namespace, chunk_start: int, chunk_end: int) -> None:
    cmd = [
        sys.executable,
        str(RECOVER_SCRIPT),
        str(metadata["event_file"]),
        "--entity",
        args.entity,
        "--project",
        args.project,
        "--run-id",
        metadata["run_id"],
        "--run-name",
        metadata["run_name"],
        "--config-yaml",
        str(metadata["config_yaml"]),
        "--min-step",
        str(chunk_start),
        "--max-step",
        str(chunk_end),
        "--progress-every",
        str(args.progress_every),
    ]
    if chunk_end >= metadata["final_step"] and metadata["summary_json"].exists():
        cmd += ["--summary-json", str(metadata["summary_json"])]

    print()
    print(f"[{metadata['run_name']}] Uploading steps {chunk_start}..{chunk_end}", flush=True)
    print("Command:", " ".join(subprocess.list2cmdline([part]) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_DIR, check=True)


def wait_for_remote_step(
    api: wandb.Api,
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
        fresh_api = wandb.Api(timeout=120)
        latest_step, latest_state = get_remote_step(fresh_api, entity, project, run_id)
        print(f"  remote step={latest_step}, state={latest_state}", flush=True)
        if latest_step >= expected_step:
            return latest_step, latest_state
        time.sleep(poll_seconds)
    return latest_step, latest_state


def recover_one(api: wandb.Api, metadata: dict[str, Any], args: argparse.Namespace) -> None:
    print(flush=True)
    print("=" * 80, flush=True)
    print(f"Experiment: {metadata['experiment_dir']}", flush=True)
    print(f"Run name : {metadata['run_name']}", flush=True)
    print(f"Run id   : {metadata['run_id']}", flush=True)
    print(f"Event    : {metadata['event_file']}", flush=True)
    print(
        "Step hints: "
        f"summary={metadata['summary_step']}, "
        f"event={metadata['event_step']}, "
        f"output_log={metadata['output_log_step']}"
        ,
        flush=True,
    )
    print(f"Final step target: {metadata['final_step']}", flush=True)

    remote_step, remote_state = get_remote_step(api, args.entity, args.project, metadata["run_id"])
    print(f"Initial remote step={remote_step}, state={remote_state}", flush=True)
    if remote_step >= metadata["final_step"]:
        print("Run already appears fully recovered, skipping.", flush=True)
        return

    next_step = 0 if remote_step < 0 else remote_step + 1
    while next_step <= metadata["final_step"]:
        chunk_end = min(next_step + args.chunk_size - 1, metadata["final_step"])
        run_chunk(metadata, args, next_step, chunk_end)
        remote_step, remote_state = wait_for_remote_step(
            api,
            args.entity,
            args.project,
            metadata["run_id"],
            chunk_end,
            args.poll_seconds,
            args.poll_timeout_seconds,
        )
        if remote_step < chunk_end:
            raise RuntimeError(
                f"W&B did not advance to step {chunk_end} for {metadata['run_id']}. "
                f"Latest remote step={remote_step}, state={remote_state}"
            )
        next_step = remote_step + 1

    final_remote_step, final_state = wait_for_remote_step(
        api,
        args.entity,
        args.project,
        metadata["run_id"],
        metadata["final_step"],
        args.poll_seconds,
        args.poll_timeout_seconds,
    )
    print(
        f"[{metadata['run_name']}] finished with remote step={final_remote_step}, "
        f"state={final_state}"
        ,
        flush=True,
    )


def main() -> None:
    args = parse_args()
    experiment_dirs = [path for path in args.experiment_dirs if path.strip()]
    for path in experiment_dirs:
        api = wandb.Api(timeout=120)
        metadata = resolve_run_metadata(Path(path).expanduser().resolve())
        recover_one(api, metadata, args)


if __name__ == "__main__":
    main()
