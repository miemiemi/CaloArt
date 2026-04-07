#!/usr/bin/env python3

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
from typing import Any

import wandb
import yaml
from tensorboard.backend.event_processing import event_accumulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover a W&B run from one TensorBoard event file."
    )
    parser.add_argument(
        "source",
        help="TensorBoard event file or experiment directory. If a directory is given, the largest event file is used.",
    )
    parser.add_argument("--entity", required=True, help="W&B entity/team name.")
    parser.add_argument("--project", required=True, help="W&B project name.")
    parser.add_argument("--run-id", required=True, help="W&B run id to create or resume.")
    parser.add_argument("--run-name", default=None, help="W&B display name.")
    parser.add_argument(
        "--config-yaml",
        default=None,
        help="Optional local W&B config.yaml to restore run config from.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional local wandb-summary.json to restore scalar summary values from.",
    )
    parser.add_argument(
        "--job-type",
        default="tb_run_recovered",
        help="W&B job_type for the recovered run.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print a progress line every N logged steps.",
    )
    parser.add_argument(
        "--min-step",
        type=int,
        default=None,
        help="Optional inclusive lower bound on global_step to upload.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Optional inclusive upper bound on global_step to upload.",
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
        raise FileNotFoundError(f"No TensorBoard event files found in: {path}")
    return max(event_files, key=lambda p: p.stat().st_size)


def load_run_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    raw = yaml.safe_load(Path(path).read_text())
    config: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "value" in value:
            config[key] = value["value"]
        else:
            config[key] = value
    return config


def load_scalar_summary(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    raw = json.loads(Path(path).read_text())
    summary: dict[str, Any] = {}
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        if isinstance(value, (int, float, str, bool)) or value is None:
            summary[key] = value
    return summary


def load_scalar_tags(event_file: Path) -> tuple[event_accumulator.EventAccumulator, list[str]]:
    accumulator = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    if not tags:
        raise ValueError(f"No scalar tags found in {event_file}")
    return accumulator, tags


def iter_step_rows(
    accumulator: event_accumulator.EventAccumulator,
    tags: list[str],
    min_step: int | None,
    max_step: int | None,
):
    scalar_lists = [accumulator.Scalars(tag) for tag in tags]
    heap: list[tuple[int, int, int]] = []
    for tag_idx, events in enumerate(scalar_lists):
        if events:
            heapq.heappush(heap, (int(events[0].step), tag_idx, 0))

    while heap:
        step, tag_idx, pos = heapq.heappop(heap)
        row: dict[str, Any] = {"global_step": step}
        row[tags[tag_idx]] = float(scalar_lists[tag_idx][pos].value)

        next_pos = pos + 1
        if next_pos < len(scalar_lists[tag_idx]):
            heapq.heappush(heap, (int(scalar_lists[tag_idx][next_pos].step), tag_idx, next_pos))

        while heap and heap[0][0] == step:
            _, other_tag_idx, other_pos = heapq.heappop(heap)
            row[tags[other_tag_idx]] = float(scalar_lists[other_tag_idx][other_pos].value)
            other_next = other_pos + 1
            if other_next < len(scalar_lists[other_tag_idx]):
                heapq.heappush(
                    heap,
                    (int(scalar_lists[other_tag_idx][other_next].step), other_tag_idx, other_next),
                )

        yield step, row


def main() -> None:
    args = parse_args()
    event_file = resolve_event_file(args.source)
    config = load_run_config(args.config_yaml)
    summary = load_scalar_summary(args.summary_json)
    accumulator, tags = load_scalar_tags(event_file)

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        resume="allow",
        reinit=True,
        name=args.run_name,
        job_type=args.job_type,
        config=config,
        settings=wandb.Settings(
            console="off",
            disable_job_creation=True,
            disable_git=True,
            save_code=False,
        ),
    )
    if run is None:
        raise RuntimeError("wandb.init returned None")

    run.define_metric("global_step")
    run.define_metric("*", step_metric="global_step")

    print(f"Recovering {len(tags)} scalar tags from {event_file}")
    print(f"Run URL: {run.url}")

    num_rows = 0
    last_step = None
    for step, row in iter_step_rows(accumulator, tags, args.min_step, args.max_step):
        if args.min_step is not None and step < args.min_step:
            continue
        if args.max_step is not None and step > args.max_step:
            continue
        run.log(row, step=step)
        num_rows += 1
        last_step = step
        if args.progress_every > 0 and num_rows % args.progress_every == 0:
            print(f"Logged {num_rows} steps; latest step={step}")

    run.summary["recovered_from_tfevents"] = str(event_file)
    run.summary["recovered_scalar_tags"] = len(tags)
    run.summary["recovered_logged_steps"] = num_rows
    if last_step is not None:
        run.summary["global_step"] = int(last_step)

    for key, value in summary.items():
        run.summary[key] = value

    run.finish()
    print(f"Finished recovery: steps={num_rows}, last_step={last_step}")


if __name__ == "__main__":
    main()
