#!/usr/bin/env python3

"""Upload one scalar series from TensorBoard event files into comparable W&B runs.

This script is intended for "recovery" cases where plain `wandb sync` preserves
experiment-specific TensorBoard prefixes in metric names, which prevents multiple
runs from landing in the same W&B plot. The script:

1. Reads one TensorBoard event file (or the largest event file in a directory).
2. Finds the source scalar tag, allowing a prefixed TensorBoard tag such as:
   `flow_xxx/Observables/.../EMD RadFirstMoment`
3. Logs that series into a fresh W&B run under the normalized target key:
   `Observables/.../EMD RadFirstMoment`

Because every recovered run uses the same target metric key, W&B can place them
on a single comparable plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from tensorboard.backend.event_processing import event_accumulator
import wandb


DEFAULT_METRIC = "Observables/Geo CCD3 E -1 Phi 0.0 Theta 1.57/EMD RadFirstMoment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover one TensorBoard scalar into comparable W&B runs."
    )
    parser.add_argument(
        "sources",
        nargs="+",
        help="TensorBoard event files or experiment directories. If a directory is given, the largest event file is used.",
    )
    parser.add_argument(
        "--entity",
        required=True,
        help="W&B entity, for example: miemimail2020-Institute of High Energy Physics",
    )
    parser.add_argument(
        "--project",
        required=True,
        help="W&B project name.",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help="Normalized W&B metric key to write. Default: %(default)s",
    )
    parser.add_argument(
        "--job-type",
        default="tb_scalar_recovered",
        help="W&B job_type for the recovered runs.",
    )
    parser.add_argument(
        "--name-suffix",
        default="",
        help="Optional suffix appended to each recovered W&B run name.",
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

    # In this repo's recovery cases, one real file is usually ~84MB and the
    # later crash/restart stubs are 88 bytes. Picking the largest file is the
    # intended default.
    return max(event_files, key=lambda p: p.stat().st_size)


def find_source_tag(tags: Sequence[str], target_metric: str) -> str:
    if target_metric in tags:
        return target_metric

    suffix_matches = [tag for tag in tags if tag.endswith("/" + target_metric)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        raise ValueError(
            "Multiple TensorBoard tags end with the requested metric. "
            f"Please disambiguate manually: {suffix_matches}"
        )

    raise KeyError(
        "Could not find the requested metric in the TensorBoard file. "
        f"Requested: {target_metric}"
    )


def load_scalar_events(event_file: Path, target_metric: str):
    accumulator = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    scalar_tags = accumulator.Tags().get("scalars", [])
    source_tag = find_source_tag(scalar_tags, target_metric)
    events = accumulator.Scalars(source_tag)
    if not events:
        raise ValueError(f"No scalar events found for tag {source_tag} in {event_file}")
    return source_tag, events


def derive_run_name(source: str, event_file: Path, name_suffix: str) -> str:
    source_path = Path(source).expanduser()
    if source_path.is_dir():
        base_name = source_path.name
    else:
        base_name = event_file.parent.name
    return f"{base_name}{name_suffix}"


def upload_one(
    source: str,
    event_file: Path,
    target_metric: str,
    source_tag: str,
    scalar_events,
    entity: str,
    project: str,
    job_type: str,
    name_suffix: str,
) -> None:
    run_name = derive_run_name(source, event_file, name_suffix)
    run = wandb.init(
        entity=entity,
        project=project,
        job_type=job_type,
        name=run_name,
        config={
            "tb_source": str(event_file),
            "tb_source_tag": source_tag,
            "tb_target_metric": target_metric,
        },
    )
    assert run is not None

    run.define_metric("global_step")
    run.define_metric(target_metric, step_metric="global_step")

    for event in scalar_events:
        run.log(
            {
                "global_step": int(event.step),
                target_metric: float(event.value),
            },
            step=int(event.step),
        )

    run.summary["recovered_from"] = str(event_file)
    run.summary["source_tb_tag"] = source_tag
    run.finish()

    print(f"{run_name}: {run.url}")


def main() -> None:
    args = parse_args()
    for source in args.sources:
        event_file = resolve_event_file(source)
        source_tag, scalar_events = load_scalar_events(event_file, args.metric)
        print(f"Using {event_file}")
        print(f"  source tag: {source_tag}")
        print(f"  num points: {len(scalar_events)}")
        upload_one(
            source=source,
            event_file=event_file,
            target_metric=args.metric,
            source_tag=source_tag,
            scalar_events=scalar_events,
            entity=args.entity,
            project=args.project,
            job_type=args.job_type,
            name_suffix=args.name_suffix,
        )


if __name__ == "__main__":
    main()
