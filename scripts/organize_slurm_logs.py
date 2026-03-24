#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


STRUCTURED_PATH_PATTERNS = (
    re.compile(r"EXPERIMENT_DIR=(\S+)"),
    re.compile(r"OUTPUT_DIR=(\S+)"),
    re.compile(r"validate\.output_dir=(\S+)"),
    re.compile(r"model\.model_path=(\S+)"),
    re.compile(r"Saving current state to (\S+)"),
    re.compile(r"Loading checkpoint from (\S+)"),
    re.compile(r"Loading weights from (\S+)"),
    re.compile(r"Saving to (\S+)"),
)
RUN_NAME_PATTERN = re.compile(r"experiment\.run_name=([^\s]+)")
GENERIC_PATH_PATTERN = re.compile(r"((?:/aifs[^\s\"':,]+|\.?/?(?:experiments|generations)/[^\s\"':,]+))")
TRAILING_CHARS = ".,:;)]}'\""
TIME_FALLBACK_SECONDS = 6 * 60 * 60
DEFAULT_EXTRA_GENERATION_ROOTS = (
    Path("/aifs/user/data/huangzhengkun/generations/caloflow/generations"),
)
DEFAULT_EXTRA_EXPERIMENT_ROOTS = (
    Path("/aifs/user/data/huangzhengkun/models/caloflow/trains/experiments"),
    Path("/aifs/user/data/huangzhengkun/generations/caloflow/condition_test/experiments"),
    Path("/aifs/user/data/huangzhengkun/ForPaper/Compare_EDM_and_Flow"),
    Path("/aifs/user/data/huangzhengkun/models/CaloFlow/EdmCheckpoint/checkpoint/experiments/paper_v3"),
)
MANUAL_STEM_TARGETS = {
    "cld_rope_dbg_19717": (
        Path("/aifs/user/data/huangzhengkun/models/caloflow/trains/experiments/caloflow_v1/flow_ccd2_pred_v_loss_v_logit_normal_calolightning_rope_sharemod_conv"),
    ),
    "cld2_3ep_dbg_19720": (
        Path("/aifs/user/data/huangzhengkun/models/caloflow/trains/experiments/caloflow_v1/flow_ccd2_pred_v_loss_v_logit_normal_calolightning_rope_sharemod_conv"),
    ),
    "cld2_3ep_dbg_19721": (
        Path("/aifs/user/data/huangzhengkun/models/caloflow/trains/experiments/caloflow_v1/flow_ccd2_pred_v_loss_v_logit_normal_calolightning_rope_sharemod_conv"),
    ),
    "cld2_3ep_dbg_19722": (
        Path("/aifs/user/data/huangzhengkun/models/caloflow/trains/experiments/caloflow_v1/flow_ccd2_pred_v_loss_v_logit_normal_calolightning_rope_sharemod_conv"),
    ),
    "cld2_3ep_dbg_19723": (
        Path("/aifs/user/data/huangzhengkun/models/caloflow/trains/experiments/caloflow_v1/flow_ccd2_pred_v_loss_v_logit_normal_calolightning_rope_sharemod_conv"),
    ),
    "cld2_dbg_19725": (
        Path("/aifs/user/data/huangzhengkun/models/caloflow/trains/experiments/caloflow_v1/flow_ccd2_pred_v_loss_v_logit_normal_calolightning_rope_sharemod_conv"),
    ),
    "litfeat_dbg_19731": (
        Path("/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/caloflow_v1/flow_ccd2_pred_v_all_lightingdit_freatures"),
    ),
    "litfeat_dbg_19733": (
        Path("/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/caloflow_v1/flow_ccd2_pred_v_all_lightingdit_freatures"),
    ),
    "litfeat_dbg_19735": (
        Path("/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/caloflow_v1/flow_ccd2_pred_v_all_lightingdit_freatures"),
    ),
    "litfeat_dbg_19756": (
        Path("/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/caloflow_v1/flow_ccd2_pred_v_all_lightingdit_freatures"),
    ),
    "litpix_ar_dbg_19764": (
        Path("/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/caloflow_v1/flow_ccd2_pred_v_all_lightingdit_freatures_pixart_contcond_ape_rope"),
    ),
    "litpix_ar_dbg_19765": (
        Path("/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/caloflow_v1/flow_ccd2_pred_v_all_lightingdit_freatures_pixart_contcond_ape_rope"),
    ),
    "flow_litpix_ar_dbg_19767": (
        Path("/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/caloflow_v1/flow_ccd2_pred_v_all_lightingdit_freatures_pixart_contcond_ape_rope"),
    ),
    "diag_cemb_20099": (
        Path("/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/caloflow_v1/diag_manualstop_legacy_init_c36_72_36_v2"),
    ),
    "diag_cemb_20100": (
        Path("/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/caloflow_v1/diag_manualstop_legacy_init_c36_72_36_v2"),
    ),
    "diag_train_20101": (
        Path("/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/caloflow_v1/diag_manualstop_legacy_init_c36_72_36_v2"),
    ),
    "diagL_leg_20103": (
        Path("/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/caloflow_v1/diag_manualstop_legacy_init_c36_72_36_v2"),
    ),
    "diagL_auto_20104": (
        Path("/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/caloflow_v1/diag_manualstop_auto_init_c36_72_36_v2"),
    ),
    "long_leg_20105": (
        Path("/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/caloflow_v1/diag_manualstop_legacy_init_c36_72_36_v2"),
    ),
    "long_auto_20106": (
        Path("/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/caloflow_v1/diag_manualstop_auto_init_c36_72_36_v2"),
    ),
    "engnosmoke_20110": (
        Path("/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/generations/ccd2_flow_diag_energy_only_nosharemod_last"),
    ),
    "engnosm2_20111": (
        Path("/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/generations/ccd2_flow_diag_energy_only_inputcheck_last"),
    ),
    "flow_c367236_20008": (
        Path("/aifs/user/data/huangzhengkun/models/caloflow/trains/experiments/caloflow_v1/flow_ccd2_pred_v_all_lightingdit_freatures_c36_72_36"),
    ),
    "flow_litE_20009": (
        Path("/aifs/user/data/huangzhengkun/models/caloflow/trains/experiments/caloflow_v1/flow_ccd2_pred_v_all_lightingdit_freatures_energy_only"),
    ),
    "mlpcond_smoke_20089": (
        Path("/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/caloflow_v1/flow_ccd2_pred_v_all_lightingdit_freatures_c72_36_36_mlpcond"),
    ),
    "edm_cdt_h258_20286": (
        Path("/aifs/user/data/huangzhengkun/ForPaper/Compare_EDM_and_Flow/edm_ccd2_0.5KeV_1en8_h258"),
    ),
    "edm_cdt_h258_20287": (
        Path("/aifs/user/data/huangzhengkun/ForPaper/Compare_EDM_and_Flow/edm_ccd2_0.5KeV_1en8_h258"),
    ),
    "edm_cld_legacy_19666": (
        Path("/aifs/user/data/huangzhengkun/models/CaloFlow/EdmCheckpoint/checkpoint/experiments/paper_v3/edm_calolightning_legacy"),
    ),
    "flow_cdt_h258_20289": (
        Path("/aifs/user/data/huangzhengkun/ForPaper/Compare_EDM_and_Flow/flow_ccd2_0.5KeV_1en8_h258"),
    ),
    "final_sel_tst_20623": (
        Path("/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/caloflow_v1/flow_ckpt_logcheck"),
    ),
    "final_sel_tst_20625": (
        Path("/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/caloflow_v1/flow_ckpt_logcheck"),
    ),
    "flow_aprh32_sm_20621": (
        Path("/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/caloflow_v1/flow_ccd2_pred_v_all_lightingdit_freatures_c72_36_36_ape_rope_sharemod_heun32_fpd"),
    ),
    "eval_gen_h5_19727": (
        Path("/aifs/user/data/huangzhengkun/generations/caloflow/generations/ccd2_flow_logit_normal_calolightning_rope_sharemod_conv_best"),
    ),
    "litpix_ar_sm_20622": (
        Path("/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/caloflow_v1/flow_ccd2_pred_v_all_lightingdit_freatures_pixart_contcond_ape_rope"),
    ),
}


@dataclass(frozen=True)
class LogGroup:
    stem: str
    files: tuple[Path, ...]

    @property
    def latest_mtime(self) -> float:
        return max(path.stat().st_mtime for path in self.files)


@dataclass(frozen=True)
class MappingResult:
    target: Path
    reason: str


def discover_generation_dirs(roots: list[Path]) -> list[Path]:
    dirs: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        dirs.extend(path for path in root.iterdir() if path.is_dir())
    return sorted(set(path.resolve() for path in dirs))


def discover_experiment_dirs(roots: list[Path]) -> list[Path]:
    marker_dirs = {
        "checkpoints",
        "plots",
        "wandb",
        "validate_checkpoint_last",
        "validate_checkpoint_last_s32",
    }
    experiment_dirs: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_dir():
                continue
            child_names = {child.name for child in path.iterdir()}
            if child_names & marker_dirs:
                experiment_dirs.append(path)
                continue
            if any(path.glob("final_model.pt")) or any(path.glob("validated_model*.pt")):
                experiment_dirs.append(path)
    return sorted(set(experiment_dirs))


def collect_log_groups(log_root: Path) -> list[LogGroup]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(log_root.glob("*")):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        suffix = path.suffix.lower()
        if suffix not in {".out", ".err"}:
            continue
        grouped[path.stem].append(path)
    return [LogGroup(stem=stem, files=tuple(sorted(files))) for stem, files in sorted(grouped.items())]


def read_group_text(group: LogGroup) -> str:
    chunks: list[str] = []
    for path in group.files:
        try:
            chunks.append(path.read_text(errors="ignore"))
        except OSError:
            continue
    return "\n".join(chunks)


def normalize_logged_path(raw: str, repo_root: Path) -> Path | None:
    cleaned = raw.strip().strip("\"'")
    cleaned = cleaned.rstrip(TRAILING_CHARS)
    if not cleaned:
        return None
    if cleaned.startswith("/"):
        return Path(cleaned).resolve()
    if cleaned.startswith(str(repo_root)):
        return Path(cleaned)
    if cleaned.startswith("./"):
        return (repo_root / cleaned[2:]).resolve()
    if cleaned.startswith("experiments/") or cleaned.startswith("generations/"):
        return (repo_root / cleaned).resolve()
    return None


def enclosing_dir(candidate: Path, known_dirs: list[Path]) -> Path | None:
    candidate = candidate.resolve()
    for root in sorted(known_dirs, key=lambda item: len(str(item)), reverse=True):
        try:
            candidate.relative_to(root)
            return root
        except ValueError:
            continue
    return None


def nearest_dir(reference_time: float, known_dirs: list[Path]) -> Path | None:
    best_path: Path | None = None
    best_delta = TIME_FALLBACK_SECONDS + 1
    for path in known_dirs:
        delta = abs(path.stat().st_mtime - reference_time)
        if delta < best_delta:
            best_path = path
            best_delta = delta
    if best_delta <= TIME_FALLBACK_SECONDS:
        return best_path
    return None


def index_by_name(paths: list[Path]) -> dict[str, list[Path]]:
    indexed: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        indexed[path.name].append(path)
    return indexed


def unique_by_name(name: str, indexed: dict[str, list[Path]]) -> Path | None:
    candidates = indexed.get(name, [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def best_suffix_match(candidate: Path, known_dirs: list[Path], min_score: int = 2) -> Path | None:
    candidate_parts = candidate.parts
    scored: list[tuple[int, Path]] = []
    for path in known_dirs:
        score = 0
        for left, right in zip(reversed(candidate_parts), reversed(path.parts)):
            if left != right:
                break
            score += 1
        if score >= min_score:
            scored.append((score, path))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score = scored[0][0]
    best_paths = [path for score, path in scored if score == best_score]
    if len(best_paths) == 1:
        return best_paths[0]
    return None


def explicit_targets_from_text(
    text: str,
    repo_root: Path,
    generation_dirs: list[Path],
    experiment_dirs: list[Path],
    generations_by_name: dict[str, list[Path]],
    experiments_by_name: dict[str, list[Path]],
) -> list[MappingResult]:
    results: dict[Path, str] = {}

    for pattern in STRUCTURED_PATH_PATTERNS:
        for match in pattern.findall(text):
            normalized = normalize_logged_path(match, repo_root)
            if normalized is None:
                continue
            target = enclosing_dir(normalized, experiment_dirs)
            if target is not None:
                results[target] = f"content:{pattern.pattern}"
                continue
            target = best_suffix_match(normalized, experiment_dirs)
            if target is not None:
                results[target] = f"content:{pattern.pattern}:suffix"
                continue
            target = unique_by_name(normalized.name, experiments_by_name)
            if target is not None:
                results[target] = f"content:{pattern.pattern}:name"
                continue
            target = enclosing_dir(normalized, generation_dirs)
            if target is not None:
                results[target] = f"content:{pattern.pattern}"
                continue
            target = best_suffix_match(normalized, generation_dirs, min_score=1)
            if target is not None:
                results[target] = f"content:{pattern.pattern}:suffix"
                continue
            target = unique_by_name(normalized.name, generations_by_name)
            if target is not None:
                results[target] = f"content:{pattern.pattern}:name"

    for match in GENERIC_PATH_PATTERN.findall(text):
        normalized = normalize_logged_path(match, repo_root)
        if normalized is None:
            continue
        target = enclosing_dir(normalized, experiment_dirs)
        if target is not None:
            results.setdefault(target, "content:path")
            continue
        target = best_suffix_match(normalized, experiment_dirs)
        if target is not None:
            results.setdefault(target, "content:path:suffix")
            continue
        target = unique_by_name(normalized.name, experiments_by_name)
        if target is not None:
            results.setdefault(target, "content:path:name")
            continue
        target = enclosing_dir(normalized, generation_dirs)
        if target is not None:
            results.setdefault(target, "content:path")
            continue
        target = best_suffix_match(normalized, generation_dirs, min_score=1)
        if target is not None:
            results.setdefault(target, "content:path:suffix")
            continue
        target = unique_by_name(normalized.name, generations_by_name)
        if target is not None:
            results.setdefault(target, "content:path:name")

    for run_name in RUN_NAME_PATTERN.findall(text):
        candidates = experiments_by_name.get(run_name, [])
        if len(candidates) == 1:
            results.setdefault(candidates[0], "content:run_name")

    return [MappingResult(target=path, reason=reason) for path, reason in sorted(results.items())]


def time_fallback_targets(
    group: LogGroup,
    text: str,
    generation_dirs: list[Path],
    experiment_dirs: list[Path],
) -> list[MappingResult]:
    results: list[MappingResult] = []
    lower_text = text.lower()
    if "scripts/validate.py" in lower_text or "validate_flow_" in group.stem:
        nearest_generation = nearest_dir(group.latest_mtime, generation_dirs)
        if nearest_generation is not None:
            results.append(MappingResult(target=nearest_generation, reason="time:generation"))
    if "scripts/train.py" in lower_text or "scripts/test_checkpoint.py" in lower_text:
        nearest_experiment = nearest_dir(group.latest_mtime, experiment_dirs)
        if nearest_experiment is not None:
            results.append(MappingResult(target=nearest_experiment, reason="time:experiment"))
    if not results:
        nearest_any = nearest_dir(group.latest_mtime, generation_dirs + experiment_dirs)
        if nearest_any is not None:
            results.append(MappingResult(target=nearest_any, reason="time:any"))
    deduped: dict[Path, str] = {}
    for result in results:
        deduped.setdefault(result.target, result.reason)
    return [MappingResult(target=path, reason=reason) for path, reason in deduped.items()]


def plan_group(
    group: LogGroup,
    repo_root: Path,
    generation_dirs: list[Path],
    experiment_dirs: list[Path],
    generations_by_name: dict[str, list[Path]],
    experiments_by_name: dict[str, list[Path]],
    skip_time_fallback: bool,
) -> list[MappingResult]:
    manual_targets = [
        MappingResult(target=path.resolve(), reason="manual:stem")
        for path in MANUAL_STEM_TARGETS.get(group.stem, ())
        if path.exists()
    ]
    if manual_targets:
        return manual_targets

    text = read_group_text(group)
    explicit = explicit_targets_from_text(
        text=text,
        repo_root=repo_root,
        generation_dirs=generation_dirs,
        experiment_dirs=experiment_dirs,
        generations_by_name=generations_by_name,
        experiments_by_name=experiments_by_name,
    )
    if explicit:
        return explicit
    if skip_time_fallback:
        return []
    return time_fallback_targets(
        group=group,
        text=text,
        generation_dirs=generation_dirs,
        experiment_dirs=experiment_dirs,
    )


def ensure_copied(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        src_stat = src.stat()
        dst_stat = dst.stat()
        if src_stat.st_size == dst_stat.st_size and int(src_stat.st_mtime) == int(dst_stat.st_mtime):
            return
    shutil.copy2(src, dst)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Copy slurm logs into the matching generation/experiment result directories.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help=f"Repository root (default: {repo_root})",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=repo_root / "logs" / "slurm",
        help="Directory that contains slurm .out/.err files.",
    )
    parser.add_argument(
        "--extra-generation-root",
        type=Path,
        action="append",
        default=list(DEFAULT_EXTRA_GENERATION_ROOTS),
        help="Additional generation root to search. Can be repeated.",
    )
    parser.add_argument(
        "--extra-experiment-root",
        type=Path,
        action="append",
        default=list(DEFAULT_EXTRA_EXPERIMENT_ROOTS),
        help="Additional experiment root to search. Can be repeated.",
    )
    parser.add_argument(
        "--target-subdir",
        default="slurm_logs",
        help="Subdirectory to create under each mapped result directory.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Copy files into the target directories. Default is dry-run only.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N log groups for debugging. 0 means all.",
    )
    parser.add_argument(
        "--skip-time-fallback",
        action="store_true",
        help="Only keep content-based matches. Logs without explicit targets stay unmatched.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    log_root = args.log_root.resolve()
    generation_roots = [repo_root / "generations"] + [Path(path).resolve() for path in args.extra_generation_root]
    experiment_roots = [repo_root / "experiments"] + [Path(path).resolve() for path in args.extra_experiment_root]
    generation_dirs = discover_generation_dirs(generation_roots)
    experiment_dirs = discover_experiment_dirs(experiment_roots)
    generations_by_name = index_by_name(generation_dirs)
    experiments_by_name = index_by_name(experiment_dirs)
    groups = collect_log_groups(log_root)
    if args.limit > 0:
        groups = groups[: args.limit]

    planned: dict[str, list[MappingResult]] = {}
    unmatched: list[str] = []
    for group in groups:
        mappings = plan_group(
            group=group,
            repo_root=repo_root,
            generation_dirs=generation_dirs,
            experiment_dirs=experiment_dirs,
            generations_by_name=generations_by_name,
            experiments_by_name=experiments_by_name,
            skip_time_fallback=args.skip_time_fallback,
        )
        if mappings:
            planned[group.stem] = mappings
        else:
            unmatched.append(group.stem)

    print("== Summary ==")
    print(f"log_groups: {len(groups)}")
    print(f"generation_dirs: {len(generation_dirs)}")
    print(f"experiment_dirs: {len(experiment_dirs)}")
    print(f"matched_groups: {len(planned)}")
    print(f"unmatched_groups: {len(unmatched)}")

    print("\n== Plan ==")
    for group in groups:
        mappings = planned.get(group.stem, [])
        files = ", ".join(path.name for path in group.files)
        print(f"- {group.stem}: {files}")
        if not mappings:
            print("  -> unmatched")
            continue
        for mapping in mappings:
            destination = mapping.target / args.target_subdir
            print(f"  -> {destination}  [{mapping.reason}]")

    if unmatched:
        print("\n== Unmatched Groups ==")
        for stem in unmatched:
            print(f"- {stem}")

    if not args.apply:
        return

    for group in groups:
        mappings = planned.get(group.stem, [])
        for mapping in mappings:
            target_dir = mapping.target / args.target_subdir
            for src in group.files:
                ensure_copied(src, target_dir / src.name)


if __name__ == "__main__":
    main()
