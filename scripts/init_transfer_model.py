import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.factory import create_model_from_config


DEFAULT_SKIP_PREFIXES = (
    "patch_embedder.",
    "final_layer.linear.",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Initialize a target model by selectively copying weights from a source model."
    )
    parser.add_argument(
        "--source-model-path",
        type=Path,
        required=True,
        help="Path to the source .pt model file.",
    )
    parser.add_argument(
        "--target-config",
        type=str,
        required=True,
        help="Target Hydra config name or yaml path under configs/.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Where to save the initialized target .pt file.",
    )
    parser.add_argument(
        "--skip-prefix",
        action="append",
        default=None,
        help="State-dict prefix to skip. Can be passed multiple times.",
    )
    return parser.parse_args()


def should_skip(key, skip_prefixes):
    return any(key.startswith(prefix) for prefix in skip_prefixes)


def resolve_config_name(target_config: str) -> str:
    config_path = Path(target_config)
    if not config_path.suffix:
        return target_config

    configs_dir = REPO_ROOT / "configs"
    resolved_path = config_path.resolve()
    try:
        relative_path = resolved_path.relative_to(configs_dir.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Target config path must live under {configs_dir}, got {resolved_path}."
        ) from exc
    return str(relative_path.with_suffix("")).replace("\\", "/")


def assign_nested(cfg, dotted_key: str, value):
    current = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in current or current[part] is None:
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def compose_config(config_path: Path, configs_dir: Path):
    raw_cfg = OmegaConf.load(config_path)
    merged_cfg = OmegaConf.create({})
    defaults = raw_cfg.pop("defaults", [])

    for entry in defaults:
        if entry == "_self_":
            merged_cfg = OmegaConf.merge(merged_cfg, raw_cfg)
            continue
        if not OmegaConf.is_dict(entry) or len(entry) != 1:
            raise ValueError(f"Unsupported defaults entry {entry!r} in {config_path}.")

        default_key, default_value = next(iter(entry.items()))
        if not default_key.startswith("/"):
            raise ValueError(
                f"Only absolute defaults are supported in {config_path}, got {default_key!r}."
            )

        default_spec = default_key[1:]
        if "@" in default_spec:
            config_group, target_key = default_spec.split("@", 1)
        else:
            config_group = default_spec
            target_key = default_spec

        default_path = configs_dir / config_group / f"{default_value}.yaml"
        child_cfg = compose_config(default_path, configs_dir)
        assign_nested(merged_cfg, target_key, child_cfg)

    if "_self_" not in defaults:
        merged_cfg = OmegaConf.merge(merged_cfg, raw_cfg)
    return merged_cfg


def load_target_config(target_config: str):
    configs_dir = REPO_ROOT / "configs"
    config_name = resolve_config_name(target_config)
    return compose_config(configs_dir / f"{config_name}.yaml", configs_dir)


def main():
    args = parse_args()
    skip_prefixes = tuple(args.skip_prefix or DEFAULT_SKIP_PREFIXES)

    source_state = torch.load(args.source_model_path, map_location="cpu", weights_only=False)
    source_model_state = source_state["model"]
    target_cfg = load_target_config(args.target_config)

    target_model = create_model_from_config(target_cfg.model, target_cfg.method)
    target_arch = target_model.model
    target_state = target_arch.state_dict()

    loaded_keys = []
    skipped_keys = []
    shape_mismatch_keys = []
    missing_in_source = []

    for key, target_value in target_state.items():
        if should_skip(key, skip_prefixes):
            skipped_keys.append(key)
            continue
        if key not in source_model_state:
            missing_in_source.append(key)
            continue
        source_value = source_model_state[key]
        if source_value.shape != target_value.shape:
            shape_mismatch_keys.append((key, tuple(source_value.shape), tuple(target_value.shape)))
            continue
        target_state[key] = source_value
        loaded_keys.append(key)

    target_arch.load_state_dict(target_state)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": target_arch.state_dict(), "config": target_cfg}, args.output_path)

    print(f"Saved initialized target model to {args.output_path}")
    print(f"Loaded {len(loaded_keys)} keys")
    print(f"Skipped {len(skipped_keys)} keys by prefix")
    print(f"Missing in source: {len(missing_in_source)} keys")
    print(f"Shape mismatch: {len(shape_mismatch_keys)} keys")

    if skipped_keys:
        print("\nSkipped keys:")
        for key in skipped_keys:
            print(f"  - {key}")

    if missing_in_source:
        print("\nMissing source keys:")
        for key in missing_in_source:
            print(f"  - {key}")

    if shape_mismatch_keys:
        print("\nShape mismatch keys:")
        for key, source_shape, target_shape in shape_mismatch_keys:
            print(f"  - {key}: source={source_shape}, target={target_shape}")


if __name__ == "__main__":
    main()
