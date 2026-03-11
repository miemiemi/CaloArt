import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


DEFAULT_CONFIGS = [
    "experiment/CaloChallenge/flow_uniform",
    "experiment/CaloChallenge/flow_logit_normal",
    "experiment/CaloChallenge/edm",
]
REQUIRED_TOP_LEVEL_KEYS = ("accelerator", "experiment", "model", "method", "sampling", "train")


def compose_config(config_name: str, overrides: list[str]):
    GlobalHydra.instance().clear()
    config_dir = Path(__file__).resolve().parent / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name=config_name, overrides=overrides)


def validate_config(config_name: str, overrides: list[str]) -> None:
    cfg = compose_config(config_name, overrides)

    missing_keys = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in cfg]
    if missing_keys:
        raise AssertionError(f"{config_name} is missing top-level keys: {missing_keys}")

    if config_name.endswith("flow_logit_normal"):
        flow_args = cfg.method.init_args
        if cfg.method.kind != "flow":
            raise AssertionError("flow_logit_normal must use method.kind=flow")
        if flow_args.time_sampler != "logit_normal":
            raise AssertionError("flow_logit_normal must use time_sampler=logit_normal")
        if flow_args.logit_normal_mean != 0.0 or flow_args.logit_normal_std != 1.0:
            raise AssertionError("flow_logit_normal must keep logit_normal_mean/std at 0.0/1.0")

    if config_name.endswith("edm") and cfg.method.kind != "edm":
        raise AssertionError("edm config must use method.kind=edm")

    print(f"[OK] {config_name}")
    print(OmegaConf.to_yaml(cfg))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose Hydra training configs and validate their shape.")
    parser.add_argument("--config-name", default=None, help="Hydra config name to validate.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional Hydra overrides such as train.max_steps=10",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_names = [args.config_name] if args.config_name else DEFAULT_CONFIGS
    for config_name in config_names:
        validate_config(config_name, args.overrides)


if __name__ == "__main__":
    main()
