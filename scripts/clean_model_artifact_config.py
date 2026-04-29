import argparse
from pathlib import Path

import torch

from src.config_utils import sanitize_config_for_artifact


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite a saved model artifact with a sanitized embedded config."
    )
    parser.add_argument("model_path", type=Path, help="Path to a saved model artifact (.pt)")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output path. Defaults to overwriting model_path in place.",
    )
    args = parser.parse_args()

    model_path = args.model_path
    output_path = args.output_path or model_path

    state = torch.load(model_path, map_location="cpu", weights_only=False)
    state["config"] = sanitize_config_for_artifact(state.get("config"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
