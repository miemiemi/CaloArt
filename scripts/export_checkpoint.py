import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.config_utils import sanitize_config_for_artifact

def main():
    parser = argparse.ArgumentParser(description="Export an Accelerate checkpoint to a single .pt file.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the accelerate checkpoint directory")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for the .pt file")
    parser.add_argument("--use_ema", action="store_true", default=True, help="Whether to export the EMA weights (default: True)")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_path = Path(args.output_path)
    
    config_path = checkpoint_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {checkpoint_dir}")
        
    config = sanitize_config_for_artifact(OmegaConf.load(config_path))

    # Accelerate saves main model in pytorch_model.bin and ema model in pytorch_model_1.bin 
    model_bin = "pytorch_model_1.bin" if args.use_ema else "pytorch_model.bin"
    model_path = checkpoint_dir / model_bin
    
    if not model_path.exists():
        print(f"Warning: {model_bin} not found, falling back to pytorch_model.bin")
        model_path = checkpoint_dir / "pytorch_model.bin"
        if not model_path.exists():
             raise FileNotFoundError(f"Neither pytorch_model_1.bin nor pytorch_model.bin found in {checkpoint_dir}")

    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # Strip "model." prefix from the wrapper state dict to fit the inner architecture state dict
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    
    print(f"Saving to {output_path}...")
    torch.save({"model": new_state_dict, "config": config}, output_path)
    print("Done!")

if __name__ == "__main__":
    main()
