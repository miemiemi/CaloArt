import argparse
import json
from pathlib import Path

import rootutils
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

rootutils.setup_root(__file__, pythonpath=True)

from src.data.dataset import CaloShowerDataset
from src.data.preprocessing import CaloShowerPreprocessor
from src.models.factory import create_model_from_config
from src.utils import import_class_by_name, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose timestep vs condition embedding norms.")
    parser.add_argument(
        "--config-name",
        default="experiment/CaloChallenge/flow_ccd2_pred_v_all_lightingdit_freatures_c36_72_36",
        help="Hydra config name.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Optional Hydra overrides, e.g. model.architecture.init_args.pe_mode=ape",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--max-num-showers", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def compose_config(config_name: str, overrides: list[str]):
    GlobalHydra.instance().clear()
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name=config_name, overrides=overrides)


def make_optimizer(model, cfg):
    optimizer_class = cfg.train.optimizer_class
    if isinstance(optimizer_class, str):
        optimizer_class = import_class_by_name(optimizer_class)
    return optimizer_class(
        model.parameters(),
        lr=cfg.train.learning_rate,
        **OmegaConf.to_container(cfg.train.optimizer_args, resolve=True),
    )


def mean_l2_norm(x: torch.Tensor) -> float:
    return x.float().norm(dim=-1).mean().item()


@torch.no_grad()
def collect_norm_stats(flow_model, conditions):
    backbone = flow_model.model
    batch_size = conditions[0].shape[0]
    t = flow_model.sample_time(batch_size, device=conditions[0].device)

    t_emb = backbone.t_embedder(t)
    c_cond = backbone._embed_conditions(conditions)

    stats = {
        "t_emb_norm": mean_l2_norm(t_emb),
        "c_cond_norm": mean_l2_norm(c_cond),
        "c_to_t_ratio": mean_l2_norm(c_cond) / max(mean_l2_norm(t_emb), 1e-12),
    }

    stats["energy_emb_norm"] = mean_l2_norm(backbone.energy_embedder(conditions[0]))
    if backbone.phi_embedder is not None:
        stats["phi_emb_norm"] = mean_l2_norm(backbone.phi_embedder(conditions[1]))
        stats["theta_emb_norm"] = mean_l2_norm(backbone.theta_embedder(conditions[2]))
    if backbone.label_embedder is not None:
        stats["label_emb_norm"] = mean_l2_norm(backbone._embed_label_condition(conditions[-1]))

    return stats


def main():
    args = parse_args()
    cfg = compose_config(args.config_name, args.overrides)
    set_seed(int(cfg.experiment.seed))

    train_data_cfg = OmegaConf.to_container(cfg.data.train, resolve=True)
    if args.max_num_showers is not None:
        train_data_cfg["max_num_showers"] = args.max_num_showers

    dataset = CaloShowerDataset(**train_data_cfg)
    preprocessor = CaloShowerPreprocessor(**OmegaConf.to_container(cfg.preprocessing, resolve=True))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=args.device.startswith("cuda"),
    )
    data_iter = iter(dataloader)

    flow_model = create_model_from_config(cfg.model, cfg.method)
    flow_model.train()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    flow_model.to(device)

    optimizer = make_optimizer(flow_model, cfg)

    results = []

    for step in range(args.num_steps + 1):
        try:
            sample = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            sample = next(data_iter)

        shower, conditions = preprocessor.transform(*sample)
        shower = shower.to(device)
        conditions = tuple(c.to(device) for c in conditions)

        stats = collect_norm_stats(flow_model, conditions)
        stats["step"] = step
        results.append(stats)

        print(
            "[diag] "
            f"step={step} "
            f"||t_emb||={stats['t_emb_norm']:.6f} "
            f"||c_cond||={stats['c_cond_norm']:.6f} "
            f"ratio={stats['c_to_t_ratio']:.6f}",
            flush=True,
        )

        if "energy_emb_norm" in stats:
            extra = f"energy={stats['energy_emb_norm']:.6f}"
            if "phi_emb_norm" in stats:
                extra += f" phi={stats['phi_emb_norm']:.6f} theta={stats['theta_emb_norm']:.6f}"
            if "label_emb_norm" in stats:
                extra += f" label={stats['label_emb_norm']:.6f}"
            print(f"[diag] component_norms {extra}", flush=True)

        if step == args.num_steps:
            break

        optimizer.zero_grad(set_to_none=True)
        loss = flow_model(shower, conditions)
        loss.backward()
        optimizer.step()
        print(f"[diag] train_step={step} loss={loss.item():.6f}", flush=True)

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config_name": args.config_name,
            "overrides": args.overrides,
            "results": results,
        }
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"[diag] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
