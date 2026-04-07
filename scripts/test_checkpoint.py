import sys

# Filter out arguments injected by accelerate/torch.distributed
# that confuse hydra
sys.argv = [arg for arg in sys.argv if not arg.startswith("--local_rank")]
sys.argv = [arg for arg in sys.argv if not arg.startswith("--local-rank")]

import hydra
import rootutils
import torch
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, pythonpath=True)

from src.data.dataset import DummyDataset
from src.data.preprocessing import CaloShowerPreprocessor
from src.models.calodit_3drope import FinalLayer as LegacyGatedFinalLayer
from src.models.factory import create_model_from_config
from src.trainer import DiffusionTrainer
from src.utils import get_logger, import_class_by_name, setup_accelerator


def build_model_with_legacy_final_layer(cfg: DictConfig):
    architecture_cfg = OmegaConf.create(cfg.model.architecture)
    method_cfg = OmegaConf.create(cfg.method)

    architecture_cls = import_class_by_name(architecture_cfg["target"])
    method_cls = import_class_by_name(method_cfg["target"])

    architecture = architecture_cls(**architecture_cfg.get("init_args", {}))
    architecture.final_layer = LegacyGatedFinalLayer(
        channels=architecture.model_channels,
        patch_size=architecture.patch_size,
        out_channels=architecture.out_channels,
        use_checkpoint=architecture.use_checkpoint,
        use_rmsnorm=architecture.use_rmsnorm,
    )
    architecture.final_layer_uses_pos_emb = False

    model = method_cls(model=architecture, **method_cfg.get("init_args", {}))
    model.load_state(cfg.model.model_path)
    model.eval()
    return model


def _runtime_overrides_to_cfg():
    override_entries = []
    for entry in HydraConfig.get().overrides.task:
        if entry.startswith(("~", "hydra.")):
            continue
        if "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        key = key.lstrip("+")
        if key == "model.model_path":
            continue
        override_entries.append(f"{key}={value}")

    return OmegaConf.from_dotlist(override_entries)


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="experiment/CaloChallenge/edm",
)
def main(cfg: DictConfig):
    accelerator = setup_accelerator(**cfg.accelerator)
    logger = get_logger()

    logger.info(f"Checkpoint test config before model load:\n{OmegaConf.to_yaml(cfg)}")

    runtime_overrides = _runtime_overrides_to_cfg()
    model_path = cfg.model.get("model_path")
    if model_path:
        saved_state = torch.load(model_path, map_location="cpu", weights_only=False)
        saved_cfg = saved_state.get("config")
        if saved_cfg is not None:
            saved_cfg = OmegaConf.create(saved_cfg)
            # Allow checkpoint-only test jobs to inject additional train-time keys
            # such as test_num_showers/save_generated even if the saved config predates them.
            OmegaConf.set_struct(saved_cfg, False)
            cfg = OmegaConf.merge(saved_cfg, runtime_overrides)
            cfg.model.model_path = model_path
            cfg.train.sampling_args = OmegaConf.create(
                OmegaConf.to_container(cfg.sampling, resolve=False)
            )

    try:
        model = create_model_from_config(cfg.model, cfg.method)
    except RuntimeError as exc:
        error_text = str(exc)
        legacy_final_layer_mismatch = (
            "final_layer.scale_shift_table" in error_text
            and "final_layer.adaLN_modulation.1.weight" in error_text
        )
        if not legacy_final_layer_mismatch:
            raise
        logger.warning(
            "Falling back to the legacy ClassicDiT final layer to load an older checkpoint."
        )
        model = build_model_with_legacy_final_layer(cfg)
    if model.config is not None:
        model_cfg = OmegaConf.create(model.config)
        OmegaConf.set_struct(model_cfg, False)
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(model_cfg, cfg)
    model.save_config(cfg)

    logger.info(f"Resolved checkpoint test config:\n{OmegaConf.to_yaml(cfg)}")

    preprocessor = CaloShowerPreprocessor(**cfg.preprocessing)

    trainer = DiffusionTrainer(
        model=model,
        train_dataset=DummyDataset(),
        valid_dataset=DummyDataset(),
        preprocessor=preprocessor,
        accelerator=accelerator,
        **cfg.experiment,
        **cfg.train,
    )
    trainer.save_config(cfg)
    trainer._setup_logging()
    accelerator.wait_for_everyone()

    test_step_override = cfg.train.get("test_step_override")
    if test_step_override is not None:
        trainer.state.step = int(test_step_override)
        if accelerator.is_main_process:
            logger.info("Overriding trainer.state.step to %s for checkpoint test output routing.", trainer.state.step)

    if accelerator.is_main_process:
        logger.info("Running trainer.test() from the loaded checkpoint...")
    trainer.test()
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info("Checkpoint test completed.")
        trainer.writer.flush()
        trainer.writer.close()


if __name__ == "__main__":
    load_dotenv()
    sys.exit(main())
