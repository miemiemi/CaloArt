import sys
# Filter out arguments injected by accelerate/torch.distributed
# that confuse hydra
sys.argv = [arg for arg in sys.argv if not arg.startswith("--local_rank")]
sys.argv = [arg for arg in sys.argv if not arg.startswith("--local-rank")]
import rootutils
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, pythonpath=True)

from src.data.dataset import CaloShowerDataset, DummyDataset
from src.data.preprocessing import CaloShowerPreprocessor
from src.models.factory import create_model_from_config
from src.trainer import DiffusionTrainer
from src.utils import get_logger, set_seed, setup_accelerator
import hydra



@hydra.main(version_base="1.3", config_path="../configs", config_name="experiment/CaloChallenge/flow_uniform")
def main(cfg: DictConfig):
    # accelerator
    accelerator = setup_accelerator(**cfg.accelerator)

    # logging
    logger = get_logger()

    print(f"Process {accelerator.process_index} using device: {accelerator.device}", flush=True)
    accelerator.wait_for_everyone()
    logger.info(f"World size: {accelerator.num_processes}")

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # seed
    process_seed = torch.randint(
        low=0,
        high=2**32 - 1,
        size=[accelerator.num_processes],
        generator=torch.Generator().manual_seed(cfg.experiment.seed),
    )
    set_seed(int(process_seed[accelerator.process_index].item()))

    # data
    preprocessor = CaloShowerPreprocessor(**cfg.preprocessing)
    if accelerator.is_main_process:
        logger.info("Loading data...")
        train_data = CaloShowerDataset(**cfg.data.train)
        valid_data = CaloShowerDataset(**cfg.data.valid)
    else:
        train_data, valid_data = DummyDataset(), DummyDataset()
    accelerator.wait_for_everyone()

    # setup model
    logger.info("Creating model...")
    model = create_model_from_config(cfg.model, cfg.method)
    model.save_config(cfg)

    if accelerator.is_main_process:
        model.summarize()

    # setup trainer
    trainer = DiffusionTrainer(
        model=model,
        train_dataset=train_data,
        valid_dataset=valid_data,
        preprocessor=preprocessor,
        accelerator=accelerator,
        **cfg.experiment,
        **cfg.train,
    )
    trainer.save_config(cfg)
    accelerator.wait_for_everyone()

    # train
    trainer.train()

    # save final model
    trainer.save_model()


if __name__ == "__main__":
    load_dotenv()
    sys.exit(main())
