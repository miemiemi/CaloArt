import copy
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.data_loader import prepare_data_loader
from accelerate.utils import broadcast
from accelerate.scheduler import AcceleratedScheduler
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.data.dataset import CaloShowerDataset
from src.data.preprocessing import CaloShowerPreprocessor, cut_below_noise_level
from src.evaluation.utils import compare_observables
from src.method_base import MethodBase
from src.models.ema import ema_update
from src.utils import (
    cycle,
    exists,
    flatten_dict,
    get_conditions_str,
    get_logger,
    get_lrs,
    import_class_by_name,
    unwrap_ddp,
)

logger = get_logger()


@dataclass
class TrainingState:
    epoch: int = 0
    step: int = 0
    elapsed_time: float = 0.0
    best_valid_loss: float = float("inf")

    def state_dict(self):
        return {
            "step": self.step,
            "epoch": self.epoch,
            "elapsed_time": self.elapsed_time,
            "best_valid_loss": self.best_valid_loss,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.step = state_dict["step"]
        self.elapsed_time = state_dict["elapsed_time"]
        self.best_valid_loss = state_dict["best_valid_loss"]

    def save_state_dict(self, path):
        OmegaConf.save(self.state_dict(), path)


class Timer:
    def __init__(self, device):
        self.device = device

    def start(self):
        if self.device.type == "cuda":
            self.start_timer = torch.cuda.Event(enable_timing=True)
            self.end_timer = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            self.start_timer.record()
        else:
            self.start_timer = time.time()

    def lap(self):
        if self.device.type == "cuda":
            self.end_timer.record()
            torch.cuda.synchronize()
            elapsed = self.start_timer.elapsed_time(self.end_timer) / 1000
        else:
            elapsed = time.time() - self.start_timer

        return elapsed


class DiffusionTrainer(object):
    def __init__(
        self,
        model: MethodBase,
        output_dir: Path,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        preprocessor: CaloShowerPreprocessor,
        accelerator: Accelerator,
        epochs=None,
        max_steps=None,
        global_batch_size=1024,
        per_device_batch_size=-1,
        learning_rate=1e-4,
        optimizer_class=torch.optim.Adam,
        optimizer_args=dict(),
        lr_scheduler_class=None,
        lr_scheduler_args=dict(),
        max_grad_value=None,
        max_grad_norm=None,
        gradient_accumulation_steps=1,
        num_workers=0,
        valid_num_workers=None,
        ema_scheduler_class=None,
        ema_scheduler_args=dict(),
        sampling_args=dict(),
        use_wandb=False,
        project_name=None,
        run_name=None,
        valid_strategy="epoch",
        valid_steps=1,
        test_strategy="epoch",
        test_steps=1,
        test_conditions=[],
        need_geo_condn=False,
        train_on=[],
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",
        save_steps=1,
        save_best_and_last_only=False,
        load_best_model_at_end=False,
        resume_from_checkpoint=None,
        **kwargs,
    ):
        super().__init__()
        self.accelerator = accelerator

        # data
        assert (
            global_batch_size > 0 or per_device_batch_size > 0
        ), "Either global_batch_size or per_device_batch_size must be provided"
        if per_device_batch_size > 0:
            batch_size = per_device_batch_size
        else:
            batch_size = global_batch_size // self.accelerator.num_processes

        self.batch_size = batch_size
        logger.info(f"Using the total batch size of {self.batch_size * gradient_accumulation_steps * self.accelerator.num_processes}")

        if valid_num_workers is None:
            valid_num_workers = num_workers

        self.train_dataloader = self._get_dataloader(train_dataset, batch_size, num_workers=num_workers)

        batches_per_epoch = torch.tensor(len(self.train_dataloader), device=self.accelerator.device)
        batches_per_epoch = broadcast(batches_per_epoch)
        self.batches_per_epoch = batches_per_epoch.item()

        self.train_dataloader = cycle(self.train_dataloader)
        self.valid_dataloader = self._get_dataloader(valid_dataset, batch_size, shuffle=False, num_workers=valid_num_workers)
        self.need_geo_condn = need_geo_condn
        self.train_on = train_on
        self.preprocessor = preprocessor

        # progress
        self.state = TrainingState()
        self.accelerator.register_for_checkpointing(self.state)
        self.timer = Timer(self.accelerator.device)

        assert epochs is not None or max_steps is not None, "Either epochs or max_steps must be provided"
        assert epochs is None or max_steps is None, "Only one of epochs or max_steps can be provided"
        if max_steps is not None:
            self.epochs = math.ceil(max_steps / self.batches_per_epoch)
            self.max_steps = max_steps
        else:
            self.epochs = epochs
            self.max_steps = epochs * self.batches_per_epoch - 1

        # model
        self.model = model
        self.model = self.accelerator.prepare(model)
        self.model.train()

        # optimization
        optimizer_class_str = None
        if isinstance(optimizer_class, str):
            optimizer_class_str = optimizer_class
            optimizer_class = import_class_by_name(optimizer_class)

        if optimizer_class_str == 'src.optimizers.BiasedAdamW':
            self.optimizer = optimizer_class(self.model, lr=learning_rate, **optimizer_args)
        else:
            self.optimizer = optimizer_class(self.model.parameters(), lr=learning_rate, **optimizer_args)
        self.optimizer = self.accelerator.prepare(self.optimizer)

        self.lr_scheduler = None
        if lr_scheduler_class is not None:
            if isinstance(lr_scheduler_class, str):
                lr_scheduler_class = import_class_by_name(lr_scheduler_class)
            self.lr_scheduler = lr_scheduler_class(self.optimizer, **lr_scheduler_args)
            self.lr_scheduler = AcceleratedScheduler(self.lr_scheduler, self.optimizer, step_with_optimizer=False)

        self.max_grad_value = max_grad_value
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # EMA
        self.ema_scheduler = None
        if ema_scheduler_class is not None:
            if isinstance(ema_scheduler_class, str):
                ema_scheduler_class = import_class_by_name(ema_scheduler_class)
            self.ema_scheduler = ema_scheduler_class(**ema_scheduler_args)
            self.accelerator.register_for_checkpointing(self.ema_scheduler)

            self.ema_model = copy.deepcopy(self.accelerator.unwrap_model(model))
            self.ema_model = self.accelerator.prepare(self.ema_model)
            self.ema_model.eval()
            self.ema_model.requires_grad_(False)

        self.use_ema = self.ema_scheduler is not None

        self.sampling_args = sampling_args

        # files and logging
        self.output_dir = Path(output_dir)
        if exists(project_name):
            self.output_dir = self.output_dir / project_name

        if exists(run_name):
            self.output_dir = self.output_dir / run_name

        self.output_dir.mkdir(exist_ok=True, parents=True)

        # logging
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.run_name = run_name
        self.config = dict()

        def parse_strategy(strategy, steps):
            if strategy in ["no", "none", False]:
                return "no", int(1e9)
            assert strategy in ["epoch", "steps"], f"Invalid strategy: {strategy}"
            return strategy, steps if strategy == "steps" else steps * self.batches_per_epoch

        self.logging_strategy, self.logging_steps = parse_strategy(logging_strategy, logging_steps)
        self.valid_strategy, self.valid_steps = parse_strategy(valid_strategy, valid_steps)
        self.test_strategy, self.test_steps = parse_strategy(test_strategy, test_steps)
        self.test_conditions = test_conditions

        self.plot_dir = self.output_dir / "plots"
        self.plot_dir.mkdir(exist_ok=True, parents=True)

        self.save_strategy, self.save_steps = parse_strategy(save_strategy, save_steps)
        self.save_best_and_last_only = save_best_and_last_only

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # load
        self.load_best_model_at_end = load_best_model_at_end
        self.resume_from_checkpoint = Path(resume_from_checkpoint) if resume_from_checkpoint else None
        if self.resume_from_checkpoint:
            self.load_state(self.resume_from_checkpoint)

    @property
    def device(self):
        return self.accelerator.device

    def _setup_logging(self):
        if self.accelerator.is_main_process:
            if self.use_wandb:
                wandb.tensorboard.patch(root_logdir=str(self.output_dir.resolve()), pytorch=True, save=False)
                wandb_entity = os.environ["WANDB_ENTITY"]
                wandb.init(
                    name=self.run_name,
                    project=self.project_name,
                    entity=wandb_entity,
                    dir=str(self.output_dir.resolve()),
                    reinit=True,
                    config=flatten_dict(OmegaConf.to_container(self.config)),
                )
            self.writer = SummaryWriter(str(self.output_dir.resolve()))

    def save_config(self, config: dict):
        self.config = config

    def _get_dataloader(self, dataset, batch_size, shuffle=True, num_workers=0):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        dataloader = prepare_data_loader(dataloader, dispatch_batches=True, put_on_device=True)
        return dataloader

    def loss_fn(self, model, x_0, x_cond, step=None):
        """Loss function. By default, it returns the loss implemented by the model class."""
        return model(x_0, x_cond)

    def _training_step(self):
        self.model.train()

        running_loss = 0.0
        for _ in range(self.gradient_accumulation_steps):
            sample = next(self.train_dataloader)
            shower, conditions = self.preprocessor.transform(*sample)

            loss = self.loss_fn(self.model, shower, conditions, step=self.state.step)
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            running_loss += self.accelerator.gather(loss).mean().item()
            self.accelerator.backward(loss)

        if self.accelerator.sync_gradients:
            if self.max_grad_value is not None:
                self.accelerator.clip_grad_value_(self.model.parameters(), self.max_grad_value)
            if self.max_grad_norm is not None:
                grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if self.accelerator.is_main_process:
                    self.writer.add_scalar("Train/Gradient norm", grad_norm.item(), global_step=self.state.step)

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.state.step += 1

        if self.accelerator.is_main_process:
            self.writer.add_scalar("Train/Loss", running_loss, global_step=self.state.step)

        if self.use_ema:
            ema_decay = self.ema_scheduler.get_decay()
            if self.accelerator.is_main_process:
                self.writer.add_scalar("Train/EMA decay", ema_decay, global_step=self.state.step)
            if self.accelerator.sync_gradients:
                ema_update(self.model, self.ema_model, ema_decay)
                self.ema_scheduler.step()

        return running_loss

    def _anneal_learning_rate(self, valid_loss=None):
        if self.lr_scheduler is not None:
            if self.accelerator.is_main_process:
                for i, lr in enumerate(get_lrs(self.optimizer)):
                    self.writer.add_scalar(f"Train/Group {i} LR", lr, global_step=self.state.step)

            if isinstance(self.lr_scheduler.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.state.step % self.valid_steps == 0:
                    self.lr_scheduler.step(valid_loss)
            else:
                self.lr_scheduler.step()

    def train(self):
        self._setup_logging()

        logger.info("Starting training...")
        self.timer.start()

        train_running_losses = []
        train_epoch_losses = []
        valid_loss = float("inf")
        try:
            while self.state.step < self.max_steps:
                train_loss = self._training_step()
                train_running_losses.append(train_loss)
                train_epoch_losses.append(train_loss)

                if self.state.step % self.logging_steps == 0:
                    train_mean_loss = sum(train_running_losses) / len(train_running_losses)
                    train_running_losses.clear()
                    logger.info(
                        f"Epoch {self.state.epoch}/{self.epochs}, step {self.state.step}/{self.max_steps}, training loss: {train_mean_loss}"
                    )

                if self.state.step % self.batches_per_epoch == 0:
                    train_epoch_loss = sum(train_epoch_losses) / len(train_epoch_losses)
                    train_epoch_losses.clear()
                    if self.accelerator.is_main_process:
                        self.writer.add_scalar("Train/Epoch Loss", train_epoch_loss, global_step=self.state.step)
                        self.writer.add_scalar("Epoch", self.state.epoch, global_step=self.state.step)

                if self.state.step % self.valid_steps == 0:
                    valid_loss = self.validate()

                if self.state.step % self.test_steps == 0:
                    self.test()

                self._anneal_learning_rate(valid_loss)

                if self.state.step % self.save_steps == 0 and self.accelerator.is_main_process:
                    self.save_state("last" if self.save_best_and_last_only else self.state.step)

                self.state.epoch = math.ceil(self.state.step / self.batches_per_epoch)

        except KeyboardInterrupt:
            logger.info("Training interrupted")

        training_time = self.timer.lap()
        logger.info(f"Training completed in {training_time:.2f}s")

        if self.load_best_model_at_end:
            self.load_state(self.checkpoint_dir / "checkpoint_best")

    @torch.inference_mode()
    def validate(self):
        model = self.ema_model if self.use_ema else self.model
        model.eval()

        logger.info("Validating...")

        running_loss = 0.0
        for sample in self.valid_dataloader:
            showers, conditions = self.preprocessor.transform(*sample)
            loss = self.loss_fn(model, showers, conditions, step=self.state.step)
            running_loss += self.accelerator.gather(loss).mean().item()

        mean_loss = running_loss / len(self.valid_dataloader)

        if self.accelerator.is_main_process:
            self.writer.add_scalar("Valid/Loss", mean_loss, global_step=self.state.step)
            logger.info(
                f"Epoch {self.state.epoch}/{self.epochs}, step {self.state.step}/{self.max_steps}, validation loss: {mean_loss}"
            )

        if mean_loss < self.state.best_valid_loss and self.accelerator.is_main_process:
            self.state.best_valid_loss = mean_loss
            self.save_state("best")

        return mean_loss

    @torch.inference_mode()
    def test(self):
        model = self.ema_model if self.use_ema else self.model
        model.eval()

        logger.info("Testing...")

        for geometry, energy, phi, theta, fullsim_path in self.test_conditions:
            conditions_str = get_conditions_str(geometry, energy, phi, theta)
            output_dir = self.plot_dir / f"{self.state.step}/{conditions_str}"

            if self.need_geo_condn:
                file_struc = [[geometry, fullsim_path],]
            else:
                file_struc = [fullsim_path,]

            max_num_showers = None
            is_ccd = False
            if geometry.startswith("CCD"):
                is_ccd = True
                max_num_showers = 1000

            dataset = CaloShowerDataset(files=file_struc, need_geo_condn=self.need_geo_condn, train_on=self.train_on, is_ccd=is_ccd, max_num_showers=max_num_showers)
            dataloader = self._get_dataloader(dataset, self.batch_size, shuffle=False)

            num_samples = len(dataset)
            logger.info(
                f"Generating {num_samples} events for geometry {geometry}, energy {energy} GeV, phi {phi} and theta {theta}"
            )

            generated_events_list = []
            orginal_events_list = []
            for sample in tqdm(dataloader):
                showers, conditions = sample
                _, conditions = self.preprocessor.transform(conditions=conditions)
                showers = cut_below_noise_level(showers, noise_level=self.preprocessor.shower_preprocessor.noise_level)
                orginal_events_list.append(showers.squeeze(1).cpu().numpy())
                generated_events = unwrap_ddp(model).sample(conditions=conditions, progress=self.accelerator.is_main_process, **self.sampling_args).squeeze(1)
                generated_events, _ = self.preprocessor.inverse_transform(generated_events, conditions)
                generated_events_list.append(generated_events.cpu().numpy())

            original_events = np.concatenate(orginal_events_list)
            generated_events = np.concatenate(generated_events_list)

            if self.accelerator.is_main_process:
                observables = compare_observables(
                    original_events, generated_events, output_dir, geometry, energy, phi, theta
                )
                for observable_name, plot_emd_dict in observables.items():
                    self.writer.add_scalar(
                        f"Observables/{conditions_str.replace('_', ' ')}/EMD {observable_name}",
                        plot_emd_dict["emd"],
                        global_step=self.state.step,
                    )
                    if self.use_wandb:
                        wandb.log({
                            f"Observables/{conditions_str.replace('_', ' ')}/{observable_name}": [wandb.Image(str(plot_emd_dict["plot_path"]))]
                        })

            self.accelerator.wait_for_everyone()

    def save_state(self, milestone="last"):
        self.state.elapsed_time = self.timer.lap()

        checkpoint_dir = (
            self.checkpoint_dir / f"checkpoint_{milestone if isinstance(milestone, str) else f'{milestone:08}'}"
        )
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.accelerator.save_state(checkpoint_dir, safe_serialization=False)
        self.state.save_state_dict(checkpoint_dir / "state.yaml")
        OmegaConf.save(self.config, checkpoint_dir / "config.yaml")

        if milestone != "last" and milestone != "best":
            latest_checkpoint = self.checkpoint_dir / "checkpoint_last"
            if latest_checkpoint.exists():
                latest_checkpoint.unlink()

            latest_checkpoint.symlink_to(checkpoint_dir.resolve())

    def load_state(self, checkpoint_dir):
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
        self.accelerator.load_state(checkpoint_dir)
        self.config = OmegaConf.load(checkpoint_dir / "config.yaml")

    def save_model(self, save_path: Union[str, Path] = None):
        self.accelerator.wait_for_everyone()
        if save_path is None:
            save_path = self.output_dir / "final_model.pt"
        model = self.ema_model if self.use_ema else self.model
        model = self.accelerator.unwrap_model(model)
        model.save_state(save_path)
