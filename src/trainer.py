import copy
import inspect
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.data_loader import prepare_data_loader
from accelerate.utils import broadcast
from accelerate.scheduler import AcceleratedScheduler
from omegaconf import ListConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.data.dataset import CaloShowerDataset, DummyDataset, preprocess_geo
from src.data.preprocessing import CaloShowerPreprocessor, cut_below_noise_level
from src.data.utils import save_showers
from src.evaluation.utils import compare_observables
from src.flow.reject_redraw import apply_reject_and_redraw, filter_model_sample_kwargs
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
    train_stage: int = 2

    def state_dict(self):
        return {
            "step": self.step,
            "epoch": self.epoch,
            "elapsed_time": self.elapsed_time,
            "best_valid_loss": self.best_valid_loss,
            "train_stage": self.train_stage,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.step = state_dict["step"]
        self.elapsed_time = state_dict["elapsed_time"]
        self.best_valid_loss = state_dict["best_valid_loss"]
        self.train_stage = state_dict.get("train_stage", 2)

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
        log_condition_diagnostics=False,
        condition_diagnostics_steps=0,
        condition_diagnostics_every=1,
        save_strategy="epoch",
        save_steps=1,
        save_steps_schedule=None,
        save_best_and_last_only=False,
        load_best_model_at_end=False,
        resume_from_checkpoint=None,
        test_num_showers=None,
        test_output_subdir=None,
        enable_plots=True,
        enable_fpd=False,
        save_test_model_artifact=False,
        save_generated=False,
        fpd_config=None,
        freeze_then_unfreeze=None,
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
            self.max_steps = epochs * self.batches_per_epoch

        # model
        self.model = model
        self.model = self.accelerator.prepare(model)
        self.model.train()

        self.base_learning_rate = learning_rate
        self.optimizer_args = dict(optimizer_args or {})
        self.lr_scheduler_args = dict(lr_scheduler_args or {})
        self.optimizer_class = import_class_by_name(optimizer_class) if isinstance(optimizer_class, str) else optimizer_class
        self.lr_scheduler_class = (
            import_class_by_name(lr_scheduler_class) if isinstance(lr_scheduler_class, str) else lr_scheduler_class
        )
        self.freeze_then_unfreeze = self._normalize_freeze_schedule(freeze_then_unfreeze)
        self._freeze_resume_state = self._peek_resume_training_state(
            Path(resume_from_checkpoint) if resume_from_checkpoint else None
        )
        self.state.train_stage = self._infer_initial_train_stage()
        self._apply_training_stage(self.state.train_stage, initial=True)

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
        self.log_condition_diagnostics = log_condition_diagnostics
        self.condition_diagnostics_steps = condition_diagnostics_steps
        self.condition_diagnostics_every = condition_diagnostics_every

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
        self.test_num_showers = test_num_showers
        self.test_output_subdir = test_output_subdir
        self.test_conditions = test_conditions
        self.enable_plots = enable_plots
        self.enable_fpd = enable_fpd
        self.save_test_model_artifact = save_test_model_artifact
        self.save_generated = save_generated
        self.fpd_config = dict(fpd_config or {})

        self.plot_dir = self.output_dir / "plots"
        self.plot_dir.mkdir(exist_ok=True, parents=True)

        self.save_strategy, self.save_steps = parse_strategy(save_strategy, save_steps)
        self.save_steps_schedule = self._normalize_step_schedule(
            schedule=save_steps_schedule,
            strategy=self.save_strategy,
            default_interval=self.save_steps,
            interval_key="save_steps",
            schedule_name="save_steps_schedule",
        )
        self.save_best_and_last_only = save_best_and_last_only

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # load
        self.load_best_model_at_end = load_best_model_at_end
        self.resume_from_checkpoint = Path(resume_from_checkpoint) if resume_from_checkpoint else None
        if self.resume_from_checkpoint:
            self.load_state(self.resume_from_checkpoint)
            self._maybe_advance_training_stage()

        unwrap_ddp(self.model).record_condition_diagnostics = self.log_condition_diagnostics

    @property
    def device(self):
        return self.accelerator.device

    @staticmethod
    def _normalize_step_schedule(schedule, strategy, default_interval, interval_key, schedule_name):
        if schedule is None:
            return []
        if strategy != "steps":
            raise ValueError(f"{schedule_name} requires `save_strategy: steps`, got {strategy!r}.")

        normalized = []
        for raw_entry in schedule:
            if raw_entry is None:
                continue

            start_step = raw_entry.get("start_step")
            interval = raw_entry.get(interval_key)
            if start_step is None or interval is None:
                raise ValueError(
                    f"Each entry in {schedule_name} must define `start_step` and `{interval_key}`."
                )

            start_step = int(start_step)
            interval = int(interval)
            if start_step < 0:
                raise ValueError(f"{schedule_name} start_step must be >= 0, got {start_step}.")
            if interval <= 0:
                raise ValueError(f"{schedule_name} {interval_key} must be > 0, got {interval}.")

            normalized.append((start_step, interval))

        normalized.sort(key=lambda item: item[0])
        if normalized and normalized[0][0] == 0:
            default_interval = normalized[0][1]

        deduped = []
        last_start_step = None
        for start_step, interval in normalized:
            if start_step == last_start_step:
                deduped[-1] = (start_step, interval)
            else:
                deduped.append((start_step, interval))
                last_start_step = start_step

        if deduped and deduped[0][0] > 0:
            deduped.insert(0, (0, int(default_interval)))

        return deduped

    @staticmethod
    def _resolve_step_interval(step, default_interval, schedule):
        interval = int(default_interval)
        for start_step, scheduled_interval in schedule:
            if step >= start_step:
                interval = scheduled_interval
            else:
                break
        return interval

    def _should_save_checkpoint(self):
        save_interval = self._resolve_step_interval(
            step=self.state.step,
            default_interval=self.save_steps,
            schedule=self.save_steps_schedule,
        )
        return self.state.step % save_interval == 0

    def _build_sampling_conditions(self, incident_energy, phi, theta, geometry):
        energy = torch.as_tensor(
            np.asarray(incident_energy).reshape(-1),
            device=self.accelerator.device,
            dtype=torch.float32,
        )
        phi_tensor = torch.full_like(energy, float(phi))
        theta_tensor = torch.full_like(energy, float(theta))
        conditions = (energy, phi_tensor, theta_tensor)
        if self.need_geo_condn:
            if self.train_on is None:
                raise ValueError("`train_on` must be set when `need_geo_condn=True`.")
            geo = preprocess_geo(len(energy), geometry, self.train_on)
            conditions = conditions + (
                torch.as_tensor(geo, device=self.accelerator.device, dtype=torch.float32),
            )
        return conditions

    def _sample_replacements(self, model, incident_energy, phi, theta, geometry):
        raw_conditions = self._build_sampling_conditions(incident_energy, phi, theta, geometry)
        _, transformed_conditions = self.preprocessor.transform(conditions=raw_conditions)
        generated_events = unwrap_ddp(model).sample(
            conditions=transformed_conditions,
            progress=False,
            **filter_model_sample_kwargs(self.sampling_args),
        ).squeeze(1)
        generated_events, _ = self.preprocessor.inverse_transform(generated_events, transformed_conditions)
        return generated_events.cpu().numpy()

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

    def _normalize_freeze_schedule(self, freeze_then_unfreeze):
        if not freeze_then_unfreeze or not freeze_then_unfreeze.get("enabled", False):
            return None

        trainable_prefixes = list(freeze_then_unfreeze.get("trainable_prefixes", []))
        stage1_learning_rate = freeze_then_unfreeze.get("stage1_learning_rate", self.base_learning_rate)
        stage1_optimizer_args = dict(freeze_then_unfreeze.get("stage1_optimizer_args", self.optimizer_args))
        stage1_lr_scheduler_args = freeze_then_unfreeze.get("stage1_lr_scheduler_args", {})
        stage1_optimizer_class = freeze_then_unfreeze.get("stage1_optimizer_class")
        stage1_lr_scheduler_class = freeze_then_unfreeze.get("stage1_lr_scheduler_class")

        if isinstance(stage1_optimizer_class, str):
            stage1_optimizer_class = import_class_by_name(stage1_optimizer_class)
        if isinstance(stage1_lr_scheduler_class, str):
            stage1_lr_scheduler_class = import_class_by_name(stage1_lr_scheduler_class)

        return {
            "unfreeze_at_step": int(freeze_then_unfreeze.get("unfreeze_at_step", 0)),
            "trainable_prefixes": trainable_prefixes,
            "stage1_learning_rate": stage1_learning_rate,
            "stage1_optimizer_class": stage1_optimizer_class or self.optimizer_class,
            "stage1_optimizer_args": stage1_optimizer_args,
            "stage1_lr_scheduler_class": stage1_lr_scheduler_class,
            "stage1_lr_scheduler_args": dict(stage1_lr_scheduler_args),
        }

    def _peek_resume_training_state(self, checkpoint_dir: Optional[Path]):
        if checkpoint_dir is None:
            return None
        state_path = checkpoint_dir / "state.yaml"
        if not state_path.exists():
            return None
        return OmegaConf.to_container(OmegaConf.load(state_path), resolve=True)

    def _infer_initial_train_stage(self):
        if self.freeze_then_unfreeze is None:
            return 2
        if self._freeze_resume_state is not None:
            if "train_stage" in self._freeze_resume_state:
                return int(self._freeze_resume_state["train_stage"])
            if int(self._freeze_resume_state.get("step", 0)) >= self.freeze_then_unfreeze["unfreeze_at_step"]:
                return 2
        return 1

    def _set_trainable_by_prefixes(self, trainable_prefixes):
        trainable_prefixes = tuple(trainable_prefixes or [])
        total_params = 0
        trainable_params = 0
        raw_model = unwrap_ddp(self.model)
        for name, param in raw_model.named_parameters():
            is_trainable = True if not trainable_prefixes else any(name.startswith(prefix) for prefix in trainable_prefixes)
            param.requires_grad_(is_trainable)
            total_params += param.numel()
            if is_trainable:
                trainable_params += param.numel()
        return trainable_params, total_params

    def _build_optimizer_and_scheduler(self, learning_rate, optimizer_class, optimizer_args, lr_scheduler_class, lr_scheduler_args):
        optimizer_init_params = inspect.signature(optimizer_class.__init__).parameters
        if "model" in optimizer_init_params:
            optimizer = optimizer_class(self.model, lr=learning_rate, **optimizer_args)
        else:
            optimizer = optimizer_class(self.model.parameters(), lr=learning_rate, **optimizer_args)
        optimizer = self.accelerator.prepare(optimizer)

        lr_scheduler = None
        if lr_scheduler_class is not None:
            lr_scheduler = lr_scheduler_class(optimizer, **lr_scheduler_args)
            lr_scheduler = AcceleratedScheduler(lr_scheduler, optimizer, step_with_optimizer=False)

        return optimizer, lr_scheduler

    def _apply_training_stage(self, stage, initial=False):
        if stage == 1 and self.freeze_then_unfreeze is not None:
            trainable_params, total_params = self._set_trainable_by_prefixes(self.freeze_then_unfreeze["trainable_prefixes"])
            learning_rate = self.freeze_then_unfreeze["stage1_learning_rate"]
            optimizer_class = self.freeze_then_unfreeze["stage1_optimizer_class"]
            optimizer_args = self.freeze_then_unfreeze["stage1_optimizer_args"]
            lr_scheduler_class = self.freeze_then_unfreeze["stage1_lr_scheduler_class"]
            lr_scheduler_args = self.freeze_then_unfreeze["stage1_lr_scheduler_args"]
            stage_label = "stage1_frozen_backbone"
        else:
            trainable_params, total_params = self._set_trainable_by_prefixes([])
            learning_rate = self.base_learning_rate
            optimizer_class = self.optimizer_class
            optimizer_args = self.optimizer_args
            lr_scheduler_class = self.lr_scheduler_class
            lr_scheduler_args = self.lr_scheduler_args
            stage_label = "stage2_full_finetune"

        self.optimizer, self.lr_scheduler = self._build_optimizer_and_scheduler(
            learning_rate=learning_rate,
            optimizer_class=optimizer_class,
            optimizer_args=optimizer_args,
            lr_scheduler_class=lr_scheduler_class,
            lr_scheduler_args=lr_scheduler_args,
        )
        self.state.train_stage = stage

        logger.info(
            "Configured %s at step %s with lr=%s; trainable params=%s/%s",
            stage_label,
            self.state.step,
            learning_rate,
            trainable_params,
            total_params,
        )
        if not initial and self.accelerator.is_main_process and hasattr(self, "writer"):
            self.writer.add_scalar("Train/Stage", stage, global_step=self.state.step)
            self.writer.add_scalar("Train/Trainable Parameters", trainable_params, global_step=self.state.step)

    def _maybe_advance_training_stage(self):
        if self.freeze_then_unfreeze is None:
            return
        should_unfreeze = self.state.step >= self.freeze_then_unfreeze["unfreeze_at_step"]
        if should_unfreeze and self.state.train_stage == 1:
            self._apply_training_stage(2)

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
            self._log_condition_diagnostics()
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

    def _should_log_condition_diagnostics(self):
        if not self.log_condition_diagnostics:
            return False
        if self.condition_diagnostics_steps > 0 and self.state.step >= self.condition_diagnostics_steps:
            return False
        return self.state.step % max(self.condition_diagnostics_every, 1) == 0


    def _log_condition_diagnostics(self):
        if not self._should_log_condition_diagnostics():
            return

        raw_model = unwrap_ddp(self.model)
        diagnostics = {}
        if hasattr(raw_model, "get_condition_diagnostics"):
            last_diagnostics = raw_model.get_condition_diagnostics()
            if last_diagnostics is not None:
                diagnostics.update(last_diagnostics)
        if hasattr(raw_model, "get_condition_gradient_diagnostics"):
            diagnostics.update(raw_model.get_condition_gradient_diagnostics())
        if not diagnostics:
            return

        reduced = {}
        for key, value in diagnostics.items():
            if not torch.is_tensor(value):
                value = torch.tensor(value, device=self.device, dtype=torch.float32)
            value = value.detach().float().reshape(1)
            reduced[key] = self.accelerator.gather(value).mean().item()

        if self.accelerator.is_main_process:
            for key, value in reduced.items():
                self.writer.add_scalar(f"Diag/{key}", value, global_step=self.state.step)
            diag_str = " ".join(f"{key}={value:.6f}" for key, value in sorted(reduced.items()))
            logger.info(f"Condition diagnostics at step {self.state.step}: {diag_str}")

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
                self._maybe_advance_training_stage()
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

                if self._should_save_checkpoint() and self.accelerator.is_main_process:
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
        step_output_dir = self.plot_dir / f"{self.state.step}"
        if exists(self.test_output_subdir):
            step_output_dir = step_output_dir / self.test_output_subdir
        model_artifact_path = None
        if self.save_test_model_artifact and self.accelerator.is_main_process:
            step_output_dir.mkdir(parents=True, exist_ok=True)
            model_filename = "ema_model_with_config.pt" if self.use_ema else "model_with_config.pt"
            model_artifact_path = step_output_dir / model_filename
            self.accelerator.unwrap_model(model).save_state(model_artifact_path)

        # Test-time sample count is shared across all test conditions.
        test_num_showers = self.test_num_showers
        if test_num_showers is not None:
            test_num_showers = int(test_num_showers)
            
        save_generated = bool(self.save_generated or self.fpd_config.get("save_generated", False))
        compute_kpd = bool(self.fpd_config.get("compute_kpd", False))
        # Split test-control flags from the kwargs that the metric function
        # actually accepts. Otherwise keys like `num_showers`/`save_generated`
        # or a duplicate `compute_kpd` can crash post-processing.
        metric_fpd_config = {
            key: value
            for key, value in self.fpd_config.items()
            if key in {"particle", "xml_filename", "cut", "min_samples", "batch_size"}
        }

        for geometry, energy, phi, theta, fullsim_path in self.test_conditions:
            conditions_str = get_conditions_str(geometry, energy, phi, theta)
            output_dir = step_output_dir / conditions_str
            postprocess_done_path = output_dir / ".postprocess_done"
            postprocess_failed_path = output_dir / ".postprocess_failed"
            needs_main_postprocess = self.enable_plots or save_generated or self.enable_fpd

            if needs_main_postprocess and self.accelerator.is_main_process:
                output_dir.mkdir(parents=True, exist_ok=True)
                postprocess_done_path.unlink(missing_ok=True)
                postprocess_failed_path.unlink(missing_ok=True)
            if needs_main_postprocess:
                # Keep collectives short: all ranks start post-processing from the same state,
                # then non-main ranks wait on filesystem markers instead of NCCL.
                self.accelerator.wait_for_everyone()

            if isinstance(fullsim_path, (list, tuple, ListConfig)):
                fullsim_files = list(fullsim_path)
            else:
                fullsim_files = [fullsim_path]

            if self.need_geo_condn:
                file_struc = [[geometry, path] for path in fullsim_files]
            else:
                file_struc = fullsim_files

            is_ccd = geometry.startswith("CCD")
            if test_num_showers is not None:
                max_num_showers = test_num_showers
            elif is_ccd:
                max_num_showers = 1000
            else:
                max_num_showers = None

            if self.accelerator.is_main_process:
                dataset = CaloShowerDataset(
                    files=file_struc,
                    need_geo_condn=self.need_geo_condn,
                    train_on=self.train_on,
                    is_ccd=is_ccd,
                    max_num_showers=max_num_showers,
                )
            else:
                dataset = DummyDataset()
            self.accelerator.wait_for_everyone()
            dataloader = self._get_dataloader(dataset, self.batch_size, shuffle=False)

            num_samples = len(dataset)
            num_batches = len(dataloader)
            logger.info(
                f"Generating {num_samples} events for geometry {geometry}, energy {energy} GeV, phi {phi} and theta {theta}"
            )
            logger.info(
                f"Sampling {num_samples} events in {num_batches} batches "
                f"(per-device batch size {self.batch_size}, world size {self.accelerator.num_processes})."
            )

            generated_events_list = []
            need_original_events = self.enable_plots or self.enable_fpd
            orginal_events_list = []
            incident_energy_list = []
            for sample in tqdm(
                dataloader,
                desc="Sampling",
                disable=not self.accelerator.is_main_process,
            ):
                showers, conditions = sample
                incident_energy, *_ = conditions
                _, conditions = self.preprocessor.transform(conditions=conditions)
                showers = cut_below_noise_level(showers, noise_level=self.preprocessor.shower_preprocessor.noise_level)
                generated_events = unwrap_ddp(model).sample(
                    conditions=conditions,
                    progress=False,
                    **filter_model_sample_kwargs(self.sampling_args),
                ).squeeze(1)
                generated_events, _ = self.preprocessor.inverse_transform(generated_events, conditions)
                if need_original_events:
                    gathered_original_events, gathered_generated_events, gathered_incident_energy = self.accelerator.gather_for_metrics(
                        (
                            showers.squeeze(1).contiguous(),
                            generated_events.contiguous(),
                            incident_energy.contiguous(),
                        )
                    )
                else:
                    gathered_generated_events, gathered_incident_energy = self.accelerator.gather_for_metrics(
                        (
                            generated_events.contiguous(),
                            incident_energy.contiguous(),
                        )
                    )
                if self.accelerator.is_main_process:
                    if need_original_events:
                        orginal_events_list.append(gathered_original_events.cpu().numpy())
                    generated_events_list.append(gathered_generated_events.cpu().numpy())
                    incident_energy_list.append(gathered_incident_energy.cpu().numpy().reshape(-1, 1))

            original_events = None
            generated_events = None
            incident_energy = None
            if self.accelerator.is_main_process:
                if need_original_events:
                    original_events = np.concatenate(orginal_events_list)
                generated_events = np.concatenate(generated_events_list)
                incident_energy = np.concatenate(incident_energy_list)
                (
                    generated_events,
                    original_events,
                    incident_energy,
                    redraw_summary,
                ) = apply_reject_and_redraw(
                    generated_events,
                    incident_energy,
                    geometry=geometry,
                    sampling_args=self.sampling_args,
                    original_events=original_events,
                    sample_fn=lambda batch_incident_energy: self._sample_replacements(
                        model,
                        batch_incident_energy,
                        phi,
                        theta,
                        geometry,
                    ),
                )
            else:
                redraw_summary = None

            if needs_main_postprocess:
                if self.accelerator.is_main_process:
                    try:
                        if redraw_summary is not None:
                            OmegaConf.save(
                                OmegaConf.create(redraw_summary),
                                output_dir / "reject_redraw_summary.yaml",
                            )

                        if self.enable_plots:           # still 1000 for keep with calodit2
                            plot_original_events = original_events
                            plot_generated_events = generated_events
                            if is_ccd and len(original_events) > 1000:
                                plot_original_events = original_events[:1000]
                                plot_generated_events = generated_events[:1000]
                            observables = compare_observables(
                                plot_original_events, plot_generated_events, output_dir, geometry, energy, phi, theta
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

                        if save_generated:
                            generated_output_path = output_dir / "generated.h5"
                            if generated_output_path.exists():
                                logger.warning(
                                    "File %s already exists, overwriting",
                                    generated_output_path,
                                )
                                generated_output_path.unlink()
                            save_showers(
                                generated_events,
                                incident_energy,
                                phi,
                                theta,
                                generated_output_path,
                                is_ccd=is_ccd,
                            )
                            logger.info("Saved generated events to %s", generated_output_path)

                        if self.enable_fpd:
                            from src.evaluation.fpd_kpd import (
                                compute_fpd_kpd,
                                prepare_fpd_inputs,
                            )

                            gen_showers_mev, gen_energy_mev = prepare_fpd_inputs(
                                generated_events, incident_energy, geometry=geometry
                            )
                            ref_showers_mev, ref_energy_mev = prepare_fpd_inputs(
                                original_events, incident_energy, geometry=geometry
                            )

                            fpd_results = compute_fpd_kpd(
                                generated_showers=gen_showers_mev,
                                reference_showers=ref_showers_mev,
                                generated_energy=gen_energy_mev,
                                reference_energy=ref_energy_mev,
                                geometry=geometry,
                                output_dir=output_dir,
                                dataset_name=conditions_str,
                                compute_kpd=compute_kpd,
                                **metric_fpd_config,
                            )

                            # Log whichever metrics were requested.
                            for key in fpd_results:
                                self.writer.add_scalar(
                                    f"FPD_KPD/{conditions_str.replace('_', ' ')}/{key}",
                                    fpd_results[key],
                                    global_step=self.state.step,
                                )
                            if self.use_wandb:
                                wandb.log({
                                    f"FPD_KPD/{conditions_str}/{k}": v
                                    for k, v in fpd_results.items()
                                })

                            score_payload = {
                                "step": self.state.step,
                                "geometry": geometry,
                                "energy": energy,
                                "phi": phi,
                                "theta": theta,
                                "model_path": str(model_artifact_path) if model_artifact_path is not None else None,
                                **{
                                    key: value.item() if isinstance(value, np.generic) else value
                                    for key, value in fpd_results.items()
                                },
                            }
                            OmegaConf.save(score_payload, output_dir / "fpd_kpd.yaml")
                    except Exception as exc:
                        postprocess_failed_path.write_text(f"{type(exc).__name__}: {exc}\n")
                        raise
                    else:
                        postprocess_done_path.write_text("done\n")
                else:
                    while True:
                        if postprocess_failed_path.exists():
                            raise RuntimeError(
                                f"Main-process test post-processing failed for {conditions_str}: "
                                f"{postprocess_failed_path.read_text().strip()}"
                            )
                        if postprocess_done_path.exists():
                            break
                        time.sleep(5)

            self.accelerator.wait_for_everyone()

    def _get_checkpointable_lr_scheduler(self):
        if self.lr_scheduler is None:
            return None
        return getattr(self.lr_scheduler, "scheduler", self.lr_scheduler)

    def save_state(self, milestone="last"):
        self.state.elapsed_time = self.timer.lap()

        checkpoint_dir = (
            self.checkpoint_dir / f"checkpoint_{milestone if isinstance(milestone, str) else f'{milestone:08}'}"
        )
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.accelerator.save_state(checkpoint_dir, safe_serialization=False)
        checkpointable_lr_scheduler = self._get_checkpointable_lr_scheduler()
        if checkpointable_lr_scheduler is not None:
            torch.save(checkpointable_lr_scheduler.state_dict(), checkpoint_dir / "scheduler.bin")
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
        checkpointable_lr_scheduler = self._get_checkpointable_lr_scheduler()
        scheduler_path = checkpoint_dir / "scheduler.bin"
        if checkpointable_lr_scheduler is not None:
            if scheduler_path.exists():
                checkpointable_lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
            else:
                logger.warning(
                    "No lr scheduler state found in %s. This checkpoint predates scheduler checkpointing, "
                    "so resuming may use an incorrect learning-rate schedule.",
                    checkpoint_dir,
                )
        self.config = OmegaConf.load(checkpoint_dir / "config.yaml")

    def save_model(self, save_path: Union[str, Path] = None):
        self.accelerator.wait_for_everyone()
        if save_path is None:
            save_path = self.output_dir / "final_model.pt"
        model = self.ema_model if self.use_ema else self.model
        model = self.accelerator.unwrap_model(model)
        model.save_state(save_path)
