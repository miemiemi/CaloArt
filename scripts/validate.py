"""Generate showers under the same conditions and compare with the full simulation."""
import sys
from pathlib import Path

import numpy as np
import hydra
import rootutils
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

rootutils.setup_root(__file__, pythonpath=True)

from src.data.dataset import CaloShowerDataset
from src.data.preprocessing import CaloShowerPreprocessor, cut_below_noise_level
from src.data.utils import save_showers
from src.evaluation.utils import compare_observables
from src.flow.reject_redraw import apply_reject_and_redraw, filter_model_sample_kwargs
from src.models.factory import create_model_from_config
from src.utils import get_conditions_str, get_logger, setup_accelerator, to_device

sys.argv = [arg for arg in sys.argv if not arg.startswith("--local_rank")]
sys.argv = [arg for arg in sys.argv if not arg.startswith("--local-rank")]


@hydra.main(version_base="1.3", config_path="../configs", config_name="experiment/CaloChallenge/validate/default")
def main(cfg: DictConfig):
    # setup accelerator and logger
    accelerator = setup_accelerator(**cfg.accelerator)

    logger = get_logger()

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # load model
    model = create_model_from_config(cfg.model, cfg.method)
    model = accelerator.prepare(model)

    preprocessor = CaloShowerPreprocessor(**model.config.preprocessing)

    batch_size = int(cfg.validate.batch_size)
    for geometry, energy, phi, theta, fullsim_path in cfg.validate.simulation_conditions:
        conditions_str = get_conditions_str(geometry, energy, phi, theta)
        if cfg.validate.is_ccd or cfg.validate.is_fpd:
            output_dir = Path(cfg.validate.output_dir)
        else:
            output_dir = Path(cfg.validate.output_dir) / conditions_str
        output_dir.mkdir(parents=True, exist_ok=True)

        if cfg.validate.need_geo_condn:
            file_struc = [[geometry, fullsim_path],]
        else:
            file_struc = [fullsim_path,]
        dataset = CaloShowerDataset(files=file_struc, need_geo_condn=cfg.validate.need_geo_condn, train_on=cfg.validate.train_on, is_ccd=cfg.validate.is_ccd)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        dataloader = accelerator.prepare(dataloader)

        num_samples = len(dataset)

        logger.info(
            f"Generating {num_samples} events for geometry {geometry}, energy {energy} GeV, phi {phi} and theta {theta}"
        )

        generated_events_list = []
        orginal_events_list = []
        energy_list = []
        for sample in tqdm(dataloader):
            showers, conditions = sample
            _, conditions = preprocessor.transform(conditions=conditions)
            energy_list.append(conditions[0].cpu().numpy())

            showers = cut_below_noise_level(showers, noise_level=preprocessor.shower_preprocessor.noise_level)
            orginal_events_list.append(showers.squeeze(1).cpu().numpy())

            generated_events = model.sample(
                conditions=conditions,
                progress=True,
                **filter_model_sample_kwargs(cfg.sampling),
            ).squeeze(1)
            generated_events, _ = preprocessor.inverse_transform(generated_events, conditions)
            generated_events_list.append(generated_events.cpu().numpy())

        original_events = np.concatenate(orginal_events_list)
        generated_events = np.concatenate(generated_events_list)
        if cfg.validate.is_ccd or cfg.validate.is_fpd:
            energies = np.concatenate(energy_list) * 1000.0
        else:
            energies = energy

        generated_events, original_events, energies, redraw_summary = apply_reject_and_redraw(
            generated_events,
            energies,
            geometry=geometry,
            sampling_args=cfg.sampling,
            original_events=original_events,
            sample_fn=lambda batch_incident_energy: _sample_replacements(
                model,
                preprocessor,
                batch_incident_energy,
                phi,
                theta,
                cfg.sampling,
            ),
        )

        output_path = (
            output_dir / f"generated.h5"
        )

        if output_path.exists():
            logger.warning(f"File {output_path} already exists, overwriting")
            output_path.unlink()

        save_showers(generated_events, energies, phi, theta, output_path, is_ccd=cfg.validate.is_ccd)
        logger.info(f"Saved generated events to {output_path}")

        if redraw_summary is not None:
            OmegaConf.save(
                OmegaConf.create(redraw_summary),
                output_dir / "reject_redraw_summary.yaml",
            )

        if generated_events.shape[0] > 1000:
            generated_events = generated_events[:1000]
            original_events = original_events[:1000]
            logger.info(f"Comparing observables for 1000 events")
        compare_observables(original_events, generated_events, output_dir, geometry, energy, phi, theta)

    logger.info("Validation completed.")

def _sample_replacements(model, preprocessor, incident_energy, phi, theta, sampling_args):
    if np.isscalar(incident_energy):
        incident_energy = np.asarray([[incident_energy]], dtype=np.float32)
    elif incident_energy.ndim == 1:
        incident_energy = incident_energy.reshape(-1, 1)

    cond_e = torch.as_tensor(incident_energy, dtype=torch.float32)
    cond_phi = torch.full_like(cond_e, phi)
    cond_theta = torch.full_like(cond_e, theta)
    _, conditions = preprocessor.transform(conditions=(cond_e, cond_phi, cond_theta))
    conditions = to_device(conditions, next(model.parameters()).device)
    generated_events = model.sample(
        conditions=conditions,
        progress=True,
        **filter_model_sample_kwargs(sampling_args),
    ).squeeze(1)
    generated_events, _ = preprocessor.inverse_transform(generated_events, conditions)
    return generated_events.detach().cpu().numpy()


if __name__ == "__main__":
    load_dotenv()
    sys.exit(main())
