"""Generate showers for given conditions using a trained generative model."""

import math
import sys
from pathlib import Path

import numpy as np
import hydra
import rootutils
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

rootutils.setup_root(__file__, pythonpath=True)

from src.models.factory import create_model_from_config
from src.data.preprocessing import CaloShowerPreprocessor
from src.data.utils import save_showers
from src.flow.reject_redraw import apply_reject_and_redraw, filter_model_sample_kwargs
from src.utils import get_logger, setup_accelerator, to_device

sys.argv = [arg for arg in sys.argv if not arg.startswith("--local_rank")]
sys.argv = [arg for arg in sys.argv if not arg.startswith("--local-rank")]


@hydra.main(version_base="1.3", config_path="../configs", config_name="experiment/CaloChallenge/generate/default")
def main(cfg: DictConfig):
    # setup accelerator and logger
    accelerator = setup_accelerator(**cfg.accelerator)

    logger = get_logger()

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    output_dir = Path(cfg.generate.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load model
    model = create_model_from_config(cfg.model, cfg.method)
    model = accelerator.prepare(model)
    model.summarize()

    preprocessor = CaloShowerPreprocessor(**model.config.preprocessing)
    num_samples = int(cfg.generate.num_samples)
    batch_size = int(cfg.generate.batch_size)
    for geometry, energy, phi, theta in cfg.generate.simulation_conditions:
        logger.info(
            f"Generating {num_samples} events for geometry {geometry}, energy {energy} GeV, phi {phi} and theta {theta}"
        )

        generated_events_list = []
        for _ in tqdm(range(math.ceil(num_samples / batch_size))):
            cond_e = torch.full((batch_size, 1), energy).float()
            cond_phi = torch.full((batch_size, 1), phi).float()
            cond_theta = torch.full((batch_size, 1), theta).float()
            _, conditions = preprocessor.transform(conditions=(cond_e, cond_phi, cond_theta))
            conditions = to_device(conditions, accelerator.device)

            generated_events = model.sample(
                conditions=conditions,
                progress=True,
                **filter_model_sample_kwargs(cfg.sampling),
            ).squeeze(1)
            generated_events, _ = preprocessor.inverse_transform(generated_events, conditions)
            generated_events_list.append(generated_events.cpu().numpy())

        generated_events = np.concatenate(generated_events_list)
        generated_events = generated_events[:num_samples]
        energies = np.full((generated_events.shape[0], 1), energy, dtype=np.float32)

        generated_events, _, energies, redraw_summary = apply_reject_and_redraw(
            generated_events,
            energies,
            geometry=geometry,
            sampling_args=cfg.sampling,
            sample_fn=lambda batch_incident_energy: _sample_replacements(
                model,
                preprocessor,
                accelerator.device,
                batch_incident_energy,
                phi,
                theta,
                cfg.sampling,
            ),
        )

        output_path = (
            output_dir / f"generated_{num_samples}events_Geo_{geometry}_E_{energy}GeV_Phi_{phi}_Theta_{theta}.h5"
        )

        if output_path.exists():
            logger.warning(f"File {output_path} already exists, overwriting")
            output_path.unlink()

        save_showers(
            generated_events,
            energies,
            phi,
            theta,
            output_path,
            is_ccd=geometry.startswith("CCD"),
        )
        logger.info(f"Saved generated events to {output_path}")

        if redraw_summary is not None:
            summary_path = output_dir / (
                f"reject_redraw_{num_samples}events_Geo_{geometry}_E_{energy}GeV_Phi_{phi}_Theta_{theta}.yaml"
            )
            OmegaConf.save(
                OmegaConf.create(redraw_summary),
                summary_path,
            )
            logger.info(f"Saved reject/redraw summary to {summary_path}")

def _sample_replacements(model, preprocessor, device, incident_energy, phi, theta, sampling_args):
    if np.isscalar(incident_energy):
        incident_energy = np.asarray([[incident_energy]], dtype=np.float32)
    elif incident_energy.ndim == 1:
        incident_energy = incident_energy.reshape(-1, 1)

    cond_e = torch.as_tensor(incident_energy, dtype=torch.float32)
    cond_phi = torch.full_like(cond_e, phi)
    cond_theta = torch.full_like(cond_e, theta)
    _, conditions = preprocessor.transform(conditions=(cond_e, cond_phi, cond_theta))
    conditions = to_device(conditions, device)

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
