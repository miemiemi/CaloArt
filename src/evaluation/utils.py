import logging
from pathlib import Path

import numpy as np

from src.data.geometry import set_geometry
from src.evaluation.observables import AzimuthalProfile, LongitudinalProfile, RadialProfile, Shower
from src.evaluation.plotters import ProfilePlotter, ShowerPlotter
from src.utils import get_logger

logger = get_logger()


def compare_observables(
    full_simulation: np.ndarray,
    ml_simulation: np.ndarray,
    save_dir: Path,
    geometry: str,
    particle_energy: int,
    phi: float,
    theta: float,
    keep_previous: bool = False,
    convert_GeV_to_MeV: bool = True,
    clip_to_fullsim: bool = True,
):
    if geometry.startswith("CCD"):
        set_geometry(geometry)

    save_dir.mkdir(parents=True, exist_ok=True)

    if not keep_previous:
        logger.info(f"Removing previous plots in {save_dir}")
        for f in save_dir.glob("*.png"):
            f.unlink()

    scale = 1000.0 if convert_GeV_to_MeV else 1.0
    full_simulation = full_simulation * scale
    ml_simulation = ml_simulation * scale

    if clip_to_fullsim:
        ml_simulation = np.clip(ml_simulation, 0.0, full_simulation.max())

    results = {}

    # longitudinal profile
    full_sim_long = LongitudinalProfile(_input=full_simulation)
    ml_sim_long = LongitudinalProfile(_input=ml_simulation)
    longitudinal_profile_plotter = ProfilePlotter(
        save_dir, particle_energy, phi, geometry, theta, full_sim_long, ml_sim_long, _plot_gaussian=False
    )
    long_results = longitudinal_profile_plotter.plot_and_save()
    results.update(long_results)

    # radial profile
    full_sim_rad = RadialProfile(_input=full_simulation)
    ml_sim_rad = RadialProfile(_input=ml_simulation)
    radial_profile_plotter = ProfilePlotter(
        save_dir, particle_energy, phi, geometry, theta, full_sim_rad, ml_sim_rad, _plot_gaussian=False
    )
    rad_results = radial_profile_plotter.plot_and_save()
    results.update(rad_results)

    # azimuthal profile
    full_sim_azim = AzimuthalProfile(_input=full_simulation)
    ml_sim_azim = AzimuthalProfile(_input=ml_simulation)
    azimuthal_profile_plotter = ProfilePlotter(
        save_dir, particle_energy, phi, geometry, theta, full_sim_azim, ml_sim_azim, _plot_gaussian=False
    )
    azim_results = azimuthal_profile_plotter.plot_and_save()
    results.update(azim_results)

    # global observables
    full_sim_shower = Shower(_input=full_simulation, _energy=particle_energy)
    ml_sim_shower = Shower(_input=ml_simulation, _energy=particle_energy)
    shower_plotter = ShowerPlotter(save_dir, particle_energy, phi, geometry, theta, full_sim_shower, ml_sim_shower)
    shower_results = shower_plotter.plot_and_save()
    results.update(shower_results)

    return results
