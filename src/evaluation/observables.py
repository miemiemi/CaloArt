from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from src.data.geometry import GEOMETRY


@dataclass
class Observable:
    """An abstract class defining interface of all observables.

    Do not use this class directly.

    Attributes:
          _input: A numpy array with shape = (NE, R, PHI, Z), where NE stands for the number of events.
    """

    _input: np.ndarray


class ProfileType(Enum):
    """Enum class of various profile types."""

    LONGITUDINAL = 0
    RADIAL = 1
    AZIMUTHAL = 2


@dataclass
class Profile(Observable):
    """An abstract class describing behaviour of profiles along the axes. Do not use this class directly."""

    def __post_init__(self):
        """
        _energies_per_event: A numpy array of shape (NE, NL) where NE stands for the number of events and NL for the number of layers along the given axis.
            An element [i, j] is a sum of energies detected in all cells located in a jth layer for an ith event.
        _total_energy_per_event: A numpy array of shape (NE, ).
            An element [i] is a sum of energies detected in all cells for an ith event.
        _hits_per_event: A numpy array of shape (NE, NL).
            An element [i, j] is a number of hits detected in a jth layer for an ith event.
        """
        self._energies_per_event = None
        self._total_energy_per_event = None
        self._hits_per_event = None
        self._cell_size = None
        self._num_cells = None

    def calc_total_energy_per_layer(self) -> np.ndarray:
        """Calculates the total energy deposited in a given layer a.k.a profile along the axis.

        A profile for a given layer l = 0, ..., NL - 1 is defined as:
        sum_{i = 0}^{NE - 1} energy_per_event[i, l]

        Returns:
            A numpy array of shape (NL, ).

        """
        return np.sum(self._energies_per_event, axis=0)

    def calc_total_hits_per_layer(self) -> np.ndarray:
        """Calculates the total number of hits in a given layer.

        A number of hits for a given layer l = 0, ..., NL - 1 is defined as:
        sum_{i = 0}^{NE - 1} hits_per_event[i, l]

        Returns:
            A numpy array of number of hits for each layer with a shape (NL, ).
        """
        return np.sum(self._hits_per_event, axis=0)

    def calc_first_moment(self) -> np.ndarray:
        """Calculates the first moment of the profile along the axis.

        A first moment of the profile for a given event e = 0, ..., NE - 1 is defined as:
        FM[e] = sum_{i = 0}^{NL - 1} (i * cell_size * energies_per_event[e, i] / total_energy_per_event[e])

        Returns:
            A numpy array of first moments of profiles for each event with a shape (NE, ).

        """
        return (
            self._cell_size
            * np.dot(self._energies_per_event, np.arange(self._num_cells))
            / self._total_energy_per_event
        )

    def calc_second_moment(self) -> np.ndarray:
        """Calculates the second moment of the profile along the axis.

        A second moment of the profile for a given event e = 0, ..., NE - 1 is defined as:
        SM[e] = sum_{i = 0}^{NL - 1} (i * cell_size - FM[e])^2 * energies_per_event[e, i] / total_energy_per_event[e]

        Returns:
            A numpy array of second moments of the profiles for each event with a shape (NE, ).
        """
        first_moment = self.calc_first_moment()
        first_moment = np.expand_dims(first_moment, axis=1)
        cell_position = self._cell_size * np.expand_dims(np.arange(self._num_cells), axis=0)
        # cell_position has now a shape = [1, NL] and first moment has a shape = [NE, 1]. There is a broadcasting in the line
        # below how that one create an array with a shape = [NE, NL]
        return (
            np.sum(np.multiply(np.power(cell_position - first_moment, 2), self._energies_per_event), axis=1)
            / self._total_energy_per_event
        )

    def calc_event_energy_per_layer(self):
        """Calculates energy deposited in a particular layer.

        Energy per layer for a given event e = 0, ..., NE - 1 is defined by an array with shape (NL, ) storing
        values of total energy detected in a particular layer.

        Returns:
            A numpy array of layer energy values with shape (NE, NL).

        """
        return np.copy(self._energies_per_event)


@dataclass
class LongitudinalProfile(Profile):
    def __post_init__(self):
        super().__post_init__()
        self._energies_per_event = np.sum(self._input, axis=(1, 2))
        self._total_energy_per_event = np.sum(self._energies_per_event, axis=1)
        self._hits_per_event = np.sum(self._input > 0, axis=(1, 2))
        self._cell_size = GEOMETRY.SIZE_Z
        self._num_cells = GEOMETRY.N_CELLS_Z


@dataclass
class RadialProfile(Profile):
    def __post_init__(self):
        super().__post_init__()
        self._energies_per_event = np.sum(self._input, axis=(2, 3))
        self._total_energy_per_event = np.sum(self._energies_per_event, axis=1)
        self._hits_per_event = np.sum(self._input > 0, axis=(2, 3))
        self._cell_size = GEOMETRY.SIZE_R
        self._num_cells = GEOMETRY.N_CELLS_R


@dataclass
class AzimuthalProfile(Profile):
    def __post_init__(self):
        super().__post_init__()
        self._energies_per_event = np.sum(self._input, axis=(1, 3))
        self._total_energy_per_event = np.sum(self._energies_per_event, axis=1)
        self._hits_per_event = np.sum(self._input > 0, axis=(1, 3))
        self._cell_size = GEOMETRY.SIZE_PHI
        self._num_cells = GEOMETRY.N_CELLS_PHI


@dataclass
class Shower(Observable):
    """A class defining global observables of the shower."""

    _energy: Optional[int]

    @property
    def shower(self):
        return self._input

    def calc_total_energy_per_event(self):
        """Calculates total energy detected in an event.

        Total energy for a given event e = 0, ..., NE - 1 is defined as a sum of energies detected in all cells
        for this event.

        Returns:
            A numpy array of total energy values with shape (NE, ).
        """
        return np.sum(self._input, axis=(1, 2, 3))

    def calc_total_hits_per_event(self):
        """Calculates total number of hits detected in an event.

        Total number of hits for a given event e = 0, ..., NE - 1 is defined as a sum of hits detected in all cells
        for this event.

        Returns:
            A numpy array of total number of hits with shape (NE, ).
        """
        return np.sum(self._input > 0, axis=(1, 2, 3))

    def calc_cell_energy_per_event(self):
        """Calculates cell energy.

        Cell energy for a given event e = 0, ..., NE - 1 is defined by an array with shape (R * PHI * Z) storing
        values of energy in particular cells.

        Returns:
            A numpy array of cell energy values with shape (NE, R * PHI * Z).

        """
        return np.copy(self._input).reshape(-1)
