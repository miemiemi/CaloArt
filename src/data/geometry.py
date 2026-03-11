"""
Geometry problem specific constants.
Based on the CaloChalenge Datset 2: https://calochallenge.github.io/homepage/
"""

import math
from types import SimpleNamespace

GEOMETRY = SimpleNamespace()

# We define the cylindrical coordinates system (r, phi, z) as follows:
# - z is the longitudinal coordinate
# - phi is the azimuthal coordinate
# - r is the radial coordinate

# Number of calorimeter cells
GEOMETRY.N_CELLS_Z = 45
GEOMETRY.N_CELLS_PHI = 16
GEOMETRY.N_CELLS_R = 9

# Cell size
GEOMETRY.SIZE_Z = 3.4 # mm = 2 x (0.3mm of Si + 1.4mm of W)
GEOMETRY.SIZE_PHI = 2 * math.pi / GEOMETRY.N_CELLS_PHI
GEOMETRY.SIZE_R = 4.65 # mm