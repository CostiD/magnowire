"""
magnowire
=========
Micromagnetic finite-difference solver for nanowire arrays.

Features
--------
- Open and true 3-D periodic boundary conditions (Bruckner 2021)
- Newell 1993 demagnetisation tensor (OOMMF-validated)
- LLG / RK4 integration, GPU-accelerated via CuPy (optional)
- Predefined material library: Co, FeCo, Fe, Ni, Permalloy, NdFeB
- Geometry builders: single nanowire, square / hexagonal array
- Quasi-static hysteresis protocol
- Automatic extraction of Hc, Mr, squareness, BHmax

Quick start
-----------
>>> import magnowire as mw
>>> mat  = mw.materials.Co
>>> geom = mw.geometry.nanowire(diameter_nm=20, length_nm=200, cell_nm=2)
>>> sim  = mw.solver.MicromagSolver(geom, mat, pbc=False)
>>> loop = mw.hysteresis.hysteresis_loop(sim, B_max=1.0, n_field=21)
>>> metrics = mw.analysis.extract_metrics(loop)
>>> print(metrics)
"""

from . import materials, geometry, demag, solver, hysteresis, analysis
from ._backend import BACKEND, GPU

__version__ = "0.1.0"
__author__  = "magnowire contributors"

__all__ = [
    "materials",
    "geometry",
    "demag",
    "solver",
    "hysteresis",
    "analysis",
    "BACKEND",
    "GPU",
]
