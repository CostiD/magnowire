"""
magnowire.geometry
~~~~~~~~~~~~~~~~~~
Grid geometry builders for nanowires and nanowire arrays.

All builders return a :class:`Geometry` object that carries:
  - grid dimensions and cell sizes
  - a boolean mask (True = magnetic material)
  - an initial magnetisation array m0
  - human-readable metadata

Coordinate convention
---------------------
  x → wire axis (easy axis for shape anisotropy)
  y, z → transverse directions
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class Geometry:
    """
    Fully described FD grid geometry.

    Parameters
    ----------
    nx, ny, nz : int    Grid cell counts.
    dx, dy, dz : float  Cell sizes [m].
    mask       : ndarray (nx, ny, nz) bool  True = magnetic cell.
    m0         : ndarray (nx, ny, nz, 3)    Initial normalised magnetisation.
    name       : str    Short description.
    meta       : dict   Extra metadata (wire diameter, aspect ratio, …).
    """
    nx: int;  ny: int;  nz: int
    dx: float; dy: float; dz: float
    mask: np.ndarray
    m0:   np.ndarray
    name: str = ""
    meta: dict = field(default_factory=dict)

    # ── Convenience properties ────────────────────────────────────

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    @property
    def cell_volume(self) -> float:
        return self.dx * self.dy * self.dz

    @property
    def fill_fraction(self) -> float:
        return float(self.mask.mean())

    def __str__(self) -> str:
        return (f"{self.name}  grid {self.nx}×{self.ny}×{self.nz}  "
                f"cell {self.dx*1e9:.1f}×{self.dy*1e9:.1f}×{self.dz*1e9:.1f} nm  "
                f"fill {self.fill_fraction:.3f}")


# ── Single nanowire (open BC) ─────────────────────────────────────────────────

def nanowire(
    diameter_nm: float,
    length_nm:   float,
    cell_nm:     float = 2.0,
    axis:        str   = "x",
    pad_cells:   int   = 2,
    initial:     str   = "uniform+x",
) -> Geometry:
    """
    Build a single cylindrical nanowire with open boundary conditions.

    The wire axis is aligned along `axis` (default x).  `pad_cells` empty
    cells are added on each transverse face to reduce boundary artefacts.

    Parameters
    ----------
    diameter_nm : float  Wire diameter [nm].
    length_nm   : float  Wire length [nm].
    cell_nm     : float  Cell size [nm] (isotropic).
    axis        : str    Wire axis: 'x', 'y', or 'z'.
    pad_cells   : int    Transverse padding (open-BC ghost cells).
    initial     : str    Seed state: 'uniform+x', 'uniform-x', 'vortex'.

    Returns
    -------
    Geometry
    """
    c   = cell_nm * 1e-9
    d   = diameter_nm * 1e-9
    L   = length_nm   * 1e-9

    n_wire_ax = max(1, int(round(L / c)))
    n_wire_tr = max(1, int(round(d / c)))
    n_tr      = n_wire_tr + 2 * pad_cells  # transverse with padding

    # Axis permutation so wire always runs along 'x' internally
    if axis == "x":
        nx, ny, nz = n_wire_ax, n_tr, n_tr
    elif axis == "y":
        nx, ny, nz = n_tr, n_wire_ax, n_tr
    elif axis == "z":
        nx, ny, nz = n_tr, n_tr, n_wire_ax
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")

    # Cylindrical mask in the transverse plane
    cy = np.arange(ny) - (ny - 1) / 2.0
    cz = np.arange(nz) - (nz - 1) / 2.0
    CY, CZ = np.meshgrid(cy, cz, indexing="ij")
    R_cell = (n_wire_tr / 2.0)
    circ   = (CY**2 + CZ**2) <= R_cell**2  # (ny, nz)

    if axis == "x":
        mask = np.broadcast_to(circ[None, :, :], (nx, ny, nz)).copy()
    elif axis == "y":
        cx2 = np.arange(nx) - (nx - 1) / 2.0
        cz2 = np.arange(nz) - (nz - 1) / 2.0
        CX2, CZ2 = np.meshgrid(cx2, cz2, indexing="ij")
        circ2 = (CX2**2 + CZ2**2) <= R_cell**2
        mask = np.broadcast_to(circ2[:, None, :], (nx, ny, nz)).copy()
    else:  # z
        cx3 = np.arange(nx) - (nx - 1) / 2.0
        cy3 = np.arange(ny) - (ny - 1) / 2.0
        CX3, CY3 = np.meshgrid(cx3, cy3, indexing="ij")
        circ3 = (CX3**2 + CY3**2) <= R_cell**2
        mask = np.broadcast_to(circ3[:, :, None], (nx, ny, nz)).copy()

    # Initial magnetisation
    m0 = _make_initial(nx, ny, nz, mask, initial)

    aspect = length_nm / diameter_nm
    return Geometry(
        nx=nx, ny=ny, nz=nz, dx=c, dy=c, dz=c,
        mask=mask, m0=m0,
        name=f"Nanowire ∅{diameter_nm:.0f}nm × L{length_nm:.0f}nm",
        meta=dict(
            diameter_nm  = diameter_nm,
            length_nm    = length_nm,
            aspect_ratio = aspect,
            axis         = axis,
            cell_nm      = cell_nm,
            pad_cells    = pad_cells,
            n_wire_ax    = n_wire_ax,
            n_wire_tr    = n_wire_tr,
        ),
    )


# ── Nanowire array (PBC in transverse plane) ─────────────────────────────────

def nanowire_array(
    diameter_nm:  float,
    pitch_nm:     float,
    length_nm:    float,
    cell_nm:      float = 2.0,
    arrangement:  str   = "square",
    initial:      str   = "uniform+x",
) -> Geometry:
    """
    Build a unit cell of a nanowire array with PBC in the transverse plane.

    The wire axis is along x.  PBC in y and z represent the infinite array.
    OBC in x represents the finite wire length.

    Parameters
    ----------
    diameter_nm  : float  Wire diameter [nm].
    pitch_nm     : float  Centre-to-centre distance [nm].  pitch > diameter.
    length_nm    : float  Wire length [nm].
    cell_nm      : float  Cell size [nm] (isotropic).
    arrangement  : str    'square' or 'hexagonal' (hex shifts every other row).
    initial      : str    Seed state.

    Returns
    -------
    Geometry
    """
    if pitch_nm <= diameter_nm:
        raise ValueError(
            f"pitch_nm ({pitch_nm}) must be larger than diameter_nm ({diameter_nm})")

    c       = cell_nm * 1e-9
    nx      = max(1, int(round(length_nm  * 1e-9 / c)))
    n_pitch = max(1, int(round(pitch_nm   * 1e-9 / c)))
    r_cell  = (diameter_nm * 1e-9 / c) / 2.0

    if arrangement == "square":
        ny = nz = n_pitch
        cy = np.arange(ny) - (ny - 1) / 2.0
        cz = np.arange(nz) - (nz - 1) / 2.0
        CY, CZ = np.meshgrid(cy, cz, indexing="ij")
        circ2d = (CY**2 + CZ**2) <= r_cell**2
    elif arrangement == "hexagonal":
        ny = n_pitch
        nz = int(round(n_pitch * math.sqrt(3) / 2))
        cy = np.arange(ny) - (ny - 1) / 2.0
        cz = np.arange(nz) - (nz - 1) / 2.0
        CY, CZ = np.meshgrid(cy, cz, indexing="ij")
        # Hex shift: alternate rows offset by 0.5 pitch
        shift  = np.where(np.arange(nz) % 2 == 0, 0.0, n_pitch / 2.0)
        CY_s   = CY - shift[None, :]
        circ2d = (CY_s**2 + CZ**2) <= r_cell**2
    else:
        raise ValueError(f"arrangement must be 'square' or 'hexagonal', got '{arrangement}'")

    mask = np.broadcast_to(circ2d[None, :, :], (nx, ny, nz)).copy()
    m0   = _make_initial(nx, ny, nz, mask, initial)

    gap_nm = pitch_nm - diameter_nm
    return Geometry(
        nx=nx, ny=ny, nz=nz, dx=c, dy=c, dz=c,
        mask=mask, m0=m0,
        name=(f"NanowireArray ∅{diameter_nm:.0f}nm "
              f"pitch {pitch_nm:.0f}nm ({arrangement})"),
        meta=dict(
            diameter_nm  = diameter_nm,
            pitch_nm     = pitch_nm,
            gap_nm       = gap_nm,
            length_nm    = length_nm,
            arrangement  = arrangement,
            cell_nm      = cell_nm,
        ),
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_initial(nx: int, ny: int, nz: int,
                  mask: np.ndarray,
                  mode: str) -> np.ndarray:
    """Create a normalised initial magnetisation array."""
    m0 = np.zeros((nx, ny, nz, 3))

    if mode == "uniform+x":
        m0[mask, 0] = 1.0
        m0[~mask, 0] = 1.0   # non-magnetic: direction irrelevant

    elif mode == "uniform-x":
        m0[..., 0] = -1.0

    elif mode == "vortex":
        # Vortex in y-z plane, uniform along x
        cy = np.arange(ny) - (ny - 1) / 2.0
        cz = np.arange(nz) - (nz - 1) / 2.0
        CY, CZ = np.meshgrid(cy, cz, indexing="ij")
        r = np.sqrt(CY**2 + CZ**2) + 1e-6
        m0[:, :, :, 1] =  (CZ / r)[None, :, :]
        m0[:, :, :, 2] = -(CY / r)[None, :, :]
        # Small axial component for stability
        m0[:, :, :, 0] = 0.1

    elif mode == "random":
        rng = np.random.default_rng(0)
        v = rng.standard_normal((nx, ny, nz, 3))
        m0 = v

    else:
        raise ValueError(f"Unknown initial state '{mode}'")

    nrm = np.linalg.norm(m0, axis=-1, keepdims=True)
    return m0 / np.maximum(nrm, 1e-10)
