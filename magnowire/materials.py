"""
magnowire.materials
~~~~~~~~~~~~~~~~~~~
Predefined magnetic material parameters and a helper for custom materials.

All values at room temperature (~300 K).

References
----------
Co (hcp) : Coey, "Magnetism and Magnetic Materials" (2010)
FeCo      : Bozorth, "Ferromagnetism" (1951); Coey (2010)
Fe (bcc)  : Coey (2010)
Ni (fcc)  : Coey (2010)
Permalloy : standard muMag benchmark parameters
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Tuple

from .constants import MU0


@dataclass
class Material:
    """
    Micromagnetic material parameters.

    Parameters
    ----------
    name : str
        Human-readable label.
    Ms : float
        Saturation magnetisation [A/m].
    A : float
        Exchange stiffness [J/m].
    Ku : float
        First uniaxial magnetocrystalline anisotropy constant [J/m³].
        Positive → easy axis, negative → easy plane.
    u_axis : tuple of float
        Unit vector of the easy axis (will be normalised automatically).
    alpha : float
        Default Gilbert damping parameter (can be overridden in solver).
    """

    name:   str
    Ms:     float
    A:      float
    Ku:     float          = 0.0
    u_axis: Tuple[float, float, float] = (0., 0., 1.)
    alpha:  float          = 0.5

    # ── Derived quantities ────────────────────────────────────────

    @property
    def l_ex(self) -> float:
        """Exchange length [m]:  l_ex = sqrt(2A / mu0 Ms²)."""
        return math.sqrt(2 * self.A / (MU0 * self.Ms ** 2))

    @property
    def mu0_Ha(self) -> float:
        """Anisotropy field [T]:  mu0 Ha = 2 Ku / Ms."""
        return 2 * self.Ku / self.Ms   # already in T (= mu0 * Ha)

    @property
    def mu0_Ms(self) -> float:
        """Saturation polarisation [T]:  Js = mu0 Ms."""
        return MU0 * self.Ms

    def __str__(self) -> str:
        return (
            f"{self.name}:  Ms={self.Ms/1e6:.3f} MA/m  "
            f"A={self.A*1e12:.1f} pJ/m  "
            f"Ku={self.Ku/1e3:.1f} kJ/m³  "
            f"l_ex={self.l_ex*1e9:.2f} nm  "
            f"mu0Ha={self.mu0_Ha*1e3:.1f} mT"
        )


# ── Predefined materials ──────────────────────────────────────────────────────

#: Cobalt hcp — strong uniaxial anisotropy, easy axis along c (z here)
Co = Material(
    name   = "Co (hcp)",
    Ms     = 1.40e6,    # A/m
    A      = 30e-12,    # J/m
    Ku     = 450e3,     # J/m³
    u_axis = (0., 0., 1.),
    alpha  = 0.01,
)

#: FeCo alloy (~50/50) — highest Ms of common materials, low Ku
FeCo = Material(
    name   = "FeCo",
    Ms     = 1.95e6,
    A      = 28e-12,
    Ku     = 10e3,
    u_axis = (0., 0., 1.),
    alpha  = 0.01,
)

#: Iron bcc
Fe = Material(
    name   = "Fe (bcc)",
    Ms     = 1.70e6,
    A      = 21e-12,
    Ku     = 48e3,
    u_axis = (0., 0., 1.),
    alpha  = 0.002,
)

#: Nickel fcc — easy plane (Ku < 0), long exchange length
Ni = Material(
    name   = "Ni (fcc)",
    Ms     = 0.484e6,
    A      = 9e-12,
    Ku     = -5.7e3,    # easy plane
    u_axis = (0., 0., 1.),
    alpha  = 0.064,
)

#: Permalloy (Ni80Fe20) — standard muMag benchmark material, zero Ku
Permalloy = Material(
    name   = "Permalloy (Ni80Fe20)",
    Ms     = 8.0e5,
    A      = 13e-12,
    Ku     = 0.0,
    u_axis = (1., 0., 0.),
    alpha  = 0.01,
)

#: NdFeB (sintered, room temperature) — reference hard magnet
NdFeB = Material(
    name   = "NdFeB (sintered)",
    Ms     = 1.28e6,
    A      = 8e-12,
    Ku     = 4.9e6,
    u_axis = (0., 0., 1.),
    alpha  = 0.1,
)


LIBRARY: dict[str, Material] = {
    "Co":         Co,
    "FeCo":       FeCo,
    "Fe":         Fe,
    "Ni":         Ni,
    "Permalloy":  Permalloy,
    "NdFeB":      NdFeB,
}


def get(name: str) -> Material:
    """Retrieve a material by name (case-insensitive)."""
    key = {k.lower(): k for k in LIBRARY}
    match = key.get(name.lower())
    if match is None:
        raise KeyError(
            f"Unknown material '{name}'. Available: {list(LIBRARY)}"
        )
    return LIBRARY[match]


def list_materials() -> None:
    """Print a summary table of all predefined materials."""
    print(f"{'Name':<25} {'Ms [MA/m]':>10} {'A [pJ/m]':>10} "
          f"{'Ku [kJ/m³]':>12} {'l_ex [nm]':>10} {'mu0Ha [mT]':>11}")
    print("-" * 82)
    for mat in LIBRARY.values():
        print(f"  {mat.name:<23} {mat.Ms/1e6:>10.3f} {mat.A*1e12:>10.1f} "
              f"{mat.Ku/1e3:>12.1f} {mat.l_ex*1e9:>10.2f} "
              f"{mat.mu0_Ha*1e3:>11.1f}")
