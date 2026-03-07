"""
magnowire.analysis
~~~~~~~~~~~~~~~~~~
Post-processing of hysteresis loops.

Extracts standard magnetic figures of merit:
  - Hc  : coercive field [T]
  - Mr  : remanent magnetisation (normalised) [dimensionless]
  - Hs  : saturation field [T]
  - BHmax : maximum energy product [kJ/m³]
  - squareness : Mr / Ms = Mr (since M is normalised)

All functions accept a :class:`~magnowire.hysteresis.HysteresisResult`
or plain arrays.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from .hysteresis import HysteresisResult


@dataclass
class LoopMetrics:
    """
    Magnetic figures of merit extracted from one hysteresis branch.

    Attributes
    ----------
    Hc_pos   : float  Coercive field (positive branch crossing) [T].
    Hc_neg   : float  Coercive field (negative branch crossing) [T].
    Hc       : float  Average coercivity = (|Hc_pos| + |Hc_neg|) / 2 [T].
    Mr_pos   : float  Remanence after positive saturation (M at H=0⁻) [norm.].
    Mr_neg   : float  Remanence after negative saturation (M at H=0⁺) [norm.].
    Mr       : float  Average remanence [normalised].
    Ms       : float  Saturation magnetisation (M at max |H|) [normalised ~1].
    squareness: float Mr / Ms.
    BHmax    : float  Maximum energy product [J/m³].
    BHmax_kJ : float  Maximum energy product [kJ/m³].
    """
    Hc_pos:     float
    Hc_neg:     float
    Hc:         float
    Mr_pos:     float
    Mr_neg:     float
    Mr:         float
    Ms:         float
    squareness: float
    BHmax:      float
    BHmax_kJ:   float

    def __str__(self) -> str:
        return (
            f"  Hc       = {self.Hc*1e3:.2f} mT  "
            f"(+{self.Hc_pos*1e3:.2f} / -{abs(self.Hc_neg)*1e3:.2f} mT)\n"
            f"  Mr       = {self.Mr:.4f}  "
            f"(+{self.Mr_pos:.4f} / {self.Mr_neg:.4f})\n"
            f"  Ms       = {self.Ms:.4f}\n"
            f"  Squareness = {self.squareness:.4f}\n"
            f"  BHmax    = {self.BHmax_kJ:.2f} kJ/m³"
        )


def extract_metrics(
    result:      HysteresisResult,
    Ms_material: Optional[float] = None,
) -> LoopMetrics:
    """
    Extract coercivity, remanence, and energy product from a loop.

    Parameters
    ----------
    result      : HysteresisResult
    Ms_material : float, optional  Physical Ms [A/m].
                  If given, BHmax is in absolute SI units [J/m³].
                  If None, BHmax is computed from normalised M (approximate).

    Returns
    -------
    LoopMetrics
    """
    B = result.B_applied
    M = result.Mx       # primary component (along field axis)

    n = len(B)
    half = n // 2

    # Two branches: descending (field goes + → -) and ascending (- → +)
    B_desc = B[:half + 1]
    M_desc = M[:half + 1]
    B_asc  = B[half:]
    M_asc  = M[half:]

    # ── Coercive fields ───────────────────────────────────────────
    Hc_neg = _zero_crossing(B_desc, M_desc)   # M=0 on descending branch → Hc⁻
    Hc_pos = _zero_crossing(B_asc,  M_asc)    # M=0 on ascending branch  → Hc⁺
    Hc     = (abs(Hc_neg) + abs(Hc_pos)) / 2.0

    # ── Remanence ─────────────────────────────────────────────────
    Mr_pos = _interpolate_at(B_desc, M_desc, 0.0)  # M at H=0, descending
    Mr_neg = _interpolate_at(B_asc,  M_asc,  0.0)  # M at H=0, ascending
    Mr     = (abs(Mr_pos) + abs(Mr_neg)) / 2.0

    # ── Saturation (M at |B| = B_max) ────────────────────────────
    Ms_norm = float(np.mean([abs(M[0]), abs(M[-1])]))

    # ── Energy product: BHmax from the second quadrant ────────────
    # In the second quadrant: B_field < 0, M > 0
    # B_total = mu0*(H + M*Ms_mat); for a magnet in a circuit:
    # Here we work with normalised M and applied μ₀H.
    # BH product (normalised): (-μ₀H) * M  for H<0, M>0
    from .constants import MU0
    if Ms_material is not None:
        # Absolute BHmax [J/m³]
        # B_magnet = mu0*(H_applied + M*Ms) (open-circuit approximation)
        H_arr  = B_desc / MU0
        M_abs  = M_desc * Ms_material
        B_tot  = MU0 * (H_arr + M_abs)
        BH     = B_tot * H_arr
        BHmax  = float(np.max(-BH[B_desc <= 0]))
    else:
        # Normalised proxy: (-μ₀H) * M_norm
        mask2q = B_desc <= 0
        if mask2q.any():
            BH_norm = (-B_desc[mask2q]) * M_desc[mask2q]
            BHmax   = float(np.max(BH_norm))
        else:
            BHmax = 0.0

    return LoopMetrics(
        Hc_pos     = float(Hc_pos),
        Hc_neg     = float(Hc_neg),
        Hc         = float(Hc),
        Mr_pos     = float(Mr_pos),
        Mr_neg     = float(Mr_neg),
        Mr         = float(Mr),
        Ms         = float(Ms_norm),
        squareness = float(Mr / Ms_norm) if Ms_norm > 0 else 0.0,
        BHmax      = float(BHmax),
        BHmax_kJ   = float(BHmax / 1e3),
    )


def plot_loop(
    result: HysteresisResult,
    metrics: Optional[LoopMetrics] = None,
    title:  str = "",
    ax=None,
    color: str = "royalblue",
):
    """
    Plot a hysteresis loop, optionally annotating Hc and Mr.

    Parameters
    ----------
    result  : HysteresisResult
    metrics : LoopMetrics, optional  If given, annotates Hc and Mr.
    title   : str  Plot title.
    ax      : matplotlib Axes, optional  (creates new figure if None).
    color   : str  Line colour.

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    B_mT = result.B_applied * 1e3
    ax.plot(B_mT, result.Mx, color=color, lw=2.0, label="⟨Mx⟩/Ms")

    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.axvline(0, color="k", lw=0.6, ls="--")

    if metrics is not None:
        # Annotate Hc
        for hc_T, lab in [(metrics.Hc_pos, "+Hc"), (metrics.Hc_neg, "−Hc")]:
            ax.axvline(hc_T * 1e3, color="firebrick", lw=1.2, ls=":")
            ax.text(hc_T * 1e3, 0.05, f"  {lab}\n  {abs(hc_T)*1e3:.1f} mT",
                    color="firebrick", fontsize=8, va="bottom")
        # Annotate Mr
        ax.axhline(metrics.Mr_pos, color="seagreen", lw=1.0, ls=":")
        ax.axhline(-metrics.Mr_pos, color="seagreen", lw=1.0, ls=":")
        ax.text(B_mT[0] * 0.05, metrics.Mr_pos + 0.03,
                f"Mr={metrics.Mr:.3f}", color="seagreen", fontsize=8)

    ax.set_xlabel(r"$\mu_0 H_{\rm ext}$ [mT]", fontsize=12)
    ax.set_ylabel(r"$\langle M_x \rangle / M_s$", fontsize=12)
    ax.set_xlim(B_mT.min() * 1.05, B_mT.max() * 1.05)
    ax.set_ylim(-1.25, 1.25)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=11)

    return fig, ax


# ── Internal helpers ──────────────────────────────────────────────────────────

def _zero_crossing(B: np.ndarray, M: np.ndarray) -> float:
    """Linear interpolation to find B where M = 0."""
    for i in range(len(M) - 1):
        if M[i] * M[i+1] <= 0:
            if M[i] == M[i+1]:
                return float(B[i])
            t = -M[i] / (M[i+1] - M[i])
            return float(B[i] + t * (B[i+1] - B[i]))
    # Fallback: return B at minimum |M|
    return float(B[np.argmin(np.abs(M))])


def _interpolate_at(B: np.ndarray, M: np.ndarray, B_target: float) -> float:
    """Linear interpolation to find M at a given B value."""
    idx = np.searchsorted(B, B_target)
    if idx == 0:
        return float(M[0])
    if idx >= len(B):
        return float(M[-1])
    t = (B_target - B[idx-1]) / (B[idx] - B[idx-1] + 1e-300)
    return float(M[idx-1] + t * (M[idx] - M[idx-1]))
