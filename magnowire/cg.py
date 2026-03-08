"""
magnowire.cg
~~~~~~~~~~~~
Conjugate-gradient energy minimiser on the unit sphere.

Mirrors the algorithm in OOMMF's Oxs_CGEvolve (cgevolve.cc):
  - Gradient on sphere  : g = m × (H_eff × m)  [= mxHxm torque]
  - CG direction update : Polak-Ribière with reset
  - Spin move           : geodesic retraction  m(t) = normalise(m√(1+t²|d|²) + t·d)
  - Line search         : bracket + golden-section on E(t)
  - Convergence         : max|mxHxm| < tol  [A/m]

References
----------
OOMMF cgevolve.cc (Donahue & Porter, NIST)
NOTES II, 29-May-2002, p156  (Donahue)
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from ._backend import xp, to_np, to_xp
from .constants import MU0


# ── helpers ───────────────────────────────────────────────────────────────────

def _mxHxm(m, H):
    """
    Torque vector  m × H × m  (gradient of energy on unit sphere).
    Equivalent to  H - (H·m) m  (tangent projection of H).
    Shape: same as m.
    """
    Hdotm = xp.sum(H * m, axis=-1, keepdims=True)
    return H - Hdotm * m


def _project_tangent(v, m):
    """Project v onto tangent plane of unit sphere at m: v -= (v·m)m."""
    return v - xp.sum(v * m, axis=-1, keepdims=True) * m


def _spin_move(m0, d, t):
    """
    Geodesic retraction (OOMMF ThreadA formula):
        m_new = normalise(m0 * sqrt(1 + t²|d|²) + t * d)
    d must be tangent to sphere at m0.
    """
    dsq  = xp.sum(d * d, axis=-1, keepdims=True)           # |d|² per cell
    mult = xp.sqrt(1.0 + t * t * dsq)
    raw  = mult * m0 + t * d
    nrm  = xp.linalg.norm(raw, axis=-1, keepdims=True)
    return raw / xp.maximum(nrm, 1e-30)


def _energy(solver, m, H_ext):
    """
    Total Zeeman + internal energy [J] (unnormalised — used only for comparison).
    We compute -μ₀ Ms Σ m·H_eff * V = -μ₀ Ms * vol * <m·H_eff>.
    """
    H   = solver.H_eff(m, H_ext)
    vol = float(solver.geom.mask.sum()) * solver.geom.cell_volume
    dot = float(to_np(xp.sum(m[solver._mask.astype(bool)] *
                              H[solver._mask.astype(bool)])))
    return -MU0 * solver.Ms * dot * solver.geom.cell_volume


# ── line search ───────────────────────────────────────────────────────────────

def _line_search(solver, m, d, H_ext, H_eff_0,
                 t_start: float = 0.05,
                 n_golden: int  = 12):
    """
    Find step size t ≥ 0 minimising E(m(t)).

    Strategy
    --------
    1. dE/dt|₀ < 0 by construction (d is downhill).
    2. Bracket by doubling t_start until E increases.
    3. Golden-section refinement.

    Returns
    -------
    t_best : float   Step size along d.
    m_best : array   New spin state.
    E_best : float   Energy at best point.
    """
    E0 = _energy(solver, m, H_ext)

    # ── bracket ──────────────────────────────────────────────────
    t_lo, t_hi = 0.0, t_start
    E_lo = E0

    for _ in range(20):
        m_try = _spin_move(m, d, t_hi)
        E_try = _energy(solver, m_try, H_ext)
        if E_try > E_lo:
            break
        t_lo, E_lo = t_hi, E_try
        t_hi *= 2.0
        if t_hi > 4 * np.pi:          # sanity cap (half rotation)
            break

    # If t_lo == 0 and E(t_hi) > E0: minimum in [0, t_hi] → search
    # If E kept decreasing and we hit cap: best we found is t_lo
    if t_lo == 0.0 and E_lo == E0:
        # No improvement found at all — return zero step
        return 0.0, m, E0

    # ── golden-section in [t_lo, t_hi] ───────────────────────────
    phi   = (np.sqrt(5) - 1) / 2   # 0.618...
    t_a, t_b = t_lo, t_hi
    t_c = t_b - phi * (t_b - t_a)
    t_d = t_a + phi * (t_b - t_a)
    E_c = _energy(solver, _spin_move(m, d, t_c), H_ext)
    E_d = _energy(solver, _spin_move(m, d, t_d), H_ext)

    for _ in range(n_golden):
        if E_c < E_d:
            t_b, t_d, E_d = t_d, t_c, E_c
            t_c = t_b - phi * (t_b - t_a)
            E_c = _energy(solver, _spin_move(m, d, t_c), H_ext)
        else:
            t_a, t_c, E_c = t_c, t_d, E_d
            t_d = t_a + phi * (t_b - t_a)
            E_d = _energy(solver, _spin_move(m, d, t_d), H_ext)

    t_best = (t_a + t_b) / 2.0
    m_best = _spin_move(m, d, t_best)
    E_best = _energy(solver, m_best, H_ext)

    if E_best > E0:          # safety: never increase energy
        return 0.0, m, E0

    return t_best, m_best, E_best


# ── main CG minimiser ─────────────────────────────────────────────────────────

def cg_minimise(
    solver,
    m0,
    H_ext,
    tol:          float = 1e3,      # A/m — max|mxHxm| convergence threshold
    max_iter:     int   = 2000,
    reset_every:  int   = 50,       # Polak-Ribière reset period (OOMMF: 5000)
    t_start:      float = 0.05,     # initial line-search step [rad]
    verbose:      bool  = False,
):
    """
    Minimise micromagnetic energy by CG on the unit sphere.

    Parameters
    ----------
    solver      : MicromagSolver
    m0          : (nx,ny,nz,3) initial normalised magnetisation.
    H_ext       : applied field [A/m].
    tol         : convergence criterion — max|mxHxm| [A/m].
    max_iter    : hard iteration cap.
    reset_every : CG direction reset period (Fletcher-Reeves restart).
    t_start     : initial line-search bracket step [rad].
    verbose     : print convergence info.

    Returns
    -------
    m_eq     : ndarray (nx,ny,nz,3) equilibrium state.
    info     : dict with 'converged', 'n_iter', 'max_mxHxm'.
    """
    m   = to_xp(m0).astype(xp.float64)
    H_ext_xp = xp.asarray(np.asarray(H_ext, dtype=np.float64))

    mask_bool = solver._mask.astype(bool)

    d_prev    = None
    g_sq_prev = None
    g_prev    = None     # for Polak-Ribière

    max_torque = np.inf
    n_iter     = 0

    for n_iter in range(max_iter):
        # ── gradient on sphere (mxHxm) ─────────────────────────
        H_eff = solver.H_eff(m, H_ext_xp)
        g     = _mxHxm(m, H_eff)          # (nx,ny,nz,3), tangent to sphere

        # convergence check
        g_mag = xp.linalg.norm(g[mask_bool], axis=-1)
        max_torque = float(to_np(xp.max(g_mag)))
        if max_torque < tol:
            break

        # ── Polak-Ribière beta ──────────────────────────────────
        g_sq = float(to_np(xp.sum(g[mask_bool] ** 2)))

        if d_prev is None or (n_iter % reset_every == 0) or g_sq_prev == 0.0:
            beta = 0.0
        else:
            # PR: β = Σ gₙ·(gₙ - gₙ₋₁) / Σ |gₙ₋₁|²
            dg   = g - g_prev
            beta = float(to_np(xp.sum(g[mask_bool] * dg[mask_bool]))) / g_sq_prev
            beta = max(beta, 0.0)   # PR+ (restart if negative)

        # ── CG direction (tangent space) ────────────────────────
        if beta == 0.0 or d_prev is None:
            d = g.copy()
        else:
            d = g + beta * _project_tangent(d_prev, m)

        d = _project_tangent(d, m)   # ensure tangency

        d_norm = float(to_np(xp.max(xp.linalg.norm(d[mask_bool], axis=-1))))
        if d_norm < 1e-30:
            break

        # ── line search ─────────────────────────────────────────
        t_s   = t_start / (d_norm + 1e-30)   # normalise step to ~0.05 rad
        t, m_new, E_new = _line_search(solver, m, d, H_ext_xp, H_eff,
                                       t_start=t_s)

        if t == 0.0:
            # No progress — hard reset CG direction next step
            d_prev, g_sq_prev, g_prev = None, None, None
            continue

        # ── update state ────────────────────────────────────────
        g_prev    = g
        g_sq_prev = g_sq
        d_prev    = d
        m         = m_new

    converged = max_torque < tol
    if verbose:
        status = "converged" if converged else "NOT converged"
        print(f"    CG {status}: {n_iter} iters, max|mxHxm|={max_torque:.2e} A/m")

    return to_np(m), {"converged": converged, "n_iter": n_iter,
                      "max_mxHxm": max_torque}
