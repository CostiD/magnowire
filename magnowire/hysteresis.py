"""
magnowire.hysteresis
~~~~~~~~~~~~~~~~~~~~
Quasi-static hysteresis loop protocol.

The field is swept step-by-step; at each field value the system is relaxed
to (near) equilibrium before recording <M>.  This mirrors the approach in
Bruckner et al. (2021) and standard experimental VSM/AGM measurements.
"""

from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from ._backend import xp, GPU, to_np, to_xp
from .constants import MU0, GAMMA
from .solver import MicromagSolver


@dataclass
class HysteresisResult:
    """
    Raw output of a single hysteresis sweep.

    Attributes
    ----------
    B_applied : ndarray (n,)    Applied field  μ₀H [T].
    Mx        : ndarray (n,)    <Mx>/Ms along sweep.
    My        : ndarray (n,)    <My>/Ms.
    Mz        : ndarray (n,)    <Mz>/Ms.
    m_states  : list of ndarray Magnetisation snapshots (optional).
    meta      : dict            Sweep parameters.
    """
    B_applied: np.ndarray
    Mx:        np.ndarray
    My:        np.ndarray
    Mz:        np.ndarray
    m_states:  list  = field(default_factory=list, repr=False)
    meta:      dict  = field(default_factory=dict)


def hysteresis_loop(
    solver:         MicromagSolver,
    B_max:          float          = 100e-3,
    n_field:        int            = 21,
    field_axis:     int            = 0,
    t_relax_ps:     float          = 2000.0,
    dt:             float          = 1e-12,
    B_min_relax_mT: float          = 3.0,
    max_relax_factor: float        = 4.0,
    save_states:    bool           = False,
    verbose:        bool           = True,
) -> HysteresisResult:
    """
    Run a full quasi-static hysteresis loop.

    The field is swept  +B_max → -B_max → +B_max along `field_axis`.
    At each field point the solver is advanced for an adaptive number of
    RK4 steps (at least `t_relax_ps` ps; more near H=0 where τ is long).

    Parameters
    ----------
    solver          : MicromagSolver  Pre-built solver (already at +B_max state).
    B_max           : float  Maximum applied field [T].
    n_field         : int    Field points per half-sweep (total = 2n-1).
    field_axis      : int    0=x, 1=y, 2=z.
    t_relax_ps      : float  Minimum relaxation per field point [ps].
    dt              : float  RK4 time step [s].
    B_min_relax_mT  : float  Effective minimum field for τ calculation [mT].
    max_relax_factor: float  Cap on adaptive n_steps (× t_relax).
    save_states     : bool   If True, save m snapshot at each field point.
    verbose         : bool   Print progress.

    Returns
    -------
    HysteresisResult
    """
    # ── Field array ──────────────────────────────────────────────
    B1      = np.linspace( B_max, -B_max, n_field)
    B2      = np.linspace(-B_max,  B_max, n_field)
    B_sweep = np.concatenate([B1, B2[1:]])

    n_base  = int(round(t_relax_ps * 1e-12 / dt))
    B_floor = B_min_relax_mT * 1e-3   # T

    if verbose:
        print(f"\n  Hysteresis loop: {len(B_sweep)} field points")
        print(f"  B_max={B_max*1e3:.0f}mT  n_field={n_field}  "
              f"dt={dt*1e12:.1f}ps  t_relax≥{t_relax_ps:.0f}ps")

    # ── Pre-saturate at +B_max ────────────────────────────────────
    m_cur = to_xp(solver.geom.m0).astype(xp.float64)
    H_init = _field_vec(B_sweep[0], field_axis)

    if verbose:
        print(f"  Pre-saturating at +{B_max*1e3:.0f} mT ...")

    for _ in range(n_base * 3):
        m_cur = solver.rk4_step(m_cur, H_init, dt)

    _check_saturation(m_cur, solver, verbose)

    # ── Main sweep ───────────────────────────────────────────────
    Mx_arr = []; My_arr = []; Mz_arr = []
    states = []
    t0 = time.time()

    for i_f, B_val in enumerate(B_sweep):
        H_ext = _field_vec(B_val, field_axis)

        # Adaptive relaxation steps
        abs_B   = max(abs(B_val), B_floor)
        tau     = 1.0 / (1.0 * GAMMA * abs_B)   # s  (α=1 relaxation time)
        n_steps = max(n_base, int(5 * tau / dt))
        n_steps = min(n_steps, int(n_base * max_relax_factor))

        for _ in range(n_steps):
            m_cur = solver.rk4_step(m_cur, H_ext, dt)

        m_np = to_np(m_cur)
        mask = solver.geom.mask
        avg  = m_np[mask].mean(axis=0)
        Mx_arr.append(avg[0]); My_arr.append(avg[1]); Mz_arr.append(avg[2])

        if save_states:
            states.append(m_np.copy())

        if verbose and ((i_f + 1) % max(1, len(B_sweep)//10) == 0 or i_f == 0):
            elapsed = time.time() - t0
            eta = elapsed / (i_f + 1) * (len(B_sweep) - i_f - 1) if i_f > 0 else 0
            print(f"    [{i_f+1:3d}/{len(B_sweep)}]  "
                  f"μ₀H={B_val*1e3:+7.1f} mT  "
                  f"⟨Mx⟩/Ms={avg[0]:+.4f}  "
                  f"steps={n_steps}  ETA {eta:.0f}s")

    if verbose:
        print(f"  Done in {time.time()-t0:.1f}s")

    return HysteresisResult(
        B_applied = B_sweep,
        Mx        = np.array(Mx_arr),
        My        = np.array(My_arr),
        Mz        = np.array(Mz_arr),
        m_states  = states,
        meta      = dict(
            B_max=B_max, n_field=n_field, field_axis=field_axis,
            t_relax_ps=t_relax_ps, dt=dt,
        ),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _field_vec(B: float, axis: int) -> np.ndarray:
    from .constants import MU0
    H = np.zeros(3)
    H[axis] = B / MU0
    return H


def _check_saturation(m_cur, solver: MicromagSolver, verbose: bool):
    m_np = to_np(m_cur)
    mask = solver.geom.mask
    Mx   = m_np[mask, 0].mean()
    ok   = Mx > 0.85
    if verbose:
        flag = "✓" if ok else "⚠ LOW — increase B_max or t_relax"
        print(f"  After pre-saturation: ⟨Mx⟩/Ms = {Mx:+.4f}  {flag}")
    if not ok:
        import warnings
        warnings.warn(
            f"Pre-saturation reached only ⟨Mx⟩/Ms={Mx:.3f} < 0.85. "
            "The demagnetisation field may exceed B_max. "
            "Increase B_max or use a higher fill-fraction geometry.",
            UserWarning, stacklevel=3,
        )
