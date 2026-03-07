"""
magnowire.solver
~~~~~~~~~~~~~~~~
Micromagnetic finite-difference solver based on the
Landau-Lifshitz-Gilbert (LLG) equation.

Supports:
  - Open boundary conditions (DemagOpenBC)
  - True 3-D periodic boundary conditions (DemagPBC)
  - Exchange with Neumann BC at material boundaries (composite geometries)
  - Uniaxial anisotropy (global uniform or per-cell)
  - RK4 time integration
  - GPU acceleration (CuPy, auto-detected)
"""

from __future__ import annotations
import time
import numpy as np
from typing import Callable, Optional, Union

from ._backend import xp, GPU, to_np, to_xp
from .constants import MU0, GAMMA
from .demag import DemagOpenBC, DemagPBC
from .materials import Material
from .geometry import Geometry

try:
    import cupy as cp
except ImportError:
    cp = None


class MicromagSolver:
    """
    Finite-difference micromagnetic solver (LLG, RK4).

    Parameters
    ----------
    geom     : Geometry  Grid geometry (dimensions, cell sizes, mask, m0).
    material : Material  Magnetic material (Ms, A, Ku, u_axis, alpha).
    pbc      : bool      True → DemagPBC, False → DemagOpenBC.
    alpha    : float     Override Gilbert damping (defaults to material.alpha).
    verbose  : bool      Print setup info.
    """

    def __init__(
        self,
        geom:     Geometry,
        material: Material,
        pbc:      bool  = False,
        alpha:    Optional[float] = None,
        verbose:  bool  = True,
    ):
        self.geom     = geom
        self.material = material
        self.pbc      = pbc

        self.nx, self.ny, self.nz = geom.nx, geom.ny, geom.nz
        self.dx, self.dy, self.dz = geom.dx, geom.dy, geom.dz

        self.Ms    = material.Ms
        self.A     = material.A
        self.Ku    = material.Ku
        self.alpha = float(alpha if alpha is not None else material.alpha)

        self.C_exch = 2.0 * material.A / (MU0 * material.Ms)
        self.C_ani  = 2.0 * material.Ku / (MU0 * material.Ms)
        self._gam   = GAMMA * MU0 / (1.0 + self.alpha**2)

        # Easy axis: geometry wire axis overrides material default
        # (e.g. Co has u_axis=z by default but a wire along x needs u_axis=x)
        u_raw = geom.meta.get("u_axis", material.u_axis)
        u_np  = np.asarray(u_raw, dtype=np.float64)
        u_np  = u_np / np.linalg.norm(u_np)
        self._u = to_xp(u_np)

        # Material mask (GPU array, float for arithmetic)
        self._mask = to_xp(geom.mask.astype(np.float64))  # (nx,ny,nz)

        # Build demag kernel
        bc_label = "PBC" if pbc else "open-BC"
        bk_label = "GPU" if GPU else "CPU"
        l_ex = material.l_ex

        if verbose:
            print(f"\nMicromagSolver [{bc_label}, {bk_label}]")
            print(f"  Grid : {self.nx}×{self.ny}×{self.nz}  "
                  f"cell {self.dx*1e9:.2f}×{self.dy*1e9:.2f}×{self.dz*1e9:.2f} nm")
            print(f"  Mat  : {material.name}")
            print(f"  Ms={self.Ms/1e6:.3f} MA/m  A={self.A*1e12:.1f} pJ/m  "
                  f"Ku={self.Ku:.0f} J/m³  α={self.alpha}")
            print(f"  l_ex={l_ex*1e9:.2f} nm  "
                  f"cell/l_ex={self.dx/l_ex:.2f}")

        if pbc:
            self.demag = DemagPBC(
                self.nx, self.ny, self.nz,
                self.dx, self.dy, self.dz, verbose=verbose)
        else:
            self.demag = DemagOpenBC(
                self.nx, self.ny, self.nz,
                self.dx, self.dy, self.dz, verbose=verbose)

    # ── Field contributions ───────────────────────────────────────────────────

    def H_demag(self, m) -> "xp.ndarray":
        """
        Demagnetisation field.
        Automatically masks non-magnetic cells so they contribute zero flux.
        """
        return self.demag(m * self._mask[..., None], self.Ms)

    def H_exchange(self, m) -> "xp.ndarray":
        """
        Exchange field with Neumann BC at mask boundaries.
        Non-magnetic neighbours do not contribute to the Laplacian.
        """
        Lap = xp.zeros_like(m)
        for ax, d in enumerate([self.dx, self.dy, self.dz]):
            n_ax = m.shape[ax]
            if n_ax == 1:
                continue
            if self.pbc:
                mp = xp.roll(m, -1, axis=ax)
                mm = xp.roll(m, +1, axis=ax)
                kp = xp.roll(self._mask, -1, axis=ax)
                km = xp.roll(self._mask, +1, axis=ax)
            else:
                # Neumann BC: ghost cell = boundary cell
                idx_p = [slice(None)] * 4; idx_p[ax] = slice(1, n_ax)
                idx_m = [slice(None)] * 4; idx_m[ax] = slice(0, n_ax - 1)
                bd_p  = [slice(None)] * 4; bd_p[ax]  = slice(-1, None)
                bd_m  = [slice(None)] * 4; bd_m[ax]  = slice(0, 1)
                mp = xp.concatenate([m[tuple(idx_p)], m[tuple(bd_p)]], axis=ax)
                mm = xp.concatenate([m[tuple(bd_m)],  m[tuple(idx_m)]], axis=ax)

                # Mask neighbours for Neumann
                mk = self._mask
                kp_sl = [slice(None)] * 3; kp_sl[ax] = slice(1, n_ax)
                km_sl = [slice(None)] * 3; km_sl[ax] = slice(0, n_ax - 1)
                bd_pk = [slice(None)] * 3; bd_pk[ax] = slice(-1, None)
                bd_mk = [slice(None)] * 3; bd_mk[ax] = slice(0, 1)
                kp = xp.concatenate([mk[tuple(kp_sl)], mk[tuple(bd_pk)]], axis=ax)
                km = xp.concatenate([mk[tuple(bd_mk)], mk[tuple(km_sl)]], axis=ax)

            n_nb = (kp + km)[..., None]
            Lap += (kp[..., None]*mp + km[..., None]*mm - n_nb*m) / d**2

        return self.C_exch * Lap * self._mask[..., None]

    def H_anisotropy(self, m) -> "xp.ndarray":
        """Uniaxial anisotropy field (supports per-cell easy axis)."""
        if self.Ku == 0.0:
            return xp.zeros_like(m)
        u = self._u
        dot = xp.sum(m * u, axis=-1, keepdims=True)
        return self.C_ani * dot * u * self._mask[..., None]

    def H_eff(self, m, H_ext) -> "xp.ndarray":
        """Total effective field [A/m]."""
        H = self.H_demag(m)
        H = H + self.H_exchange(m)
        H = H + self.H_anisotropy(m)
        H = H + xp.asarray(H_ext, dtype=xp.float64)
        return H

    # ── LLG integrator ───────────────────────────────────────────────────────

    @staticmethod
    def _normalize(m):
        return m / xp.maximum(xp.linalg.norm(m, axis=-1, keepdims=True), 1e-30)

    def _dm_dt(self, m, H_ext):
        """Landau-Lifshitz form: dm/dt = -γ/(1+α²)[m×H + α m×(m×H)]."""
        H   = self.H_eff(m, H_ext)
        mxH = xp.cross(m, H)
        return -self._gam * (mxH + self.alpha * xp.cross(m, mxH))

    def rk4_step(self, m, H_ext, dt: float):
        """Single RK4 step."""
        k1 = self._dm_dt(m,              H_ext)
        k2 = self._dm_dt(m + 0.5*dt*k1, H_ext)
        k3 = self._dm_dt(m + 0.5*dt*k2, H_ext)
        k4 = self._dm_dt(m +     dt*k3, H_ext)
        return self._normalize(m + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4))

    # ── High-level integration ────────────────────────────────────────────────

    def run(
        self,
        m0,
        t_end:      float,
        dt:         float,
        H_ext:      Union[np.ndarray, Callable],
        save_every: int  = 50,
        verbose:    bool = True,
    ):
        """
        Integrate LLG from t=0 to t=t_end.

        Parameters
        ----------
        m0         : (nx,ny,nz,3) initial normalised magnetisation.
        t_end      : float  End time [s].
        dt         : float  Time step [s].
        H_ext      : array [A/m] or callable t → array [A/m].
        save_every : int    Steps between saved averages.
        verbose    : bool   Print progress.

        Returns
        -------
        t_arr  : ndarray (n_save,)      Saved times [s].
        M_avg  : ndarray (n_save, 3)    Volume-averaged <m>.
        m_final: ndarray (nx,ny,nz,3)   Final state (CPU).
        """
        m   = to_xp(m0).astype(xp.float64)
        Hfn = H_ext if callable(H_ext) else (lambda t, _H=np.asarray(H_ext): _H)

        n_steps = int(round(t_end / dt))
        t_arr   = [0.0]
        M_avg   = [to_np(m[self.geom.mask].mean(axis=0))]

        t0  = time.time()
        log = max(1, n_steps // 10)

        if verbose:
            print(f"  RK4: {n_steps} steps, dt={dt*1e12:.1f} ps, "
                  f"t_end={t_end*1e9:.3f} ns")

        for i in range(n_steps):
            m = self.rk4_step(m, Hfn(i * dt), dt)
            if (i + 1) % save_every == 0 or i == n_steps - 1:
                avg = to_np(m[self.geom.mask].mean(axis=0))
                t_arr.append((i + 1) * dt)
                M_avg.append(avg)
            if verbose and (i + 1) % log == 0:
                elapsed = time.time() - t0
                eta     = elapsed / (i + 1) * (n_steps - i - 1)
                print(f"    step {i+1}/{n_steps}  "
                      f"t={(i+1)*dt*1e9:.3f} ns  "
                      f"<m>=({M_avg[-1][0]:+.3f},{M_avg[-1][1]:+.3f})  "
                      f"ETA {eta:.0f}s")

        if verbose:
            print(f"  Done in {time.time()-t0:.1f}s")

        return np.array(t_arr), np.array(M_avg), to_np(m)

    def relax(
        self,
        m0,
        t_relax: float = 2e-9,
        dt:      float = 5e-13,
        H_ext:   Optional[np.ndarray] = None,
        verbose: bool  = True,
    ):
        """
        Relax to equilibrium using α=1 (maximum damping).

        Parameters
        ----------
        m0      : Initial magnetisation.
        t_relax : Relaxation duration [s].
        dt      : Time step [s].
        H_ext   : Optional applied field [A/m].
        verbose : Print progress.

        Returns
        -------
        m_eq : ndarray (nx,ny,nz,3) equilibrium state (CPU).
        """
        if H_ext is None:
            H_ext = np.zeros(3)

        alpha0, gam0  = self.alpha, self._gam
        self.alpha    = 1.0
        self._gam     = GAMMA * MU0 / 2.0

        if verbose:
            print(f"  Relaxing (α=1), t={t_relax*1e9:.2f} ns ...")

        _, _, m_eq = self.run(
            m0, t_relax, dt, H_ext,
            save_every=max(1, int(t_relax / dt) // 5),
            verbose=verbose,
        )

        self.alpha = alpha0
        self._gam  = gam0

        return m_eq

    # ── Energy densities ──────────────────────────────────────────────────────

    def energy_density(self, m) -> dict:
        """
        Compute energy densities [J/m³].

        Returns dict with keys: 'demag', 'exchange', 'anisotropy', 'total'.
        """
        m_xp  = to_xp(m).astype(xp.float64)
        vol   = float(self.geom.mask.sum()) * self.geom.cell_volume

        def _dot(H_fn):
            H  = H_fn(m_xp)
            return float(to_np(xp.sum(H * m_xp))) * self.geom.cell_volume

        E_demag = -0.5 * MU0 * self.Ms * _dot(self.H_demag) / vol
        E_exch  = -0.5 * MU0 * self.Ms * _dot(self.H_exchange) / vol
        E_ani   = -0.5 * MU0 * self.Ms * _dot(self.H_anisotropy) / vol

        return {
            "demag":      E_demag,
            "exchange":   E_exch,
            "anisotropy": E_ani,
            "total":      E_demag + E_exch + E_ani,
        }
