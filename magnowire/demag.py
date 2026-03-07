"""
magnowire.demag
~~~~~~~~~~~~~~~
Stray-field (demagnetisation field) kernels.

Two implementations:
  DemagOpenBC  — open boundary conditions, Newell 1993 kernel + FFT convolution
  DemagPBC     — true 3-D periodic boundary conditions, Bruckner 2021

References
----------
[N93]  Newell, Williams & Dunlop, J. Geophys. Res. 98, 9551 (1993)
[A15]  Abert et al., J. Magn. Magn. Mater. 387, 13 (2015)
[B21]  Bruckner et al., Sci. Rep. 11, 9202 (2021)
"""

from __future__ import annotations
import time
import numpy as np

from ._backend import xp, GPU, _rfftn, _irfftn, _fftn, _ifftn, to_np

try:
    import cupy as cp
except ImportError:
    cp = None


# ── Newell auxiliary functions (CPU only, computed once) ─────────────────────

def _f(x, y, z):
    """Auxiliary function F — Newell 1993 eq. B4.  Even in x, y, z."""
    x = np.abs(np.asarray(x, dtype=np.float64))
    y = np.abs(np.asarray(y, dtype=np.float64))
    z = np.abs(np.asarray(z, dtype=np.float64))
    x2, y2, z2 = x*x, y*y, z*z
    R = np.sqrt(x2 + y2 + z2)

    txz = x2 + z2
    txy = x2 + y2

    # Branch z > 0
    szp  = 2*(2*x2 - y2 - z2)*R
    szp -= 12*x*y*z * np.arctan2(y*z, x*R + 1e-300)
    szp += 3*y*(z2 - x2) * np.where(
        (y > 0) & (txz > 0),
        np.log1p(2*y*(y + R) / np.where(txz > 0, txz, 1.)), 0.)
    szp += 3*z*(y2 - x2) * np.where(
        txy > 0,
        np.log1p(2*z*(z + R) / np.where(txy > 0, txy, 1.)), 0.)

    # Branch z == 0
    K   = 2*np.sqrt(2.) - 6*np.log(1 + np.sqrt(2.))
    sz0 = np.where(
        x == y,
        K * x * x2,
        2*(2*x2 - y2)*R
        - 3*y*x2 * np.where(
            (y > 0) & (x > 0),
            np.log1p(2*y*(y + R) / np.where(x2 > 0, x2, 1.)), 0.))

    return np.where(R > 0, np.where(z > 0, szp, sz0) / 12., 0.)


def _g(x, y, z):
    """Auxiliary function G — Newell 1993 eq. B5.  Odd in x, y; even in z."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    sign = np.sign(x) * np.sign(y)
    x, y, z = np.abs(x), np.abs(y), np.abs(z)
    x2, y2, z2 = x*x, y*y, z*z
    R = np.sqrt(x2 + y2 + z2)

    txy = x2 + y2; tyz = y2 + z2; txz = x2 + z2

    szp  = -2*x*y*R
    szp -= z*z2 * np.arctan2(x*y,  z*R + 1e-300)
    szp -= 3*z*y2 * np.arctan2(x*z, y*R + 1e-300)
    szp -= 3*z*x2 * np.arctan2(y*z, x*R + 1e-300)
    szp += 3*x*y*z * np.where(
        txy > 0, np.log1p(2*z*(z + R) / np.where(txy > 0, txy, 1.)), 0.)
    szp += 0.5*y*(3*z2 - y2) * np.where(
        (y > 0) & (tyz > 0),
        np.log1p(2*x*(x + R) / np.where(tyz > 0, tyz, 1.)), 0.)
    szp += 0.5*x*(3*z2 - x2) * np.where(
        (x > 0) & (txz > 0),
        np.log1p(2*y*(y + R) / np.where(txz > 0, txz, 1.)), 0.)

    sz0  = -2*x*y*R
    sz0 -= 0.5*y*y2 * np.where(
        y > 0, np.log1p(2*x*(x + R) / np.where(y2 > 0, y2, 1.)), 0.)
    sz0 -= 0.5*x*x2 * np.where(
        x > 0, np.log1p(2*y*(y + R) / np.where(x2 > 0, x2, 1.)), 0.)

    return np.where(R > 0, sign * np.where(z > 0, szp, sz0) / 6., 0.)


def _D6f(x, y, z, dx, dy, dz):
    """D2x·D2y·D2z applied to f, divided by 4π dx dy dz → Nxx component."""
    f = _f
    r = (- f(x+dx, y+dy, z+dz) - f(x+dx, y-dy, z+dz)
         - f(x+dx, y-dy, z-dz) - f(x+dx, y+dy, z-dz)
         - f(x-dx, y+dy, z-dz) - f(x-dx, y+dy, z+dz)
         - f(x-dx, y-dy, z+dz) - f(x-dx, y-dy, z-dz)
         + 2*(f(x, y-dy, z-dz) + f(x, y-dy, z+dz)
            + f(x, y+dy, z+dz) + f(x, y+dy, z-dz)
            + f(x+dx, y+dy, z) + f(x+dx, y, z+dz)
            + f(x+dx, y, z-dz) + f(x+dx, y-dy, z)
            + f(x-dx, y-dy, z) + f(x-dx, y, z+dz)
            + f(x-dx, y, z-dz) + f(x-dx, y+dy, z))
         - 4*(f(x, y-dy, z) + f(x, y+dy, z) + f(x, y, z-dz)
            + f(x, y, z+dz) + f(x+dx, y, z) + f(x-dx, y, z))
         + 8*f(x, y, z))
    return r / (4 * np.pi * dx * dy * dz)


def _D6g(x, y, z, dx, dy, dz):
    """D2x·D2y·D2z applied to g, divided by 4π dx dy dz → Nxy component."""
    g = _g
    r = (- g(x-dx, y-dy, z-dz) - g(x-dx, y-dy, z+dz)
         - g(x+dx, y-dy, z+dz) - g(x+dx, y-dy, z-dz)
         - g(x+dx, y+dy, z-dz) - g(x+dx, y+dy, z+dz)
         - g(x-dx, y+dy, z+dz) - g(x-dx, y+dy, z-dz)
         + 2*(g(x, y+dy, z-dz) + g(x, y+dy, z+dz)
            + g(x, y-dy, z+dz) + g(x, y-dy, z-dz)
            + g(x-dx, y-dy, z) + g(x-dx, y+dy, z)
            + g(x-dx, y, z-dz) + g(x-dx, y, z+dz)
            + g(x+dx, y, z+dz) + g(x+dx, y, z-dz)
            + g(x+dx, y-dy, z) + g(x+dx, y+dy, z))
         - 4*(g(x-dx, y, z) + g(x+dx, y, z) + g(x, y, z+dz)
            + g(x, y, z-dz) + g(x, y-dy, z) + g(x, y+dy, z))
         + 8*g(x, y, z))
    return r / (4 * np.pi * dx * dy * dz)


# ── DemagOpenBC ───────────────────────────────────────────────────────────────

class DemagOpenBC:
    """
    Demagnetisation field with open (free) boundary conditions.

    Kernel: Newell 1993 [N93], computed once on CPU.
    Field:  H = -N * M, evaluated via FFT convolution with zero-padding.

    Parameters
    ----------
    nx, ny, nz : int   Grid dimensions.
    dx, dy, dz : float Cell sizes [m].
    verbose    : bool  Print timing info during kernel construction.
    """

    def __init__(self, nx: int, ny: int, nz: int,
                 dx: float, dy: float, dz: float,
                 verbose: bool = True):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz

        kx, ky, kz = 2*nx, 2*ny, 2*nz
        ix = np.arange(kx); iy = np.arange(ky); iz = np.arange(kz)
        px = (np.where(ix < nx, ix, ix - kx) * dx).reshape(-1, 1, 1)
        py = (np.where(iy < ny, iy, iy - ky) * dy).reshape(1, -1, 1)
        pz = (np.where(iz < nz, iz, iz - kz) * dz).reshape(1,  1, -1)

        if verbose:
            print(f"  DemagOpenBC {nx}×{ny}×{nz}: building kernel ...",
                  end=" ", flush=True)
        t0 = time.time()

        axes = (0, 1, 2)
        Kxx = np.fft.rfftn(_D6f(px, py, pz, dx, dy, dz), axes=axes)
        Kyy = np.fft.rfftn(_D6f(py, pz, px, dy, dz, dx), axes=axes)
        Kzz = np.fft.rfftn(_D6f(pz, px, py, dz, dx, dy), axes=axes)
        Kxy = np.fft.rfftn(_D6g(px, py, pz, dx, dy, dz), axes=axes)
        Kxz = np.fft.rfftn(_D6g(px, pz, py, dx, dz, dy), axes=axes)
        Kyz = np.fft.rfftn(_D6g(py, pz, px, dy, dz, dx), axes=axes)

        elapsed = time.time() - t0
        if verbose:
            print(f"{elapsed:.1f}s")

        _to = (lambda a: cp.asarray(a)) if GPU else (lambda a: a)
        self.Kxx = _to(Kxx); self.Kxy = _to(Kxy); self.Kxz = _to(Kxz)
        self.Kyy = _to(Kyy); self.Kyz = _to(Kyz); self.Kzz = _to(Kzz)

        if GPU and verbose:
            mb = 6 * Kxx.nbytes / 2**20
            print(f"  Kernel transferred to GPU ({mb:.1f} MB)")

    def __call__(self, m, Ms: float):
        """
        Compute H_demag [A/m].

        Parameters
        ----------
        m  : array (nx, ny, nz, 3), normalised magnetisation.
        Ms : float, saturation magnetisation [A/m].

        Returns
        -------
        H : array (nx, ny, nz, 3) [A/m]
        """
        nx, ny, nz = self.nx, self.ny, self.nz
        pad = [(0, nx), (0, ny), (0, nz), (0, 0)]
        M   = xp.pad(Ms * m, pad)
        axes = (0, 1, 2)
        s    = (2*nx, 2*ny, 2*nz)
        Mx = _rfftn(M[..., 0], axes=axes)
        My = _rfftn(M[..., 1], axes=axes)
        Mz = _rfftn(M[..., 2], axes=axes)
        Hx = _irfftn(-(self.Kxx*Mx + self.Kxy*My + self.Kxz*Mz),
                     s=s, axes=axes)[:nx, :ny, :nz]
        Hy = _irfftn(-(self.Kxy*Mx + self.Kyy*My + self.Kyz*Mz),
                     s=s, axes=axes)[:nx, :ny, :nz]
        Hz = _irfftn(-(self.Kxz*Mx + self.Kyz*My + self.Kzz*Mz),
                     s=s, axes=axes)[:nx, :ny, :nz]
        return xp.stack([Hx, Hy, Hz], axis=-1)


# ── DemagPBC ──────────────────────────────────────────────────────────────────

class DemagPBC:
    """
    Demagnetisation field with true 3-D periodic boundary conditions.

    Method: Bruckner et al. [B21], eq. (4)–(8).
    Solves  Δu = div m  in Fourier space → h = ∇u.

    No zero-padding, no tensor storage — O(N³ log N) per step.

    Parameters
    ----------
    nx, ny, nz : int   Grid dimensions.
    dx, dy, dz : float Cell sizes [m].
    verbose    : bool  Print init message.
    """

    def __init__(self, nx: int, ny: int, nz: int,
                 dx: float, dy: float, dz: float,
                 verbose: bool = True):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz

        kx = 2*np.pi * np.fft.fftfreq(nx)
        ky = 2*np.pi * np.fft.fftfreq(ny)
        kz = 2*np.pi * np.fft.fftfreq(nz)

        # Eigenvalues of the discrete Laplacian
        L  = ((-4/dx**2 * np.sin(kx/2)**2).reshape(-1, 1, 1)
            + (-4/dy**2 * np.sin(ky/2)**2).reshape(1, -1, 1)
            + (-4/dz**2 * np.sin(kz/2)**2).reshape(1,  1, -1))
        with np.errstate(divide="ignore", invalid="ignore"):
            Linv = np.where(L != 0, 1.0 / L, 0.)  # L[0,0,0]=0 → ũ(0,0,0)=0

        # Phase factors for discrete divergence (backward difference)
        qx = ((1 - np.exp(-1j*kx)) / dx).reshape(-1, 1, 1)
        qy = ((1 - np.exp(-1j*ky)) / dy).reshape(1, -1, 1)
        qz = ((1 - np.exp(-1j*kz)) / dz).reshape(1,  1, -1)

        # Phase factors for discrete gradient (forward difference)
        px = ((np.exp(1j*kx) - 1) / dx).reshape(-1, 1, 1)
        py = ((np.exp(1j*ky) - 1) / dy).reshape(1, -1, 1)
        pz = ((np.exp(1j*kz) - 1) / dz).reshape(1,  1, -1)

        _to = (lambda a: cp.asarray(a)) if GPU else (lambda a: a)
        self.Linv = _to(Linv)
        self.qx = _to(qx); self.qy = _to(qy); self.qz = _to(qz)
        self.px = _to(px); self.py = _to(py); self.pz = _to(pz)

        if verbose:
            print(f"  DemagPBC {nx}×{ny}×{nz}: initialised "
                  f"({'GPU' if GPU else 'CPU'})")

    def __call__(self, m, Ms: float):
        """
        Compute H_demag [A/m] with periodic boundary conditions.

        Parameters
        ----------
        m  : array (nx, ny, nz, 3), normalised magnetisation.
        Ms : float, saturation magnetisation [A/m].

        Returns
        -------
        H : array (nx, ny, nz, 3) [A/m]
        """
        m   = xp.asarray(m, dtype=xp.complex128)
        axes = (0, 1, 2)
        Mx = _fftn(Ms * m[..., 0], axes=axes)
        My = _fftn(Ms * m[..., 1], axes=axes)
        Mz = _fftn(Ms * m[..., 2], axes=axes)

        div_h = self.qx*Mx + self.qy*My + self.qz*Mz
        u_h   = self.Linv * div_h

        Hx = xp.real(_ifftn(self.px * u_h, axes=axes))
        Hy = xp.real(_ifftn(self.py * u_h, axes=axes))
        Hz = xp.real(_ifftn(self.pz * u_h, axes=axes))

        return xp.stack([Hx, Hy, Hz], axis=-1)
