"""
magnowire._backend
~~~~~~~~~~~~~~~~~~
GPU/CPU autodetection. All modules import `xp`, `_fftn`, etc. from here.
"""

import numpy as np
from numpy.fft import fftn, ifftn, rfftn, irfftn

try:
    import cupy as cp
    cp.array([1.0])          # probe allocation
    xp       = cp
    _fftn    = cp.fft.fftn
    _ifftn   = cp.fft.ifftn
    _rfftn   = cp.fft.rfftn
    _irfftn  = cp.fft.irfftn
    GPU      = True
    _dev     = cp.cuda.Device(0)
    BACKEND  = f"CuPy/GPU  (VRAM {_dev.mem_info[1] // 2**20} MB)"
except Exception:
    xp       = np
    _fftn    = fftn
    _ifftn   = ifftn
    _rfftn   = rfftn
    _irfftn  = irfftn
    GPU      = False
    BACKEND  = "NumPy/CPU"


def to_np(a) -> np.ndarray:
    """Move array from GPU to CPU if necessary."""
    if GPU and isinstance(a, __import__("cupy").ndarray):
        return a.get()
    return np.asarray(a)


def to_xp(a):
    """Move array to the active backend (GPU or CPU)."""
    return xp.asarray(a)
