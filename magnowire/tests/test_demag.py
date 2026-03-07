"""
magnowire.tests.test_demag
~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the demagnetisation field kernels.

Tests run in < 30 s on CPU.  All pass with numpy without GPU.

Run with:
    python -m pytest magnowire/tests/test_demag.py -v
or:
    python -m magnowire.tests.test_demag
"""

import numpy as np
import sys

# ── Import package ────────────────────────────────────────────────────────────
try:
    from magnowire.demag import DemagOpenBC, DemagPBC, _D6f, _D6g
    from magnowire._backend import to_np, xp
except ImportError:
    # Allow running from repo root without install
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from magnowire.demag import DemagOpenBC, DemagPBC, _D6f, _D6g
    from magnowire._backend import to_np, xp


# ── Newell tensor validation ──────────────────────────────────────────────────

def test_newell_tensor():
    """
    Check values from OOMMF source (demagcoef.cc).
    Tolerance: 1e-9 (matches OOMMF).
    """
    def nxx(x, y, z, dx, dy, dz):
        return float(_D6f(
            np.array([[[float(x)]]]), np.array([[[float(y)]]]),
            np.array([[[float(z)]]]), dx, dy, dz).ravel()[0])

    def nxy(x, y, z, dx, dy, dz):
        if x == 0 or y == 0:
            return 0.0
        return float(_D6g(
            np.array([[[float(x)]]]), np.array([[[float(y)]]]),
            np.array([[[float(z)]]]), dx, dy, dz).ravel()[0])

    tol = 1e-9
    cases = [
        ("Nxx(0,0,0,1,1,1)",    nxx(0,0,0,1,1,1),     1/3),
        ("Nxx(0,0,0,50,10,1)",  nxx(0,0,0,50,10,1),   0.021829576458713811),
        ("Nxx(1,0,0,1,1,1)",    nxx(1,0,0,1,1,1),    -0.13501718054449527),
        ("Nxy(1,1,1,1,1,1)",    nxy(1,1,1,1,1,1),    -0.016062127810508234),
        ("Nxy(1,2,3,1,2,3)",    nxy(1,2,3,1,2,3),    -0.0088226536707711039),
    ]
    all_ok = True
    for label, val, ref in cases:
        err = abs(val - ref)
        ok  = err < tol
        status = "✓" if ok else "✗"
        print(f"  {label} = {val:.10f}  ref={ref:.10f}  err={err:.1e}  {status}")
        all_ok &= ok
    assert all_ok, "Newell tensor check failed"
    print("  Newell tensor: PASSED ✓\n")


# ── T1: Uniform magnetisation → zero field ────────────────────────────────────

def test_t1_uniform_m():
    """
    PBC: uniform magnetisation → h = 0 exactly.
    div m = 0 everywhere → Δu = 0 → u = const → h = 0.
    """
    pbc = DemagPBC(16, 16, 16, 5e-9, 5e-9, 5e-9, verbose=False)
    Ms  = 8e5
    dirs = [(1,0,0), (0,1,0), (0,0,1), (1,1,1)]
    all_ok = True
    for d in dirs:
        m = np.zeros((16, 16, 16, 3))
        dn = np.array(d, float); dn /= np.linalg.norm(dn)
        m[..., 0] = dn[0]; m[..., 1] = dn[1]; m[..., 2] = dn[2]
        h = to_np(pbc(m, Ms))
        ok = np.max(np.abs(h)) < 1e-4
        status = "✓" if ok else "✗"
        print(f"  T1 m={np.round(dn,2)} → max|h|={np.max(np.abs(h)):.1e}  {status}")
        all_ok &= ok
    assert all_ok, "T1 failed: uniform m should give zero field"
    print("  T1: PASSED ✓\n")


# ── T2: Thin-film stack — analytical solution ─────────────────────────────────

def test_t2_thin_film():
    """
    PBC: infinite stack of thin films.
    Analytical solution (Bruckner 2021):
      h_in  = +d0 / (d0 + d1) * Ms
      h_out = -d1 / (d0 + d1) * Ms
    """
    d1, d0 = 2, 6
    period  = d1 + d0
    Ms      = 8e5
    pbc     = DemagPBC(4, 4, period, 5e-9, 5e-9, 3e-9, verbose=False)

    m = np.zeros((4, 4, period, 3))
    for k in range(d1):
        m[:, :, k, 2] = 1.0

    h    = to_np(pbc(m, Ms))
    hz   = np.real(h[:, :, :, 2].mean(axis=(0, 1))) / Ms
    h_in  =  d0 / (d0 + d1)
    h_out = -d1 / (d0 + d1)
    err = max(abs(hz[0] - h_in), abs(hz[d1] - h_out)) / abs(h_in)
    ok  = err < 1e-10
    print(f"  T2 h_in:  num={hz[0]:.6f}  analytic={h_in:.6f}")
    print(f"  T2 h_out: num={hz[d1]:.6f}  analytic={h_out:.6f}")
    print(f"  T2 rel_err={err:.1e}  {'✓' if ok else '✗'}")
    assert ok, f"T2 failed: relative error {err:.1e} > 1e-10"
    print("  T2: PASSED ✓\n")


# ── T3: PBC ≠ open-BC ────────────────────────────────────────────────────────

def test_t3_pbc_vs_obc():
    """
    PBC and open-BC must give *different* fields — they model different physics.
    The periodic images in PBC significantly change the demag field.
    """
    nx_m = 4; pad = 4; n = nx_m + 2*pad; d = 5e-9
    obc = DemagOpenBC(n, n, n, d, d, d, verbose=False)
    pbc = DemagPBC   (n, n, n, d, d, d, verbose=False)

    m3 = np.zeros((n, n, n, 3))
    sl = slice(pad, pad + nx_m)
    m3[sl, sl, sl, 0] = 1.0

    Ms   = 8e5
    h_ob = to_np(obc(xp.asarray(m3), Ms))
    h_pb = to_np(pbc(m3, Ms))
    diff = np.max(np.abs(h_pb[sl, sl, sl] - h_ob[sl, sl, sl]))
    ok   = diff > 0.01 * Ms
    print(f"  T3 max field diff (PBC vs OBC) = {diff/Ms:.4f} Ms  {'✓' if ok else '✗'}")
    print(f"  (Large difference is CORRECT — periodic images contribute)")
    assert ok, "T3 failed: PBC and OBC should differ significantly"
    print("  T3: PASSED ✓\n")


# ── T4: Fundamental properties ────────────────────────────────────────────────

def test_t4_properties():
    """
    a) Net flux = 0 for any M (divergence theorem).
    b) Anti-symmetry: h(-m) = -h(m).
    """
    pbc = DemagPBC(8, 10, 6, 4e-9, 5e-9, 6e-9, verbose=False)
    Ms  = 8e5

    rng = np.random.default_rng(42)
    mr  = rng.standard_normal((8, 10, 6, 3))
    mr /= np.linalg.norm(mr, axis=-1, keepdims=True)

    h  = to_np(pbc(mr, Ms))
    h2 = to_np(pbc(-mr, Ms))

    net   = np.max(np.abs(np.real(h).mean(axis=(0, 1, 2))))
    antisym = np.max(np.abs(np.real(h) + np.real(h2)))

    ok_a = net     < 1e-4
    ok_b = antisym < 1e-10
    print(f"  T4a net flux   = {net:.1e}     {'✓' if ok_a else '✗'}")
    print(f"  T4b anti-symm  = {antisym:.1e}  {'✓' if ok_b else '✗'}")
    assert ok_a, f"T4a failed: net flux {net:.1e} > 1e-4"
    assert ok_b, f"T4b failed: anti-symmetry error {antisym:.1e} > 1e-10"
    print("  T4: PASSED ✓\n")


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all():
    tests = [
        ("Newell tensor",    test_newell_tensor),
        ("T1 uniform m",     test_t1_uniform_m),
        ("T2 thin film",     test_t2_thin_film),
        ("T3 PBC vs OBC",    test_t3_pbc_vs_obc),
        ("T4 properties",    test_t4_properties),
    ]
    results = {}
    for name, fn in tests:
        print(f"── {name} " + "─"*(50 - len(name)))
        try:
            fn()
            results[name] = True
        except AssertionError as e:
            print(f"  FAILED: {e}")
            results[name] = False
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = False

    print("═" * 52)
    print("SUMMARY")
    print("═" * 52)
    all_passed = True
    for name, ok in results.items():
        flag = "PASSED ✓" if ok else "FAILED ✗"
        print(f"  {name:<30}  {flag}")
        all_passed &= ok
    print("═" * 52)
    print("ALL PASSED ✓" if all_passed else "SOME TESTS FAILED ✗")
    return all_passed


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
