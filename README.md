# magnowire

Micromagnetic finite-difference solver for magnetic nanowire arrays.

**Motivation**: Arrays of magnetic nanowires are a candidate replacement for
sintered NdFeB magnets — cheaper to fabricate and more thermally stable.
This package enables systematic study of how wire geometry and inter-wire
spacing affect coercivity, remanence, and energy product.

## Features

| Feature | Detail |
|---|---|
| **Boundary conditions** | Open (OBC) and true 3-D periodic (PBC) |
| **Demag kernel** | Newell 1993, OOMMF-validated |
| **PBC method** | Bruckner et al., *Sci. Rep.* **11**, 9202 (2021) |
| **Integrator** | LLG / RK4, normalised magnetisation |
| **GPU support** | CuPy (auto-detected, falls back to NumPy) |
| **Materials** | Co, FeCo, Fe, Ni, Permalloy, NdFeB + user-defined |
| **Geometry** | Cylindrical nanowire, square/hexagonal array |
| **Analysis** | Hc, Mr, squareness, BHmax extraction |

## Installation

```bash
git clone https://github.com/yourname/magnowire
cd magnowire
pip install -e .          # CPU only
pip install -e ".[gpu]"   # + CuPy for CUDA 12
```

## Quick start

```python
import magnowire as mw

# 1. Material
mat = mw.materials.Co

# 2. Geometry — single Co nanowire, ∅20 nm × 200 nm
geom = mw.geometry.nanowire(diameter_nm=20, length_nm=200, cell_nm=2)

# 3. Solver (open BC for single wire)
sim = mw.solver.MicromagSolver(geom, mat, pbc=False)

# 4. Hysteresis loop
loop = mw.hysteresis.hysteresis_loop(sim, B_max=1.0, n_field=21)

# 5. Figures of merit
metrics = mw.analysis.extract_metrics(loop, Ms_material=mat.Ms)
print(metrics)

# 6. Plot
mw.analysis.plot_loop(loop, metrics, title="Co nanowire ∅20 nm")
```

## Examples

| Notebook | Description |
|---|---|
| `examples/01_single_wire_Co.ipynb` | Single Co nanowire: SW validation, aspect ratio sweep |

## Physical model

### LLG equation

$$\frac{d\mathbf{m}}{dt} = -\frac{\gamma}{1+\alpha^2}
\left[\mathbf{m} \times \mathbf{H}_{\rm eff}
+ \alpha\, \mathbf{m} \times (\mathbf{m} \times \mathbf{H}_{\rm eff})\right]$$

### Effective field

$$\mathbf{H}_{\rm eff} = \mathbf{H}_{\rm demag} + \mathbf{H}_{\rm exch}
+ \mathbf{H}_{\rm anis} + \mathbf{H}_{\rm ext}$$

### Open-BC demagnetisation field

Computed via Newell 1993 tensor + FFT convolution with zero-padding.

### PBC demagnetisation field (Bruckner 2021)

Solves $\Delta u = \nabla \cdot \mathbf{m}$ in Fourier space:

$$\tilde{u}_{l,m,n} = \frac{\widetilde{\nabla \cdot \mathbf{m}}_{l,m,n}}{\lambda_{l,m,n}}, \qquad
\tilde{\mathbf{h}} = \nabla \tilde{u}$$

No zero-padding, no tensor storage — O(N³ log N) per step.

## Tests

```bash
pytest                                 # all tests
python magnowire/tests/test_demag.py  # standalone (no pytest needed)
```

Tests complete in < 30 s on CPU and validate:

- **Newell tensor** against OOMMF reference values
- **T1** Uniform magnetisation → zero field
- **T2** Thin-film stack → analytical Bruckner result
- **T3** PBC ≠ open-BC (correct physical difference)
- **T4** Net flux = 0, anti-symmetry

## References

1. Newell, Williams & Dunlop, *J. Geophys. Res.* **98**, 9551 (1993)
2. Abert et al., *J. Magn. Magn. Mater.* **387**, 13 (2015)
3. Bruckner et al., *Sci. Rep.* **11**, 9202 (2021)
4. Stoner & Wohlfarth, *Phil. Trans. R. Soc.* **240**, 599 (1948)

## Roadmap

- [ ] Nanowire array example (PBC, square/hex)
- [ ] Material comparison sweep (Co vs FeCo vs Fe)
- [ ] Switching mechanism visualisation (coherent vs curling)
- [ ] Parameter optimisation (BHmax vs diameter, pitch)
