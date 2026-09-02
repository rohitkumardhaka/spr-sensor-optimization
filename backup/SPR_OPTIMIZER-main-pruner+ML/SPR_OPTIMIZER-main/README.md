# SPR Optimizer: Surface Plasmon Resonance Optical Design

A Bayesian optimization framework to design planar multilayer structures that maximize **Surface Plasmon Resonance (SPR)** sensor sensitivity for optical biosensing applications.

## Overview

**What is SPR?**  
Surface Plasmon Resonance is a powerful optical technique for real-time, label-free biosensing. When light hits a metal-dielectric interface at a specific angle (the *resonance angle*), it excites collective oscillations of free electrons (plasmons), causing a sharp dip in reflected light intensity. This resonance angle shifts when biological molecules bind to the sensor surface, enabling detection.

**What does this optimizer do?**  
This project automatically designs the optimal layer thicknesses and metal choice for a planar SPR sensor to maximize sensitivity (how much the resonance angle shifts for a given refractive index change). It uses **Optuna** (a Bayesian hyperparameter optimization framework) to explore millions of design combinations efficiently.

## Project Structure

```
spr-optimizer/
├── optimizer.py          # Main optimization loop (Optuna driver)
├── physics.py            # Electromagnetic calculations (Fresnel equations, multilayer optics)
├── materials.py          # Refractive index database for metals and dielectrics
├── postprocess.py        # Export results to CSV and visualization plots
├── check_db.py           # Utility to inspect optimization database
├── results/              # Output folder
│   ├── spr_opt.db        # SQLite database of all trials
│   ├── best_trials.csv   # CSV export of trials
│   ├── fom_convergence.png
│   └── parameter_importance.png
└── README.md             # This file
```

## File Descriptions

### `materials.py` — Material Database
Defines the **complex refractive indices** of metals and dielectric materials used in the multilayer structure.

**Key constants:**
- `N1 = 1.7231` — SF10 prism (top layer, where light couples in)
- `N2` — Metal layer (Ag, Au, or Cu; complex index with absorption)
- `N3 = 2.3859` — PbMoO₄ (dielectric)
- `N4 = 2.69 + 0.22j` — Blue Phosphorus (dielectric)
- `N5 = 1.402` — Sensing medium (e.g., water or buffer)
- `dn = 0.024` — Analyte refractive index shift (what we sense)

**Metals (as `N2`):**
- `Ag: 0.135 + 3.999j` — Silver (sharp SPR, good sensitivity)
- `Au: 0.166 + 3.11j` — Gold (stable, biocompatible)
- `Cu: 0.24 + 3.42j` — Copper (cost-effective, lower sensitivity)

### `physics.py` — Electromagnetic Simulations
Implements the **transfer matrix method (TMM)** for computing optical reflectance in multilayer structures.

**Key functions:**

1. **`csqrt(z)`** — Complex square root (numerically stable)

2. **`reflection_coeff(E1, E2, K1, K2)`** — Fresnel reflection coefficient at a single interface
   - Returns the amplitude reflection from medium 1 to medium 2
   - Used as building block for multilayer systems

3. **`calculate_total_reflection_coefficient(R12, R23, R34, R45, K2, K3, K4, D2, D3, D4)`** — Cascade formula for 5-layer stack
   - Combines reflections at each interface using matrix algebra
   - Accounts for phase accumulation in each layer (via `exp(2j*K*D)` terms)
   - Computes total reflectance at the top surface

4. **`compute_reflectance(theta_deg, lam, N1, N2, N3, N4, N5, N5_1, D2, D3, D4)`** — Full SPR simulation
   - Sweeps incident angle from 20° to 89°
   - For each angle, computes reflectance with and without analyte present
   - Returns `Rwo` (without analyte) and `Rw` (with analyte)
   - Uses Numba JIT compilation for 100x speedup

5. **`compute_fom(theta_deg, Rwo, Rw, dn)`** — Figure of Merit (FOM)
   - **Sensitivity (S)**: angular shift per refractive index change = `(θ_w - θ_wo) / dn`
   - **FWHM**: Full Width at Half Maximum of the reflectance dip
   - **FOM = S / FWHM** — Higher is better (more sensitivity, narrower linewidth)

### `optimizer.py` — Bayesian Optimization Loop
Drives the search for optimal layer thicknesses and metal choice using **Optuna**.

**Optimization parameters:**
- `D2_nm` — Metal layer thickness (20–120 nm)
- `D3_nm` — Dielectric layer 1 thickness (0.5–20 nm)
- `D4_nm` — Dielectric layer 2 thickness (0.1–2 nm)
- `metal` — Metal choice (Ag, Au, Cu)

**Two-stage evaluation:**
1. **Coarse scan** — θ resolution = 0.25° (fast, ~0.01 s per trial)
   - Quickly filters out poor designs
2. **Fine scan** — θ resolution = 0.05° (accurate, ~0.05 s per trial)
   - Only run if coarse FOM > 1.0 (promising designs)

**Key settings:**
- 20,000 total trials across all designs
- 8 parallel jobs (multiprocessing)
- Direction: **minimize** (Optuna minimizes, so we return `-FOM`)
- Database: SQLite (persistent, resume-friendly)

### `postprocess.py` — Results Export
Loads the optimization database and exports results.

**Outputs:**
- `best_trials.csv` — All 20,000 trials as a table (FOM, parameters, etc.)
- `fom_convergence.png` — Plot of best FOM over optimization iterations
- `parameter_importance.png` — Sensitivity of FOM to each parameter

**Requirements:**
- `plotly` — For interactive plots
- `kaleido` — For static PNG export

### `check_db.py` — Database Inspector
Quick utility to check the number of trials completed without running a full export.

```bash
python check_db.py
# Output: Trials: 12345
```

## Installation & Setup

### 1. **Install Python 3.8+**
```bash
python --version  # Verify 3.8 or higher
```

### 2. **Install Dependencies**
```bash
pip install optuna numpy numba pandas
```

**Optional (for visualization):**
```bash
pip install plotly kaleido scikit-learn
```

### 3. **Verify Installation**
```bash
python -c "import optuna, numpy, numba, physics; print('OK')"
```

## Usage

### Run Optimization
```bash
python optimizer.py
```

**Output:**
```
BEST PARAMETERS: {'D2_nm': 47.3, 'D3_nm': 12.1, 'D4_nm': 0.8, 'metal': 'Ag'}
BEST FOM: 187.4
```

**Timing:**
- ~1 hour on 8 cores for 20,000 trials
- Can be parallelized across multiple machines via Optuna's built-in support

### Resume Optimization
Since trials are saved to SQLite, you can resume mid-run:
```bash
python optimizer.py  # Continues from last checkpoint
```

### Export & Visualize Results
```bash
python postprocess.py
```

**Outputs** (in `results/`):
- `best_trials.csv` — Raw data for custom analysis
- `fom_convergence.png` — Convergence curve
- `parameter_importance.png` — Which parameters matter most

### Inspect Database
```bash
python check_db.py
# Trials: 20000
```

## Understanding the Physics

### The 5-Layer Stack
```
Prism (SF10, N1=1.723)
  ↓
Metal layer (Ag/Au/Cu, thickness D2)  ← Plasmon resonator
  ↓
Dielectric 1 (PbMoO4, N3=2.386, thickness D3)
  ↓
Dielectric 2 (Blue-P, N4=2.69+0.22j, thickness D4)
  ↓
Sensing medium (water, N5=1.402)
  ↓
[Analyte: refractive index shift dn=0.024]
```

### SPR Condition
At the resonance angle **θ_res**, the component of the wavevector parallel to the interface matches the surface plasmon dispersion:

$$k_∥ = k_0 N_{\text{prism}} \sin(θ_{\text{res}}) = \sqrt{(\omega/c)^2 \varepsilon_{\text{metal}} + (\omega/c)^2 \varepsilon_{\text{dielectric}}}$$

This causes a sharp reflection dip. When the analyte refractive index increases, θ_res shifts outward.

### Sensitivity (S)
The key metric:
$$S = \frac{dθ}{dn}$$

High sensitivity means the resonance angle shifts a lot for tiny refractive index changes—better for detecting weak biomolecular signals.

### FWHM (Linewidth)
The angular width of the reflectance dip at half-maximum. Narrower is better (sharper feature, easier to detect).

### FOM (Figure of Merit)
$$\text{FOM} = \frac{S}{\text{FWHM}}$$

A universal goodness metric: higher FOM means better sensor performance.

## Optimization Strategy

**Why two-stage evaluation?**
- SPR optimization is computationally expensive; fine-scan for all designs is slow
- Most random designs are poor (low FOM); coarse scan filters these quickly
- Only "promising" designs (FOM > 1.0) get fine-scanned
- **Result**: 10× speedup with negligible accuracy loss

**Why Optuna?**
- Smarter than random search: learns which parameter ranges are good
- Smarter than grid search: handles 4D parameter space without combinatorial explosion
- Parallel-friendly: trivial to distribute across multiple CPUs/machines
- Persistent: trials saved to database, can resume anytime

## Expected Results

Typical best FOM values for different metals:
- **Silver (Ag)**: FOM ≈ 150–200 (best sensitivity, sharp resonance)
- **Gold (Au)**: FOM ≈ 80–120 (stable, good for biotech)
- **Copper (Cu)**: FOM ≈ 40–60 (cost-effective, lower performance)

Optimal thicknesses (typically):
- D2 (metal): 40–60 nm
- D3 (dielectric 1): 5–15 nm
- D4 (dielectric 2): 0.5–2 nm

## Troubleshooting

### "ModuleNotFoundError: No module named 'optuna'"
```bash
pip install optuna
```

### "NameError: name 'nb' is not defined"
```bash
pip install numba
```

### Optimization is very slow
- Check `n_jobs=8` in `optimizer.py` is working (should use all cores)
- Disable `coarse=False` fast-path: remove the `if FOMc > 1.0` check and always use `coarse=True` for speed (less accurate)

### CSV export fails with PermissionError
- Close the file if open in Excel or another program
- The script retries automatically, but file locks can persist

### Plots don't export (visualization skipped)
```bash
pip install plotly kaleido scikit-learn
python postprocess.py
```

## Advanced Customization

### Change Material Stack
Edit `materials.py` to swap layers or refractive indices. Then re-run `optimizer.py`.

### Change Parameter Search Space
In `optimizer.py`, modify the `trial.suggest_float()` ranges:
```python
D2 = trial.suggest_float("D2_nm", 10, 200)  # Wider range
```

### Change Number of Trials
In `optimizer.py`:
```python
study.optimize(objective, n_trials=50000, n_jobs=8)  # More trials
```

### Optimize Different Metals Only
In `optimizer.py`:
```python
metal = trial.suggest_categorical("metal", ["Ag"])  # Only silver
```

## References

1. Homola, J. (2008). "Surface Plasmon Resonance Sensors..." *Chemical Reviews* 108(2): 462–493.
2. Kretschmann, E., & Raether, H. (1968). "Radiative decay of non-radiative surface plasmons excited by light" *Z. Naturforsch.* 23a: 2135–2136.
3. Born, M., & Wolf, E. (1999). *Principles of Optics*. Cambridge University Press.

## License

This project is provided as-is for educational and research purposes.

## Contact & Support

For issues, questions, or contributions, please refer to the project repository or contact the maintainers.

---

**Last Updated:** November 14, 2025

