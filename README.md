# SPR Sensor Optimization Project

This repository contains a research and optimization workspace for designing Surface Plasmon Resonance (SPR) biosensors using multilayer optical stacks and Bayesian optimization. The project explores several variations of the same core idea: tuning layer thicknesses and material choices to maximize the sensor response, sensitivity, and figure of merit (FOM).

The workspace includes multiple experiment variants built around the same physics model but with different optimization strategies:

- raw optimization baseline
- pruning-based optimization
- machine learning assisted optimization
- regression-based evaluation
- classifier-based filtering
- backup/reference copies of earlier runs

---

## 1. Overview

SPR sensors are widely used in biosensing because they detect tiny refractive index changes near a metal-dielectric interface. When biomolecules bind to the sensor surface, the resonance angle shifts. This shift can be measured and correlated to concentration, binding events, or molecular properties.

This project models a multilayer SPR structure and searches for the parameter set that produces the strongest resonance response. The design variables typically include:

- metal thickness `D2_nm`
- dielectric layer thickness `D3_nm`
- dielectric layer thickness `D4_nm`
- metal type (`Ag`, `Au`, `Cu`)

The optimization objective is to improve the sensor figure of merit by balancing:

- resonance angle shift
- sensitivity to refractive index change
- narrow resonance linewidth
- strong reflectance contrast

---

## 2. Problem Statement

The goal is to optimize a planar multilayer SPR sensor so that it has:

- high angular sensitivity
- narrow resonance dip
- strong signal change for small refractive-index variations
- robust design under realistic material constraints

Mathematically, the project evaluates a sensor through its reflectance profile and derives a FOM based on displacement of the resonance dip and the width of that dip.

---

## 3. Repository Structure

```text
Minor Project/
├── .git/
├── .gitignore
├── .venv/
├── README.md
├── Material Database.xlsx
├── spr_ml_test.py
├── backup/
│   ├── SPR_OPTIMIZER-main-classifier/
│   ├── SPR_OPTIMIZER-main-pruner+ML/
│   ├── SPR_OPTIMIZER-main-regression/
│   └── SPR_OPTIMIZER-main-w/
├── SPR_OPTIMIZER-main-breast/
│   └── SPR_OPTIMIZER-main/
├── SPR_OPTIMIZER-main-classifier/
│   ├── README.md
│   ├── check_db.py
│   ├── config.json
│   ├── generate_report_graphs.py
│   ├── materials.py
│   ├── optimizer.py
│   ├── physics.py
│   ├── postprocess.py
│   ├── result.pu
│   ├── result1.py
│   └── results/
├── SPR_OPTIMIZER-main-pruner/
│   ├── README.md
│   ├── check_db.py
│   ├── config.json
│   ├── materials.py
│   ├── optimizer.py
│   ├── physics.py
│   ├── postprocess.py
│   ├── result.pu
│   ├── result1.py
│   └── results/
├── SPR_OPTIMIZER-main-pruner+ML/
│   ├── README.md
│   ├── check_db.py
│   ├── config.json
│   ├── generate_report_graphs.py
│   ├── materials.py
│   ├── optimizer.py
│   ├── physics.py
│   ├── postprocess.py
│   ├── result.pu
│   ├── result1.py
│   └── results/
├── SPR_OPTIMIZER-main-raw/
│   ├── README.md
│   ├── check_db.py
│   ├── config.json
│   ├── generate_report_graphs.py
│   ├── materials.py
│   ├── optimizer.py
│   ├── physics.py
│   ├── postprocess.py
│   ├── result.pu
│   ├── result1.py
│   └── results/
├── SPR_OPTIMIZER-main-regression/
│   ├── README.md
│   ├── check_db.py
│   ├── config.json
│   ├── generate_report_graphs.py
│   ├── materials.py
│   ├── optimizer.py
│   ├── physics.py
│   ├── postprocess.py
│   ├── result.pu
│   ├── result1.py
│   └── results/
├── SPR_OPTIMIZER-main-raw.zip
├── SPR_OPTIMIZER-main.zip
└── .venv/
```

---

## 4. Project Variants

Each subfolder represents a different optimization strategy applied to the same SPR design problem.

| Variant | Purpose |
| --- | --- |
| `SPR_OPTIMIZER-main-raw` | Baseline without enhancement strategies |
| `SPR_OPTIMIZER-main-pruner` | Uses pruning to avoid poor trials |
| `SPR_OPTIMIZER-main-pruner+ML` | Combines pruning and learning-based guidance |
| `SPR_OPTIMIZER-main-regression` | Regression-based or surrogate-style optimization |
| `SPR_OPTIMIZER-main-classifier` | Classification-based filtering of promising designs |
| `SPR_OPTIMIZER-main-breast` | Specialized variant for a breast-related sensor case |
| `backup/` | Saved copies of earlier experiments |

These folders are useful for benchmarking how different optimization strategies affect convergence, performance, and runtime.

---

## 5. Core Design and Physics

### 5.1 Multilayer structure
The SPR sensor is modeled as a multilayer stack with a prism, metal layer, dielectric layers, and sensing medium. The typical stack is conceptually:

```text
Prism
  ↓
Metal layer
  ↓
Dielectric layer 1
  ↓
Dielectric layer 2
  ↓
Sensing medium
```

The resonant condition occurs when the light wave vector matches the surface plasmon condition at the metal-dielectric interface. Small changes in refractive index near the sensing surface cause a measurable angular shift.

### 5.2 Material definitions
The `materials.py` file contains the optical constants for the materials used in the stack, including:

- prism material
- metal layer (`Ag`, `Au`, `Cu`)
- dielectric layers
- sensing medium
- refractive index perturbation `dn`

### 5.3 Reflectance simulation
The `physics.py` file contains the optical simulation used to compute reflectance as a function of incident angle. It generally includes:

- complex refractive indices
- Fresnel reflection coefficients
- propagation through each film layer
- multilayer transfer-matrix or equivalent cascade logic
- reflectance curve generation

The code then evaluates parameter combinations by analyzing the resonance dip in the reflectance profile.

### 5.4 Performance metric
The sensor performance is estimated using a figure of merit (FOM), which normally captures the relationship between:

- resonance shift
- sensitivity
- linewidth (FWHM)

A high-performing design typically shows a pronounced resonance shift with a narrow and sharp resonance dip.

---

## 6. Optimization Workflow

The optimization process is built around Optuna and uses Python-based numerical simulation.

### Main workflow
1. Define the optical stack and materials
2. Generate a candidate design with random or guided parameters
3. Simulate reflectance response for the design
4. Compute sensitivity and FOM
5. Store the trial in the Optuna database
6. Repeat until a high-performance design is found

The optimizer tunes parameters like thicknesses and metal selection while evaluating candidate structures against the objective function.

---

## 7. Files Commonly Used

### `optimizer.py`
Main optimization driver for each project variant.

### `physics.py`
Optical simulation and reflectance calculations.

### `materials.py`
Material database and refractive-index constants.

### `postprocess.py`
Exports optimization results and visualizations.

### `check_db.py`
Checks the number of completed trials in the SQLite database.

### `generate_report_graphs.py`
Generates analysis plots and report-oriented visual outputs.

### `results/`
Contains generated outputs such as SQLite databases, CSV files, and plot images.

---

## 8. Running the Project

### Environment setup
From the project root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install numpy numba optuna pandas scikit-learn matplotlib plotly kaleido
```

### Run a variant
Example for the regression folder:

```bash
cd SPR_OPTIMIZER-main-regression
python optimizer.py
```

You can run any other variant similarly by switching to its directory.

---

## 9. Interpreting Results

The generated outputs typically include:

- SQLite trial database (`spr_opt.db`)
- CSV files with parameter and FOM values
- convergence plots
- parameter importance charts
- resonance dip and sensitivity visualizations

A good result is not just a high FOM, but a design that clearly demonstrates a sharp, measurable SPR response with a strong and repeatable shift under refractive index change.

---

## 10. Use Cases

This project is suitable for:

- SPR biosensor design optimization
- optical multilayer stack analysis
- comparative evaluation of optimization techniques
- Bayesian optimization experiments in optics
- research exploration of sensor performance trade-offs

---

## 11. Practical Recommendations

For a structured workflow, consider the following order:

1. Run the raw version to establish a baseline
2. Compare with the pruner version to evaluate search efficiency
3. Try the pruner + ML version for guided optimization
4. Compare with regression/classifier variants
5. Choose the setup that offers the best trade-off between runtime and sensor quality

This gives a clearer understanding of which optimization strategy is most effective for the problem.

---

## 12. Summary

This repository is a comparative SPR optimization workspace for multilayer optical sensor design. It combines electromagnetic modeling, Bayesian optimization, and performance evaluation to find high-performing sensor stacks. The multiple project folders represent different optimization strategies applied to the same fundamental design problem, making the workspace useful for analysis, experimentation, and benchmarking.

---

## 13. Quick Start

```bash
cd "D:/Minor Project"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
cd SPR_OPTIMIZER-main-regression
python optimizer.py
```

This will start the optimization process for one of the primary SPR variants in the project.

---

This README provides a project-level overview of the repository and its optimization strategy across the different SPR experiment variants.
