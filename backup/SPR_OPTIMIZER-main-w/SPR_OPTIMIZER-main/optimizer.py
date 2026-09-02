import optuna
import numpy as np
from physics import compute_reflectance, compute_fom
from materials import materials, N1, N3, N4, N5, dn
import os

# Guarantee results folder exists
os.makedirs("results", exist_ok=True)

# ------------------------------------------
# Evaluate a single trial
# ------------------------------------------


def evaluate(D2_nm, D3_nm, D4_nm, metal, coarse=True):
    lam = 633e-9
    N2 = materials[metal]

    # nm -> meters
    D2 = D2_nm * 1e-9
    D3 = D3_nm * 1e-9
    D4 = D4_nm * 1e-9

    theta = np.arange(20, 89, 0.25 if coarse else 0.05)

    Rwo, Rw = compute_reflectance(theta, lam,
                                  N1, N2, N3, N4, N5, N5+dn,
                                  D2, D3, D4)

    FOM = compute_fom(theta, Rwo, Rw, dn)

    # safety:
    if np.isnan(FOM) or np.isinf(FOM):
        return -1e9

    return FOM

# ------------------------------------------
# Optuna objective
# ------------------------------------------


def objective(trial):

    D2 = trial.suggest_float("D2_nm", 20, 120)
    D3 = trial.suggest_float("D3_nm", 0.5, 20)
    D4 = trial.suggest_float("D4_nm", 0.1, 2)
    metal = trial.suggest_categorical("metal", ["Ag", "Au", "Cu"])

    # coarse run
    FOMc = evaluate(D2, D3, D4, metal, coarse=True)

    # refine only if promising
    if FOMc > 1.0:
        FOMf = evaluate(D2, D3, D4, metal, coarse=False)
        return -FOMf

    return -FOMc


# ------------------------------------------
# MAIN EXECUTION
# ------------------------------------------
if __name__ == "__main__":

    study = optuna.create_study(
        direction="minimize",
        study_name="spr_opt",
        storage="sqlite:///results/spr_opt.db",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=20000, n_jobs=8)

    print("BEST PARAMETERS:", study.best_params)
    print("BEST FOM:", -study.best_value)
