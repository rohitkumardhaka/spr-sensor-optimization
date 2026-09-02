import optuna
import numpy as np
from physics import compute_reflectance, compute_fom
from materials import materials, N1, N3, N4, N5, dn
import os
# from sklearn.ensemble import RandomForestClassifier
# import random

# metal_map = {"Ag": 0, "Au": 1, "Cu": 2}

# Guarantee results folder exists
os.makedirs("results", exist_ok=True)

# ------------------------------------------
# LOAD EXISTING DATA
# ------------------------------------------
def load_existing_data():
    try:
        study = optuna.load_study(
            study_name="spr_opt",
            storage="sqlite:///results/spr_opt.db"
        )

        X, y = [], []

        for trial in study.trials:
            if trial.value is None:
                continue

            params = trial.params

            if not all(k in params for k in ["D2_nm", "D3_nm", "D4_nm", "metal"]):
                continue

            X.append([
                params["D2_nm"],
                params["D3_nm"],
                params["D4_nm"],
                metal_map[params["metal"]]
            ])

            y.append(-trial.value)

        print(f"[ML] Loaded {len(X)} samples")
        return X, y

    except Exception as e:
        print("[ML] Load error:", e)
        return [], []

# ML_X, ML_y = load_existing_data()

# # ------------------------------------------
# # TRAIN CLASSIFIER
# # ------------------------------------------
# ML_model = None

# if len(ML_X) > 100:
#     print("[ML] Training classifier...")

#     threshold = np.percentile(ML_y, 70)
#     labels = [1 if val >= threshold else 0 for val in ML_y]

#     ML_model = RandomForestClassifier(
#         n_estimators=50,
#         max_depth=8,
#         random_state=42,
#         n_jobs=-1
#     )

#     ML_model.fit(ML_X, labels)
#     print("[ML] Classifier ready")

# else:
#     print("[ML] Not enough data")

# ------------------------------------------
# EVALUATION FUNCTION
# ------------------------------------------
def evaluate(D2_nm, D3_nm, D4_nm, metal, coarse=True):
    lam = 633e-9
    N2 = materials[metal]

    D2 = D2_nm * 1e-9
    D3 = D3_nm * 1e-9
    D4 = D4_nm * 1e-9

    theta = np.arange(20, 89, 0.5 if coarse else 0.02)

    Rwo, Rw = compute_reflectance(
        theta, lam,
        N1, N2, N3, N4, N5, N5 + dn,
        D2, D3, D4
    )

    FOM = compute_fom(theta, Rwo, Rw, dn)

    if np.isnan(FOM) or np.isinf(FOM):
        return -1e9

    return FOM

# ------------------------------------------
# OBJECTIVE FUNCTION
# ------------------------------------------
def objective(trial):
    global ML_model, ML_y

    D2_nm = trial.suggest_float("D2_nm", 20, 120)
    D3_nm = trial.suggest_float("D3_nm", 0.5, 20)
    D4_nm = trial.suggest_float("D4_nm", 0.1, 2)
    metal = trial.suggest_categorical("metal", ["Ag", "Au", "Cu"])

    # ================= SAFE SOFT SKIPPING =================
    # if ML_model is not None and len(ML_y) > 50:

    #     x_input = [[
    #         D2_nm,
    #         D3_nm,
    #         D4_nm,
    #         metal_map.get(metal, 0)
    #     ]]

    #     pred = ML_model.predict(x_input)[0]

    #     # pred = 0 (bad), 1 (good)
    #     if pred == 0:
    #         if random.random() < 0.5:
    #             return -1.0   # soft penalty
    # =====================================================

    # -------- COARSE EVALUATION --------
    FOMc = evaluate(D2_nm, D3_nm, D4_nm, metal, coarse=True)

    # -------- PRUNING --------
    trial.report(-FOMc, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    # -------- FINE EVALUATION --------
    if FOMc > 1.0:
        FOMf = evaluate(D2_nm, D3_nm, D4_nm, metal, coarse=False)
        return -FOMf

    return -FOMc

# ------------------------------------------
# MAIN EXECUTION
# ------------------------------------------
if __name__ == "__main__":

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=200,
        multivariate=True,
        group=True
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=50
    )

    study = optuna.create_study(
        direction="minimize",
        study_name="spr_opt",
        storage="sqlite:///results/spr_opt.db",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner
    )

    study.optimize(objective, n_trials=50, n_jobs=8)

    print("BEST PARAMETERS:", study.best_params)
    print("BEST FOM:", -study.best_value)