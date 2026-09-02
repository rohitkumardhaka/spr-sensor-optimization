import optuna
import numpy as np
from physics import compute_reflectance, compute_fom
from materials import materials, N1, N3, N4, N5, dn
import os
import random

metal_map = {"Ag": 0, "Au": 1, "Cu": 2}

# ================= ML DATA LOADING =================


def load_existing_data():
    try:
        study = optuna.load_study(
            study_name="spr_opt",   # make sure this matches your study name
            storage="sqlite:///results/spr_opt.db"
        )

        X = []
        y = []

        for trial in study.trials:
            if trial.value is None:
                continue

            params = trial.params

            # Ensure all required params exist
            if not all(k in params for k in ["D2_nm", "D3_nm", "D4_nm", "metal"]):
                continue

            # Convert metal to numeric
            metal_map = {"Ag": 0, "Au": 1, "Cu": 2}

            X.append([
                params["D2_nm"],
                params["D3_nm"],
                params["D4_nm"],
                metal_map[params["metal"]]
            ])

            # Convert back to positive FOM
            y.append(-trial.value)

        print(f"[ML] Loaded {len(X)} samples from database")

        return X, y

    except Exception as e:
        print("[ML] Error loading data:", e)
        return [], []

# Load data once at startup
ML_X, ML_y = load_existing_data()
# ==================================================

# ================= ML MODEL TRAINING =================
from sklearn.ensemble import RandomForestRegressor
ML_model = None

if len(ML_X) > 50:
    print("[ML] Training model...")

    ML_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

    # ================= FILTER BEST DATA =================
    threshold = np.percentile(ML_y, 70)

    X_filtered = [
        x for x, y_val in zip(ML_X, ML_y) if y_val >= threshold
    ]

    y_filtered = [
        y_val for y_val in ML_y if y_val >= threshold
    ]

    if len(X_filtered) > 20:
        ML_model.fit(X_filtered, y_filtered)
        print(f"[ML] Trained on {len(X_filtered)} BEST samples")
    else:
        ML_model.fit(ML_X, ML_y)
        print("[ML] Fallback to full data")

    print("[ML] Model trained successfully")

else:
    print("[ML] Not enough data to train model")
# ====================================================

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

    global ML_X, ML_y, ML_model

    # ================= SMART PARAMETER SUGGESTION =================

    if ML_model is not None and len(ML_X) > 100 and random.random() < 0.7:
        # Find top 20% best data
        threshold = np.percentile(ML_y, 80)

        good_samples = [
            x for x, y_val in zip(ML_X, ML_y) if y_val >= threshold
        ]

        if len(good_samples) > 10:
            good_samples = np.array(good_samples)

            # Sample around best region (mean + small variation)
            mean_vals = np.mean(good_samples, axis=0)
            std_vals = np.std(good_samples, axis=0)

            D2_nm = trial.suggest_float(
                "D2_nm",
                max(20, mean_vals[0] - std_vals[0]),
                min(120, mean_vals[0] + std_vals[0])
            )

            D3_nm = trial.suggest_float(
                "D3_nm",
                max(0.5, mean_vals[1] - std_vals[1]),
                min(20, mean_vals[1] + std_vals[1])
            )

            D4_nm = trial.suggest_float(
                "D4_nm",
                max(0.1, mean_vals[2] - std_vals[2]),
                min(2, mean_vals[2] + std_vals[2])
            )

            # ✅ FIXED METAL BLOCK
            if random.random() < 0.2:
                metal = random.choice(["Ag", "Au", "Cu"])
            else:
                metal_index = int(round(mean_vals[3]))
                metal = ["Ag", "Au", "Cu"][min(max(metal_index, 0), 2)]

        else:
            # fallback
            D2_nm = trial.suggest_float("D2_nm", 20, 120)
            D3_nm = trial.suggest_float("D3_nm", 0.5, 20)
            D4_nm = trial.suggest_float("D4_nm", 0.1, 2)
            metal = trial.suggest_categorical("metal", ["Ag", "Au", "Cu"])

    else:
        # fallback (no ML)
        D2_nm = trial.suggest_float("D2_nm", 20, 120)
        D3_nm = trial.suggest_float("D3_nm", 0.5, 20)
        D4_nm = trial.suggest_float("D4_nm", 0.1, 2)
        metal = trial.suggest_categorical("metal", ["Ag", "Au", "Cu"])

    # ================= ML PREDICTION =================
    if ML_model is not None:

        x_input = [[
            D2_nm,
            D3_nm,
            D4_nm,
            metal_map.get(metal, 0)
        ]]

        pred_fom = ML_model.predict(x_input)[0]

        # Soft filtering (safe threshold)
        if ML_y:
            threshold = np.percentile(ML_y, 20)
        else:
            threshold = 20

        if pred_fom < threshold and random.random() < 0.5:
         return -1.0
    # =================================================

    # ================= ORIGINAL SIMULATION =================
    FOMc = evaluate(D2_nm, D3_nm, D4_nm, metal, coarse=True)
    # ======================================================

    # ================= ORIGINAL LOGIC (UNCHANGED) =================
    if FOMc > 1.0:
        FOMf = evaluate(D2_nm, D3_nm, D4_nm, metal, coarse=False)
        return -FOMf
    else:
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

    study.optimize(objective, n_trials=20, n_jobs=4)

    print("BEST PARAMETERS:", study.best_params)
    print("BEST FOM:", -study.best_value)
