import numpy as np
import matplotlib.pyplot as plt
import optuna
import os

from physics import compute_reflectance
from materials import materials, N1, N3, N4


# ==============================
# BREAST CANCER (MDA-MB-231)
# ==============================
N_normal = 1.385
N_cancer = 1.399
dn = N_cancer - N_normal


# ==============================
# CREATE RESULTS DIRECTORY
# ==============================

os.makedirs("results/report_graphs", exist_ok=True)


# ==============================
# LOAD BEST CONFIGURATION
# ==============================

study = optuna.load_study(
    study_name="spr_opt",
    storage="sqlite:///results/spr_opt.db"
)

best = study.best_params

metal = best["metal"]
D2_nm = best["D2_nm"]
D3_nm = best["D3_nm"]
D4_nm = best["D4_nm"]

print("\nBest configuration loaded:")
print(best)

D2 = D2_nm * 1e-9
D3 = D3_nm * 1e-9
D4 = D4_nm * 1e-9

N2 = materials[metal]

lam = 633e-9
theta = np.arange(50, 90, 0.005)


# =====================================================
# GRAPH 1 — DIELECTRIC THICKNESS COMPARISON
# =====================================================

thickness_list = [
    D3_nm - 5,
    D3_nm - 2,
    D3_nm,
    D3_nm + 2,
    D3_nm + 5
]

plt.figure(figsize=(8,6))

for t in thickness_list:

    D3_temp = t * 1e-9

    Rwo, Rw = compute_reflectance(
        theta,
        lam,
        N1, N2, N3, N4, N_normal, N_cancer,
        D2, D3_temp, D4
    )

    if abs(t - D3_nm) < 0.01:
        plt.plot(theta, Rwo, linewidth=3,
                 label=f"{metal}_{t:.2f} nm (Optimal)")
    else:
        plt.plot(theta, Rwo, linestyle="--",
                 label=f"{metal}_{t:.2f} nm")

plt.xlabel("Incident Angle (°)")
plt.ylabel("Reflectance")
plt.title("SPR Reflectance for Different Dielectric Thickness")
plt.legend()
plt.grid(True)

plt.savefig("results/report_graphs/thickness_comparison.png", dpi=300)
plt.close()


# =====================================================
# GRAPH 2 — SENSOR SENSITIVITY CURVES
# =====================================================

ri_values = [
    N_normal,
    (N_normal + N_cancer)/2,
    N_cancer
]

plt.figure(figsize=(8,6))

for n in ri_values:

    Rwo, Rw = compute_reflectance(
        theta,
        lam,
        N1, N2, N3, N4, n, n,
        D2, D3, D4
    )

    plt.plot(theta, Rwo, label=f"n = {n:.4f}")

plt.xlabel("Incident Angle (°)")
plt.ylabel("Reflectance")
plt.title("SPR Sensor Sensitivity (Breast Cancer RI Change)")
plt.legend()
plt.grid(True)

plt.savefig("results/report_graphs/sensitivity_curve.png", dpi=300)
plt.close()


# =====================================================
# GRAPH 3 — WAVELENGTH RESPONSE
# =====================================================

lam_range = np.linspace(400e-9, 800e-9, 300)

theta_res = theta[np.argmin(Rwo)]
R_normal = []
R_cancer = []

for lam_val in lam_range:

    Rwo, Rw = compute_reflectance(
        np.array([theta_res]),
        lam_val,
        N1, N2, N3, N4, N_normal, N_cancer,
        D2, D3, D4
    )

    R_normal.append(Rwo[0])
    R_cancer.append(Rw[0])

plt.figure(figsize=(8,6))

plt.plot(lam_range*1e9, R_normal, label="Normal Cell")
plt.plot(lam_range*1e9, R_cancer, label="Cancer Cell")

plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.title("SPR Response vs Wavelength (Breast Cancer)")
plt.legend()
plt.grid(True)

plt.savefig("results/report_graphs/wavelength_response.png", dpi=300)
plt.close()


# =====================================================
# GRAPH 4 — RESONANCE DIP (FWHM BASIS)
# =====================================================

Rwo, Rw = compute_reflectance(
    theta,
    lam,
    N1, N2, N3, N4, N_normal, N_cancer,
    D2, D3, D4
)

plt.figure(figsize=(8,6))

plt.plot(theta, Rwo, label="Reflectance Curve")

min_idx = np.argmin(Rwo)
res_angle = theta[min_idx]

plt.scatter(res_angle, Rwo[min_idx],
            color="red",
            label=f"Resonance Angle = {res_angle:.2f}°")

plt.xlabel("Incident Angle (°)")
plt.ylabel("Reflectance")
plt.title("SPR Resonance Dip (Breast Cancer)")
plt.legend()
plt.grid(True)

plt.savefig("results/report_graphs/resonance_dip.png", dpi=300)
plt.close()


# =====================================================
# GRAPH 5 — RESONANCE ANGLE SHIFT vs RI
# =====================================================

ri_range = np.linspace(N_normal, N_cancer, 6)

res_angles = []

for n in ri_range:

    Rwo, Rw = compute_reflectance(
        theta,
        lam,
        N1, N2, N3, N4, n, n,
        D2, D3, D4
    )

    res_angles.append(theta[np.argmin(Rwo)])

plt.figure(figsize=(8,6))

plt.plot(ri_range, res_angles, marker='o')

plt.xlabel("Refractive Index")
plt.ylabel("Resonance Angle (°)")
plt.title("Resonance Angle Shift vs RI (Breast Cancer)")

plt.grid(True)

plt.savefig("results/report_graphs/resonance_shift.png", dpi=300)
plt.close()


print("\nAll report graphs generated successfully!")
print("Saved in: results/report_graphs/")