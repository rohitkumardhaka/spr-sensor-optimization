import numpy as np
import numba as nb


# -------------------------------------------------------
# Complex sqrt (numba safe)
# -------------------------------------------------------
@nb.njit
def csqrt(z):
    return np.sqrt(z)


# -------------------------------------------------------
# Fresnel reflection coefficient
# -------------------------------------------------------
@nb.njit
def reflection_coeff(E1, E2, K1, K2):
    return (E2/K2 - E1/K1) / (E2/K2 + E1/K1)


# -------------------------------------------------------
# MULTILAYER CASCADE REFLECTION COEFFICIENT
# MUST BE DEFINED BEFORE compute_reflectance
# -------------------------------------------------------
@nb.njit
def calculate_total_reflection_coefficient(R12, R23, R34, R45,
                                           K2, K3, K4,
                                           D2, D3, D4):

    R345 = (R34 + R45 * np.exp(2j*K4*D4)) / (1 + R34 * R45 * np.exp(2j*K4*D4))
    R2345 = (R23 + R345 * np.exp(2j*K3*D3)) / \
        (1 + R23 * R345 * np.exp(2j*K3*D3))
    Rtot = (R12 + R2345 * np.exp(2j*K2*D2)) / \
        (1 + R12 * R2345 * np.exp(2j*K2*D2))

    return Rtot


# -------------------------------------------------------
# REFLECTANCE CALCULATOR
# -------------------------------------------------------
@nb.njit
def compute_reflectance(theta_deg, lam,
                        N1, N2, N3, N4, N5, N5_1,
                        D2, D3, D4):

    # force complex arithmetic
    N1c = N1 + 0j
    N2c = N2 + 0j
    N3c = N3 + 0j
    N4c = N4 + 0j
    N5c = N5 + 0j
    N5_1c = N5_1 + 0j

    w = 2 * np.pi / lam

    E1 = N1c * N1c
    E2 = N2c * N2c
    E3 = N3c * N3c
    E4 = N4c * N4c
    E5 = N5c * N5c
    E5_1 = N5_1c * N5_1c

    Rwo = np.zeros(theta_deg.size)
    Rw = np.zeros(theta_deg.size)

    for i in range(theta_deg.size):

        th = np.deg2rad(theta_deg[i])

        Kz = np.sqrt(E1) * w * np.sin(th)

        K1 = csqrt(E1*w*w - Kz*Kz)
        K2 = csqrt(E2*w*w - Kz*Kz)
        K3 = csqrt(E3*w*w - Kz*Kz)
        K4 = csqrt(E4*w*w - Kz*Kz)
        K5 = csqrt(E5*w*w - Kz*Kz)
        K5p = csqrt(E5_1*w*w - Kz*Kz)

        R12 = reflection_coeff(E1, E2, K1, K2)
        R23 = reflection_coeff(E2, E3, K2, K3)
        R34 = reflection_coeff(E3, E4, K3, K4)
        R45 = reflection_coeff(E4, E5, K4, K5)
        R45p = reflection_coeff(E4, E5_1, K4, K5p)

        R_wo = calculate_total_reflection_coefficient(R12, R23, R34, R45,
                                                      K2, K3, K4,
                                                      D2, D3, D4)

        R_w = calculate_total_reflection_coefficient(R12, R23, R34, R45p,
                                                     K2, K3, K4,
                                                     D2, D3, D4)

        Rwo[i] = abs(R_wo)**2
        Rw[i] = abs(R_w)**2

    return Rwo, Rw


# -------------------------------------------------------
# FOM CALCULATION
# -------------------------------------------------------
@nb.njit
def compute_fom(theta_deg, Rwo, Rw, dn):

    idx_wo = np.argmin(Rwo)
    idx_w = np.argmin(Rw)

    th_wo = theta_deg[idx_wo]
    th_w = theta_deg[idx_w]

    S = (th_w - th_wo) / dn

    Rmin = np.min(Rw)
    half = (np.max(Rw) + Rmin) / 2

    left = 0
    for i in range(idx_w, -1, -1):
        if Rw[i] > half:
            left = i
            break

    right = idx_w
    for i in range(idx_w, theta_deg.size):
        if Rw[i] > half:
            right = i
            break

    FWHM = theta_deg[right] - theta_deg[left]

    return S / (FWHM + 1e-12)
