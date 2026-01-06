# -*- coding: utf-8 -*-
"""
Avaliação de "saúde" do aterramento da torre a partir de:
 - Correntes reconstruídas nos 8 eletrodos (4 fundações + 4 cabos contrapeso)
 - Matriz de impedâncias Z (8x8)
 - Matriz de transferência T (5x8) corrente->potencial
 - Regressão da tensão da torre a partir dos 5 sensores (vetor a_vec)

Saídas:
 - Índices de saúde globais:
     H_Z_glob   : a partir de Z (equipotencialidade nos eletrodos)
     H_T_glob   : a partir de T (coerência V_med vs T*I)
     H_I,F_glob : a partir das frações de corrente nas fundações (β)
     H_I,CP_glob: a partir das frações de corrente nos cabos (α_CP)
 - Índices de saúde por eletrodo / por sensor.
 - Tabelas LaTeX para uso no artigo.
"""

import numpy as np

# Ordem dos eletrodos
electrodes = ["FA", "FB", "FC", "FD", "CA", "CB", "CC", "CD"]

# ==== MATRIZ Z (8 x 8) – obtida por identificação em baixa frequência ====
Z = np.array([
    [4.0376, 0.1157, 0.2339, 1.4337, 1.5107, 0.5729, 0.3916, 0.6488],
    [0.1157, 3.2139, 1.7983, 1.5395, 0.6981, 1.2090, 0.8420, 0.7007],
    [0.2339, 1.7983, 3.4648, 1.3695, 0.6334, 0.8973, 1.1087, 0.7162],
    [1.4337, 1.5395, 1.3695, 1.7111, 0.9491, 1.0106, 0.7293, 0.7661],
    [1.5107, 0.6981, 0.6334, 0.9491, 3.1737, 0.7949, 0.7658, 0.6444],
    [0.5729, 1.2090, 0.8973, 1.0106, 0.7949, 5.3168, 0.6041, 0.6199],
    [0.3916, 0.8420, 1.1087, 0.7293, 0.7658, 0.6041, 4.6417, 0.7096],
    [0.6488, 0.7007, 0.7162, 0.7661, 0.6444, 0.6199, 0.7096, 2.7489]
], dtype=float)

# ==== MATRIZ T (5 x 8) – transferência corrente->potencial (V/A) ====
T = np.array([
    [-1.5003,  2.1478,  1.5500,  0.3675, -0.4896,  0.2338,  0.2263,  0.0472],
    [ 0.4186,  1.7336,  0.1604,  0.2623, -0.2356, -0.0032,  0.0129,  0.0286],
    [-0.6880,  1.5729,  1.6048,  0.2831, -0.1772, -0.0632,  0.0619,  0.0381],
    [-0.3408,  1.4764,  0.4328,  0.4416, -0.0160,  0.2767, -0.1855, -0.0373],
    [-0.0654,  0.8117,  1.0296,  0.3379, -0.3091,  0.1317,  0.2193,  0.0354]
], dtype=float)

# ==== VETOR a – V_torre ≈ a0*V0 + ... + a4*V4 (já calibrado) ====
a_vec = np.array([
    -0.170765,
    -17.341725,
    16.536161,
    -13.135335,
    18.206203
], dtype=float)

# ==== Tolerâncias internas para índices de saúde ====
V_TOL_Z = 0.05   # ~ desvio típico aceitável para resíduos em Z (V)
V_TOL_T = 0.05   # ~ desvio típico aceitável para resíduos em T (V)
BETA_TOL = 0.05  # tolerância para diferenças em frações β (fundações)
ALPHA_TOL = 0.05 # tolerância para diferenças em frações α_CP (cabos)


# ======================================================================
#  Funções de health
# ======================================================================

def health_index_Z(Z, I_full, V_torre_true, V_tol=V_TOL_Z):
    """
    Índice de saúde baseado em Z:
      V_model = Z * I_full
      resid_i = V_model[i] - V_torre_true
      health_i = 1 / (1 + |resid_i| / V_tol)
    """
    I_full = np.asarray(I_full, dtype=float).reshape(8)
    V_model = Z @ I_full
    residuals = V_model - float(V_torre_true)

    V_tol = max(V_tol, 1e-9)
    health_e = 1.0 / (1.0 + np.abs(residuals) / V_tol)
    health_global = float(np.mean(health_e))
    return health_e, health_global, residuals


def health_index_T(T_ref, I_full, V_meas, V_tol=V_TOL_T):
    """
    Índice de saúde baseado em T:
      V_pred = T_ref * I_full
      resid_s = V_meas[s] - V_pred[s]
      health_s = 1 / (1 + |resid_s| / V_tol)
    """
    I_full = np.asarray(I_full, dtype=float).reshape(8)
    V_meas = np.asarray(V_meas, dtype=float).reshape(5)

    V_pred = T_ref @ I_full
    residuals = V_meas - V_pred

    V_tol = max(V_tol, 1e-9)
    health_s = 1.0 / (1.0 + np.abs(residuals) / V_tol)
    health_global = float(np.mean(health_s))
    return health_s, health_global, residuals


def health_I_foundations(I_full, beta_ref, beta_tol=BETA_TOL):
    """
    Health baseado em fração de corrente nas fundações (β):

      β_i = I_F_i / sum_j I_F_j   (i = FA..FD)
      health_{F,i} = 1 / (1 + |β_i - β_ref_i| / beta_tol)

    Retorna:
      - health_F (4,)   : por fundação
      - health_F_glob   : média
    """
    I_full = np.asarray(I_full, dtype=float).reshape(8)
    I_F = I_full[0:4]
    sum_F = float(np.sum(np.abs(I_F)))
    if sum_F < 1e-9:
        health_F = np.zeros(4)
        return health_F, 0.0

    beta = I_F / sum_F
    beta_ref = np.asarray(beta_ref, dtype=float).reshape(4)

    beta_tol = max(beta_tol, 1e-9)
    diff = np.abs(beta - beta_ref)
    health_F = 1.0 / (1.0 + diff / beta_tol)
    health_F_glob = float(np.mean(health_F))
    return health_F, health_F_glob


def health_I_cps(I_full, alpha_ref_cp, alpha_tol=ALPHA_TOL):
    """
    Health baseado em fração de corrente nos cabos contrapeso (α_CP):

      α_CP,j = I_CP_j / I_tot   (j = CA..CD)
      health_{CP,j} = 1 / (1 + |α_CP,j - α_ref_cp,j| / alpha_tol)
    """
    I_full = np.asarray(I_full, dtype=float).reshape(8)
    I_CP = I_full[4:8]
    I_tot = float(np.sum(np.abs(I_full)))
    if I_tot < 1e-9:
        health_CP = np.zeros(4)
        return health_CP, 0.0

    alpha_cp = I_CP / I_tot
    alpha_ref_cp = np.asarray(alpha_ref_cp, dtype=float).reshape(4)

    alpha_tol = max(alpha_tol, 1e-9)
    diff = np.abs(alpha_cp - alpha_ref_cp)
    health_CP = 1.0 / (1.0 + diff / alpha_tol)
    health_CP_glob = float(np.mean(health_CP))
    return health_CP, health_CP_glob


# ======================================================================
#  Dados de campo – 16 ensaios (correntes, R_ref, Itot, V_sens)
# ======================================================================

# Padrão de CPs (F = fechado, A = aberto)
cp_labels = [
    "FFFF", "AFFF", "FAFF", "FFAF",
    "FFFA", "AAFF", "FAAF", "FFAA",
    "AFAF", "FAFA", "AFFA", "AAAF",
    "AFAA", "AAFA", "FAAA", "AAAA"
]

# Correntes totais (mA) e resistências (Ω) dos 16 ensaios
Itot_mA = np.array([
    631, 580, 715, 805, 877, 860, 767, 744,
    801, 764, 748, 773, 717, 696, 699, 638
], dtype=float)

R_ref = np.array([
    1.150, 1.195, 1.166, 1.190, 1.303, 1.267, 1.253, 1.385,
    1.278, 1.333, 1.371, 1.342, 1.497, 1.454, 1.458, 1.596
], dtype=float)

Itot_ref = Itot_mA / 1000.0  # A

# Potenciais do método IA (em mV) – V0..V4 para os 16 ensaios
V_sens_mV = np.array([
    [126, 145, 146, 116, 126],
    [154, 192, 158, 125, 165],
    [175, 208, 212, 155, 165],
    [195, 204, 233, 200, 175],
    [230, 239, 253, 238, 235],
    [265, 330, 284, 210, 267],
    [215, 236, 268, 211, 192],
    [233, 232, 273, 261, 221],
    [246, 280, 264, 225, 246],
    [230, 255, 263, 219, 214],
    [254, 298, 260, 235, 277],
    [285, 340, 322, 248, 270],
    [298, 322, 313, 289, 292],
    [280, 330, 298, 234, 279],
    [256, 265, 309, 267, 225],
    [310, 360, 350, 295, 300]
], dtype=float)

V_sens_ref = V_sens_mV / 1000.0  # V

# Correntes reconstruídas (em mA) – fundações
I_F_mA = np.array([
    [ 71.4,  61.5,  58.4,  76.0],
    [107.0,  72.0,  63.0, 106.7],
    [ 80.0,  78.0,  63.0,  98.0],
    [ 83.5,  75.7,  85.0, 135.7],
    [105.0,  83.2,  88.0, 251.0],
    [147.0, 115.0,  86.4, 194.5],
    [ 98.0, 101.0,  99.0, 172.7],
    [104.0,  82.0, 112.0, 251.0],
    [131.0,  91.0,  98.0, 178.0],
    [101.9, 101.0,  83.0, 221.0],
    [137.0,  89.7,  85.0, 235.0],
    [141.0, 114.0, 105.0, 188.6],
    [144.0,  99.8, 116.0, 269.0],
    [146.0, 113.0,  92.3, 243.0],
    [104.7, 106.0, 113.0, 233.0],
    [148.0, 118.0, 116.0, 253.0]
], dtype=float)

# Correntes reconstruídas (em mA) – cabos contrapeso
I_CP_mA = np.array([
    [ 94.6,  50.5,  68.6, 140.0],
    [  0.0,  61.0,  75.0, 145.3],
    [102.0,   0.0,  77.0, 151.0],
    [120.5,  65.3,   0.0, 194.3],
    [152.0,  78.8, 131.0,   0.0],
    [  0.0,   0.0, 115.6, 236.5],
    [138.0,   0.0,   0.0, 220.3],
    [142.0,  89.0,   0.0,   0.0],
    [  0.0,  80.0,   0.0, 223.0],
    [140.1,  0.0, 119.0,   0.0],
    [  0.0, 80.3, 119.0,   0.0],
    [  0.0,  0.0,   0.0, 220.4],
    [  0.0, 87.2,   0.0,   0.0],
    [  0.0,  0.0, 122.7,   0.0],
    [142.3,  0.0,   0.0,   0.0],
    [  0.0,  0.0,   0.0,   0.0]
], dtype=float)


# ======================================================================
#  Baseline – Ensaio 1 (FFFF)
# ======================================================================

# Ensaio 1 – correntes em A
I_F_1 = I_F_mA[0, :] / 1000.0
I_CP_1 = I_CP_mA[0, :] / 1000.0
I_full_1 = np.concatenate([I_F_1, I_CP_1])
I_tot_1 = Itot_ref[0]
R_ref_1 = R_ref[0]
V_meas_1 = V_sens_ref[0, :]

# Tensão verdadeira da torre (método passivo)
V_torre_true_1 = I_tot_1 * R_ref_1

# Tensão estimada a partir dos sensores (regressão a_vec)
V_torre_est_1 = float(a_vec @ V_meas_1)
R_est_1 = V_torre_est_1 / I_tot_1

# Frações de corrente – baseline
#   β_ref: frações dentro do conjunto de fundações
sum_F_1 = float(np.sum(np.abs(I_F_1)))
beta_ref = I_F_1 / sum_F_1

#   α_ref_cp: frações globais para os cabos contrapeso
alpha_ref_cp = I_CP_1 / I_tot_1

print("=== Baseline (Ensaio 1 - FFFF) ===")
print(f"I_tot_1          = {I_tot_1:.4f} A")
print(f"V_torre_true_1   = {V_torre_true_1:.4f} V")
print(f"V_torre_est_1    = {V_torre_est_1:.4f} V")
print(f"R_ref_1          = {R_ref_1:.4f} Ω")
print(f"R_est_1          = {R_est_1:.4f} Ω\n")

print("Frações β_ref (apenas fundações, normalizadas entre FA..FD):")
for name, b in zip(electrodes[0:4], beta_ref):
    print(f"  {name}: beta_ref = {b:.3f}")
print("\nFrações α_ref,CP (cabos, normalizadas pelo I_tot):")
for name, a in zip(electrodes[4:8], alpha_ref_cp):
    print(f"  {name}: alpha_ref_CP = {a:.3f}")
print()

# Health em Z / T / I (baseline)
hZ_e_1, hZ_g_1, resZ_1 = health_index_Z(Z, I_full_1, V_torre_true_1)
hT_s_1, hT_g_1, resT_1 = health_index_T(T, I_full_1, V_meas_1)
hF_e_1, hF_g_1 = health_I_foundations(I_full_1, beta_ref)
hCP_e_1, hCP_g_1 = health_I_cps(I_full_1, alpha_ref_cp)

print("=== Health baseado em Z (baseline) ===")
print(f"Health_Z_global = {hZ_g_1:.3f}")
for name, h, r in zip(electrodes, hZ_e_1, resZ_1):
    print(f"  {name}: health_Z={h:.3f}, resid_Z={r:.4f} V")
print()

print("=== Health baseado em T (baseline) ===")
print(f"Health_T_global = {hT_g_1:.3f}")
for i, (h, r) in enumerate(zip(hT_s_1, resT_1)):
    print(f"  V{i}: health_T={h:.3f}, resid_T={r:.4f} V")
print()

print("=== Health baseado em fração de corrente β (fundacoes) – baseline ===")
print(f"Health_I,F_global = {hF_g_1:.3f}")
for name, h in zip(electrodes[0:4], hF_e_1):
    print(f"  {name}: health_I,F={h:.3f}")
print()

print("=== Health baseado em fração de corrente α_CP (cabos) – baseline ===")
print(f"Health_I,CP_global = {hCP_g_1:.3f}")
for name, h in zip(electrodes[4:8], hCP_e_1):
    print(f"  {name}: health_I,CP={h:.3f}")
print()


# ======================================================================
#  Varrendo os 16 ensaios e montando as tabelas
# ======================================================================

results = []

for k in range(16):
    test_id = k + 1
    cp_label = cp_labels[k]

    I_F_k = I_F_mA[k, :] / 1000.0
    I_CP_k = I_CP_mA[k, :] / 1000.0
    I_full_k = np.concatenate([I_F_k, I_CP_k])
    I_tot_k = Itot_ref[k]
    R_ref_k = R_ref[k]
    V_meas_k = V_sens_ref[k, :]

    # Tensão verdadeira da torre (método passivo)
    V_t_true_k = I_tot_k * R_ref_k

    # Tensão estimada a partir dos sensores (a_vec)
    V_t_est_k = float(a_vec @ V_meas_k)
    R_est_k = V_t_est_k / I_tot_k
    delta_R_k = R_est_k - R_ref_k

    # Health Z/T
    HZ_e_k, HZ_g_k, _ = health_index_Z(Z, I_full_k, V_t_true_k)
    HT_s_k, HT_g_k, _ = health_index_T(T, I_full_k, V_meas_k)

    # Health I – fundações (β) e cabos (α_CP)
    HIF_e_k, HIF_g_k = health_I_foundations(I_full_k, beta_ref)
    HICP_e_k, HICP_g_k = health_I_cps(I_full_k, alpha_ref_cp)

    results.append({
        "test": test_id,
        "cp": cp_label,
        "R_ref": R_ref_k,
        "R_est": R_est_k,
        "delta_R": delta_R_k,
        "HZ_global": HZ_g_k,
        "HT_global": HT_g_k,
        "HIF_global": HIF_g_k,
        "HICP_global": HICP_g_k,
        "HZ_e": HZ_e_k,
        "HIF_e": HIF_e_k,
        "HICP_e": HICP_e_k,
        "HT_s": HT_s_k
    })


# ======================================================================
#  Impressão das tabelas em LaTeX
# ======================================================================

print()
print("% ===== TABELA GLOBAL (R_ref, R_est, índices globais) =====")
print(r"\begin{table}[!t]")
print(r"\centering")
print(r"\caption{Global health indices for the 16 tests.}")
print(r"\label{tab:health_global}")
print(r"\begin{tabular}{c c c c c c c c c}")
print(r"\hline")
print(r"Test & CPs & $R_{\mathrm{ref}}$ [\si{\ohm}] & $R_{\mathrm{est}}$ [\si{\ohm}] & $\Delta R$ [\si{\ohm}] & $H_Z^{\mathrm{glob}}$ & $H_T^{\mathrm{glob}}$ & $H_{I,F}^{\mathrm{glob}}$ & $H_{I,CP}^{\mathrm{glob}}$ \\")
print(r"\hline")
for res in results:
    print(f"{res['test']:d} & {res['cp']} & "
          f"{res['R_ref']:.3f} & {res['R_est']:.3f} & {res['delta_R']:+.3f} & "
          f"{res['HZ_global']:.3f} & {res['HT_global']:.3f} & "
          f"{res['HIF_global']:.3f} & {res['HICP_global']:.3f} \\\\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")
print()

# Tabela H_Z por eletrodo
print("% ===== TABELA HEALTH_Z POR ELETRODO =====")
print(r"\begin{table}[!t]")
print(r"\centering")
print(r"\caption{Per electrode impedance-based health index $H_Z$ for the 16 tests.}")
print(r"\label{tab:health_Z}")
print(r"\begin{tabular}{c c c c c c c c c c}")
print(r"\hline")
print(r"Test & CPs & FA & FB & FC & FD & CA & CB & CC & CD \\")
print(r"\hline")
for res in results:
    HZ = res["HZ_e"]
    print(f"{res['test']:d} & {res['cp']} & "
          f"{HZ[0]:.3f} & {HZ[1]:.3f} & {HZ[2]:.3f} & {HZ[3]:.3f} & "
          f"{HZ[4]:.3f} & {HZ[5]:.3f} & {HZ[6]:.3f} & {HZ[7]:.3f} \\\\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")
print()

# Tabela H_I,F por fundação (β)
print("% ===== TABELA HEALTH_I POR FUNDAÇÃO (β) =====")
print(r"\begin{table}[!t]")
print(r"\centering")
print(r"\caption{Per foundation current-fraction-based health index $H_{I,F}$ (using $\beta_i$) for the 16 tests.}")
print(r"\label{tab:health_I_F}")
print(r"\begin{tabular}{c c c c c c}")
print(r"\hline")
print(r"Test & CPs & FA & FB & FC & FD \\")
print(r"\hline")
for res in results:
    HIF = res["HIF_e"]
    print(f"{res['test']:d} & {res['cp']} & "
          f"{HIF[0]:.3f} & {HIF[1]:.3f} & {HIF[2]:.3f} & {HIF[3]:.3f} \\\\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")
print()

# Tabela H_I,CP por cabo contrapeso (α_CP)
print("% ===== TABELA HEALTH_I POR CABO CONTRAPESO (α_CP) =====")
print(r"\begin{table}[!t]")
print(r"\centering")
print(r"\caption{Per counterpoise current-fraction-based health index $H_{I,CP}$ (using $\alpha_{\mathrm{CP},j}$) for the 16 tests.}")
print(r"\label{tab:health_I_CP}")
print(r"\begin{tabular}{c c c c c c}")
print(r"\hline")
print(r"Test & CPs & CA & CB & CC & CD \\")
print(r"\hline")
for res in results:
    HICP = res["HICP_e"]
    print(f"{res['test']:d} & {res['cp']} & "
          f"{HICP[0]:.3f} & {HICP[1]:.3f} & {HICP[2]:.3f} & {HICP[3]:.3f} \\\\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")
print()

# Tabela H_T por sensor
print("% ===== TABELA HEALTH_T POR SENSOR =====")
print(r"\begin{table}[!t]")
print(r"\centering")
print(r"\caption{Per sensor transfer-matrix-based health index $H_T$ for the 16 tests.}")
print(r"\label{tab:health_T}")
print(r"\begin{tabular}{c c c c c c c}")
print(r"\hline")
print(r"Test & CPs & $V_0$ & $V_1$ & $V_2$ & $V_3$ & $V_4$ \\")
print(r"\hline")
for res in results:
    HT = res["HT_s"]
    print(f"{res['test']:d} & {res['cp']} & "
          f"{HT[0]:.3f} & {HT[1]:.3f} & {HT[2]:.3f} & {HT[3]:.3f} & {HT[4]:.3f} \\\\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")
print()

