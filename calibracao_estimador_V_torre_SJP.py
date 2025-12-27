# -*- coding: utf-8 -*-
"""
Monitoramento da resistência de aterramento de uma torre usando:
 - Matriz de impedâncias Z (8x8)
 - Matriz de transferência T (5x8) corrente->potencial
 - Modelo linear V_torre ≈ a_V^T * V_sens + a_I^T * I_cp

Entradas "online" (edge):
 - Correntes nos cabos contrapeso: I_cp = [I_CA, I_CB, I_CC, I_CD] (A)
 - Potenciais em 5 sensores: V_meas = [V0, V1, V2, V3, V4] (V)

Saídas principais:
 - R_torre_est: resistência equivalente estimada (Ω)
 - I_tot_est: corrente total estimada (A)
 - I_full_hat: correntes estimadas nos 8 eletrodos [FA..CD] (A)
 - Índices de saúde baseados em Z (por eletrodo e global)
 - Índices de saúde baseados em T (por sensor e global)
"""

import numpy as np

# ---------------------------------------------------------
# 1) DADOS DOS 16 ENSAIOS (comissionamento)
# ---------------------------------------------------------

# Potenciais nos 5 sensores (mV) - Método IA (V0torre, V1, V2, V3, V4)
V_mV = [
    [126, 145, 146, 116, 126],  # Ensaio 1
    [154, 192, 158, 125, 165],  # 2
    [175, 208, 212, 155, 165],  # 3
    [195, 204, 233, 200, 175],  # 4
    [230, 239, 253, 238, 235],  # 5
    [265, 330, 284, 210, 267],  # 6
    [215, 236, 268, 211, 192],  # 7
    [233, 232, 273, 261, 221],  # 8
    [246, 280, 264, 225, 246],  # 9
    [230, 255, 263, 219, 214],  # 10
    [254, 298, 260, 235, 277],  # 11
    [285, 340, 322, 248, 270],  # 12
    [298, 322, 313, 289, 292],  # 13
    [280, 330, 298, 234, 279],  # 14
    [256, 265, 309, 267, 225],  # 15
    [310, 360, 350, 295, 300],  # 16
]

# Corrente total (mA) e resistência global (Ω) – método passivo
Itot_mA = [631, 580, 715, 805, 877, 860, 767, 744,
           801, 764, 748, 773, 717, 696, 699, 638]

R_ref = [1.150, 1.195, 1.166, 1.190,
         1.303, 1.267, 1.253, 1.385,
         1.278, 1.333, 1.371, 1.342,
         1.497, 1.454, 1.458, 1.596]

# Correntes nos cabos contrapeso (Hioki, mA) – CA, CB, CC, CD
cp_mA = [
    [94.600, 50.500, 68.600, 140.000],  # 1: FFFF
    [0.000,  61.000, 75.000, 145.300],  # 2: AFFF
    [102.000, 0.000, 77.000, 151.000],  # 3: FAFF
    [120.500, 65.300, 0.000, 194.300],  # 4: FFAF
    [152.000, 78.800, 131.000, 0.000],  # 5: FFFA
    [0.000,   0.000, 115.600, 236.500], # 6: AAFF
    [138.000, 0.000, 0.000, 220.300],   # 7: FAAF
    [142.000, 89.000, 0.000, 0.000],    # 8: FFAA
    [0.000,   80.000, 0.000, 223.000],  # 9: AFAF
    [140.100, 0.000, 119.000, 0.000],   # 10: FAFA
    [0.000,   80.300, 119.000, 0.000],  # 11: AFFA
    [0.000,   0.000, 0.000, 220.400],   # 12: AAAF
    [0.000,   87.200, 0.000, 0.000],    # 13: AFAA
    [0.000,   0.000, 122.700, 0.000],   # 14: AAFA
    [142.300, 0.000, 0.000, 0.000],     # 15: FAAA
    [0.000,   0.000, 0.000, 0.000],     # 16: AAAA
]

# Conversão para SI
V_sens_ref = np.array(V_mV, dtype=float) / 1000.0  # (16 x 5) em V
Itot_ref   = np.array(Itot_mA, dtype=float) / 1000.0  # (16,) em A
I_cp_ref   = np.array(cp_mA, dtype=float) / 1000.0    # (16 x 4) em A
R_ref      = np.array(R_ref, dtype=float)            # (16,) em Ω

# Tensão verdadeira da torre (via método passivo)
V_t_true = Itot_ref * R_ref  # (16,)

# ---------------------------------------------------------
# 2) CALIBRAÇÃO DO MODELO V_torre = f(V_sens, I_cp)
# ---------------------------------------------------------

# X = [V0..V4, I_CA..I_CD]  (16 x 9)
X_calib = np.hstack([V_sens_ref, I_cp_ref])

# Mínimos quadrados: X * a_full ≈ V_t_true
a_full, residuals, rank, s = np.linalg.lstsq(X_calib, V_t_true, rcond=None)

a_V = a_full[:5]   # coeficientes dos sensores de potencial
a_I = a_full[5:]   # coeficientes das correntes dos CPs (CA..CD)

V_t_est_calib = X_calib @ a_full
R_est_calib   = V_t_est_calib / Itot_ref

rmse_V_calib = np.sqrt(np.mean((V_t_est_calib - V_t_true)**2))
rmse_R_calib = np.sqrt(np.mean((R_est_calib - R_ref)**2))
mean_err_R   = np.mean(R_est_calib - R_ref)
max_err_R    = np.max(np.abs(R_est_calib - R_ref))

print("===== CALIBRAÇÃO DO ESTIMADOR V_torre = f(V_sens, I_cp) =====\n")

print("Coeficientes a_V (V_torre ≈ a_V^T * V + a_I^T * I_cp):")
for i, val in enumerate(a_V):
    print(f"  a{i} (V{i}) = {val:.6f}")
print("\nCoeficientes a_I (ganhos nas correntes dos CPs):")
for nome, val in zip(["CA", "CB", "CC", "CD"], a_I):
    print(f"  b_{nome} = {val:.6f}  [V/A ≈ ohms]")

print("\nResumo dos erros de calibração:")
print(f"  RMSE(V_torre) = {rmse_V_calib:.6f} V")
print(f"  RMSE(R_torre) = {rmse_R_calib:.6f} Ω")
print(f"  Erro médio em R_torre = {mean_err_R:+.6f} Ω")
print(f"  Erro máximo |R_est - R_ref| = {max_err_R:.6f} Ω\n")

# ---------------------------------------------------------
# 3) MATRIZES Z E T  (modelo de impedância e transferência)
# ---------------------------------------------------------

electrodes = ["FA", "FB", "FC", "FD", "CA", "CB", "CC", "CD"]

# Matriz Z (8x8)
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

# Matriz T (5 x 8) – transferência corrente->potencial (V/A)
T = np.array([
    [-1.5003,  2.1478,  1.5500,  0.3675, -0.4896,  0.2338,  0.2263,  0.0472],
    [ 0.4186,  1.7336,  0.1604,  0.2623, -0.2356, -0.0032,  0.0129,  0.0286],
    [-0.6880,  1.5729,  1.6048,  0.2831, -0.1772, -0.0632,  0.0619,  0.0381],
    [-0.3408,  1.4764,  0.4328,  0.4416, -0.0160,  0.2767, -0.1855, -0.0373],
    [-0.0654,  0.8117,  1.0296,  0.3379, -0.3091,  0.1317,  0.2193,  0.0354]
], dtype=float)

# Submatrizes de T: fundações (FA..FD) e cabos (CA..CD)
T_F  = T[:, 0:4]  # FA, FB, FC, FD
T_CP = T[:, 4:8]  # CA, CB, CC, CD

# Tolerâncias para índices de saúde (ajuste fino depois com dados reais)
V_TOL_Z = 0.05  # V
V_TOL_T = 0.05  # V

# ---------------------------------------------------------
# 4) FUNÇÕES PRINCIPAIS (estimação e health)
# ---------------------------------------------------------

def estimate_foundation_currents(T_F, T_CP, I_cp, V_meas):
    """
    Estima correntes nas fundações [FA..FD] a partir de:
      - V_meas (5,) -> [V0..V4] (V)
      - I_cp   (4,) -> [I_CA, I_CB, I_CC, I_CD] (A)

    Resolve por mínimos quadrados:
      V_meas ≈ T_F * I_F + T_CP * I_cp
      => T_F * I_F ≈ V_meas - T_CP * I_cp
    """
    V_meas = np.asarray(V_meas, dtype=float).reshape(5)
    I_cp   = np.asarray(I_cp,   dtype=float).reshape(4)

    rhs = V_meas - T_CP @ I_cp
    I_F_est, *_ = np.linalg.lstsq(T_F, rhs, rcond=None)
    return I_F_est  # (4,): [I_FA, I_FB, I_FC, I_FD]


def estimate_online_from_edge(Z, T_F, T_CP, a_V, a_I, I_cp, V_meas):
    """
    Estima estado da torre a partir de medições do edge:
      - I_cp: correntes nos cabos [CA,CB,CC,CD] (A)
      - V_meas: potenciais [V0..V4] (V)

    Retorna:
      - R_torre_est (Ω)
      - V_torre_est (V)
      - I_tot_est (A)
      - I_full (8,) correntes estimadas [FA..CD]
    """
    V_meas = np.asarray(V_meas, dtype=float).reshape(5)
    I_cp   = np.asarray(I_cp,   dtype=float).reshape(4)

    # 1) Estima correntes nas fundações via T
    I_F_est = estimate_foundation_currents(T_F, T_CP, I_cp, V_meas)

    # 2) Vetor completo de correntes
    I_full = np.concatenate([I_F_est, I_cp])  # [FA..FD, CA..CD]

    # 3) Corrente total (soma das correntes nos 8 eletrodos)
    I_tot_est = I_full.sum()

    # 4) V_torre estimado (modelo calibrado com V e I_cp)
    V_torre_est = float(a_V @ V_meas + a_I @ I_cp)

    # 5) R_torre estimada = V/I
    R_torre_est = V_torre_est / I_tot_est if I_tot_est != 0 else np.nan

    return R_torre_est, V_torre_est, I_tot_est, I_full


def health_index_Z(Z, I_full, V_torre_est, V_tol=V_TOL_Z):
    """
    Índice de saúde baseado em Z:
      - V_model = Z * I_full
      - Residual por eletrodo: r_i = V_model[i] - V_torre_est
      - health_i = 1 / (1 + |r_i| / V_tol)
    """
    I_full = np.asarray(I_full, dtype=float).reshape(8)
    V_model = Z @ I_full
    residuals = V_model - V_torre_est  # devia ser ~0 se tudo ok

    V_tol = max(V_tol, 1e-6)
    health_e = 1.0 / (1.0 + np.abs(residuals) / V_tol)
    health_global = float(np.mean(health_e))
    return health_e, health_global, residuals


def health_index_T(T_ref, I_full, V_meas, V_tol=V_TOL_T):
    """
    Índice de saúde baseado em T:
      - V_pred = T_ref * I_full
      - Residual por sensor: r_s = V_meas[s] - V_pred[s]
      - health_s = 1 / (1 + |r_s| / V_tol)
    """
    I_full = np.asarray(I_full, dtype=float).reshape(8)
    V_meas = np.asarray(V_meas, dtype=float).reshape(5)

    V_pred = T_ref @ I_full
    residuals = V_meas - V_pred

    V_tol = max(V_tol, 1e-6)
    health_s = 1.0 / (1.0 + np.abs(residuals) / V_tol)
    health_global = float(np.mean(health_s))

    return health_s, health_global, residuals


def print_diagnostic_snapshot(Z, T, T_F, T_CP, a_V, a_I, I_cp_meas, V_meas):
    """
    Snapshot completo de diagnóstico, a partir de:
      - I_cp_meas (4,)
      - V_meas (5,)
    """
    R_hat, V_t_hat, I_tot_hat, I_full_hat = estimate_online_from_edge(
        Z, T_F, T_CP, a_V, a_I, I_cp_meas, V_meas
    )

    print("====== SNAPSHOT ONLINE ======")
    print(f"I_tot_est   = {I_tot_hat:.4f} A")
    print(f"V_torre_est = {V_t_hat:.4f} V")
    print(f"R_torre_est = {R_hat:.4f} Ω\n")

    print("Correntes estimadas [FA..CD]:")
    for name, I_val in zip(electrodes, I_full_hat):
        print(f"  {name}: {I_val:.4f} A")
    print()

    # Health baseado em Z
    health_Z_e, health_Z_global, res_Z = health_index_Z(Z, I_full_hat, V_t_hat)
    print("=== Health baseado em Z ===")
    print(f"Health_Z_global = {health_Z_global:.3f}")
    for name, h, r in zip(electrodes, health_Z_e, res_Z):
        print(f"  {name}: health={h:.3f}, resid_Z={r:.4f} V")
    print()

    # Health baseado em T
    health_T_s, health_T_global, res_T = health_index_T(T, I_full_hat, V_meas)
    print("=== Health baseado em T ===")
    print(f"Health_T_global = {health_T_global:.3f}")
    for s_idx, (h, r) in enumerate(zip(health_T_s, res_T)):
        print(f"  V{s_idx}: health={h:.3f}, resid_T={r:.4f} V")


# ---------------------------------------------------------
# 5) FUNÇÃO ENXUTA PARA USO NO EDGE
# ---------------------------------------------------------

def estimate_R_torre_from_edge(V_meas, I_cp):
    """
    Função compacta para uso no edge.

    Entradas:
      - V_meas: array-like (5,) com [V0, V1, V2, V3, V4] em volts
      - I_cp:   array-like (4,) com [I_CA, I_CB, I_CC, I_CD] em ampères

    Saída:
      - R_hat: resistência estimada da torre (Ω)
      - V_hat: tensão estimada da torre (V)
      - I_tot_hat: corrente total estimada (A)
      - I_full_hat: correntes estimadas nos 8 eletrodos [FA..CD] (A)
    """
    R_hat, V_hat, I_tot_hat, I_full_hat = estimate_online_from_edge(
        Z, T_F, T_CP, a_V, a_I, I_cp, V_meas
    )
    return R_hat, V_hat, I_tot_hat, I_full_hat


# ---------------------------------------------------------
# 6) EXEMPLOS E VALIDAÇÃO COM OS 16 ENSAIOS
# ---------------------------------------------------------

if __name__ == "__main__":

    # Exemplo: usar os dados do Ensaio 1 como se fossem leitura "online"
    print("\n===== EXEMPLO: SNAPSHOT USANDO ENSAIO 1 =====")
    I_cp_ex = I_cp_ref[0, :]      # [I_CA, I_CB, I_CC, I_CD]
    V_meas_ex = V_sens_ref[0, :]  # [V0..V4]
    print_diagnostic_snapshot(Z, T, T_F, T_CP, a_V, a_I, I_cp_ex, V_meas_ex)

    # -------------------------------------------------
    # Validação: aplicar o estimador em todos os 16 ensaios
    # -------------------------------------------------
    print("\n===== VALIDAÇÃO DO ESTIMADOR COM OS 16 ENSAIOS =====")

    R_hat_edge_list = []    # usando I_tot_est (caso edge)
    R_hat_trueI_list = []   # usando Itot_ref (melhor caso)
    V_hat_list = []

    for k in range(16):
        V_meas_k = V_sens_ref[k, :]
        I_cp_k   = I_cp_ref[k, :]
        Itot_k   = Itot_ref[k]
        R_ref_k  = R_ref[k]
        V_t_true_k = V_t_true[k]

        # Estima via modelo completo
        R_hat_edge, V_hat_k, I_tot_est_k, I_full_k = estimate_online_from_edge(
            Z, T_F, T_CP, a_V, a_I, I_cp_k, V_meas_k
        )

        # "Melhor caso": usa V_hat_k, mas divide pela corrente verdadeira Itot_k
        R_hat_trueI = V_hat_k / Itot_k

        R_hat_edge_list.append(R_hat_edge)
        R_hat_trueI_list.append(R_hat_trueI)
        V_hat_list.append(V_hat_k)

        print(f"Ensaio {k+1:2d} | "
              f"R_ref = {R_ref_k:6.3f} Ω | "
              f"R_hat_trueI = {R_hat_trueI:6.3f} Ω | "
              f"R_hat_edge = {R_hat_edge:6.3f} Ω | "
              f"I_tot_est = {I_tot_est_k:6.3f} A | "
              f"I_tot_ref = {Itot_k:6.3f} A")

    R_hat_edge_arr   = np.array(R_hat_edge_list)
    R_hat_trueI_arr  = np.array(R_hat_trueI_list)

    rmse_R_trueI = np.sqrt(np.mean((R_hat_trueI_arr - R_ref)**2))
    rmse_R_edge  = np.sqrt(np.mean((R_hat_edge_arr  - R_ref)**2))

    print("\nResumo da validação:")
    print(f"  RMSE(R_hat_trueI vs R_ref) = {rmse_R_trueI:.6f} Ω  (usa Itot_ref)")
    print(f"  RMSE(R_hat_edge  vs R_ref) = {rmse_R_edge:.6f} Ω  (usa I_tot_est)")


import numpy as np
import matplotlib.pyplot as plt

# Valores de referência (método passivo) e estimados pelo “edge”
R_ref = np.array([1.150, 1.195, 1.166, 1.190,
                  1.303, 1.267, 1.253, 1.385,
                  1.278, 1.333, 1.371, 1.342,
                  1.497, 1.454, 1.458, 1.596])

R_hat_edge = np.array([1.145, 1.200, 1.207, 1.237,
                       1.259, 1.226, 1.205, 1.364,
                       1.256, 1.358, 1.325, 1.331,
                       1.520, 1.463, 1.457, 1.533])

ensaios = np.arange(1, 17)

# Métricas globais
rmse = np.sqrt(np.mean((R_hat_edge - R_ref)**2))
mean_rel = np.mean((R_hat_edge - R_ref) / R_ref * 100)
max_rel = np.max(np.abs((R_hat_edge - R_ref) / R_ref * 100))

print(f"RMSE = {rmse:.4f} Ω")
print(f"Erro médio relativo = {mean_rel:+.2f} %")
print(f"Erro máx. relativo  = {max_rel:.2f} %")

# ---- Figura 1: R_hat_edge vs R_ref ----
plt.figure(figsize=(5, 4))
plt.scatter(R_ref, R_hat_edge)
plt.plot([1.1, 1.65], [1.1, 1.65], '--')  # reta identidade
plt.xlabel(r'$R_{\mathrm{ref}}$ ($\Omega$)')
plt.ylabel(r'$R_{\mathrm{hat,edge}}$ ($\Omega$)')
plt.grid(True, linestyle=':')
plt.title(r'Estimativa de $R_{\mathrm{torre}}$ a partir de $I_{\mathrm{CP}}$ e $V_{0\ldots4}$')

# Opcional: anotação com RMSE
plt.text(1.11, 1.63,
         rf'RMSE = {rmse:.03f} $\Omega$' '\n'
         rf'Erro médio = {mean_rel:+.2f}\%',
         fontsize=8,
         va='top')

plt.tight_layout()
plt.savefig('Req_edge_vs_ref.png', dpi=300, bbox_inches='tight')

# ---- Figura 2: resíduos por ensaio (opcional) ----
res = R_hat_edge - R_ref

plt.figure(figsize=(5, 3.5))
plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
plt.stem(ensaios, res, use_line_collection=True)
plt.xlabel('Ensaio')
plt.ylabel(r'$\Delta R$ ($\Omega$)')
plt.grid(True, linestyle=':')
plt.title(r'Resíduos $R_{\mathrm{hat,edge}} - R_{\mathrm{ref}}$')
plt.tight_layout()
plt.savefig('Req_edge_residuos.png', dpi=300, bbox_inches='tight')

plt.show()
