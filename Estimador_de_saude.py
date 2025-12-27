# -*- coding: utf-8 -*-
"""
Monitoramento online da "saúde" do aterramento de uma torre usando:
 - Matriz de impedâncias Z (8x8)
 - Matriz de transferência T (5x8) corrente->potencial
 - Vetor a (V_torre ≈ a^T V)

Entradas "online" (do edge):
 - Correntes nos cabos contrapeso: I_cp = [I_CA, I_CB, I_CC, I_CD] (A)
 - Potenciais em 5 sensores: V_meas = [V0, V1, V2, V3, V4] (V)

Saídas principais:
 - R_torre_est: resistência equivalente estimada (Ω)
 - I_tot_est: corrente total estimada (A)
 - I_full_hat: correntes estimadas em cada eletrodo [FA..CD] (A)
 - Índices de saúde baseados em Z (por eletrodo e global)
 - Índices de saúde baseados em T (por sensor e global)
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

# Separar T em fundações (FA..FD) e cabos (CA..CD)
T_F  = T[:, 0:4]  # colunas 0..3 -> FA, FB, FC, FD
T_CP = T[:, 4:8]  # colunas 4..7 -> CA, CB, CC, CD

# ==== VETOR a – V_torre ≈ a0*V0 + ... + a4*V4 (calibrado offline) ====
a_vec = np.array([
    -0.170765,
    -17.341725,
    16.536161,
    -13.135335,
    18.206203
], dtype=float)

# ==== Tolerâncias internas para índices de saúde (ajustar com dados reais) ====
V_TOL_Z = 0.05  # [V] ~ desvio típico aceitável para resíduos em Z
V_TOL_T = 0.05  # [V] ~ desvio típico aceitável para resíduos em T


def estimate_foundation_currents(T_F, T_CP, I_cp, V_meas):
    """
    Estima correntes nas fundações [FA..FD] a partir de:
      - V_meas (5,) -> [V0..V4]
      - I_cp   (4,) -> [I_CA, I_CB, I_CC, I_CD]

    Resolve por mínimos quadrados:
      V_meas ≈ T_F * I_F + T_CP * I_cp
      => T_F * I_F ≈ V_meas - T_CP * I_cp
    """
    V_meas = np.asarray(V_meas, dtype=float).reshape(5)
    I_cp   = np.asarray(I_cp,   dtype=float).reshape(4)

    rhs = V_meas - T_CP @ I_cp
    I_F_est, *_ = np.linalg.lstsq(T_F, rhs, rcond=None)
    return I_F_est  # (4,): [I_FA, I_FB, I_FC, I_FD]


def estimate_online_R_torre(Z, T_F, T_CP, a_vec, I_cp, V_meas):
    """
    Usa Z, T e a_vec para estimar:
      - R_torre_est,
      - V_torre_est,
      - I_tot_est,
      - I_full_hat (correntes nos 8 eletrodos).

    Entradas:
      - Z: matriz de impedâncias 8x8
      - T_F, T_CP: submatrizes de T
      - a_vec: vetor de combinação linear dos sensores
      - I_cp: correntes medidas nos CPs [CA,CB,CC,CD] (A)
      - V_meas: potenciais medidos [V0..V4] (V)
    """
    # 1) Estima correntes nas fundações
    I_F_est = estimate_foundation_currents(T_F, T_CP, I_cp, V_meas)

    # 2) Correntes completas
    I_full = np.concatenate([I_F_est, np.asarray(I_cp, dtype=float).reshape(4)])

    # 3) Corrente total
    I_tot_est = I_full.sum()

    # 4) V_torre estimado via combinação linear dos sensores
    V_meas = np.asarray(V_meas, dtype=float).reshape(5)
    V_torre_est = float(a_vec @ V_meas)

    # 5) R_torre estimado
    R_torre_est = V_torre_est / I_tot_est if I_tot_est != 0 else np.nan

    return R_torre_est, V_torre_est, I_tot_est, I_full


def health_index_Z(Z, I_full, V_torre_est, V_tol=V_TOL_Z):
    """
    Índice de saúde baseado em Z:
      - Calcula V_model = Z * I_full
      - Compara cada V_model[i] com V_torre_est.
      - Define health_i = 1 / (1 + |residuo| / V_tol).

    Retorna:
      - health_electrodes (8,) por eletrodo
      - health_global (média)
      - residuals (8,) em volts
    """
    I_full = np.asarray(I_full, dtype=float).reshape(8)
    V_model = Z @ I_full
    residuals = V_model - V_torre_est  # devia ser ~0 se Z estiver “perfeita”

    # Evita divisão por zero
    V_tol = max(V_tol, 1e-6)

    health_e = 1.0 / (1.0 + np.abs(residuals) / V_tol)
    health_global = float(np.mean(health_e))
    return health_e, health_global, residuals


def health_index_T(T_ref, I_full, V_meas, V_tol=V_TOL_T):
    """
    Índice de saúde baseado em T:
      - Calcula V_pred = T_ref * I_full
      - Compara com V_meas (5 sensores).
      - health_s = 1 / (1 + |residuo| / V_tol).

    Retorna:
      - health_sensors (5,)
      - health_global (média)
      - residuals (5,)
    """
    I_full = np.asarray(I_full, dtype=float).reshape(8)
    V_meas = np.asarray(V_meas, dtype=float).reshape(5)

    V_pred = T_ref @ I_full
    residuals = V_meas - V_pred

    V_tol = max(V_tol, 1e-6)
    health_s = 1.0 / (1.0 + np.abs(residuals) / V_tol)
    health_global = float(np.mean(health_s))

    return health_s, health_global, residuals


def print_diagnostic_snapshot(Z, T, T_F, T_CP, a_vec, I_cp_meas, V_meas):
    """
    Rodar um snapshot completo:
      - Estima R_torre, V_torre, I_tot, I_full
      - Calcula health baseado em Z
      - Calcula health baseado em T
      - Imprime tudo de forma amigável
    """
    R_hat, V_t_hat, I_tot_hat, I_full_hat = estimate_online_R_torre(
        Z, T_F, T_CP, a_vec, I_cp_meas, V_meas
    )

    print("====== SNAPSHOT ONLINE ======")
    print(f"I_tot_est      = {I_tot_hat:.4f} A")
    print(f"V_torre_est    = {V_t_hat:.4f} V")
    print(f"R_torre_est    = {R_hat:.44f} Ω\n")

    print("Correntes estimadas em cada eletrodo:")
    for name, I_val in zip(electrodes, I_full_hat):
        print(f"  {name}: {I_val:.4f} A")
    print()

    # Índice de saúde baseado em Z
    health_Z_e, health_Z_global, res_Z = health_index_Z(Z, I_full_hat, V_t_hat)
    print("=== Health baseado em Z (equipotencialidade nos eletrodos) ===")
    print(f"Health_Z_global = {health_Z_global:.3f}")
    for name, h, r in zip(electrodes, health_Z_e, res_Z):
        print(f"  {name}: health={h:.3f}, resid_Z={r:.4f} V")
    print()

    # Índice de saúde baseado em T
    health_T_s, health_T_global, res_T = health_index_T(T, I_full_hat, V_meas)
    print("=== Health baseado em T (sensores de potencial) ===")
    print(f"Health_T_global = {health_T_global:.3f}")
    for s_idx, (h, r) in enumerate(zip(health_T_s, res_T)):
        print(f"  V{s_idx}: health={h:.3f}, resid_T={r:.4f} V")


if __name__ == "__main__":
    # ======================================================================
    # EXEMPLO DE USO
    # Aqui eu só uso valores fictícios para I_cp e V_meas, só pra ver o fluxo.
    # Na prática, você vai substituir por valores medidos pelo edge.
    # ======================================================================

    # Correntes nos cabos contrapeso (A) – exemplo
    I_cp_meas = np.array([0.0946, 0.0505, 0.0686, 0.140])  # [I_CA, I_CB, I_CC, I_CD]

    # Potenciais medidos nos 5 sensores (V) – exemplo
    V_meas = np.array([0.126, 0.145, 0.146, 0.116, 0.126])
   


    print_diagnostic_snapshot(Z, T, T_F, T_CP, a_vec, I_cp_meas, V_meas)


if __name__ == "__main__":
    # ======================================================================
    # EXEMPLO DE USO ONLINE (snapshot)
    # Aqui eu só uso valores fictícios para I_cp e V_meas, só pra ver o fluxo.
    # Na prática, você vai substituir por valores medidos pelo edge.
    # ======================================================================

    # Correntes nos cabos contrapeso (A) – exemplo (Ensaio 1, FFFF)
    I_cp_meas = np.array([0.0946, 0.0505, 0.0686, 0.1400])  # [I_CA, I_CB, I_CC, I_CD]

    # Potenciais medidos nos 5 sensores (V) – exemplo (Ensaio 1, método IA, em volts)
    V_meas = np.array([0.126, 0.145, 0.146, 0.116, 0.126])

    print_diagnostic_snapshot(Z, T, T_F, T_CP, a_vec, I_cp_meas, V_meas)


    # ============================================================
    #  DADOS DOS 16 ENSAIOS (comissionamento)
    #  - Itot_ref em A (corrente residual)
    #  - R_ref em ohms (método passivo)
    #  - V_sens_ref em V (método IA, V0..V4 a 1 m de profundidade)
    # ============================================================

    # Correntes totais (mA) e resistências (Ω) da Tabela de campo
    Itot_mA = np.array([
        631, 580, 715, 805, 877, 860, 767, 744,
        801, 764, 748, 773, 717, 696, 699, 638
    ], dtype=float)

    R_ref = np.array([
        1.150, 1.195, 1.166, 1.190, 1.303, 1.267, 1.253, 1.385,
        1.278, 1.333, 1.371, 1.342, 1.497, 1.454, 1.458, 1.596
    ], dtype=float)

    # Converter Itot para ampères
    Itot_ref = Itot_mA / 1000.0

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

    # Converter para volts
    V_sens_ref = V_sens_mV / 1000.0

    # ============================================================
    #  CHECK FINAL: validar o estimador com os 16 ensaios
    # ============================================================

    print("\n===== VALIDAÇÃO DO ESTIMADOR COM OS 16 ENSAIOS =====")

    R_est_list = []
    V_torre_est_list = []
    V_torre_true_list = []

    for k in range(len(Itot_ref)):
        V_sens_k = V_sens_ref[k, :]      # [V0..V4] do ensaio k (em V)
        Itot_k   = Itot_ref[k]           # corrente total medida (A)
        R_ref_k  = R_ref[k]              # resistência de referência (Ω)

        # Tensão verdadeira da torre pelo método passivo
        V_torre_true = Itot_k * R_ref_k

        # Tensão estimada a partir dos 5 sensores (usando a_vec)
        V_torre_est  = float(V_sens_k @ a_vec)

        # Resistência estimada
        R_est = V_torre_est / Itot_k

        R_est_list.append(R_est)
        V_torre_est_list.append(V_torre_est)
        V_torre_true_list.append(V_torre_true)

        print(f"Ensaio {k+1:2d} | "
              f"V_t_true = {V_torre_true:6.4f} V | "
              f"V_t_est = {V_torre_est:6.4f} V | "
              f"R_ref = {R_ref_k:6.3f} Ω | "
              f"R_est = {R_est:6.3f} Ω | "
              f"Erro_R = {R_est - R_ref_k:+6.3f} Ω")

    R_est_arr        = np.array(R_est_list)
    V_torre_est_arr  = np.array(V_torre_est_list)
    V_torre_true_arr = np.array(V_torre_true_list)

    # Métricas globais
    rmse_V = np.sqrt(np.mean((V_torre_est_arr - V_torre_true_arr)**2))
    rmse_R = np.sqrt(np.mean((R_est_arr - R_ref)**2))
    mean_err_R = np.mean(R_est_arr - R_ref)
    max_err_R  = np.max(np.abs(R_est_arr - R_ref))

    print("\nResumo da validação (16 ensaios):")
    print(f"  RMSE(V_torre) = {rmse_V:.6f} V")
    print(f"  RMSE(R_torre) = {rmse_R:.6f} Ω")
    print(f"  Erro médio em R_torre = {mean_err_R:+.6f} Ω")
    print(f"  Erro máximo |R_est - R_ref| = {max_err_R:.6f} Ω")

