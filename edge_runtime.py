# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 15:55:14 2025

@author: alexandre
"""

# -*- coding: utf-8 -*-
"""
Runtime para monitoramento online da "saúde" do aterramento de uma torre,
usando apenas parâmetros calibrados previamente:

 - Matriz de impedâncias Z (8x8)
 - Matriz de transferência T (5x8) corrente->potencial
 - Coeficientes a_V e a_I para estimar V_torre a partir de V_sensores e I_cp

Entradas em tempo real (do edge):
 - I_cp_meas: correntes nos cabos contrapeso [I_CA, I_CB, I_CC, I_CD] (A)
 - V_meas: potenciais em 5 sensores [V0, V1, V2, V3, V4] (V)

Saídas principais:
 - R_torre_est  (Ω)
 - V_torre_est  (V)
 - I_tot_est    (A)
 - I_full_hat   (A) correntes estimadas em cada eletrodo [FA..CD]
 - Índices de saúde baseados em Z e em T
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
# Linhas: sensores V0..V4; colunas: FA, FB, FC, FD, CA, CB, CC, CD
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

# ==== COEFICIENTES FIXOS PARA V_torre (calibrados offline) ====
# V_torre_est ≈ a_V^T * V_meas + a_I^T * I_cp_meas
a_V = np.array([
    2.949004,   # V0
   -0.180613,   # V1
    0.075346,   # V2
    1.281876,   # V3
   -0.800849    # V4
], dtype=float)

a_I = np.array([
    0.814810,   # CA
    1.094683,   # CB
    1.224316,   # CC
    0.615223    # CD
], dtype=float)

# ==== Tolerâncias para índices de saúde (ajuste fino depois com dados reais) ====
V_TOL_Z = 0.05  # [V] – resíduo típico aceitável em Z
V_TOL_T = 0.05  # [V] – resíduo típico aceitável em T


def estimate_foundation_currents(T_F, T_CP, I_cp, V_meas):
    """
    Estima correntes nas fundações [FA..FD] a partir de:
      - V_meas (5,) -> [V0..V4]
      - I_cp   (4,) -> [I_CA, I_CB, I_CC, I_CD]

    Problema de mínimos quadrados:
      V_meas ≈ T_F * I_F + T_CP * I_cp
      => T_F * I_F ≈ V_meas - T_CP * I_cp
    """
    V_meas = np.asarray(V_meas, dtype=float).reshape(5)
    I_cp   = np.asarray(I_cp,   dtype=float).reshape(4)

    rhs = V_meas - T_CP @ I_cp
    I_F_est, *_ = np.linalg.lstsq(T_F, rhs, rcond=None)
    return I_F_est  # (4,) -> [I_FA, I_FB, I_FC, I_FD]


def estimate_online_from_runtime(Z, T_F, T_CP, a_V, a_I, I_cp, V_meas):
    """
    Estimador runtime:

    Entradas:
      - Z    : matriz de impedâncias (8x8)
      - T_F  : submatriz de T (5x4) – fundações
      - T_CP : submatriz de T (5x4) – cabos contrapeso
      - a_V  : coeficientes dos sensores de potencial (5,)
      - a_I  : coeficientes das correntes dos CPs (4,)
      - I_cp : correntes medidas nos CPs [I_CA, I_CB, I_CC, I_CD] (A)
      - V_meas: potenciais medidos [V0..V4] (V)

    Saídas:
      - R_torre_est (Ω)
      - V_torre_est (V)
      - I_tot_est   (A)
      - I_full      (8,) correntes estimadas [FA..CD]
    """
    # 1) Correntes nas fundações via T_F
    I_F_est = estimate_foundation_currents(T_F, T_CP, I_cp, V_meas)

    # 2) Vetor completo de correntes [FA..FD, CA..CD]
    I_cp = np.asarray(I_cp, dtype=float).reshape(4)
    I_full = np.concatenate([I_F_est, I_cp])  # (8,)

    # 3) Corrente total
    I_tot_est = float(I_full.sum())

    # 4) Tensão da torre via combinação linear V e I_cp
    V_meas = np.asarray(V_meas, dtype=float).reshape(5)
    V_torre_est = float(a_V @ V_meas + a_I @ I_cp)

    # 5) Resistência equivalente da torre
    if abs(I_tot_est) > 1e-9:
        R_torre_est = V_torre_est / I_tot_est
    else:
        R_torre_est = np.nan

    return R_torre_est, V_torre_est, I_tot_est, I_full


def health_index_Z(Z, I_full, V_torre_est, V_tol=V_TOL_Z):
    """
    Índice de saúde baseado em Z (equipotencialidade na torre):

      V_model = Z * I_full  (tensão em cada eletrodo)
      Resíduo por eletrodo: r_i = V_model[i] - V_torre_est

      health_i = 1 / (1 + |r_i| / V_tol)

    Retorna:
      - health_electrodes (8,)
      - health_global (média)
      - residuals (8,)
    """
    I_full = np.asarray(I_full, dtype=float).reshape(8)
    V_model = Z @ I_full
    residuals = V_model - V_torre_est

    V_tol = max(V_tol, 1e-6)
    health_e = 1.0 / (1.0 + np.abs(residuals) / V_tol)
    health_global = float(np.mean(health_e))
    return health_e, health_global, residuals


def health_index_T(T_ref, I_full, V_meas, V_tol=V_TOL_T):
    """
    Índice de saúde baseado em T (coerência dos sensores de potencial):

      V_pred = T_ref * I_full
      Resíduo por sensor: r_s = V_meas[s] - V_pred[s]

      health_s = 1 / (1 + |r_s| / V_tol)

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


def print_diagnostic_snapshot(I_cp_meas, V_meas):
    """
    Roda um snapshot completo com os dados de entrada:

      - Estima R_torre, V_torre, I_tot, I_full
      - Calcula health baseado em Z
      - Calcula health baseado em T
      - Imprime tudo
    """
    R_hat, V_t_hat, I_tot_hat, I_full_hat = estimate_online_from_runtime(
        Z, T_F, T_CP, a_V, a_I, I_cp_meas, V_meas
    )

    print("====== SNAPSHOT ONLINE (RUNTIME) ======")
    print(f"I_cp_meas  = {I_cp_meas}")
    print(f"V_meas     = {V_meas}")
    print()
    print(f"I_tot_est  = {I_tot_hat:.4f} A")
    print(f"V_torre_est= {V_t_hat:.4f} V")
    print(f"R_torre_est= {R_hat:.4f} Ω\n")

    print("Correntes estimadas em cada eletrodo:")
    for name, I_val in zip(electrodes, I_full_hat):
        print(f"  {name}: {I_val:.4f} A")
    print()

    # Health baseado em Z
    hZ_e, hZ_global, resZ = health_index_Z(Z, I_full_hat, V_t_hat)
    print("=== Health baseado em Z (equipotencialidade) ===")
    print(f"Health_Z_global = {hZ_global:.3f}")
    for name, h, r in zip(electrodes, hZ_e, resZ):
        print(f"  {name}: health={h:.3f}, resid_Z={r:.4f} V")
    print()

    # Health baseado em T
    hT_s, hT_global, resT = health_index_T(T, I_full_hat, V_meas)
    print("=== Health baseado em T (sensores de potencial) ===")
    print(f"Health_T_global = {hT_global:.3f}")
    for s_idx, (h, r) in enumerate(zip(hT_s, resT)):
        print(f"  V{s_idx}: health={h:.3f}, resid_T={r:.4f} V")
    print()

def edge_step(I_cp_meas, V_meas):
    """
    ÚNICO PASSO DO EDGE.

    Entradas:
      - I_cp_meas: array-like (4,)  -> [I_CA, I_CB, I_CC, I_CD] em A
      - V_meas   : array-like (5,)  -> [V0, V1, V2, V3, V4] em V

    Saída:
      dict com:
        - R_hat        : resistência equivalente da torre (Ω)
        - V_torre_hat  : tensão da torre (V)
        - I_tot_hat    : corrente total no aterramento (A)
        - I_full       : correntes estimadas nos 8 eletrodos (A)
        - health_Z     : {'global':..., 'by_electrode':{nome:valor}}
        - health_T     : {'global':..., 'by_sensor':{'V0':..., ...}}
    """
    # Converte para numpy arrays
    I_cp_meas = np.asarray(I_cp_meas, dtype=float).reshape(4)
    V_meas    = np.asarray(V_meas,    dtype=float).reshape(5)

    # Estimativa principal
    R_hat, V_t_hat, I_tot_hat, I_full = estimate_online_from_runtime(
        Z, T_F, T_CP, a_V, a_I, I_cp_meas, V_meas
    )

    # Health baseado em Z
    hZ_e, hZ_global, _ = health_index_Z(Z, I_full, V_t_hat)
    health_Z = {
        "global": hZ_global,
        "by_electrode": {name: float(h) for name, h in zip(electrodes, hZ_e)}
    }

    # Health baseado em T
    hT_s, hT_global, _ = health_index_T(T, I_full, V_meas)
    health_T = {
        "global": hT_global,
        "by_sensor": {f"V{i}": float(h) for i, h in enumerate(hT_s)}
    }

    return {
        "R_hat":      float(R_hat),
        "V_torre_hat": float(V_t_hat),
        "I_tot_hat":  float(I_tot_hat),
        "I_full":     I_full.copy(),
        "health_Z":   health_Z,
        "health_T":   health_T,
    }


if __name__ == "__main__":
    # EXEMPLO: usar os dados do Ensaio 1 (apenas para teste local)
    # Correntes nos cabos contrapeso (A) – Ensaio 1
    I_cp_meas = np.array([0.0946, 0.0505, 0.0686, 0.1400])  # [I_CA, I_CB, I_CC, I_CD]

    # Potenciais medidos nos 5 sensores (V) – Ensaio 1 (em torno de 0.7-1.0 V, aqui escala mV->V)
    V_meas = np.array([0.126, 0.145, 0.146, 0.116, 0.126])  # [V0..V4]

    print_diagnostic_snapshot(I_cp_meas, V_meas)
    
    
    
