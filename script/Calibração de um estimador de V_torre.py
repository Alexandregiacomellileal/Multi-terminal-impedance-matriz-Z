# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 09:13:55 2025

@author: Alexandre
"""

# -*- coding: utf-8 -*-
"""
Calibração de um estimador de V_torre (e R_torre) usando:
  - Potenciais V0..V4 (Método IA, 1 m abaixo da superfície, em 5 pontos)
  - Correntes nos cabos contrapeso CA..CD (Hioki)
  - 16 ensaios de comissionamento

Modelo:
    V_torre ≈ a0*V0 + a1*V1 + a2*V2 + a3*V3 + a4*V4
              + b0*I_CA + b1*I_CB + b2*I_CC + b3*I_CD

Depois: R_est = V_t_est / I_tot

Autor: Alexandre + Chat :)
"""

import numpy as np

# ---------------------------------------------------------
# 1) Dados dos 16 ensaios (como na tua planilha)
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
# (mesmo que IA_Hioki, IB_Hioki, IC_Hioki, ID_Hioki, em mA)
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

# ---------------------------------------------------------
# 2) Conversão para unidades de SI
#    V em volts, I em ampères
# ---------------------------------------------------------
V_sens_ref = np.array(V_mV, dtype=float) / 1000.0  # (16 x 5) em V
Itot_ref = np.array(Itot_mA, dtype=float) / 1000.0  # (16,) em A
I_cp_ref = np.array(cp_mA, dtype=float) / 1000.0    # (16 x 4) em A
R_ref = np.array(R_ref, dtype=float)                # (16,) em Ω

# Tensão verdadeira da torre (via método passivo)
V_t_true = Itot_ref * R_ref  # (16,)

# ---------------------------------------------------------
# 3) Monta matriz de regressão: X = [V0..V4, I_CA..I_CD]
# ---------------------------------------------------------
# X tem dimensão 16 x 9
X = np.hstack([V_sens_ref, I_cp_ref])

# Resolve mínimos quadrados: X * a_full ≈ V_t_true
# a_full = [a0..a4, b0..b3]
a_full, residuals, rank, s = np.linalg.lstsq(X, V_t_true, rcond=None)

a_V = a_full[:5]  # coeficientes dos sensores de potencial
a_I = a_full[5:]  # coeficientes das correntes dos CPs

# Estimativas com o modelo ajustado
V_t_est = X @ a_full          # (16,)
R_est = V_t_est / Itot_ref    # (16,)

# ---------------------------------------------------------
# 4) Métricas de erro
# ---------------------------------------------------------
rmse_V = np.sqrt(np.mean((V_t_est - V_t_true)**2))
rmse_R = np.sqrt(np.mean((R_est - R_ref)**2))
mean_err_R = np.mean(R_est - R_ref)
max_err_R = np.max(np.abs(R_est - R_ref))

# ---------------------------------------------------------
# 5) Impressão dos resultados
# ---------------------------------------------------------
print("===== CALIBRAÇÃO DO ESTIMADOR V_torre = f(V_sens, I_cp) =====\n")

print("Coeficientes a_V (V_torre ≈ a_V^T * V + a_I^T * I_cp):")
for i, val in enumerate(a_V):
    print(f"  a{i} (V{i}) = {val:.6f}")
print("\nCoeficientes a_I (ganhos em relação às correntes dos CPs):")
for nome, val in zip(["CA", "CB", "CC", "CD"], a_I):
    print(f"  b_{nome} = {val:.6f}  [ohms aproximadamente]")

print("\nResumo dos erros:")
print(f"  RMSE(V_torre) = {rmse_V:.6f} V")
print(f"  RMSE(R_torre) = {rmse_R:.6f} Ω")
print(f"  Erro médio em R_torre = {mean_err_R:+.6f} Ω")
print(f"  Erro máximo |R_est - R_ref| = {max_err_R:.6f} Ω\n")

print("Comparação ensaio a ensaio:")
print("Ensaio | V_t_true (V) | V_t_est (V) | R_ref (Ω) | R_est (Ω) | Erro_R (Ω)")
for k in range(16):
    print(f"{k+1:6d} | "
          f"{V_t_true[k]:11.4f} | {V_t_est[k]:11.4f} | "
          f"{R_ref[k]:8.3f} | {R_est[k]:8.3f} | {R_est[k] - R_ref[k]:+8.3f}")

# ---------------------------------------------------------
# 6) Exemplo de uso "online" com um dos ensaios (por ex., Ensaio 1)
# ---------------------------------------------------------
k_ex = 0  # índice 0 -> Ensaio 1
V_meas_ex = V_sens_ref[k_ex, :]  # [V0..V4] em V
I_cp_ex = I_cp_ref[k_ex, :]      # [I_CA..I_CD] em A
Itot_ex = Itot_ref[k_ex]         # corrente total de referência (só para exemplo)

V_t_hat_ex = float(a_V @ V_meas_ex + a_I @ I_cp_ex)
R_hat_ex = V_t_hat_ex / Itot_ex

print("\nExemplo (Ensaio 1 reaplicado pelo modelo):")
print(f"  V_t_true = {V_t_true[k_ex]:.4f} V, R_ref = {R_ref[k_ex]:.3f} Ω")
print(f"  V_t_hat  = {V_t_hat_ex:.4f} V, R_hat = {R_hat_ex:.3f} Ω")
