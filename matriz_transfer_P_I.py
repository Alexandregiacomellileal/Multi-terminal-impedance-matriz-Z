# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 17:22:06 2025

@author: alexa
"""

# -*- coding: utf-8 -*-
"""
Identificação de matriz de transferência T entre correntes nos eletrodos
(FA, FB, FC, FD, CA, CB, CC, CD) e potenciais medidos no solo
(V0_torre, V1, V2, V3, V4), usando mínimos quadrados.

Alexandre G. Leal
"""

import numpy as np

# Ordem dos eletrodos (coerente com a matriz Z e demais scripts)
labels_eletrodos = ["FA", "FB", "FC", "FD", "CA", "CB", "CC", "CD"]

# Correntes medidas em cada eletrodo (A)
# 16 ensaios × 8 eletrodos
# Extraído do output do teste_final_torre.py (I_meas)
currents_meas = np.array([
    [0.0714, 0.0615, 0.0584, 0.0760, 0.0946, 0.0505, 0.0686, 0.1400],  # Ensaio 1
    [0.1070, 0.0720, 0.0630, 0.1067, 0.0000, 0.0610, 0.0750, 0.1453],  # Ensaio 2
    [0.0800, 0.0780, 0.0630, 0.0980, 0.1020, 0.0000, 0.0770, 0.1510],  # Ensaio 3
    [0.0835, 0.0757, 0.0850, 0.1357, 0.1205, 0.0653, 0.0000, 0.1943],  # Ensaio 4
    [0.1050, 0.0832, 0.0880, 0.2510, 0.1520, 0.0788, 0.1310, 0.0000],  # Ensaio 5
    [0.1470, 0.1150, 0.0864, 0.1945, 0.0000, 0.0000, 0.1156, 0.2365],  # Ensaio 6
    [0.0980, 0.1010, 0.0990, 0.1727, 0.1380, 0.0000, 0.0000, 0.2203],  # Ensaio 7
    [0.1040, 0.0820, 0.1120, 0.2510, 0.1420, 0.0890, 0.0000, 0.0000],  # Ensaio 8
    [0.1310, 0.0910, 0.0980, 0.1780, 0.0000, 0.0800, 0.0000, 0.2230],  # Ensaio 9
    [0.1019, 0.1010, 0.0830, 0.2210, 0.1401, 0.0000, 0.1190, 0.0000],  # Ensaio 10
    [0.1370, 0.0897, 0.0850, 0.2350, 0.0000, 0.0803, 0.1190, 0.0000],  # Ensaio 11
    [0.1410, 0.1140, 0.1050, 0.1886, 0.0000, 0.0000, 0.0000, 0.2204],  # Ensaio 12
    [0.1440, 0.0998, 0.1160, 0.2690, 0.0000, 0.0872, 0.0000, 0.0000],  # Ensaio 13
    [0.1460, 0.1130, 0.0923, 0.2430, 0.0000, 0.0000, 0.1227, 0.0000],  # Ensaio 14
    [0.1047, 0.1060, 0.1130, 0.2330, 0.1423, 0.0000, 0.0000, 0.0000],  # Ensaio 15
    [0.1480, 0.1180, 0.1160, 0.2530, 0.0000, 0.0000, 0.0000, 0.0000],  # Ensaio 16
])

# Potenciais medidos – Método IA (mV)
# Colunas: V0_torre, V1, V2, V3, V4
potentials_mV = np.array([
    [126, 145, 146, 116, 126],  # Ensaio 1
    [154, 192, 158, 125, 165],  # Ensaio 2
    [175, 208, 212, 155, 165],  # Ensaio 3
    [195, 204, 233, 200, 175],  # Ensaio 4
    [230, 239, 253, 238, 235],  # Ensaio 5
    [265, 330, 284, 210, 267],  # Ensaio 6
    [215, 236, 268, 211, 192],  # Ensaio 7
    [233, 232, 273, 261, 221],  # Ensaio 8
    [246, 280, 264, 225, 246],  # Ensaio 9
    [230, 255, 263, 219, 214],  # Ensaio 10
    [254, 298, 260, 235, 277],  # Ensaio 11
    [285, 340, 322, 248, 270],  # Ensaio 12
    [298, 322, 313, 289, 292],  # Ensaio 13
    [280, 330, 298, 234, 279],  # Ensaio 14
    [256, 265, 309, 267, 225],  # Ensaio 15
    [310, 360, 350, 295, 300],  # Ensaio 16
])

sensor_labels = ["V0_torre", "V1", "V2", "V3", "V4"]

# Converte potenciais para volts
V_sensors = potentials_mV / 1000.0  # [16 × 5] em V
I_elet = currents_meas              # [16 × 8] em A

# Matrizes para o problema de mínimos quadrados
X = I_elet        # 16 × 8  (correntes)
Y = V_sensors     # 16 × 5  (potenciais)

# Matriz de transferência T (5 sensores × 8 eletrodos)
T = np.zeros((Y.shape[1], X.shape[1]))

for s in range(Y.shape[1]):
    # resolve: X * t_s ≈ Y[:, s]
    t_s, *_ = np.linalg.lstsq(X, Y[:, s], rcond=None)
    T[s, :] = t_s

# --- Saída dos resultados ---

print("Eletrodos (ordem em T):", labels_eletrodos)
print("\nMatriz de transferência T (sensores × eletrodos) em [V/A]:\n")

for s_idx, s_name in enumerate(sensor_labels):
    linha = ", ".join(f"{coef:.4f}" for coef in T[s_idx, :])
    print(f"{s_name}: [{linha}]")

# Checagem de ajuste (reconstrução dos potenciais)
V_hat = X @ T.T           # [16 × 5]
res = V_hat - Y           # resíduos
rmse = np.sqrt(np.mean(res**2, axis=0))

print("\nRMSE por sensor:")
for s_idx, s_name in enumerate(sensor_labels):
    print(
        f"{s_name}: RMSE = {rmse[s_idx]:.6f} V  "
        f"({rmse[s_idx]*1000:.3f} mV)"
    )

# Se quiser ver as primeiras linhas de comparação:
print("\nPrimeiros ensaios – medido vs modelado (mV):")
for k in range(5):
    med = Y[k, :] * 1000
    mod = V_hat[k, :] * 1000
    print(f"Ensaio {k+1:2d}: med = {med},  mod = {mod}")



# Check

import numpy as np

# Ordem dos eletrodos deve bater com a usada na regressão:
labels = ["FA", "FB", "FC", "FD", "CA", "CB", "CC", "CD"]

T = np.array([
    [-1.5003,  2.1478,  1.5500,  0.3675, -0.4896,  0.2338,  0.2263,  0.0472],
    [ 0.4186,  1.7336,  0.1604,  0.2623, -0.2356, -0.0032,  0.0129,  0.0286],
    [-0.6880,  1.5729,  1.6048,  0.2831, -0.1772, -0.0632,  0.0619,  0.0381],
    [-0.3408,  1.4764,  0.4328,  0.4416, -0.0160,  0.2767, -0.1855, -0.0373],
    [-0.0654,  0.8117,  1.0296,  0.3379, -0.3091,  0.1317,  0.2193,  0.0354],
])

def predict_potentials(I_dict):
    """
    I_dict: dicionário com correntes em A por eletrodo.
            Exemplo:
            {
              "FA": 0.10, "FB": 0.08, "FC": 0.09, "FD": 0.20,
              "CA": 0.15, "CB": 0.07, "CC": 0.12, "CD": 0.00
            }
    Retorna: vetor V (5,) com tensões em volts [V0, V1, V2, V3, V4]
    """
    I_vec = np.array([I_dict.get(name, 0.0) for name in labels], dtype=float)
    V_vec = T @ I_vec
    return V_vec

# Exemplo de uso: correntes parecidas com um ensaio qualquer
I_exemplo = {
    "FA": 0.10,
    "FB": 0.08,
    "FC": 0.09,
    "FD": 0.20,
    "CA": 0.15,
    "CB": 0.07,
    "CC": 0.12,
    "CD": 0.00,
}

V_pred = predict_potentials(I_exemplo)
print("Potenciais previstos (V0...V4) em volts:", V_pred)
print("Em mV:", V_pred * 1000)
