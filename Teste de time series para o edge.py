# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 16:15:52 2025

@author: alexa
"""

# -*- coding: utf-8 -*-
"""
Teste de time series para o edge:
 - Gera 24 amostras (uma por hora)
 - Horas 0..11: sistema saudável
 - Horas 12..23: defeito sintético em CA (corrente "esperada" diminuída)
 - Para cada hora:
      -> gera I_full (correntes em todos eletrodos) sintético
      -> calcula V_meas = T * I_full + ruído pequeno
      -> edge_step() vê apenas I_cp_meas e V_meas
      -> grava resultados em tela e em CSV

Requer:
  - edge_runtime.py na mesma pasta, com:
      Z, T, T_F, T_CP, a_V, a_I, electrodes, edge_step
"""

import numpy as np
import csv
from edge_runtime import T, electrodes, edge_step  # importa do runtime

# === 1) Definir um perfil de referência de correntes (torre "saudável") ===
# Vamos usar como referência os valores do snapshot (Ensaio 1) que você já tinha:
#  FA: 0.0707 A
#  FB: 0.0597 A
#  FC: 0.0564 A
#  FD: 0.0764 A
#  CA: 0.0946 A
#  CB: 0.0505 A
#  CC: 0.0686 A
#  CD: 0.1400 A

I_full_ref = np.array([
    0.0707,  # FA
    0.0597,  # FB
    0.0564,  # FC
    0.0764,  # FD
    0.0946,  # CA
    0.0505,  # CB
    0.0686,  # CC
    0.1400   # CD
], dtype=float)

# V_ref correspondente (sensores) pela matriz T
V_ref = T @ I_full_ref   # (5,)

# === 2) Configuração da simulação ===
N = 24                   # 24 horas
hours = np.arange(N)     # 0..23

# Ruído pequeno (p.ex. 1% do valor)
NOISE_V_STD   = 0.005    # ~5 mV
NOISE_I_STD   = 0.002    # ~2 mA

# Fator de "carga diária" (variação senoidal suave)
load_factor = 1.0 + 0.1 * np.sin(2.0 * np.pi * hours / 24.0)


# === 3) Geração dos dados e chamada ao edge_step ===
rows = []

print("=== SIMULAÇÃO 24h (com defeito sintético em CA a partir da hora 12) ===\n")

for h in hours:
    lf = load_factor[h]

    # -- Caso base: todos eletrodos saudáveis
    I_full = I_full_ref * lf

    # A partir da hora 12, aplicar um defeito sintético em CA (reduzido 40%)
    if h >= 12:
        I_full_fault = I_full.copy()
        I_full_fault[4] *= 0.6  # coluna 4 = CA
        I_full = I_full_fault

    # Correntes reais com ruído
    I_full_meas = I_full + np.random.normal(0.0, NOISE_I_STD, size=8)

    # Potenciais nos sensores (V) = T * I_full + ruído
    V_meas = T @ I_full_meas + np.random.normal(0.0, NOISE_V_STD, size=5)

    # O edge só enxerga I_cp (cabos) e V_meas
    I_cp_meas = I_full_meas[4:8]  # CA, CB, CC, CD

    # Chama o passo do edge
    out = edge_step(I_cp_meas, V_meas)

    R_hat      = out["R_hat"]
    V_t_hat    = out["V_torre_hat"]
    I_tot_hat  = out["I_tot_hat"]
    health_Z_g = out["health_Z"]["global"]
    health_T_g = out["health_T"]["global"]

    # Marcar se é hora com defeito
    fault_flag = 1 if h >= 12 else 0

    print(f"Hora {h:02d} | fault={fault_flag} | "
          f"R_hat={R_hat:.4f} Ω | "
          f"Health_Z={health_Z_g:.3f} | "
          f"Health_T={health_T_g:.3f}")

    # Monta linha para CSV
    row = {
        "hour": int(h),
        "fault_flag": int(fault_flag),
        "I_CA": I_cp_meas[0],
        "I_CB": I_cp_meas[1],
        "I_CC": I_cp_meas[2],
        "I_CD": I_cp_meas[3],
        "V0": V_meas[0],
        "V1": V_meas[1],
        "V2": V_meas[2],
        "V3": V_meas[3],
        "V4": V_meas[4],
        "R_hat": R_hat,
        "V_torre_hat": V_t_hat,
        "I_tot_hat": I_tot_hat,
        "health_Z_global": health_Z_g,
        "health_T_global": health_T_g,
    }

    # Também pode gravar health por eletrodo e sensor, se quiser
    for name, hval in out["health_Z"]["by_electrode"].items():
        row[f"healthZ_{name}"] = hval
    for sname, hval in out["health_T"]["by_sensor"].items():
        row[f"healthT_{sname}"] = hval

    rows.append(row)

# === 4) Salvar CSV para análise offline ===
csv_file = "edge_test_24h.csv"
fieldnames = list(rows[0].keys())

with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"\nArquivo CSV gerado: {csv_file}")
