# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 15:39:40 2025

@author: alexa
"""

# -*- coding: utf-8 -*-
"""
Reconstrução da matriz de impedâncias Z (8x8) a partir das medições de campo,
e exportação das 96 equações em formato CSV para uso no Excel.

Passos:
1) Definir rótulos dos eletrodos e mapa (i,j) -> índice no vetor z (36 incógnitas).
2) Inserir as medições de campo (correntes AEMC/Hioki em mA, Itotal, R).
3) Reconstruir as correntes em cada eletrodo (FA..FD, CA..CD) em ampères.
4) Calcular V_torre^(k) = Itot^(k) * R^(k).
5) Montar as equações linha a linha: V_torre = sum_j Z_ij * I_j.
6) Resolver o sistema superdeterminado A z ≈ v por mínimos quadrados.
7) Reconstruir a matriz Z (8x8) a partir de z.
8) Exportar as 96 equações para um arquivo CSV ("equacoes_Z.csv").
"""

import numpy as np
import csv

# ------------------------------------------------------------
# 1) Definições básicas
# ------------------------------------------------------------

# Ordem fixa dos eletrodos
labels = ["FA", "FB", "FC", "FD", "CA", "CB", "CC", "CD"]
n_eletrodos = len(labels)

# Mapeia nome -> índice (0..7)
idx_map = {name: i for i, name in enumerate(labels)}

# Combinações dos cabos contrapeso (CPA..CPD) por ensaio (como na tabela)
cp_patterns = [
    "FFFF",  # 1
    "AFFF",  # 2
    "FAFF",  # 3
    "FFAF",  # 4
    "FFFA",  # 5
    "AAFF",  # 6
    "FAAF",  # 7
    "FFAA",  # 8
    "AFAF",  # 9
    "FAFA",  # 10
    "AFFA",  # 11
    "AAAF",  # 12
    "AFAA",  # 13
    "AAFA",  # 14
    "FAAA",  # 15
    "AAAA",  # 16
]

# ------------------------------------------------------------
# 2) Medições de campo (copiadas da Tabela de medições)
#    Todas as correntes em mA, R em ohms
# ------------------------------------------------------------

# Correntes nas pernas da torre pelo AEMC (mA)
IA_AEMC = np.array([166, 107, 182, 204, 257, 147, 236, 246,
                    131, 242, 137, 141, 144, 146, 247, 148], dtype=float)

IB_AEMC = np.array([112, 133,  78, 141, 162, 115, 101, 171,
                    171, 101, 170, 114, 187, 113, 106, 118], dtype=float)

IC_AEMC = np.array([127, 138, 140,  85, 219, 202,  99, 112,
                     98, 202, 204, 105, 116, 215, 113, 116], dtype=float)

ID_AEMC = np.array([216, 252, 249, 330, 251, 431, 393, 251,
                    401, 221, 235, 409, 269, 243, 233, 253], dtype=float)

# Correntes nos cabos CP pelo Hioki (mA)
IA_Hioki = np.array([ 94.6,   0.0, 102.0, 120.5, 152.0,   0.0, 138.0, 142.0,
                       0.0, 140.1,   0.0,   0.0,   0.0,   0.0, 142.3,   0.0], dtype=float)

IB_Hioki = np.array([ 50.5,  61.0,   0.0,  65.3,  78.8,   0.0,   0.0,  89.0,
                      80.0,   0.0,  80.3,   0.0,  87.2,   0.0,   0.0,   0.0], dtype=float)

IC_Hioki = np.array([ 68.6,  75.0,  77.0,   0.0, 131.0, 115.6,   0.0,   0.0,
                       0.0, 119.0, 119.0,   0.0,   0.0, 122.7,   0.0,   0.0], dtype=float)

ID_Hioki = np.array([140.0, 145.3, 151.0, 194.3,   0.0, 236.5, 220.3,   0.0,
                     223.0,   0.0,   0.0, 220.4,   0.0,   0.0,   0.0,   0.0], dtype=float)

# Corrente total Itotal (mA) e resistência global R (ohms) por ensaio
Itot_mA = np.array([631, 580, 715, 805, 877, 860, 767, 744,
                    801, 764, 748, 773, 717, 696, 699, 638], dtype=float)

R_ohm   = np.array([1.150, 1.195, 1.166, 1.190, 1.303, 1.267, 1.253, 1.385,
                    1.278, 1.333, 1.371, 1.342, 1.497, 1.454, 1.458, 1.596], dtype=float)

# ------------------------------------------------------------
# 3) Reconstrução das correntes em cada eletrodo (em A)
#    Usando a regra: I_FA = (IA_AEMC - IA_Hioki)/1000, I_CA = IA_Hioki/1000, etc.
# ------------------------------------------------------------

n_ensaio = len(Itot_mA)
I_eletrodos = np.zeros((n_ensaio, n_eletrodos), dtype=float)  # 16 x 8

for k in range(n_ensaio):
    # fundações (corrente na perna - corrente no cabo, em A)
    I_FA = (IA_AEMC[k] - IA_Hioki[k]) / 1000.0
    I_FB = (IB_AEMC[k] - IB_Hioki[k]) / 1000.0
    I_FC = (IC_AEMC[k] - IC_Hioki[k]) / 1000.0
    I_FD = (ID_AEMC[k] - ID_Hioki[k]) / 1000.0

    # cabos contrapeso (corrente Hioki em A)
    I_CA = IA_Hioki[k] / 1000.0
    I_CB = IB_Hioki[k] / 1000.0
    I_CC = IC_Hioki[k] / 1000.0
    I_CD = ID_Hioki[k] / 1000.0

    I_eletrodos[k, :] = [I_FA, I_FB, I_FC, I_FD, I_CA, I_CB, I_CC, I_CD]

# ------------------------------------------------------------
# 4) Cálculo de V_torre^(k) para cada ensaio
#    V_torre = Itot(A) * R
# ------------------------------------------------------------

Itot_A = Itot_mA / 1000.0
V_torre = Itot_A * R_ohm  # vetor (16,)

# ------------------------------------------------------------
# 5) Montagem do sistema A z ≈ v
#    z empilha os 36 elementos independentes de Z (diagonal + superior)
# ------------------------------------------------------------

# 5.1) Montar a lista de pares (i,j) com i <= j
pairs = []
for i in range(n_eletrodos):
    pairs.append((i, i))  # diagonais
for i in range(n_eletrodos):
    for j in range(i + 1, n_eletrodos):
        pairs.append((i, j))  # parte superior

n_vars = len(pairs)  # deve ser 36

# 5.2) Mapa (i,j) (i <= j) -> índice no vetor z
pair_to_idx = {pair: idx for idx, pair in enumerate(pairs)}

def idx_Z(i, j):
    """
    Retorna o índice no vetor z correspondente a Z_ij,
    usando simetria Z_ij = Z_ji.
    """
    if i <= j:
        return pair_to_idx[(i, j)]
    else:
        return pair_to_idx[(j, i)]

# Listas para as equações
A_rows = []
v_rows = []
eq_ensaio = []   # qual ensaio gerou a equação
eq_eletrodo_i = []  # qual eletrodo i (0..7) gerou a equação

for k in range(n_ensaio):
    # conjunto S_k = fundações (0..3) + CPs fechados (4..7)
    Sk = set([0, 1, 2, 3])  # FA..FD sempre ligadas
    pattern = cp_patterns[k]  # ex: "FFFF", "AFFF", ...

    # posições: 0->CA, 1->CB, 2->CC, 3->CD
    if pattern[0] == "F":
        Sk.add(idx_map["CA"])
    if pattern[1] == "F":
        Sk.add(idx_map["CB"])
    if pattern[2] == "F":
        Sk.add(idx_map["CC"])
    if pattern[3] == "F":
        Sk.add(idx_map["CD"])

    I_k = I_eletrodos[k, :]
    V_k = V_torre[k]

    # Para cada eletrodo i ligado naquele ensaio, gera uma equação
    for i in sorted(Sk):
        row = np.zeros(n_vars, dtype=float)
        for j in Sk:
            idx_var = idx_Z(i, j)
            row[idx_var] += I_k[j]  # coeficiente = corrente em j

        A_rows.append(row)
        v_rows.append(V_k)
        eq_ensaio.append(k + 1)     # ensaio numerado de 1 a 16
        eq_eletrodo_i.append(i)     # índice 0..7 (FA..CD)

# Converter listas em arrays
A = np.vstack(A_rows)   # (n_eq, 36)
v = np.array(v_rows)    # (n_eq,)

print(f"Número de equações montadas: {A.shape[0]}")
print(f"Número de incógnitas (elementos independentes de Z): {A.shape[1]}\n")

# ------------------------------------------------------------
# 6) Resolver o sistema por mínimos quadrados
# ------------------------------------------------------------

z_star, residuals, rank, s = np.linalg.lstsq(A, v, rcond=None)

print("Informações da solução de mínimos quadrados:")
print(f"  Resíduo total (soma dos quadrados): {residuals}")
print(f"  posto (rank) da matriz A: {rank}")
print(f"  número de incógnitas: {n_vars}\n")

# ------------------------------------------------------------
# 7) Reconstruir a matriz Z (8x8) a partir do vetor z_star
# ------------------------------------------------------------

Z_rec = np.zeros((n_eletrodos, n_eletrodos), dtype=float)

for idx, (i, j) in enumerate(pairs):
    Z_rec[i, j] = z_star[idx]
    Z_rec[j, i] = z_star[idx]  # simetria

print("Matriz de impedâncias Z identificada (Ω):\n")
for i, name_i in enumerate(labels):
    row_str = " ".join(f"{Z_rec[i, j]:7.4f}" for j in range(n_eletrodos))
    print(f"{name_i}: {row_str}")

print("\n=== Formato LaTeX (tabular) ===\n")
print("\\begin{tabular}{l" + "r" * n_eletrodos + "}")
print("\\toprule")
header = " & ".join([" "] + labels) + " \\\\"
print(header)
print("\\midrule")
for i, name_i in enumerate(labels):
    row_vals = " & ".join(f"{Z_rec[i, j]:.4f}" for j in range(n_eletrodos))
    print(f"{name_i} & {row_vals} \\\\")
print("\\bottomrule")
print("\\end{tabular}")

# ------------------------------------------------------------
# 8) Exportar as 96 equações para CSV (para abrir no Excel)
# ------------------------------------------------------------

# Nome das variáveis Z_ij na mesma ordem de "pairs"
var_names = []
for (i, j) in pairs:
    name_i = labels[i]
    name_j = labels[j]
    var_names.append(f"Z_{name_i}_{name_j}")

csv_name = "equacoes_Z.csv"
with open(csv_name, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter=';')  # ; ajuda no Excel PT-BR

    # Cabeçalho
    header_row = ["eq_id", "ensaio", "eletrodo_i", "V_torre"] + var_names
    writer.writerow(header_row)

    # Linhas (uma por equação)
    for eq_id, (k_ensaio, i_elec, V_eq, coeffs) in enumerate(
        zip(eq_ensaio, eq_eletrodo_i, v_rows, A_rows), start=1
    ):
        eletrode_name = labels[i_elec]
        row = [eq_id, k_ensaio, eletrode_name, V_eq] + list(coeffs)
        writer.writerow(row)

print(f"\nArquivo '{csv_name}' gerado com sucesso!")
print("Cada linha corresponde a uma equação do tipo:")
print("  V_torre(k) = soma_j { coef_ij * Z_ij }")
print("onde os coeficientes são as correntes nos eletrodos (A) para aquele ensaio.")
