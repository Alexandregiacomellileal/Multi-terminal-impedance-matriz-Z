# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 16:12:23 2025

@author: alexa
"""

# -*- coding: utf-8 -*-
"""
Análise da impedância de laço "vista" pelo alicate Hioki em cada cabo contrapeso,
para as 16 configurações de ensaio da torre.

Compara:
- RatA, RatB, RatC, RatD (medidos em campo pelo Hioki)
- Z_loop (modelo) ≈ Z_self(cabo) + Z_eq(resto da malha) usando a matriz Z 8x8
"""

import numpy as np

# Ordem dos eletrodos na matriz Z
labels = ["FA", "FB", "FC", "FD", "CA", "CB", "CC", "CD"]

# Matriz de impedâncias Z (Ω) em baixa frequência
Z = np.array([
    [4.0376, 0.1157, 0.2339, 1.4337, 1.5107, 0.5729, 0.3916, 0.6488],
    [0.1157, 3.2139, 1.7983, 1.5395, 0.6981, 1.2090, 0.8420, 0.7007],
    [0.2339, 1.7983, 3.4648, 1.3695, 0.6334, 0.8973, 1.1087, 0.7162],
    [1.4337, 1.5395, 1.3695, 1.7111, 0.9491, 1.0106, 0.7293, 0.7661],
    [1.5107, 0.6981, 0.6334, 0.9491, 3.1737, 0.7949, 0.7658, 0.6444],
    [0.5729, 1.2090, 0.8973, 1.0106, 0.7949, 5.3168, 0.6041, 0.6199],
    [0.3916, 0.8420, 1.1087, 0.7293, 0.7658, 0.6041, 4.6417, 0.7096],
    [0.6488, 0.7007, 0.7162, 0.7661, 0.6444, 0.6199, 0.7096, 2.7489]
])


def compute_parallel_equiv_impedance(Zsub):
    """
    Calcula a impedância equivalente vista na torre para um conjunto de eletrodos
    descritos pela submatriz Zsub, assumindo I_tot = 1 A distribuído entre eles.

    Resolve o sistema:
        [ Zsub  -1 ] [ I ] = [ 0 ]
        [ 1^T   0 ] [ V ]   [ 1 ]

    Retorna:
        Z_eq = V  (em ohms)
        I     = vetor de correntes em cada eletrodo (A)
    """
    Zsub = np.asarray(Zsub, dtype=float)
    n = Zsub.shape[0]
    one = np.ones((n, 1))

    A = np.block([
        [Zsub, -one],
        [one.T, np.zeros((1, 1))]
    ])

    b = np.zeros((n + 1, 1))
    b[-1, 0] = 1.0  # I_tot = 1 A

    x = np.linalg.solve(A, b)
    I = x[:n, 0]
    V = x[n, 0]
    return V, I  # Z_eq, correntes


def compute_loop_for_cable(Z, labels, cabo_label, ativos=None):
    """
    Aproxima a impedância de laço 'vista' pelo alicate em um cabo específico.

    - Z: matriz 8x8 de impedâncias.
    - labels: ordem dos eletrodos em Z.
    - cabo_label: "CA", "CB", "CC" ou "CD".
    - ativos: lista de rótulos de eletrodos efetivamente conectados na configuração
      (FFFF, AFFF, ...). Se None, assume todos ativos.

    Modelo aproximado:
        Z_loop ≈ Z_self(cabo) + Z_eq(resto da malha)
    onde Z_eq é calculado com todos os mútuos do sub-sistema "resto".
    """
    cabo_idx = labels.index(cabo_label)

    if ativos is None:
        ativos = labels.copy()

    if cabo_label not in ativos:
        raise ValueError(f"Cabo {cabo_label} não está na lista de eletrodos ativos: {ativos}")

    # Índices dos eletrodos ativos
    idx_ativos = [labels.index(name) for name in ativos]

    # Índices do "resto" = todos os ativos, exceto o cabo medido
    idx_resto = [i for i in idx_ativos if i != cabo_idx]

    # Submatriz do resto da malha
    Z_rest = Z[np.ix_(idx_resto, idx_resto)]

    # Impedância equivalente do resto (inclui mútuos)
    Z_eq_resto, _ = compute_parallel_equiv_impedance(Z_rest)

    # Impedância própria do cabo (diagonal de Z)
    Z_self = Z[cabo_idx, cabo_idx]

    # Aproximação de laço: série cabo + resto
    Z_loop = Z_self + Z_eq_resto

    return Z_loop, Z_self, Z_eq_resto


# Padrões de CPA, CPB, CPC, CPD por ensaio (16 configurações)
patterns = {
    1: "FFFF",
    2: "AFFF",
    3: "FAFF",
    4: "FFAF",
    5: "FFFA",
    6: "AAFF",
    7: "FAAF",
    8: "FFAA",
    9: "AFAF",
    10: "FAFA",
    11: "AFFA",
    12: "AAAF",
    13: "AFAA",
    14: "AAFA",
    15: "FAAA",
    16: "AAAA",
}


def ativos_for_pattern(pat):
    """
    A partir de um padrão CPA/CPB/CPC/CPD (F/A),
    devolve a lista de eletrodos ativos (ligados à torre).
    """
    ativos = ["FA", "FB", "FC", "FD"]  # fundações sempre ligadas
    cp_labels = ["CA", "CB", "CC", "CD"]
    for state, lab in zip(pat, cp_labels):
        if state == "F":
            ativos.append(lab)
    return ativos


# Valores de RatA, RatB, RatC, RatD medidos em campo (Hioki), em ohms
# RatA ↔ CA, RatB ↔ CB, RatC ↔ CC, RatD ↔ CD
Rats = {
    1: dict(CA=4.540, CB=6.540, CC=5.900, CD=4.020),
    2: dict(CA=0.000, CB=6.630, CC=5.930, CD=4.060),
    3: dict(CA=4.590, CB=float("inf"), CC=5.930, CD=4.030),
    4: dict(CA=4.550, CB=6.580, CC=float("inf"), CD=4.110),
    5: dict(CA=4.600, CB=6.560, CC=6.030, CD=float("inf")),
    6: dict(CA=float("inf"), CB=float("inf"), CC=5.960, CD=4.060),
    7: dict(CA=4.630, CB=float("inf"), CC=float("inf"), CD=4.130),
    8: dict(CA=4.620, CB=6.610, CC=float("inf"), CD=float("inf")),
    9: dict(CA=float("inf"), CB=6.670, CC=float("inf"), CD=4.150),
    10: dict(CA=4.700, CB=float("inf"), CC=6.090, CD=float("inf")),
    11: dict(CA=float("inf"), CB=6.680, CC=6.080, CD=float("inf")),
    12: dict(CA=float("inf"), CB=float("inf"), CC=float("inf"), CD=4.170),
    13: dict(CA=float("inf"), CB=6.720, CC=float("inf"), CD=float("inf")),
    14: dict(CA=float("inf"), CB=float("inf"), CC=6.110, CD=float("inf")),
    15: dict(CA=4.670, CB=float("inf"), CC=float("inf"), CD=float("inf")),
    16: dict(CA=float("inf"), CB=float("inf"), CC=float("inf"), CD=float("inf")),
}


def generate_latex_table():
    """
    Gera o código LaTeX de uma tabela com:
    Ensaio, CPs, Cabo, Rat (Hioki), Z_loop (modelo), Diferença (Hioki − modelo).
    Só inclui linhas para cabos que estão em estado F na configuração.
    """
    lines = []
    header = (
        "\\begin{sidewaystable}[htp]\n"
        "\\centering\n"
        "\\scriptsize\n"
        "\\caption{Comparação entre as resistências de laço medidas pelo alicate Hioki (RatA/B/C/D)\n"
        "e as impedâncias de laço aproximadas $Z_{\\text{loop}}$ obtidas a partir da matriz de\n"
        "impedâncias $\\mathbf{Z}$, para cada ensaio e cabo contrapeso conectado.}\n"
        "\\label{tab:loop_vs_rat}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{cllrrr}\n"
        "\\toprule\n"
        "Ensaio & CPs & Cabo & Rat$_{\\text{Hioki}}$ ($\\Omega$) & "
        "$Z_{\\text{loop}}$ (modelo) ($\\Omega$) & Diferença ($\\Omega$) \\\\\n"
        "\\midrule\n"
    )
    lines.append(header)

    for ens in range(1, 17):
        pat = patterns[ens]
        ativos = ativos_for_pattern(pat)
        for cabo_label in ["CA", "CB", "CC", "CD"]:
            # Só faz sentido comparar se o cabo estiver ligado (estado F)
            if cabo_label in ativos:
                Rat = Rats[ens][cabo_label]
                # Se Rat veio como 0.0 em casos "A", você pode filtrar aqui se preferir
                if not np.isfinite(Rat) or Rat == 0.0:
                    continue

                Z_loop, Z_self, Z_eq_resto = compute_loop_for_cable(Z, labels, cabo_label, ativos)
                diff = Rat - Z_loop

                lines.append(
                    f"{ens} & {pat} & {cabo_label} & "
                    f"{Rat:6.3f} & {Z_loop:6.3f} & {diff:6.3f} \\\\\n"
                )

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}}\n"
        "\\end{sidewaystable}\n"
    )
    lines.append(footer)

    return "".join(lines)


if __name__ == "__main__":
    latex_table = generate_latex_table()
    print(latex_table)
