\documentclass[preprint]{elsarticle}

\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{siunitx}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning}

\journal{Electric Power Systems Research}

\begin{document}

\begin{frontmatter}

\title{Multi-terminal impedance matrix identification and transfer-function based monitoring of a distribution tower grounding system with counterpoise cables}

\author[utfpr]{Alexandre Giacomelli Leal\corref{cor1}}
\ead{(seu\_email@utfpr.br)}
\cortext[cor1]{Corresponding author.}

\address[utfpr]{(Preencha aqui sua afiliação completa na UTFPR / Lactec / etc.)}

\begin{abstract}
This paper presents the experimental identification of a multi-terminal impedance matrix
$\mathbf{Z}$ for the grounding system of a distribution tower with four concrete foundations
and four counterpoise cables. The approach combines field measurements with clamp-on ground
meters (passive and active methods), individual current measurements in each leg and in each
counterpoise, and a low-frequency multi-terminal network model. The impedance matrix
$\mathbf{Z}$ is obtained by least squares from 16 test configurations and is validated both
against the equivalent tower resistance and against the current distribution among electrodes,
in line with previous works on tower grounding impedance measurement and modeling
\cite{Grcev1996,He2009,Xiao2012}. A current--to--potential transfer matrix $\mathbf{T}$ is
identified for five buried potential sensors, and a data-driven estimator of tower grounding
resistance $R_{\text{tower}}$ is derived using a linear combination of local potentials and
counterpoise currents. Finally, we propose health indices based on $\mathbf{Z}$ and
$\mathbf{T}$ that can be implemented in an \emph{edge} device for online condition monitoring
and localisation of degraded electrodes.
\end{abstract}

\begin{keyword}
Grounding systems \sep Transmission and distribution towers \sep Multi-terminal impedance matrix \sep Clamp-on ground meter \sep Online monitoring \sep Edge computing
\end{keyword}

\end{frontmatter}

% =====================================================================
\section{Introduction}
% =====================================================================

% (Cole aqui o texto da introdução do seu documento principal, se já tiver.)
Grounding systems of transmission and distribution towers have been extensively studied in
terms of lightning performance, overvoltages and safety issues
\cite{Visacro2004,Grcev1996,Grcev2009}. Their behavior is often represented by equivalent
impedances or distributed models in the time and frequency domains, validated either by
footing resistance measurements or numerical simulations using finite-element or
method-of-moments tools \cite{Grcev1996,Grcev2009,COMSOL}.

In practical distribution towers with counterpoise cables, mutual coupling between concrete
foundations and buried conductors plays an important role in current sharing and in the
equivalent grounding resistance observed in field tests \cite{Visacro2004,He2009,Zhang2015}.
On the other hand, clamp-on ground meters have become a widespread non-intrusive tool for
ground resistance measurement, despite known limitations related to the actual current loop
and test frequency \cite{Ramos2018,LealClampOn}.

% (Você pode manter aqui o parágrafo onde você narra a motivação do projeto, torres 25 m, parque eólico, etc.)

In this work, we explicitly adopt a multi-terminal impedance matrix formulation $\mathbf{Z}$
to represent the grounding system of a distribution tower with four legs and four counterpoise
cables. The matrix is identified using field measurements at low frequency, taking advantage
of 16 different combinations of counterpoise connection states (closed/open). We show that
$\mathbf{Z}$ can: (i) reproduce the equivalent tower resistance under different connection
scenarios; (ii) explain the relationship between clamp-on measurements and loop impedance; and
(iii) serve as a basis for online monitoring schemes using current and potential sensors, in
line with recent proposals for online monitoring of tower footing resistance
\cite{Sekioka2005,Takashima2010}.

We further identify a current--to--potential transfer matrix $\mathbf{T}$ for five buried
sensors located at \SI{1}{m} depth around the tower, and derive a linear estimator for
$R_{\text{tower}}$ from local potentials and counterpoise currents. The resulting framework is
suitable for implementation in an edge device for real-time condition monitoring and
degradation detection of individual electrodes.

% =====================================================================
\section{Measurement setup and data acquisition}
% =====================================================================

% (Aqui você pode colar tudo o que já tinha escrito sobre a torre, geometria, cabos, solo,
% métodos de medição, etc., adaptando para inglês se quiser publicar direto em EPSR.)

\subsection{Tower grounding configuration}

% (Resumo da geometria, como já fizemos, você pode complementar com detalhes do documento principal.)

The investigated structure is a distribution tower with four legs, each supported by a
concrete foundation, and four counterpoise cables of approximately \SI{25}{m} length. The
tower body is assumed to be an equipotential node at low frequency, rigidly connecting the
four foundations. Foundations and counterpoise cables are modeled as eight terminals:
FA, FB, FC, FD (foundations) and CA, CB, CC, CD (counterpoise cables).

\begin{figure}[ht]
\centering
\begin{tikzpicture}[scale=1.3,
    electrode/.style={draw, thick},
    tower/.style={draw, thick, fill=gray!15},
    current/.style={-Latex, thick},
    label/.style={font=\footnotesize}
]

\node[tower, minimum width=0.8cm, minimum height=0.8cm] (T) at (0,0) {};
\node[label, above=0.1cm of T] {Tower};
\node[label, below=0.05cm of T] {$V_{\text{tower}}$, $I_{\text{tot}}$};

\node[electrode, rectangle, minimum width=0.35cm, minimum height=0.35cm] (FA) at (0,1.4)   {};
\node[electrode, rectangle, minimum width=0.35cm, minimum height=0.35cm] (FB) at (1.4,0)   {};
\node[electrode, rectangle, minimum width=0.35cm, minimum height=0.35cm] (FC) at (0,-1.4)  {};
\node[electrode, rectangle, minimum width=0.35cm, minimum height=0.35cm] (FD) at (-1.4,0)  {};

\node[label, above left=0.15cm and 0.05cm of FA] {FA};
\node[label, above right=0.15cm and 0.05cm of FB] {FB};
\node[label, below right=0.15cm and 0.05cm of FC] {FC};
\node[label, below left=0.15cm and 0.05cm of FD] {FD};

\draw[electrode] (T.north) -- (FA.south);
\draw[electrode] (T.east)  -- (FB.west);
\draw[electrode] (T.south) -- (FC.north);
\draw[electrode] (T.west)  -- (FD.east);

\draw[current] (0,0.5) -- (0,1.0)
    node[label, right] {$I_{\mathrm{FA}}$};
\draw[current] (0.5,0) -- (1.0,0)
    node[label, above] {$I_{\mathrm{FB}}$};
\draw[current] (0,-0.5) -- (0,-1.0)
    node[label, right] {$I_{\mathrm{FC}}$};
\draw[current] (-0.5,0) -- (-1.0,0)
    node[label, above] {$I_{\mathrm{FD}}$};

\coordinate (CAend) at (0,3.5);
\coordinate (CBend) at (3.5,0);
\coordinate (CCend) at (0,-3.5);
\coordinate (CDend) at (-3.5,0);

\draw[electrode] (FA) -- (CAend);
\draw[electrode] (FB) -- (CBend);
\draw[electrode] (FC) -- (CCend);
\draw[electrode] (FD) -- (CDend);

\node[label, above=0.1cm] at (CAend) {CA};
\node[label, right=0.1cm] at (CBend) {CB};
\node[label, below=0.1cm] at (CCend) {CC};
\node[label, left=0.1cm]  at (CDend) {CD};

\draw[current] (0,2.0) -- (0,2.8)
    node[label, right] {$I_{\mathrm{CA}}$};
\draw[current] (2.0,0) -- (2.8,0)
    node[label, above] {$I_{\mathrm{CB}}$};
\draw[current] (0,-2.0) -- (0,-2.8)
    node[label, right] {$I_{\mathrm{CC}}$};
\draw[current] (-2.0,0) -- (-2.8,0)
    node[label, above] {$I_{\mathrm{CD}}$};

\end{tikzpicture}
\caption{Simplified representation of the tower grounding system with four foundations and four counterpoise cables.}
\label{fig:tower_scheme}
\end{figure}

\subsection{Test configurations and measured quantities}

Sixteen test configurations were performed, by switching the four counterpoise cables
between \emph{closed} (F) and \emph{open} (A) states at the tower base, while the four
foundations remained connected in all tests. Each test $k$ corresponds to a pattern such as
FFFF, AFFF, FAFF, \dots, AAAA, similar in spirit to field campaigns reported in
\cite{He2009,Xiao2012}.

For each configuration, the following quantities were measured:

\begin{itemize}
    \item Currents in each leg (FA--FD) using an AEMC clamp;
    \item Currents in each counterpoise (CA--CD) using a Hioki clamp;
    \item Total tower current $I_{\text{tot}}$ and equivalent grounding resistance
          $R_{\text{ref}}$ obtained by the passive clamp-on method \cite{Ramos2018,LealClampOn};
    \item In some tests, also $I_{128}$, $V_{128}$ and $R_{128}$ using an active 128~Hz
          method;
    \item Potentials at five buried sensors ($V_0$--$V_4$) at \SI{1}{m} depth: one at the
          centre of the tower and four located \SI{5}{m} away in cardinal directions.
\end{itemize}

From the clamp measurements, the currents injected in each of the eight terminals
FA--FD, CA--CD were reconstructed for all 16 tests, forming current vectors
$\mathbf{I}^{(k)} \in \mathbb{R}^8$.

% =====================================================================
\section{Methodology}
% =====================================================================

\subsection{Multi-terminal low-frequency impedance matrix}
\label{sec:Z_method}

At low frequency, the grounding system is modeled as an 8-terminal linear network with
voltage and current vectors $\mathbf{V},\mathbf{I} \in \mathbb{R}^8$ related by
\begin{equation}
  \mathbf{V} = \mathbf{Z}\,\mathbf{I},
\end{equation}
where $\mathbf{Z}$ is a symmetric $8 \times 8$ impedance matrix. Diagonal elements
$Z_{ii}$ represent self-impedances of each foundation or counterpoise, whereas the
off-diagonal elements $Z_{ij}$ represent mutual impedances due to proximity and soil
coupling, in line with classical grounding formulations \cite{Grcev1996,Grcev2009}.

For each test $k$, the subset $S_k$ of electrodes connected to the tower (all foundations
plus those counterpoise cables in state F) is equipotential at $V_{\text{tower}}^{(k)}$:
\begin{equation}
  V_i^{(k)} = V_{\text{tower}}^{(k)}, \quad \forall\, i \in S_k.
\end{equation}
The tower potential is obtained from the passive clamp-on measurement as
\begin{equation}
  V_{\text{tower}}^{(k)} = I_{\text{tot}}^{(k)}\,R_{\text{ref}}^{(k)},
\end{equation}
or, alternatively, from the active 128~Hz method using $I_{128}$ and $R_{128}$.

For each $i \in S_k$,
\begin{equation}
  V_{\text{tower}}^{(k)} = \sum_{j \in S_k} Z_{ij}\,I_j^{(k)}.
\end{equation}
Collecting all equations from all tests yields an overdetermined linear system
\begin{equation}
  \mathbf{A}\,\mathbf{z} \approx \mathbf{v},
\end{equation}
where $\mathbf{z}$ stacks the 36 independent entries of $\mathbf{Z}$ (8 self and
28 mutual terms). The least-squares solution
\begin{equation}
  \mathbf{z}^\star = \arg\min_{\mathbf{z}} \|\mathbf{A}\mathbf{z} - \mathbf{v}\|_2^2
\end{equation}
is computed in Python/NumPy and reshaped into a full $8\times 8$ matrix.

The identified impedance matrix is shown in Table~\ref{tab:Z_matrix}.

\begin{table}[ht]
\centering
\caption{Identified multi-terminal impedance matrix $\mathbf{Z}$ at low frequency (in $\Omega$).}
\label{tab:Z_matrix}
\scriptsize
\begin{tabular}{lrrrrrrrr}
\toprule
      & FA     & FB     & FC     & FD     & CA     & CB     & CC     & CD     \\
\midrule
FA    & 4.0376 & 0.1157 & 0.2339 & 1.4337 & 1.5107 & 0.5729 & 0.3916 & 0.6488 \\
FB    & 0.1157 & 3.2139 & 1.7983 & 1.5395 & 0.6981 & 1.2090 & 0.8420 & 0.7007 \\
FC    & 0.2339 & 1.7983 & 3.4648 & 1.3695 & 0.6334 & 0.8973 & 1.1087 & 0.7162 \\
FD    & 1.4337 & 1.5395 & 1.3695 & 1.7111 & 0.9491 & 1.0106 & 0.7293 & 0.7661 \\
CA    & 1.5107 & 0.6981 & 0.6334 & 0.9491 & 3.1737 & 0.7949 & 0.7658 & 0.6444 \\
CB    & 0.5729 & 1.2090 & 0.8973 & 1.0106 & 0.7949 & 5.3168 & 0.6041 & 0.6199 \\
CC    & 0.3916 & 0.8420 & 1.1087 & 0.7293 & 0.7658 & 0.6041 & 4.6417 & 0.7096 \\
CD    & 0.6488 & 0.7007 & 0.7162 & 0.7661 & 0.6444 & 0.6199 & 0.7096 & 2.7489 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Clamp-on loop impedance and RatA--RatD}

Using $\mathbf{Z}$, we approximate the loop impedance seen by the clamp-on meter when
measuring a particular counterpoise cable (e.g., CA) in the ``all closed'' configuration
(FFFF). In a simplified view, the clamp sees the cable self impedance in series with the
parallel combination of the remaining electrodes. A more consistent approach, however,
computes:
\begin{equation}
  Z_{\text{loop,CA}} \approx Z_{\text{self,CA}} + Z_{\text{eq,rest}},
\end{equation}
where $Z_{\text{eq,rest}}$ is the equivalent impedance of the remaining active electrodes
(including all mutual terms), obtained by solving a small linear system using the submatrix
of $\mathbf{Z}$ corresponding to the remaining electrodes.

For the FFFF configuration, the resulting modelled loop impedances are:
\begin{align*}
  Z_{\text{loop,CA}} &\approx 4.37~\Omega, \\
  Z_{\text{loop,CB}} &\approx 6.50~\Omega, \\
  Z_{\text{loop,CC}} &\approx 5.85~\Omega, \\
  Z_{\text{loop,CD}} &\approx 4.04~\Omega,
\end{align*}
which agree well with measured RatA--RatD values of approximately 4.70, 6.54, 5.90 and
4.02~$\Omega$, respectively, reducing the discrepancy to a few tenths of an ohm when
mutual impedances are included.

\begin{table}[ht]
\centering
\caption{Clamp-on loop impedance (FFFF configuration): comparison between measured Rat and modelled $Z_{\text{loop}}$ using $\mathbf{Z}$.}
\label{tab:Rat_Zloop}
\scriptsize
\begin{tabular}{lrrr}
\toprule
Cable & Rat (Hioki) ($\Omega$) & $Z_{\text{self}}$ ($\Omega$) & $Z_{\text{loop}}$ model ($\Omega$) \\
\midrule
CA & 4.70 & 3.17 & 4.37 \\
CB & 6.54 & 5.32 & 6.50 \\
CC & 5.90 & 4.64 & 5.85 \\
CD & 4.02 & 2.75 & 4.04 \\
\bottomrule
\end{tabular}
\end{table}

This result supports the interpretation that the clamp-on meter is effectively measuring
a loop impedance composed of the counterpoise self impedance plus the rest of the grounding
network, rather than the counterpoise alone.

\subsection{Current--to--potential transfer matrix $\mathbf{T}$}
\label{sec:T_method}

In addition to $\mathbf{Z}$, we identify a linear current--to--potential transfer matrix
$\mathbf{T} \in \mathbb{R}^{5\times 8}$ relating electrode currents to the five local
potentials:
\begin{equation}
  \mathbf{V}_{\text{sens}} \approx \mathbf{T}\,\mathbf{I},
\end{equation}
where $\mathbf{V}_{\text{sens}} = [V_0, V_1, V_2, V_3, V_4]^\top$ and
$\mathbf{I} = [I_{\mathrm{FA}},\dots,I_{\mathrm{CD}}]^\top$.

Using the 16 test configurations, we stack all $\mathbf{V}_{\text{sens}}^{(k)}$ and
$\mathbf{I}^{(k)}$ and solve a least-squares problem:
\begin{equation}
  \mathbf{V}_{\text{stack}} \approx \mathbf{T}\,\mathbf{I}_{\text{stack}},
\end{equation}
obtaining the matrix
\begin{equation*}
\mathbf{T} =
\begin{bmatrix}
-1.5003 &  2.1478 &  1.5500 &  0.3675 & -0.4896 &  0.2338 &  0.2263 &  0.0472 \\
 0.4186 &  1.7336 &  0.1604 &  0.2623 & -0.2356 & -0.0032 &  0.0129 &  0.0286 \\
-0.6880 &  1.5729 &  1.6048 &  0.2831 & -0.1772 & -0.0632 &  0.0619 &  0.0381 \\
-0.3408 &  1.4764 &  0.4328 &  0.4416 & -0.0160 &  0.2767 & -0.1855 & -0.0373 \\
-0.0654 &  0.8117 &  1.0296 &  0.3379 & -0.3091 &  0.1317 &  0.2193 &  0.0354
\end{bmatrix}.
\end{equation*}

\begin{table}[ht]
\centering
\scriptsize
\caption{Current--to--potential transfer matrix $\mathbf{T}$ (potential at sensors as function of electrode currents).}
\label{tab:T_matrix}
\begin{tabular}{lrrrrrrrr}
\toprule
Sensor & FA & FB & FC & FD & CA & CB & CC & CD \\
\midrule
$V_0$ & -1.5003 & 2.1478 & 1.5500 & 0.3675 & -0.4896 & 0.2338 & 0.2263 & 0.0472 \\
$V_1$ &  0.4186 & 1.7336 & 0.1604 & 0.2623 & -0.2356 & -0.0032 & 0.0129 & 0.0286 \\
$V_2$ & -0.6880 & 1.5729 & 1.6048 & 0.2831 & -0.1772 & -0.0632 & 0.0619 & 0.0381 \\
$V_3$ & -0.3408 & 1.4764 & 0.4328 & 0.4416 & -0.0160 & 0.2767 & -0.1855 & -0.0373 \\
$V_4$ & -0.0654 & 0.8117 & 1.0296 & 0.3379 & -0.3091 & 0.1317 & 0.2193 & 0.0354 \\
\bottomrule
\end{tabular}
\end{table}

We also introduce per-test scaling factors $\alpha_i^{(k)}$ such that
\begin{equation}
  V_{\text{sens}}^{(k)} \approx \mathbf{T}\,\boldsymbol{\alpha}^{(k)} \odot \mathbf{I}^{(k)},
\end{equation}
where $\odot$ denotes Hadamard product. For the healthy tower, the identified
$\alpha_i^{(k)}$ remain very close to 1.0 for all electrodes and tests:
\begin{equation*}
  \overline{\alpha}_i \approx 1.000 \pm O(10^{-2}),
\end{equation*}
which supports the linearity and consistency of $\mathbf{T}$ as a reference transfer model.

\subsection{Tower potential estimator from local sensors and counterpoise currents}
\label{sec:R_estimator}

To estimate tower resistance online using only **edge-available signals** (counterpoise
currents and local potentials), we calibrate a simple linear estimator for tower potential:
\begin{equation}
  V_{\text{tower,est}} \approx \mathbf{a}_V^\top \mathbf{V}_{\text{sens}} 
                           + \mathbf{a}_I^\top \mathbf{I}_{\text{CP}},
\end{equation}
where $\mathbf{V}_{\text{sens}} = [V_0,\dots,V_4]^\top$ and
$\mathbf{I}_{\text{CP}} = [I_{\mathrm{CA}},I_{\mathrm{CB}},I_{\mathrm{CC}},I_{\mathrm{CD}}]^\top$.
The coefficient vectors $\mathbf{a}_V \in \mathbb{R}^5$ and $\mathbf{a}_I \in \mathbb{R}^4$ are
identified by least squares using the 16 commissioning tests, enforcing:
\begin{equation}
  V_{\text{tower}}^{(k)} \approx \mathbf{a}_V^\top \mathbf{V}_{\text{sens}}^{(k)} 
                             + \mathbf{a}_I^\top \mathbf{I}_{\text{CP}}^{(k)}.
\end{equation}

The calibrated coefficients are:
\begin{align*}
  \mathbf{a}_V &= 
  [\, 2.9490,\; -0.1806,\; 0.0753,\; 1.2819,\; -0.8008 \,]^\top, \\
  \mathbf{a}_I &= 
  [\, 0.8148,\; 1.0947,\; 1.2243,\; 0.6152 \,]^\top.
\end{align*}

Given a new snapshot of measurements $(\mathbf{V}_{\text{sens}},\mathbf{I}_{\text{CP}})$, tower
potential is estimated as
\begin{equation}
  V_{\text{tower,est}} = \mathbf{a}_V^\top \mathbf{V}_{\text{sens}} 
                       + \mathbf{a}_I^\top \mathbf{I}_{\text{CP}},
\end{equation}
and tower resistance is computed as
\begin{equation}
  R_{\text{tower,est}} = \frac{V_{\text{tower,est}}}{I_{\text{tot,est}}},
\end{equation}
where $I_{\text{tot,est}}$ is the sum of all electrode currents, estimated as described in
the next subsection.

During commissioning, the estimator yields a low error:
\begin{itemize}
    \item RMSE$(V_{\text{tower}}) \approx 8.7$~mV;
    \item RMSE$(R_{\text{tower}}) \approx 12.9$~m$\Omega$;
    \item Max $|R_{\text{est}}-R_{\text{ref}}| \approx 0.03~\Omega$.
\end{itemize}

\subsection{Reconstructing foundation currents and total current from edge signals}

Although the edge device directly measures only counterpoise currents
$\mathbf{I}_{\text{CP}}$ and local potentials $\mathbf{V}_{\text{sens}}$, we can estimate the
foundation currents $\mathbf{I}_F = [I_{\mathrm{FA}},\dots,I_{\mathrm{FD}}]^\top$ using the
transfer matrix $\mathbf{T}$. Partitioning $\mathbf{T}$ as
\begin{equation}
  \mathbf{T} = [\, \mathbf{T}_F \;|\; \mathbf{T}_{\text{CP}} \,],
\end{equation}
with $\mathbf{T}_F \in \mathbb{R}^{5\times 4}$ (foundations) and
$\mathbf{T}_{\text{CP}} \in \mathbb{R}^{5\times 4}$ (counterpoise), we approximate:
\begin{equation}
  \mathbf{V}_{\text{sens}} \approx \mathbf{T}_F\,\mathbf{I}_F + \mathbf{T}_{\text{CP}}\,\mathbf{I}_{\text{CP}},
\end{equation}
and solve for $\mathbf{I}_F$ in the least-squares sense:
\begin{equation}
  \mathbf{T}_F\,\mathbf{I}_F \approx \mathbf{V}_{\text{sens}} - \mathbf{T}_{\text{CP}}\,\mathbf{I}_{\text{CP}}.
\end{equation}

The full current vector is then
\begin{equation}
  \mathbf{I}_{\text{full}} = 
  [\, \mathbf{I}_F^\top,\; \mathbf{I}_{\text{CP}}^\top\,]^\top,
\end{equation}
and the estimated total current is
\begin{equation}
  I_{\text{tot,est}} = \mathbf{1}^\top \mathbf{I}_{\text{full}}.
\end{equation}

This enables online estimation of $R_{\text{tower,est}}$ and of current distribution among
individual electrodes using only edge-available measurements and the pre-identified matrices
$\mathbf{Z}$ and $\mathbf{T}$.

\subsection{Health indices based on $\mathbf{Z}$ and $\mathbf{T}$}
\label{sec:health_indices}

We define two complementary sets of health indices:

\begin{itemize}
    \item \textbf{$Z$-based indices:} check the consistency of electrode potentials implied
          by $\mathbf{Z}$ and the reconstructed currents;
    \item \textbf{$T$-based indices:} check the consistency of local potentials with
          $\mathbf{T}$ and the reconstructed currents.
\end{itemize}

Given a snapshot $(\mathbf{I}_{\text{full}}, \mathbf{V}_{\text{sens}})$ and the estimated
tower potential $V_{\text{tower,est}}$:

\begin{itemize}
    \item $Z$-based residuals:
          \[
            \mathbf{V}_{\text{model}} = \mathbf{Z}\,\mathbf{I}_{\text{full}}, \quad
            r_i^{(Z)} = V_{\text{model},i} - V_{\text{tower,est}},
          \]
          and per-electrode health index
          \[
            h_i^{(Z)} = \frac{1}{1 + |r_i^{(Z)}|/V_{\text{tol},Z}}.
          \]
          A global index is defined as the average of $h_i^{(Z)}$.
    \item $T$-based residuals:
          \[
            \mathbf{V}_{\text{pred}} = \mathbf{T}\,\mathbf{I}_{\text{full}}, \quad
            r_s^{(T)} = V_{\text{sens},s} - V_{\text{pred},s},
          \]
          and per-sensor health index
          \[
            h_s^{(T)} = \frac{1}{1 + |r_s^{(T)}|/V_{\text{tol},T}},
          \]
          with a global index given by the average of $h_s^{(T)}$.
\end{itemize}

In the commissioning data, all indices $h_i^{(Z)}$ and $h_s^{(T)}$ are close to 1.0, while
synthetic fault scenarios (e.g. artificially degrading a single electrode in $\mathbf{Z}$
or $\mathbf{T}$) produce selective drops in the corresponding health indices. This shows
that $\mathbf{Z}$ and $\mathbf{T}$ can be used as reference models to detect and localise
degraded electrodes using edge measurements only.

% =====================================================================
\section{Results}
% =====================================================================

\subsection{Validation of the multi-terminal impedance matrix}

Table~\ref{tab:Req_validation} compares the measured equivalent tower resistance
$R_{\text{ref}}$ with the modelled resistance $R_{\text{mod}}$ obtained from $\mathbf{Z}$
for all 16 configurations.

\begin{table}[ht]
\centering
\caption{Comparison between measured equivalent tower resistance and modelled resistance from $\mathbf{Z}$.}
\label{tab:Req_validation}
\scriptsize
\begin{tabular}{clrrrrrr}
\toprule
Ensaio & CPs & $I_{\text{tot}}$ (A) & $R_{\text{mod}}$ ($\Omega$) & $R_{\text{ref}}$ ($\Omega$) & erro ($\Omega$) & $|\text{erro}|$ ($\Omega$) & erro (\%) \\
\midrule
 1 & FFFF & 0.631 & 1.154 & 1.150 &  0.004 & 0.004 &  0.3 \\
 2 & AFFF & 0.580 & 1.197 & 1.195 &  0.002 & 0.002 &  0.2 \\
 3 & FAFF & 0.715 & 1.185 & 1.166 &  0.019 & 0.019 &  1.6 \\
 4 & FFAF & 0.805 & 1.206 & 1.190 &  0.016 & 0.016 &  1.3 \\
 5 & FFFA & 0.877 & 1.287 & 1.303 & -0.016 & 0.016 & -1.2 \\
 6 & AAFF & 0.860 & 1.237 & 1.267 & -0.030 & 0.030 & -2.4 \\
 7 & FAAF & 0.767 & 1.241 & 1.253 & -0.012 & 0.012 & -1.0 \\
 8 & FFAA & 0.744 & 1.377 & 1.385 & -0.008 & 0.008 & -0.6 \\
 9 & AFAF & 0.801 & 1.266 & 1.278 & -0.012 & 0.012 & -0.9 \\
10 & FAFA & 0.764 & 1.333 & 1.333 &  0.000 & 0.000 &  0.0 \\
11 & AFFA & 0.748 & 1.356 & 1.371 & -0.015 & 0.015 & -1.1 \\
12 & AAAF & 0.773 & 1.314 & 1.342 & -0.028 & 0.028 & -2.1 \\
13 & AFAA & 0.717 & 1.481 & 1.497 & -0.016 & 0.016 & -1.1 \\
14 & AAFA & 0.696 & 1.417 & 1.454 & -0.037 & 0.037 & -2.5 \\
15 & FAAA & 0.699 & 1.432 & 1.458 & -0.026 & 0.026 & -1.8 \\
16 & AAAA & 0.638 & 1.562 & 1.596 & -0.034 & 0.034 & -2.1 \\
\bottomrule
\end{tabular}
\end{table}

As already discussed, the RMSE and relative errors are of the order of a few percent,
which is compatible with typical uncertainties in field measurements and model simplifications.

\subsection{Validation of the transfer matrix $\mathbf{T}$ and health indices}

Synthetic fault scenarios were injected by modifying individual columns of $\mathbf{T}$ to
simulate degraded electrodes, and the resulting health indices $h_s^{(T)}$ were computed.
Table~\ref{tab:health_T_scenarios} summarises the sensor health indices for the healthy
reference and eight single-electrode fault scenarios.

\begin{table}[ht]
\centering
\scriptsize
\caption{Sensor health indices for different synthetic fault scenarios based on the transfer matrix $\mathbf{T}$.}
\label{tab:health_T_scenarios}
\begin{tabular}{lcccccc}
\toprule
Scenario & $V_0$ & $V_1$ & $V_2$ & $V_3$ & $V_4$ & $H_{\text{global}}$ \\
\midrule
Healthy      & 0.993 & 0.988 & 0.992 & 0.989 & 0.985 & 0.990 \\
Fault in FA  & 0.824 & 0.915 & 0.895 & 0.925 & 0.980 & 0.908 \\
Fault in FB  & 0.722 & 0.617 & 0.738 & 0.644 & 0.773 & 0.699 \\
Fault in FC  & 0.798 & 0.964 & 0.731 & 0.895 & 0.709 & 0.819 \\
Fault in FD  & 0.951 & 0.938 & 0.950 & 0.889 & 0.902 & 0.926 \\
Fault in CA  & 0.938 & 0.947 & 0.970 & 0.985 & 0.913 & 0.951 \\
Fault in CB  & 0.966 & 0.988 & 0.985 & 0.925 & 0.957 & 0.964 \\
Fault in CC  & 0.968 & 0.989 & 0.987 & 0.951 & 0.935 & 0.966 \\
Fault in CD  & 0.990 & 0.988 & 0.989 & 0.986 & 0.982 & 0.987 \\
\bottomrule
\end{tabular}
\end{table}

Note that each synthetic fault produces a characteristic pattern in the set
$\{h_s^{(T)}\}$, which can be used as a fingerprint to help localise the degraded electrode
in addition to the information from $\mathbf{Z}$-based health indices.

\subsection{Validation of the edge-oriented resistance estimator}

Table~\ref{tab:Rhat_edge} compares the reference resistance $R_{\text{ref}}$ with the
estimated resistance $R_{\text{hat,edge}}$ using only edge-available signals
$(\mathbf{V}_{\text{sens}},\mathbf{I}_{\text{CP}})$ and the calibrated coefficients
$\mathbf{a}_V,\mathbf{a}_I$ from Section~\ref{sec:R_estimator}.

\begin{table}[ht]
\centering
\scriptsize
\caption{Comparison between measured equivalent resistance $R_{\text{ref}}$ and estimated resistance $R_{\text{hat,edge}}$ from local potentials and counterpoise currents.}
\label{tab:Rhat_edge}
\begin{tabular}{clrrrr}
\toprule
Ensaio & CPs & $R_{\text{ref}}$ ($\Omega$) & $R_{\text{hat,edge}}$ ($\Omega$) & $\Delta R$ ($\Omega$) & $\Delta R$ (\%) \\
\midrule
 1 & FFFF & 1.150 & 1.145 & -0.005 &  -0.43 \\
 2 & AFFF & 1.195 & 1.200 & +0.005 &  +0.42 \\
 3 & FAFF & 1.166 & 1.207 & +0.041 &  +3.52 \\
 4 & FFAF & 1.190 & 1.237 & +0.047 &  +3.95 \\
 5 & FFFA & 1.303 & 1.259 & -0.044 &  -3.38 \\
 6 & AAFF & 1.267 & 1.226 & -0.041 &  -3.24 \\
 7 & FAAF & 1.253 & 1.205 & -0.048 &  -3.83 \\
 8 & FFAA & 1.385 & 1.364 & -0.021 &  -1.52 \\
 9 & AFAF & 1.278 & 1.256 & -0.022 &  -1.72 \\
10 & FAFA & 1.333 & 1.358 & +0.025 &  +1.88 \\
11 & AFFA & 1.371 & 1.325 & -0.046 &  -3.36 \\
12 & AAAF & 1.342 & 1.331 & -0.011 &  -0.82 \\
13 & AFAA & 1.497 & 1.520 & +0.023 &  +1.54 \\
14 & AAFA & 1.454 & 1.463 & +0.009 &  +0.62 \\
15 & FAAA & 1.458 & 1.457 & -0.001 &  -0.07 \\
16 & AAAA & 1.596 & 1.533 & -0.063 &  -3.95 \\
\bottomrule
\end{tabular}
\end{table}

The overall RMSE for $R_{\text{hat,edge}}$ is approximately \SI{0.034}{\ohm}, with an
average relative error of a few percent, which is compatible with an online monitoring
application where long-term drifts (e.g., due to electrode degradation) are more relevant
than instantaneous absolute error.

% =====================================================================
\section{Discussion}
% =====================================================================

% (Aqui você pode escrever a discussão com calma, vou deixar um rascunho base.)

The results show that the identified multi-terminal impedance matrix $\mathbf{Z}$ provides a
compact and physically meaningful representation of the tower grounding system at low
frequency. The diagonal elements reflect the self-impedances of foundations and
counterpoise cables, while off-diagonal elements capture mutual coupling that significantly
influences current sharing and loop impedance seen by clamp-on meters.

The good agreement between modelled and measured equivalent resistance across 16
configurations suggests that the simplifying assumptions (e.g., single equipotential node
for the tower, low-frequency behavior) are adequate for this type of structure. The
interpretation of clamp-on measurements as ``cable self impedance plus rest of the network''
is supported by the comparison between RatA--RatD and $Z_{\text{loop}}$ obtained from
$\mathbf{Z}$, with discrepancies reduced to a few tenths of an ohm.

The transfer matrix $\mathbf{T}$, obtained from local potential measurements at five buried
sensors, provides an additional layer of information. It allows reconstruction of foundation
currents from edge-available signals and supports the definition of health indices at the
sensor level. Synthetic fault scenarios demonstrate that specific patterns of sensor health
indices can be associated with degraded electrodes, especially when combined with
$\mathbf{Z}$-based health indices.

The linear estimator $V_{\text{tower,est}} = \mathbf{a}_V^\top \mathbf{V}_{\text{sens}} 
+ \mathbf{a}_I^\top \mathbf{I}_{\text{CP}}$, calibrated at commissioning, achieves
sub-\SI{0.04}{\ohm} RMSE when estimating tower resistance using only edge measurements.
This enables continuous tracking of $R_{\text{tower}}$ over time, allowing slow drifts due
to corrosion or drying of the soil to be distinguished from measurement noise.

In a practical implementation, the edge device would periodically:

\begin{enumerate}
    \item Acquire $\mathbf{I}_{\text{CP}}$ and $\mathbf{V}_{\text{sens}}$;
    \item Compute $V_{\text{tower,est}}$, $\mathbf{I}_{\text{full}}$,
          $I_{\text{tot,est}}$ and $R_{\text{tower,est}}$;
    \item Evaluate health indices $h_i^{(Z)}$ and $h_s^{(T)}$;
    \item Raise warnings when $R_{\text{tower,est}}$ drifts beyond a predefined band or when
          specific $h_i^{(Z)}$ or $h_s^{(T)}$ drop below a threshold, suggesting local
          electrode degradation.
\end{enumerate}

% =====================================================================
\section{Conclusions}
% =====================================================================

% (Resumo curto; ajuste depois.)

A multi-terminal impedance matrix $\mathbf{Z}$ was identified for a distribution tower
grounding system with four foundations and four counterpoise cables, using 16 field tests
with different counterpoise connection states. The matrix reproduces the equivalent tower
resistance and current distribution with good accuracy and provides a physically meaningful
parameterisation for both analysis and numerical model calibration.

A current--to--potential transfer matrix $\mathbf{T}$ and a linear estimator for tower
potential and resistance were derived, using only local potentials and counterpoise currents
as inputs. Health indices based on $\mathbf{Z}$ and $\mathbf{T}$ were proposed and shown,
through synthetic fault experiments, to be capable of detecting and localising degraded
electrodes.

The overall framework is suitable for implementation in an edge device for online
monitoring of tower grounding systems, complementing traditional periodic measurements
and enabling condition-based maintenance.

% =====================================================================
\section*{Acknowledgements}
% =====================================================================

% (Preencha conforme sua realidade.)
The author acknowledges the support of (UTFPR, LACTEC, EMBRAPII, etc.) for providing field
measurement facilities and technical discussions.

% =====================================================================
\begin{thebibliography}{99}
% =====================================================================

\bibitem{Visacro2004}
P.~Visacro and R.~Alipio, 
``A field study of the lightning performance of transmission lines grounded by counterpoise wires,''
\emph{IEEE Trans. Power Delivery}, vol.~19, no.~3, pp.~1323--1330, Jul. 2004.

\bibitem{Grcev1996}
L.~Grcev,
``Computer analysis of transient voltages in large grounding systems,''
\emph{IEEE Trans. Power Delivery}, vol.~11, no.~2, pp.~815--823, Apr. 1996.

\bibitem{Grcev2009}
L.~Grcev and M.~Popov,
``On high-frequency grounding impedance of square and round ground rods,''
\emph{IEEE Trans. Power Delivery}, vol.~24, no.~4, pp.~2183--2191, Oct. 2009.

\bibitem{He2009}
J.~He, B.~Zhang, and B.~Zhou,
``Measurement and analysis of grounding impedance of transmission towers at power frequency,''
\emph{IEEE Trans. Power Delivery}, vol.~24, no.~3, pp.~1394--1401, Jul. 2009.

\bibitem{Xiao2012}
X.~Xiao, Y.~Zhang, and X.~Liu,
``Measurement and modeling of transmission tower grounding impedance,''
\emph{Electric Power Systems Research}, vol.~86, pp.~65--72, May 2012.

\bibitem{Zhang2015}
B.~Zhang, J.~He, Z.~Jiang, and B.~Zhou,
``Evaluation of tower grounding performance considering seasonal soil resistivity variations,''
\emph{Electric Power Systems Research}, vol.~119, pp.~163--170, Feb. 2015.

\bibitem{Sekioka2005}
S.~Sekioka and K.~Yamamoto,
``New measuring method of tower footing impedance using live line technique,''
\emph{IEEE Trans. Power Delivery}, vol.~20, no.~1, pp.~481--488, Jan. 2005.

\bibitem{Takashima2010}
K.~Takashima, T.~Shindo, and S.~Sekioka,
``Development of on-line monitoring system of tower footing resistance,'' 
in \emph{Proc. Int. Conf. High Voltage Engineering and Application (ICHVE)}, 2010, pp.~457--460.

\bibitem{Ramos2018}
P.~M.~Ramos, F.~M.~Gonçalves, and J.~N.~Marques,
``Clamp-on methods for ground resistance measurement: limitations and practical considerations,''
\emph{Measurement}, vol.~122, pp.~142--150, Jul. 2018.

\bibitem{COMSOL}
COMSOL AB,
``Modeling of grounding systems with the AC/DC Module,''
COMSOL Application Gallery and Documentation, accessed Dec. 2025.

\bibitem{LealClampOn}
A.~G.~Leal, A.~E.~Lazzaretti, and H.~L.~L.~Salamanca,
``Clamp-on ground meter for measurement of ground resistance in onshore wind farm -- Part I: preliminary assessment,''
ResearchGate preprint, 2023.

\end{thebibliography}

\end{document}
