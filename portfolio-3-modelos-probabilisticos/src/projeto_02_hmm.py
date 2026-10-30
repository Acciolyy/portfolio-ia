from __future__ import annotations

"""
Projeto 2 – Modelo Oculto de Markov: Detecção de Padrão de Uso em Aplicativo
=============================================================================
Disciplina : FGA0221 – Inteligência Artificial
Autor      : Thiago Viriato Accioly
Matrícula  : 231029340
Professor  : Dr. Fabiano Araujo Soares

Descrição
---------
Modela o comportamento de um usuário em um aplicativo de e-commerce como um
HMM. Os estados ocultos representam a intenção real do usuário (Navegando,
Comprando, Inativo), enquanto as observações são eventos discretos registrados
pelo sistema (clique em produto, adição ao carrinho, tempo sem interação, etc.).

O objetivo é aplicar o Algoritmo de Viterbi para decodificar a sequência mais
provável de intenções do usuário a partir dos eventos observados, simulando
um sistema de análise de comportamento (UX Analytics).

Estados ocultos (S)
-------------------
0 – Navegando  : usuário explorando o catálogo sem intenção definida.
1 – Comprando  : usuário com intenção clara de adquirir um produto.
2 – Inativo    : usuário sem interação ativa (janela aberta, sem ação).

Observações (O)
---------------
0 – CliqueCategoria  : navegou por uma categoria de produtos.
1 – CliqueProduto    : clicou em um produto específico.
2 – AdicionouCarrinho: adicionou item ao carrinho.
3 – TempoSemAcao     : sem evento por mais de 30 segundos.

Bibliotecas
-----------
- numpy      : operações vetoriais e numéricas do HMM (Viterbi, forward).
- matplotlib : geração dos gráficos de resultados e comparativos.
"""

import argparse
import statistics
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Constantes de domínio
# ---------------------------------------------------------------------------

STATE_NAMES = ["Navegando", "Comprando", "Inativo"]
OBS_NAMES   = ["CliqueCategoria", "CliqueProduto", "AdicionouCarrinho", "TempoSemAcao"]

NUM_STATES = len(STATE_NAMES)   # M = 3
NUM_OBS    = len(OBS_NAMES)     # K = 4


# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------

@dataclass
class HMMParameters:
    """Parâmetros completos de um HMM com emissões discretas."""

    pi: np.ndarray   # Vetor de probabilidade inicial (M,)
    A:  np.ndarray   # Matriz de transição (M x M)
    B:  np.ndarray   # Matriz de emissão (M x K)

    def __post_init__(self) -> None:
        assert self.pi.shape == (NUM_STATES,),          "pi deve ter forma (M,)"
        assert self.A.shape  == (NUM_STATES, NUM_STATES), "A deve ter forma (M, M)"
        assert self.B.shape  == (NUM_STATES, NUM_OBS),    "B deve ter forma (M, K)"


@dataclass
class ViterbiResult:
    """Resultado da decodificação pelo Algoritmo de Viterbi."""

    decoded_states: List[int]
    log_probability: float
    elapsed_ms: float

    def state_names(self) -> List[str]:
        return [STATE_NAMES[s] for s in self.decoded_states]

    def __str__(self) -> str:
        seq = " → ".join(self.state_names()[:10])
        if len(self.decoded_states) > 10:
            seq += f" ... (+{len(self.decoded_states) - 10} passos)"
        lines = [
            "[Viterbi] Decodificação de estados ocultos",
            f"  Sequência     : {seq}",
            f"  Log-prob      : {self.log_probability:.4f}",
            f"  Tempo         : {self.elapsed_ms:.4f} ms",
        ]
        return "\n".join(lines)


@dataclass
class ForwardResult:
    """Resultado do algoritmo Forward (distribuição filtrada a cada passo)."""

    alpha: np.ndarray   # (T, M) – probabilidades a posteriori normalizadas
    most_likely: List[int]
    elapsed_ms: float

    def __str__(self) -> str:
        seq = " → ".join(STATE_NAMES[s] for s in self.most_likely[:10])
        if len(self.most_likely) > 10:
            seq += f" ... (+{len(self.most_likely) - 10} passos)"
        lines = [
            "[Forward] Distribuição filtrada de estados",
            f"  Sequência ML  : {seq}",
            f"  Tempo         : {self.elapsed_ms:.4f} ms",
        ]
        return "\n".join(lines)


@dataclass
class HMMReport:
    """Agrega todos os resultados do experimento."""

    sequence_length: int
    true_states: List[int]
    observations: List[int]
    viterbi: ViterbiResult | None = None
    forward: ForwardResult | None = None
    accuracy_viterbi: float = 0.0
    accuracy_forward: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Sequência gerada   : {self.sequence_length} eventos",
            f"Acurácia Viterbi   : {self.accuracy_viterbi:.2%}",
            f"Acurácia Forward   : {self.accuracy_forward:.2%}",
            "",
            str(self.viterbi),
            "",
            str(self.forward),
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Modelo HMM
# ---------------------------------------------------------------------------

class AppUsageHMM:
    """
    HMM para modelagem de intenção de usuário em aplicativo de e-commerce.

    Implementa manualmente os algoritmos Viterbi e Forward usando numpy,
    sem dependência de bibliotecas de HMM externas, para fins didáticos.
    """

    def __init__(self, params: HMMParameters) -> None:
        self._p = params

    # ------------------------------------------------------------------
    # Simulação de sequências
    # ------------------------------------------------------------------

    def simulate(self, length: int, seed: int = 42) -> Tuple[List[int], List[int]]:
        """
        Simula uma sequência de estados ocultos e observações.

        Parâmetros
        ----------
        length : número de passos de tempo T.
        seed   : semente para reprodutibilidade.

        Retorna
        -------
        Tupla (estados_verdadeiros, observações).
        """
        rng = np.random.RandomState(seed)
        states = np.zeros(length, dtype=int)
        obs    = np.zeros(length, dtype=int)

        # Estado inicial amostrado de pi
        states[0] = rng.choice(NUM_STATES, p=self._p.pi)
        obs[0]    = rng.choice(NUM_OBS,    p=self._p.B[states[0]])

        for t in range(1, length):
            # Transição de estado
            states[t] = rng.choice(NUM_STATES, p=self._p.A[states[t - 1]])
            # Emissão da observação a partir do novo estado
            obs[t]    = rng.choice(NUM_OBS,    p=self._p.B[states[t]])

        return states.tolist(), obs.tolist()

    # ------------------------------------------------------------------
    # Algoritmo de Viterbi
    # ------------------------------------------------------------------

    def viterbi(self, observations: List[int]) -> ViterbiResult:
        """
        Decodifica a sequência mais provável de estados ocultos (Viterbi).

        Usa escala logarítmica para evitar underflow numérico em sequências longas.

        Parâmetros
        ----------
        observations : sequência de índices de observação (0..K-1).

        Retorna
        -------
        ViterbiResult com a sequência decodificada e a log-probabilidade.
        """
        t0 = time.perf_counter()

        T = len(observations)
        # delta[t, j] = log-prob do caminho mais provável até o estado j no tempo t
        delta = np.full((T, NUM_STATES), -np.inf)
        # psi[t, j] = estado predecessor no caminho mais provável
        psi   = np.zeros((T, NUM_STATES), dtype=int)

        # --- Inicialização (t = 0) ---
        b0 = self._p.B[:, observations[0]]
        delta[0] = np.log(self._p.pi + 1e-12) + np.log(b0 + 1e-12)

        # --- Recursão ---
        for t in range(1, T):
            b_t = np.log(self._p.B[:, observations[t]] + 1e-12)
            log_A = np.log(self._p.A + 1e-12)

            for j in range(NUM_STATES):
                # Para cada estado j, encontra o predecessor com maior log-prob
                scores    = delta[t - 1] + log_A[:, j]
                psi[t, j] = int(np.argmax(scores))
                delta[t, j] = np.max(scores) + b_t[j]

        # --- Terminação: estado final ---
        path = np.zeros(T, dtype=int)
        path[T - 1] = int(np.argmax(delta[T - 1]))
        log_prob    = float(np.max(delta[T - 1]))

        # --- Backtracking ---
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        elapsed = (time.perf_counter() - t0) * 1000.0
        return ViterbiResult(
            decoded_states=path.tolist(),
            log_probability=log_prob,
            elapsed_ms=elapsed,
        )

    # ------------------------------------------------------------------
    # Algoritmo Forward
    # ------------------------------------------------------------------

    def forward(self, observations: List[int]) -> ForwardResult:
        """
        Calcula a distribuição filtrada P(state_t | o_1..o_t) via algoritmo Forward.

        A distribuição é normalizada a cada passo para estabilidade numérica.

        Parâmetros
        ----------
        observations : sequência de índices de observação.

        Retorna
        -------
        ForwardResult com a matriz alpha (T x M) e os estados mais prováveis.
        """
        t0 = time.perf_counter()

        T     = len(observations)
        alpha = np.zeros((T, NUM_STATES))

        # Inicialização
        alpha[0] = self._p.pi * self._p.B[:, observations[0]]
        norm      = alpha[0].sum()
        if norm > 0:
            alpha[0] /= norm

        # Recursão com normalização
        for t in range(1, T):
            for j in range(NUM_STATES):
                alpha[t, j] = self._p.B[j, observations[t]] * np.dot(alpha[t - 1], self._p.A[:, j])
            norm = alpha[t].sum()
            if norm > 0:
                alpha[t] /= norm

        most_likely = np.argmax(alpha, axis=1).tolist()

        elapsed = (time.perf_counter() - t0) * 1000.0
        return ForwardResult(
            alpha=alpha,
            most_likely=most_likely,
            elapsed_ms=elapsed,
        )


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------

class HMMVisualizer:
    """Gera as figuras do experimento de HMM."""

    # Paleta de cores por estado
    STATE_COLORS = ["#5B9BD5", "#ED7D31", "#A9D18E"]

    @staticmethod
    def plot_state_comparison(
        true_states: List[int],
        viterbi_states: List[int],
        forward_states: List[int],
        observations: List[int],
        output_path: str = "figura_03_hmm_estados.png",
        window: int = 80,
    ) -> None:
        """
        Plota comparativo entre estado real, Viterbi e Forward.

        Exibe apenas os primeiros `window` passos para legibilidade.
        """
        T   = min(window, len(true_states))
        t   = np.arange(T)
        obs = observations[:T]

        fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
        fig.patch.set_facecolor("#F8F8F8")

        # Painel 1: Observações
        axes[0].step(t, obs, where="mid", color="#7030A0", lw=1.5)
        axes[0].set_ylabel("Observação", fontsize=9)
        axes[0].set_yticks(range(NUM_OBS))
        axes[0].set_yticklabels(OBS_NAMES, fontsize=7)
        axes[0].set_title("Sequência de Eventos Observados no App", fontsize=10, fontweight="bold")
        axes[0].set_facecolor("#F8F8F8")

        # Painéis 2-4: estados
        datasets = [
            (true_states[:T],   "Estado Real (ground truth)", axes[1]),
            (viterbi_states[:T], "Viterbi (decodificado)",     axes[2]),
            (forward_states[:T], "Forward (estado ML)",        axes[3]),
        ]
        for states_seq, title, ax in datasets:
            colors = [HMMVisualizer.STATE_COLORS[s] for s in states_seq]
            for i in range(len(states_seq) - 1):
                ax.axvspan(t[i], t[i + 1], color=HMMVisualizer.STATE_COLORS[states_seq[i]], alpha=0.7)
            ax.step(t, states_seq, where="mid", color="#333333", lw=0.8)
            ax.set_ylabel("Estado", fontsize=9)
            ax.set_yticks(range(NUM_STATES))
            ax.set_yticklabels(STATE_NAMES, fontsize=7)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_facecolor("#F8F8F8")

        # Legenda de estados
        patches = [
            mpatches.Patch(color=HMMVisualizer.STATE_COLORS[i], label=STATE_NAMES[i])
            for i in range(NUM_STATES)
        ]
        fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
                   bbox_to_anchor=(0.5, 0.01), framealpha=0.9)

        axes[-1].set_xlabel("Passo de tempo (eventos)", fontsize=9)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Figura salva] {output_path}")

    @staticmethod
    def plot_posterior(
        alpha: np.ndarray,
        output_path: str = "figura_04_hmm_posterior.png",
        window: int = 80,
    ) -> None:
        """Plota a distribuição posterior P(state_t | obs_1..obs_t) ao longo do tempo."""

        T = min(window, alpha.shape[0])
        t = np.arange(T)

        fig, ax = plt.subplots(figsize=(13, 4))
        fig.patch.set_facecolor("#F8F8F8")
        ax.set_facecolor("#F8F8F8")

        for i in range(NUM_STATES):
            ax.plot(t, alpha[:T, i], label=STATE_NAMES[i],
                    color=HMMVisualizer.STATE_COLORS[i], lw=1.8)

        ax.set_xlabel("Passo de tempo", fontsize=9)
        ax.set_ylabel("P(estado | observações)", fontsize=9)
        ax.set_title("Distribuição Posterior dos Estados (Filtro Forward)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Figura salva] {output_path}")


# ---------------------------------------------------------------------------
# Parâmetros do modelo
# ---------------------------------------------------------------------------

def build_parameters() -> HMMParameters:
    """
    Define os parâmetros do HMM para o domínio de UX Analytics.

    Probabilidades elicitadas com base em comportamento típico de e-commerce:
    - Usuários passam mais tempo Navegando antes de Comprar.
    - O estado Inativo tem alta auto-transição (sessão congelada).
    - Adicionar ao carrinho é forte indicador do estado Comprando.
    """

    # Distribuição inicial: maioria começa navegando
    pi = np.array([0.70, 0.20, 0.10])

    # Matriz de transição A[i, j] = P(ir para estado j | estado atual i)
    # Linha: estado atual | Coluna: próximo estado
    #            Nav    Comp   Inat
    A = np.array([
        [0.60,  0.25,  0.15],   # Navegando → ...
        [0.30,  0.55,  0.15],   # Comprando → ...
        [0.20,  0.10,  0.70],   # Inativo   → ...
    ])

    # Matriz de emissão B[i, k] = P(observação k | estado i)
    # Linha: estado | Coluna: observação
    #               CliqueCateg  CliqueProd  AddCarrinho  TempoSemAção
    B = np.array([
        [0.50,        0.35,       0.05,        0.10],   # Navegando
        [0.10,        0.40,       0.40,        0.10],   # Comprando
        [0.10,        0.05,       0.02,        0.83],   # Inativo
    ])

    return HMMParameters(pi=pi, A=A, B=B)


# ---------------------------------------------------------------------------
# Experimento principal
# ---------------------------------------------------------------------------

def run_experiment() -> HMMReport:
    """Executa a simulação, decodificação e filtragem; retorna o relatório."""

    params = build_parameters()
    model  = AppUsageHMM(params)

    # Simular 200 eventos de sessão
    SEQUENCE_LENGTH = 200
    true_states, observations = model.simulate(length=SEQUENCE_LENGTH, seed=7)

    report = HMMReport(
        sequence_length=SEQUENCE_LENGTH,
        true_states=true_states,
        observations=observations,
    )

    print("=" * 60)
    print("Projeto 2 – HMM: Padrão de Uso em Aplicativo")
    print("=" * 60)
    print(f"Sequência simulada: {SEQUENCE_LENGTH} eventos\n")

    # --- Viterbi ---
    viterbi_result = model.viterbi(observations)
    report.viterbi = viterbi_result
    print(viterbi_result)
    print()

    # --- Forward ---
    forward_result = model.forward(observations)
    report.forward = forward_result
    print(forward_result)
    print()

    # --- Métricas de acurácia ---
    acc_v = float(np.mean(
        [v == t for v, t in zip(viterbi_result.decoded_states, true_states)]
    ))
    acc_f = float(np.mean(
        [f == t for f, t in zip(forward_result.most_likely, true_states)]
    ))
    report.accuracy_viterbi = acc_v
    report.accuracy_forward = acc_f

    print(f"Acurácia Viterbi : {acc_v:.2%}")
    print(f"Acurácia Forward : {acc_f:.2%}")

    return report


# ---------------------------------------------------------------------------
# Geração das figuras
# ---------------------------------------------------------------------------

def generate_figures(report: HMMReport) -> None:
    """Gera e salva todas as figuras do projeto."""

    HMMVisualizer.plot_state_comparison(
        true_states=report.true_states,
        viterbi_states=report.viterbi.decoded_states,
        forward_states=report.forward.most_likely,
        observations=report.observations,
        output_path="figura_03_hmm_estados.png",
    )
    HMMVisualizer.plot_posterior(
        alpha=report.forward.alpha,
        output_path="figura_04_hmm_posterior.png",
    )


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def evaluate_seeds(n_seeds: int, first_seed: int = 7, length: int = 200) -> None:
    """Repete a simulação com várias sementes e compara as acurácias.

    As sequências são amostradas do próprio modelo usado na decodificação, então
    a acurácia mede a qualidade da inferência, não o ajuste do modelo a dados
    reais. Inclui uma linha de base sem memória temporal (estado mais provável
    dada só a observação atual) e a diferença pareada Viterbi - Forward.
    """
    model = AppUsageHMM(build_parameters())
    B = model._p.B
    acc = {"Viterbi": [], "Forward": [], "Sem tempo (argmax de B)": []}
    for seed in range(first_seed, first_seed + n_seeds):
        true_states, obs = model.simulate(length=length, seed=seed)
        truth = np.array(true_states)
        acc["Viterbi"].append(float(np.mean(np.array(model.viterbi(obs).decoded_states) == truth)))
        acc["Forward"].append(float(np.mean(np.array(model.forward(obs).most_likely) == truth)))
        acc["Sem tempo (argmax de B)"].append(float(np.mean(np.argmax(B[:, obs], axis=0) == truth)))

    print(f"  {n_seeds} sequências de {length} eventos (sementes {first_seed}..{first_seed + n_seeds - 1})")
    for name, values in acc.items():
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"  {name:<24}: {statistics.mean(values):.2%} ± {std:.2%} "
              f"(min {min(values):.2%}, max {max(values):.2%})")
    diff = [v - f for v, f in zip(acc["Viterbi"], acc["Forward"])]
    wins = sum(d > 0 for d in diff)
    ties = sum(d == 0 for d in diff)
    print(f"  Viterbi - Forward: {statistics.mean(diff):+.2%} ± "
          f"{statistics.stdev(diff) if len(diff) > 1 else 0.0:.2%} "
          f"| Viterbi melhor em {wins}, empate em {ties}, Forward melhor em {n_seeds - wins - ties}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1,
                        help="com N > 1, apenas avalia N sequências simuladas e imprime média ± desvio")
    args = parser.parse_args()
    if args.seeds > 1:
        evaluate_seeds(args.seeds)
        raise SystemExit(0)

    report = run_experiment()
    generate_figures(report)
    print("\n" + "=" * 60)
    print("Resumo do experimento:")
    print("=" * 60)
    print(report.summary())
