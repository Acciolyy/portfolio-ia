from __future__ import annotations

"""
Projeto 1 – Rede Bayesiana: Diagnóstico de Qualidade de Café em Torrefadora
============================================================================
Disciplina : FGA0221 – Inteligência Artificial
Autor      : Thiago Viriato Accioly
Matrícula  : 231029340
Professor  : Dr. Fabiano Araujo Soares

Descrição
---------
Modela o processo de torra de café como uma Rede Bayesiana. Variáveis
relacionadas ao processo (temperatura de torra, umidade inicial do grão e
tempo de exposição ao calor) influenciam probabilisticamente a ocorrência de
defeito no sabor do lote. O objetivo é realizar inferências causais e
diagnósticas para auxiliar o controle de qualidade da produção.

Estrutura causal
----------------
    UmidadeAlta ──┐
                  ▼
    TempAlta ───► DefeitoSabor
                  ▲
    TempoExcessivo┘

Bibliotecas
-----------
- pgmpy  : modelagem de redes bayesianas e inferência por eliminação de variáveis.
- matplotlib : geração do diagrama visual da rede e dos gráficos de resultado.
- networkx : construção e layout do grafo direcional para visualização.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork


# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Encapsula o resultado de uma consulta de inferência na rede bayesiana."""

    query_label: str
    variable: str
    evidence: Dict[str, int]
    prob_false: float
    prob_true: float
    elapsed_ms: float
    inference_type: str

    def __str__(self) -> str:
        lines = [
            f"[{self.inference_type}] {self.query_label}",
            f"  Variável consultada : {self.variable}",
            f"  Evidências          : {self.evidence}",
            f"  P({self.variable}=False) = {self.prob_false:.4f}",
            f"  P({self.variable}=True)  = {self.prob_true:.4f}",
            f"  Tempo de inferência : {self.elapsed_ms:.4f} ms",
        ]
        return "\n".join(lines)


@dataclass
class BayesianNetworkReport:
    """Agrega todos os resultados gerados pela execução do experimento."""

    model_name: str
    nodes: List[str]
    edges: List[Tuple[str, str]]
    query_results: List[QueryResult] = field(default_factory=list)
    model_valid: bool = False

    def summary(self) -> str:
        lines = [
            f"Modelo  : {self.model_name}",
            f"Nós     : {self.nodes}",
            f"Arestas : {self.edges}",
            f"Válido  : {self.model_valid}",
            "",
        ]
        for qr in self.query_results:
            lines.append(str(qr))
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Construção do modelo
# ---------------------------------------------------------------------------

class CoffeeQualityBayesianNetwork:
    """
    Rede Bayesiana para diagnóstico de defeito de sabor em lotes de café torrado.

    Variáveis (binárias: 0 = False, 1 = True)
    ------------------------------------------
    UmidadeAlta    : grão com teor de umidade acima do ideal (> 12 %).
    TempAlta       : temperatura de torra acima do range recomendado (> 220 °C).
    TempoExcessivo : tempo de torra além do ponto ideal (> 14 min).
    DefeitoSabor   : lote apresenta defeito perceptível no sabor (queimado/amargo).
    """

    # Nomes das variáveis
    UMIDADE      = "UmidadeAlta"
    TEMPERATURA  = "TempAlta"
    TEMPO        = "TempoExcessivo"
    DEFEITO      = "DefeitoSabor"

    def __init__(self) -> None:
        self._model: DiscreteBayesianNetwork | None = None
        self._engine: VariableElimination | None = None

    # ------------------------------------------------------------------
    # Construção e validação
    # ------------------------------------------------------------------

    def build(self) -> "CoffeeQualityBayesianNetwork":
        """Instancia a estrutura da rede e define todas as CPDs."""

        # Estrutura causal: causas → efeito
        self._model = DiscreteBayesianNetwork([
            (self.UMIDADE,     self.DEFEITO),
            (self.TEMPERATURA, self.DEFEITO),
            (self.TEMPO,       self.DEFEITO),
        ])

        # --- CPDs das variáveis raiz (probabilidades a priori) ---

        # Umidade acima do ideal: ocorre em ~30 % dos lotes recebidos
        cpd_umidade = TabularCPD(
            variable=self.UMIDADE,
            variable_card=2,
            values=[[0.70], [0.30]],  # [False, True]
        )

        # Temperatura de torra elevada: operador ajusta mal em ~25 % das sessões
        cpd_temp = TabularCPD(
            variable=self.TEMPERATURA,
            variable_card=2,
            values=[[0.75], [0.25]],
        )

        # Tempo excessivo: risco independente de ~20 % por distração/falha de timer
        cpd_tempo = TabularCPD(
            variable=self.TEMPO,
            variable_card=2,
            values=[[0.80], [0.20]],
        )

        # --- CPD condicional do defeito de sabor ---
        # Ordem das evidências: [UmidadeAlta, TempAlta, TempoExcessivo]
        # 2³ = 8 combinações de evidências (False/True para cada variável pai)
        #
        # Lógica das probabilidades:
        #   - Sem nenhum fator de risco       → P(Defeito) ≈ 0.03  (erro residual)
        #   - Apenas umidade alta             → P(Defeito) ≈ 0.20
        #   - Apenas temperatura alta         → P(Defeito) ≈ 0.40
        #   - Apenas tempo excessivo          → P(Defeito) ≈ 0.30
        #   - Umidade + temperatura           → P(Defeito) ≈ 0.65
        #   - Umidade + tempo                 → P(Defeito) ≈ 0.55
        #   - Temperatura + tempo             → P(Defeito) ≈ 0.80
        #   - Todos os três fatores ativos    → P(Defeito) ≈ 0.97
        #
        # Colunas (produto cartesiano):
        # U=0,T=0,Te=0 | U=1,T=0,Te=0 | U=0,T=1,Te=0 | U=1,T=1,Te=0 |
        # U=0,T=0,Te=1 | U=1,T=0,Te=1 | U=0,T=1,Te=1 | U=1,T=1,Te=1

        cpd_defeito = TabularCPD(
            variable=self.DEFEITO,
            variable_card=2,
            values=[
                # P(Defeito=False | ...)
                [0.97, 0.80, 0.60, 0.35, 0.70, 0.45, 0.20, 0.03],
                # P(Defeito=True  | ...)
                [0.03, 0.20, 0.40, 0.65, 0.30, 0.55, 0.80, 0.97],
            ],
            evidence=[self.UMIDADE, self.TEMPERATURA, self.TEMPO],
            evidence_card=[2, 2, 2],
        )

        self._model.add_cpds(cpd_umidade, cpd_temp, cpd_tempo, cpd_defeito)
        self._engine = VariableElimination(self._model)
        return self

    def is_valid(self) -> bool:
        """Verifica a consistência do modelo (soma de CPDs = 1)."""
        return self._model.check_model()

    # ------------------------------------------------------------------
    # Inferência
    # ------------------------------------------------------------------

    def query(
        self,
        variable: str,
        evidence: Dict[str, int],
        label: str,
        inference_type: str,
    ) -> QueryResult:
        """
        Executa uma consulta de inferência por eliminação de variáveis.

        Parâmetros
        ----------
        variable       : variável-alvo da consulta.
        evidence       : dicionário {variável: valor_observado}.
        label          : descrição legível da consulta (para o relatório).
        inference_type : tipo de raciocínio ('Causal', 'Diagnóstica', 'Intercausal').
        """
        t0 = time.perf_counter()
        result = self._engine.query(variables=[variable], evidence=evidence, show_progress=False)
        elapsed = (time.perf_counter() - t0) * 1000.0

        probs = result.values  # [P(False), P(True)]
        return QueryResult(
            query_label=label,
            variable=variable,
            evidence=evidence,
            prob_false=float(probs[0]),
            prob_true=float(probs[1]),
            elapsed_ms=elapsed,
            inference_type=inference_type,
        )

    # ------------------------------------------------------------------
    # Propriedades auxiliares
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> List[str]:
        return list(self._model.nodes())

    @property
    def edges(self) -> List[Tuple[str, str]]:
        return list(self._model.edges())


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------

class CoffeeNetworkVisualizer:
    """Gera as figuras do experimento (diagrama da rede e gráfico de resultados)."""

    # Rótulos amigáveis para exibição
    LABELS = {
        "UmidadeAlta"    : "Umidade\nAlta",
        "TempAlta"       : "Temperatura\nAlta",
        "TempoExcessivo" : "Tempo\nExcessivo",
        "DefeitoSabor"   : "Defeito\nno Sabor",
    }

    COLORS = {
        "UmidadeAlta"    : "#5B9BD5",
        "TempAlta"       : "#ED7D31",
        "TempoExcessivo" : "#A9D18E",
        "DefeitoSabor"   : "#FF6B6B",
    }

    @staticmethod
    def plot_network(
        edges: List[Tuple[str, str]],
        output_path: str = "figura_01_rede_bayesiana.png",
    ) -> None:
        """Plota o grafo direcional da rede bayesiana."""

        G = nx.DiGraph()
        G.add_edges_from(edges)

        # Layout manual para melhor legibilidade
        pos = {
            "UmidadeAlta"    : (-1.5,  1.0),
            "TempAlta"       : ( 0.0,  1.0),
            "TempoExcessivo" : ( 1.5,  1.0),
            "DefeitoSabor"   : ( 0.0, -0.5),
        }

        node_colors = [
            CoffeeNetworkVisualizer.COLORS.get(n, "#CCCCCC") for n in G.nodes()
        ]
        labels = {
            n: CoffeeNetworkVisualizer.LABELS.get(n, n) for n in G.nodes()
        }

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#F8F8F8")
        ax.set_facecolor("#F8F8F8")

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=3200,
            alpha=0.95,
        )
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=9, font_weight="bold")
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=22,
            edge_color="#555555",
            width=2.0,
            connectionstyle="arc3,rad=0.05",
        )

        # Legenda
        patches = [
            mpatches.Patch(color=CoffeeNetworkVisualizer.COLORS[n], label=CoffeeNetworkVisualizer.LABELS[n].replace("\n", " "))
            for n in ["UmidadeAlta", "TempAlta", "TempoExcessivo", "DefeitoSabor"]
        ]
        ax.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.8)
        ax.set_title("Rede Bayesiana – Diagnóstico de Qualidade do Café", fontsize=12, fontweight="bold", pad=12)
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Figura salva] {output_path}")

    @staticmethod
    def plot_query_results(
        results: List[QueryResult],
        output_path: str = "figura_02_inferencias.png",
    ) -> None:
        """Plota gráfico de barras com as probabilidades de defeito para cada consulta."""

        labels      = [f"Consulta {i+1}" for i in range(len(results))]
        probs_true  = [r.prob_true  for r in results]
        probs_false = [r.prob_false for r in results]
        descriptions = [r.query_label for r in results]

        x = range(len(results))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#F8F8F8")
        ax.set_facecolor("#F8F8F8")

        bars_true  = ax.bar([xi - width/2 for xi in x], probs_true,  width, label="P(Defeito=True)",  color="#FF6B6B", alpha=0.85)
        bars_false = ax.bar([xi + width/2 for xi in x], probs_false, width, label="P(Defeito=False)", color="#5B9BD5", alpha=0.85)

        # Valores sobre as barras
        for bar in bars_true + bars_false:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.01,
                f"{h:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Probabilidade")
        ax.set_title("Resultados das Inferências – P(DefeitoSabor)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        # Anotações com descrição curta
        for i, desc in enumerate(descriptions):
            ax.text(i, -0.10, desc, ha="center", va="top", fontsize=7,
                    color="#444444", wrap=True, transform=ax.get_xaxis_transform())

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Figura salva] {output_path}")


# ---------------------------------------------------------------------------
# Experimento principal
# ---------------------------------------------------------------------------

def run_experiment() -> BayesianNetworkReport:
    """Constrói a rede, executa as inferências e retorna o relatório agregado."""

    # --- Construção e validação ---
    network = CoffeeQualityBayesianNetwork()
    network.build()

    report = BayesianNetworkReport(
        model_name="Diagnóstico de Qualidade de Café",
        nodes=network.nodes,
        edges=network.edges,
        model_valid=network.is_valid(),
    )

    print("=" * 60)
    print("Projeto 1 – Rede Bayesiana: Qualidade do Café")
    print("=" * 60)
    print(f"Modelo válido: {report.model_valid}\n")

    # --- Inferência 1 – Causal ---
    # Dado que a temperatura está alta, qual é a probabilidade de defeito?
    q1 = network.query(
        variable=CoffeeQualityBayesianNetwork.DEFEITO,
        evidence={CoffeeQualityBayesianNetwork.TEMPERATURA: 1},
        label="Temp. Alta",
        inference_type="Causal",
    )
    report.query_results.append(q1)
    print(q1)
    print()

    # --- Inferência 2 – Causal com múltiplas causas ---
    # Temperatura alta E tempo excessivo → qual a chance de defeito?
    q2 = network.query(
        variable=CoffeeQualityBayesianNetwork.DEFEITO,
        evidence={
            CoffeeQualityBayesianNetwork.TEMPERATURA: 1,
            CoffeeQualityBayesianNetwork.TEMPO: 1,
        },
        label="Temp.+Tempo Altos",
        inference_type="Causal",
    )
    report.query_results.append(q2)
    print(q2)
    print()

    # --- Inferência 3 – Diagnóstica ---
    # Observamos defeito no sabor → qual a probabilidade de temperatura ter sido alta?
    q3 = network.query(
        variable=CoffeeQualityBayesianNetwork.TEMPERATURA,
        evidence={CoffeeQualityBayesianNetwork.DEFEITO: 1},
        label="Defeito → Temp.?",
        inference_type="Diagnóstica",
    )
    report.query_results.append(q3)
    print(q3)
    print()

    # --- Inferência 4 – Diagnóstica com explicação alternativa ---
    # Defeito observado, mas sabemos que o tempo foi normal → umidade suspeita?
    q4 = network.query(
        variable=CoffeeQualityBayesianNetwork.UMIDADE,
        evidence={
            CoffeeQualityBayesianNetwork.DEFEITO: 1,
            CoffeeQualityBayesianNetwork.TEMPO: 0,
        },
        label="Defeito+Tempo OK → Umidade?",
        inference_type="Diagnóstica",
    )
    report.query_results.append(q4)
    print(q4)
    print()

    # --- Inferência 5 – Cenário ideal (sem nenhum fator de risco) ---
    q5 = network.query(
        variable=CoffeeQualityBayesianNetwork.DEFEITO,
        evidence={
            CoffeeQualityBayesianNetwork.UMIDADE: 0,
            CoffeeQualityBayesianNetwork.TEMPERATURA: 0,
            CoffeeQualityBayesianNetwork.TEMPO: 0,
        },
        label="Sem fatores de risco",
        inference_type="Causal",
    )
    report.query_results.append(q5)
    print(q5)
    print()

    return report


# ---------------------------------------------------------------------------
# Geração das figuras
# ---------------------------------------------------------------------------

def generate_figures(report: BayesianNetworkReport) -> None:
    """Gera e salva todas as figuras do projeto."""

    CoffeeNetworkVisualizer.plot_network(
        edges=report.edges,
        output_path="figura_01_rede_bayesiana.png",
    )

    # Para o gráfico de barras, usamos apenas as consultas sobre DefeitoSabor
    defeito_queries = [
        r for r in report.query_results
        if r.variable == CoffeeQualityBayesianNetwork.DEFEITO
    ]
    CoffeeNetworkVisualizer.plot_query_results(
        results=defeito_queries,
        output_path="figura_02_inferencias.png",
    )


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    report = run_experiment()
    generate_figures(report)
    print("\n" + "=" * 60)
    print("Resumo completo do experimento:")
    print("=" * 60)
    print(report.summary())
