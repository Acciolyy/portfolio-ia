"""
Projeto 2: Aprendizado Nao Supervisionado - Segmentacao de Amostras de Vinho
via K-Means e Analise de Componentes Principais (PCA).

Disciplina: FGA0221 - Inteligencia Artificial
Portfolio 4 - Aprendizado de Maquina
Autor: Thiago Viriato Accioly

Este projeto reutiliza o mesmo dataset quimico de vinhos do Projeto 1
(Wine recognition dataset, UCI/scikit-learn), porem sob uma otica diferente:
os rotulos de cultivar sao descartados durante o treinamento. O objetivo
passa a ser descobrir agrupamentos naturais nas amostras a partir apenas
dos atributos quimicos, simulando o cenario de uma vinicola que recebe um
lote de uvas sem informacao de origem e precisa segmenta-las por
similaridade quimica antes de decidir o processo de vinificacao. Os
rotulos verdadeiros sao usados apenas ao final, como validacao externa da
qualidade do agrupamento encontrado, nunca como sinal de treinamento.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix

# Pasta onde as figuras geradas por este script sao salvas.
FIGS_DIR = Path("imagens")
FIGS_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42


@dataclass
class ClusteringResult:
    """DTO com o resultado de uma rodada de agrupamento K-Means."""

    k: int
    inertia: float
    silhouette: float
    labels: np.ndarray


@dataclass
class ExternalValidationResult:
    """
    DTO com metricas de validacao externa do agrupamento, comparando os
    clusters encontrados (sem supervisao) com os rotulos reais de cultivar
    (usados apenas para auditoria, nao para treino).
    """

    adjusted_rand_index: float
    confusion: np.ndarray


class WineClusterAnalyzer:
    """
    Encapsula o pipeline de agrupamento nao supervisionado das amostras
    de vinho: padronizacao, reducao de dimensionalidade via PCA para
    visualizacao 2D, e agrupamento via K-Means.

    A padronizacao (StandardScaler) e essencial aqui porque o K-Means
    usa distancia Euclidiana: sem ela, o atributo "Prolina" (escala de
    centenas) dominaria completamente o calculo de distancia sobre
    atributos como "Hue" (escala de decimos).
    """

    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        self._scaler = StandardScaler()
        self._pca = PCA(n_components=2, random_state=random_state)
        self._x_scaled: np.ndarray | None = None
        self._x_pca: np.ndarray | None = None

    def prepare(self, x: np.ndarray) -> "WineClusterAnalyzer":
        self._x_scaled = self._scaler.fit_transform(x)
        self._x_pca = self._pca.fit_transform(self._x_scaled)
        return self

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self._pca.explained_variance_ratio_

    def sweep_k(self, k_range: range) -> list[ClusteringResult]:
        """Executa K-Means para uma faixa de k, no espaco padronizado original (13D)."""
        results = []
        for k in k_range:
            model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
            labels = model.fit_predict(self._x_scaled)
            sil = silhouette_score(self._x_scaled, labels) if k > 1 else float("nan")
            results.append(ClusteringResult(k=k, inertia=model.inertia_, silhouette=sil, labels=labels))
        return results

    def fit_final(self, k: int) -> ClusteringResult:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = model.fit_predict(self._x_scaled)
        sil = silhouette_score(self._x_scaled, labels)
        return ClusteringResult(k=k, inertia=model.inertia_, silhouette=sil, labels=labels)

    @property
    def x_pca(self) -> np.ndarray:
        return self._x_pca


def validate_against_ground_truth(cluster_labels: np.ndarray, true_labels: np.ndarray) -> ExternalValidationResult:
    ari = adjusted_rand_score(true_labels, cluster_labels)

    # Alinha os indices de cluster aos rotulos reais por correspondencia de maioria,
    # apenas para fins de visualizacao da matriz de confusao (o ARI acima e
    # invariante a permutacao e nao depende desse alinhamento).
    n_labels = len(np.unique(true_labels))
    remap = {}
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        majority_label = np.bincount(true_labels[mask]).argmax()
        remap[cluster_id] = majority_label
    aligned = np.array([remap[c] for c in cluster_labels])

    return ExternalValidationResult(
        adjusted_rand_index=ari,
        confusion=confusion_matrix(true_labels, aligned, labels=list(range(n_labels))),
    )


def plot_elbow_and_silhouette(results: list[ClusteringResult], filename: str) -> None:
    ks = [r.k for r in results]
    inertias = [r.inertia for r in results]
    silhouettes = [r.silhouette for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(ks, inertias, marker="o", color="#7B2D26")
    ax1.set_xlabel("Numero de Clusters (k)")
    ax1.set_ylabel("Inercia (WCSS)")
    ax1.set_title("Metodo do Cotovelo")
    ax1.grid(alpha=0.3)

    ax2.plot(ks, silhouettes, marker="o", color="#B08968")
    ax2.set_xlabel("Numero de Clusters (k)")
    ax2.set_ylabel("Coeficiente de Silhueta")
    ax2.set_title("Score de Silhueta por k")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGS_DIR / filename, dpi=150)
    plt.close(fig)


def plot_clusters_pca(x_pca: np.ndarray, cluster_labels: np.ndarray, true_labels: np.ndarray,
                       explained_variance: np.ndarray, filename: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5))

    scatter1 = ax1.scatter(x_pca[:, 0], x_pca[:, 1], c=cluster_labels, cmap="Set1", s=35, edgecolor="k", linewidth=0.3)
    ax1.set_title("Clusters Encontrados (K-Means, sem rotulos)")
    ax1.set_xlabel(f"PC1 ({explained_variance[0]*100:.1f}% var.)")
    ax1.set_ylabel(f"PC2 ({explained_variance[1]*100:.1f}% var.)")
    legend1 = ax1.legend(*scatter1.legend_elements(), title="Cluster", loc="upper right", fontsize=8)
    ax1.add_artist(legend1)

    scatter2 = ax2.scatter(x_pca[:, 0], x_pca[:, 1], c=true_labels, cmap="Set1", s=35, edgecolor="k", linewidth=0.3)
    ax2.set_title("Cultivares Reais (referencia, nao usada no treino)")
    ax2.set_xlabel(f"PC1 ({explained_variance[0]*100:.1f}% var.)")
    ax2.set_ylabel(f"PC2 ({explained_variance[1]*100:.1f}% var.)")
    legend2 = ax2.legend(*scatter2.legend_elements(), title="Cultivar", loc="upper right", fontsize=8)
    ax2.add_artist(legend2)

    fig.tight_layout()
    fig.savefig(FIGS_DIR / filename, dpi=150)
    plt.close(fig)


def main() -> None:
    dataset = load_wine()
    x, y_true = dataset.data, dataset.target

    analyzer = WineClusterAnalyzer().prepare(x)
    print(f"Variancia explicada pelas 2 primeiras componentes: "
          f"{analyzer.explained_variance_ratio.sum()*100:.2f}% "
          f"(PC1={analyzer.explained_variance_ratio[0]*100:.2f}%, "
          f"PC2={analyzer.explained_variance_ratio[1]*100:.2f}%)")

    sweep_results = analyzer.sweep_k(range(2, 8))
    print("\n=== Varredura de k (metodo do cotovelo + silhueta) ===")
    for r in sweep_results:
        print(f"k={r.k}: inercia={r.inertia:.2f} silhueta={r.silhouette:.4f}")

    best = max(sweep_results, key=lambda r: r.silhouette)
    print(f"\nMelhor k por silhueta: k={best.k} (silhueta={best.silhouette:.4f})")

    final = analyzer.fit_final(k=3)
    print("\n=== Agrupamento final com k=3 (numero real de cultivares) ===")
    print(f"Inercia: {final.inertia:.2f} | Silhueta: {final.silhouette:.4f}")

    validation = validate_against_ground_truth(final.labels, y_true)
    print("\n=== Validacao externa (auditoria pos-hoc, nao usada no treino) ===")
    print(f"Adjusted Rand Index vs. cultivares reais: {validation.adjusted_rand_index:.4f}")
    print("Matriz de confusao (cluster alinhado x cultivar real):")
    print(validation.confusion)

    plot_elbow_and_silhouette(sweep_results, "p2_elbow_silhouette.png")
    plot_clusters_pca(analyzer.x_pca, final.labels, y_true, analyzer.explained_variance_ratio, "p2_clusters_pca.png")

    print("\nFiguras salvas em:", FIGS_DIR)


if __name__ == "__main__":
    main()
