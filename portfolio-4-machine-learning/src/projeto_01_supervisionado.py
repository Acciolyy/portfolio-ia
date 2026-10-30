"""
Projeto 1: Aprendizado Supervisionado - Classificacao de Cultivares de Uva
via Gradient Boosting.

Disciplina: FGA0221 - Inteligencia Artificial
Portfolio 4 - Aprendizado de Maquina
Autor: Thiago Viriato Accioly

O problema simula o controle de qualidade agricola de uma vinicola que
precisa identificar, a partir de uma analise quimica da uva, de qual
cultivar (variedade) ela se origina. O dataset utilizado e o "Wine
recognition dataset", um repositorio publico classico de aprendizado de
maquina (UCI Machine Learning Repository, distribuido junto ao scikit-learn),
contendo 178 amostras com 13 atributos quimicos (teor alcoolico, acidez
malica, fenois totais, flavonoides, intensidade de cor, entre outros) e
3 classes de cultivar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# Pasta onde as figuras geradas por este script sao salvas.
FIGS_DIR = Path("imagens")
FIGS_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42


@dataclass
class ClassificationResult:
    """DTO com as metricas de avaliacao de um classificador treinado."""

    model_name: str
    accuracy: float
    f1_macro: float
    roc_auc_ovr: float
    training_time_ms: float
    confusion: np.ndarray
    report: str


@dataclass
class CrossValidationResult:
    """DTO com o resultado da validacao cruzada estratificada repetida.

    As pontuacoes de folds de repeticoes diferentes reutilizam as mesmas
    amostras, entao nao sao independentes: o desvio e descritivo da
    variabilidade entre folds, nao um erro-padrao da media.
    """

    model_name: str
    fold_scores: np.ndarray = field(repr=False)
    fold_f1: np.ndarray = field(repr=False)

    @property
    def mean(self) -> float:
        return float(np.mean(self.fold_scores))

    @property
    def std(self) -> float:
        return float(np.std(self.fold_scores, ddof=1))

    @property
    def f1_mean(self) -> float:
        return float(np.mean(self.fold_f1))

    @property
    def f1_std(self) -> float:
        return float(np.std(self.fold_f1, ddof=1))


class WineCultivarClassifier:
    """
    Encapsula o pipeline de classificacao supervisionada de cultivares de
    uva: padronizacao dos atributos, treinamento do modelo e avaliacao.

    A escolha do Gradient Boosting se justifica por ser um metodo de
    ensemble sequencial (boosting), diferente dos ensembles paralelos
    (bagging, como Random Forest) e tambem diferente de redes neurais.
    Cada arvore da sequencia corrige os residuos deixados pelas arvores
    anteriores, o que costuma produzir fronteiras de decisao mais
    refinadas em datasets pequenos e com atributos numericos continuos,
    como e o caso da analise quimica de vinhos.
    """

    def __init__(self, n_estimators: int = 150, learning_rate: float = 0.08,
                 max_depth: int = 2, random_state: int = RANDOM_STATE) -> None:
        self._scaler = StandardScaler()
        self._model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )
        self._feature_names: list[str] = []
        self._class_names: list[str] = []

    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            feature_names: list[str], class_names: list[str]) -> "WineCultivarClassifier":
        self._feature_names = feature_names
        self._class_names = class_names
        x_scaled = self._scaler.fit_transform(x_train)
        self._model.fit(x_scaled, y_train)
        return self

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray, model_name: str) -> ClassificationResult:
        x_scaled = self._scaler.transform(x_test)
        start = perf_counter()
        y_pred = self._model.predict(x_scaled)
        elapsed_ms = (perf_counter() - start) * 1000
        y_proba = self._model.predict_proba(x_scaled)

        return ClassificationResult(
            model_name=model_name,
            accuracy=accuracy_score(y_test, y_pred),
            f1_macro=f1_score(y_test, y_pred, average="macro"),
            roc_auc_ovr=roc_auc_score(y_test, y_proba, multi_class="ovr"),
            training_time_ms=elapsed_ms,
            confusion=confusion_matrix(y_test, y_pred),
            report=classification_report(y_test, y_pred, target_names=self._class_names),
        )

    def feature_importances(self) -> np.ndarray:
        return self._model.feature_importances_

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names


class BaselineLogisticClassifier:
    """Classificador linear utilizado como baseline de comparacao."""

    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        self._scaler = StandardScaler()
        self._model = LogisticRegression(max_iter=2000, random_state=random_state)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "BaselineLogisticClassifier":
        x_scaled = self._scaler.fit_transform(x_train)
        self._model.fit(x_scaled, y_train)
        return self

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray, model_name: str) -> ClassificationResult:
        x_scaled = self._scaler.transform(x_test)
        start = perf_counter()
        y_pred = self._model.predict(x_scaled)
        elapsed_ms = (perf_counter() - start) * 1000
        y_proba = self._model.predict_proba(x_scaled)

        return ClassificationResult(
            model_name=model_name,
            accuracy=accuracy_score(y_test, y_pred),
            f1_macro=f1_score(y_test, y_pred, average="macro"),
            roc_auc_ovr=roc_auc_score(y_test, y_proba, multi_class="ovr"),
            training_time_ms=elapsed_ms,
            confusion=confusion_matrix(y_test, y_pred),
            report="",
        )


def run_cross_validation(estimator, x: np.ndarray, y: np.ndarray, model_name: str,
                          n_splits: int = 5, n_repeats: int = 10) -> CrossValidationResult:
    """Validacao cruzada estratificada repetida. Os folds dependem so da
    semente, entao dois modelos avaliados aqui veem exatamente as mesmas
    particoes (comparacao pareada). O estimador deve ser um Pipeline quando
    ha pre-processamento, para o scaler ser ajustado so no treino de cada fold."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    scores = cross_validate(estimator, x, y, cv=cv, scoring=("accuracy", "f1_macro"))
    return CrossValidationResult(model_name=model_name, fold_scores=scores["test_accuracy"],
                                 fold_f1=scores["test_f1_macro"])


def plot_confusion_matrix(result: ClassificationResult, class_names: list[str], filename: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(result.confusion, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Classe Predita")
    ax.set_ylabel("Classe Real")
    ax.set_title(f"Matriz de Confusao - {result.model_name}")
    for i in range(result.confusion.shape[0]):
        for j in range(result.confusion.shape[1]):
            ax.text(j, i, str(result.confusion[i, j]), ha="center", va="center",
                     color="white" if result.confusion[i, j] > result.confusion.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / filename, dpi=150)
    plt.close(fig)


def plot_feature_importance(importances: np.ndarray, feature_names: list[str], filename: str) -> None:
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([feature_names[i] for i in order][::-1], importances[order][::-1], color="#7B2D26")
    ax.set_xlabel("Importancia Relativa")
    ax.set_title("Importancia dos Atributos - Gradient Boosting")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / filename, dpi=150)
    plt.close(fig)


def plot_model_comparison(results: list[CrossValidationResult], filename: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    names = [r.model_name for r in results]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, [r.mean for r in results], width, yerr=[r.std for r in results],
           capsize=4, label="Acuracia", color="#7B2D26")
    ax.bar(x + width / 2, [r.f1_mean for r in results], width, yerr=[r.f1_std for r in results],
           capsize=4, label="F1-macro", color="#B08968")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.8, 1.02)
    ax.set_ylabel("Score (media +- desvio entre folds)")
    ax.set_title("Validacao cruzada 5x10 folds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS_DIR / filename, dpi=150)
    plt.close(fig)


def main() -> None:
    dataset = load_wine()
    x, y = dataset.data, dataset.target
    feature_names = list(dataset.feature_names)
    class_names = [f"Cultivar {i}" for i in range(len(dataset.target_names))]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    gb_classifier = WineCultivarClassifier().fit(x_train, y_train, feature_names, class_names)
    gb_result = gb_classifier.evaluate(x_test, y_test, "Gradient Boosting")

    baseline = BaselineLogisticClassifier().fit(x_train, y_train)
    baseline_result = baseline.evaluate(x_test, y_test, "Regressao Logistica")

    # Pipelines: o scaler e ajustado so no treino de cada fold (sem vazamento).
    gb_cv = run_cross_validation(make_pipeline(StandardScaler(), GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.08, max_depth=2, random_state=RANDOM_STATE
    )), x, y, "Gradient Boosting")
    lr_cv = run_cross_validation(make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    ), x, y, "Regressao Logistica")

    print(f"=== Split unico de teste ({len(y_test)} amostras; 1 erro = {100/len(y_test):.1f} pontos) ===")
    for r in (gb_result, baseline_result):
        print(f"{r.model_name}: acc={r.accuracy:.4f} f1_macro={r.f1_macro:.4f} "
              f"roc_auc_ovr={r.roc_auc_ovr:.4f} tempo={r.training_time_ms:.3f}ms")

    print("\n=== Relatorio de classificacao (Gradient Boosting) ===")
    print(gb_result.report)

    n_folds = len(gb_cv.fold_scores)
    print(f"=== Validacao cruzada estratificada repetida (5x10 = {n_folds} folds, mesmas particoes) ===")
    for cv in (gb_cv, lr_cv):
        print(f"{cv.model_name}: acc={cv.mean:.4f} +- {cv.std:.4f} | f1_macro={cv.f1_mean:.4f} +- {cv.f1_std:.4f}")
    diff = lr_cv.fold_scores - gb_cv.fold_scores
    print(f"Diferenca pareada (LR - GB) por fold: media={diff.mean():+.4f} "
          f"| LR melhor em {(diff > 0).sum()}, empate em {(diff == 0).sum()}, GB melhor em {(diff < 0).sum()} de {n_folds}")

    plot_confusion_matrix(gb_result, class_names, "p1_confusion_matrix.png")
    plot_feature_importance(gb_classifier.feature_importances(), feature_names, "p1_feature_importance.png")
    plot_model_comparison([gb_cv, lr_cv], "p1_model_comparison.png")

    print("\nFiguras salvas em:", FIGS_DIR)


if __name__ == "__main__":
    main()
