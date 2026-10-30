"""
Projeto 4: Deep Learning - Classificacao de Digitos Manuscritos com CNN
estilo LeNet-5.

Disciplina: FGA0221 - Inteligencia Artificial
Portfolio 4 - Aprendizado de Maquina
Autor: Thiago Viriato Accioly

O problema consiste em classificar imagens de digitos manuscritos (0 a 9)
a partir de pixels brutos, usando o dataset MNIST (LeCun et al.), um dos
repositorios publicos mais utilizados para validar arquiteturas de visao
computacional. A arquitetura escolhida e uma variacao da LeNet-5 (LeCun,
1998), uma das primeiras CNNs bem-sucedidas em aplicacoes praticas, com
duas camadas convolucionais seguidas de pooling e um classificador denso
ao final.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Pasta onde as figuras geradas por este script sao salvas.
FIGS_DIR = Path("imagens")
FIGS_DIR.mkdir(exist_ok=True, parents=True)

# Semente da particao treino/validacao (fixa em todas as execucoes) e semente
# padrao de inicializacao e embaralhamento. Ver --seeds em main().
RANDOM_STATE = 42
N_VAL = 5_000


@dataclass
class TrainingConfig:
    """DTO com os hiperparametros de treinamento."""

    batch_size: int = 128
    n_epochs: int = 8
    learning_rate: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EpochMetrics:
    """DTO com as metricas de uma epoca de treinamento."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    epoch_time_s: float


@dataclass
class TrainingHistory:
    """DTO agregando o historico de metricas ao longo do treinamento."""

    epochs: list[EpochMetrics] = field(default_factory=list)

    def add(self, metrics: EpochMetrics) -> None:
        self.epochs.append(metrics)

    @property
    def train_losses(self) -> list[float]:
        return [e.train_loss for e in self.epochs]

    @property
    def val_losses(self) -> list[float]:
        return [e.val_loss for e in self.epochs]

    @property
    def train_accuracies(self) -> list[float]:
        return [e.train_accuracy for e in self.epochs]

    @property
    def val_accuracies(self) -> list[float]:
        return [e.val_accuracy for e in self.epochs]


class LeNet5(nn.Module):
    """
    Variacao da arquitetura LeNet-5 (LeCun et al., 1998) adaptada para
    entradas 28x28 de canal unico (padrao MNIST).

    Estrutura:
        Conv(1->6, k=5, pad=2) -> ReLU -> MaxPool(2)
        Conv(6->16, k=5)       -> ReLU -> MaxPool(2)
        Flatten
        Linear(16*5*5 -> 120)  -> ReLU
        Linear(120 -> 84)      -> ReLU
        Linear(84 -> 10)

    A convolucao com padding=2 na primeira camada preserva o mapeamento
    original de 32x32 da LeNet-5 classica mesmo recebendo entradas 28x28,
    o que mantem as dimensoes das camadas seguintes identicas ao artigo
    original. A funcao de ativacao foi trocada de tanh (original de 1998)
    para ReLU, prática padrao atual que acelera a convergencia e mitiga o
    problema de gradientes que desaparecem.
    """

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MnistDataModule:
    """Encapsula o download, a normalizacao e o particionamento do MNIST.

    Os 60k exemplos de treino sao divididos em 55k treino / 5k validacao com
    uma semente fixa. O conjunto de teste (10k) so e usado na avaliacao final,
    uma unica vez, para que nenhuma decisao de treino o enxergue.
    """

    MEAN, STD = 0.1307, 0.3081  # estatisticas padrao do MNIST (LeCun et al.)

    def __init__(self, data_dir: str = "./mnist_data", batch_size: int = 128,
                 n_val: int = N_VAL, split_seed: int = RANDOM_STATE) -> None:
        self._batch_size = batch_size
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self.MEAN,), (self.STD,)),
        ])
        full_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        self.test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        self.train_set, self.val_set = random_split(
            full_train, [len(full_train) - n_val, n_val],
            generator=torch.Generator().manual_seed(split_seed),
        )

    def train_loader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self._batch_size, shuffle=True, num_workers=2)

    def val_loader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self._batch_size, shuffle=False, num_workers=2)

    def test_loader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=self._batch_size, shuffle=False, num_workers=2)


class LeNetTrainer:
    """Orquestra o loop de treinamento e avaliacao da LeNet-5."""

    def __init__(self, model: LeNet5, config: TrainingConfig) -> None:
        self._model = model.to(config.device)
        self._config = config
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=config.learning_rate)
        self._criterion = nn.CrossEntropyLoss()

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        self._model.train(mode=train)
        total_loss, correct, total = 0.0, 0, 0
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for images, labels in loader:
                images = images.to(self._config.device)
                labels = labels.to(self._config.device)

                if train:
                    self._optimizer.zero_grad()
                logits = self._model(images)
                loss = self._criterion(logits, labels)
                if train:
                    loss.backward()
                    self._optimizer.step()

                total_loss += loss.item() * images.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += images.size(0)
        return total_loss / total, correct / total

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingHistory:
        history = TrainingHistory()
        for epoch in range(1, self._config.n_epochs + 1):
            start = perf_counter()
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val_loss, val_acc = self._run_epoch(val_loader, train=False)
            elapsed = perf_counter() - start

            history.add(EpochMetrics(epoch, train_loss, train_acc, val_loss, val_acc, elapsed))
            print(f"Epoca {epoch}/{self._config.n_epochs} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                  f"tempo={elapsed:.1f}s")
        return history

    def collect_confusion_matrix(self, loader: DataLoader, n_classes: int = 10) -> np.ndarray:
        self._model.eval()
        matrix = np.zeros((n_classes, n_classes), dtype=int)
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self._config.device)
                preds = self._model(images).argmax(dim=1).cpu().numpy()
                for true_label, pred_label in zip(labels.numpy(), preds):
                    matrix[true_label, pred_label] += 1
        return matrix

    @property
    def model(self) -> LeNet5:
        return self._model


def plot_training_curves(history: TrainingHistory, filename: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4))
    epochs = [e.epoch for e in history.epochs]

    ax1.plot(epochs, history.train_losses, marker="o", label="Treino", color="#7B2D26")
    ax1.plot(epochs, history.val_losses, marker="o", label="Validacao", color="#B08968")
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Loss (Entropia Cruzada)")
    ax1.set_title("Curva de Perda")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history.train_accuracies, marker="o", label="Treino", color="#7B2D26")
    ax2.plot(epochs, history.val_accuracies, marker="o", label="Validacao", color="#B08968")
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Acuracia")
    ax2.set_title("Curva de Acuracia")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGS_DIR / filename, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(matrix: np.ndarray, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Digito Predito")
    ax.set_ylabel("Digito Real")
    ax.set_title("Matriz de Confusao - LeNet-5 no MNIST (conjunto de teste)")
    for i in range(10):
        for j in range(10):
            if matrix[i, j] > 0:
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7,
                         color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / filename, dpi=150)
    plt.close(fig)


def plot_sample_predictions(model: LeNet5, loader: DataLoader, device: str, filename: str, n_samples: int = 10) -> None:
    model.eval()
    images, labels = next(iter(loader))
    images_device = images[:n_samples].to(device)
    with torch.no_grad():
        preds = model(images_device).argmax(dim=1).cpu().numpy()

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for idx, ax in enumerate(axes.flat):
        img = images[idx][0] * MnistDataModule.STD + MnistDataModule.MEAN
        ax.imshow(img, cmap="gray")
        true_label = labels[idx].item()
        pred_label = preds[idx]
        color = "green" if true_label == pred_label else "red"
        ax.set_title(f"Real: {true_label} | Pred: {pred_label}", color=color, fontsize=9)
        ax.axis("off")
    fig.suptitle("Amostras de Predicao no Conjunto de Teste")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / filename, dpi=150)
    plt.close(fig)


@dataclass
class RunResult:
    """DTO com o resultado de uma execucao completa (uma semente)."""

    seed: int
    history: TrainingHistory
    confusion: np.ndarray
    val_accuracy: float
    test_accuracy: float
    trainer: LeNetTrainer


def run_experiment(data: MnistDataModule, config: TrainingConfig, seed: int) -> RunResult:
    """Treina com 'seed', acompanha a validacao e avalia o teste uma unica vez."""
    torch.manual_seed(seed)
    trainer = LeNetTrainer(LeNet5(), config)
    history = trainer.fit(data.train_loader(), data.val_loader())

    confusion = trainer.collect_confusion_matrix(data.test_loader())
    test_accuracy = float(np.trace(confusion) / confusion.sum())
    return RunResult(seed, history, confusion, history.val_accuracies[-1], test_accuracy, trainer)


def main() -> None:
    parser = argparse.ArgumentParser(description="LeNet-5 no MNIST")
    parser.add_argument("--seeds", type=int, default=1,
                        help="numero de execucoes com sementes consecutivas a partir de 42")
    args = parser.parse_args()

    config = TrainingConfig()
    print(f"Dispositivo de treinamento: {config.device}")

    data = MnistDataModule(batch_size=config.batch_size)
    print(f"Particao: {len(data.train_set)} treino / {len(data.val_set)} validacao / "
          f"{len(data.test_set)} teste")

    results = []
    for seed in range(RANDOM_STATE, RANDOM_STATE + args.seeds):
        print(f"\n--- Semente {seed} ---")
        results.append(run_experiment(data, config, seed))
        print(f"Validacao (ultima epoca): {results[-1].val_accuracy*100:.2f}% | "
              f"Teste: {results[-1].test_accuracy*100:.2f}%")

    val_accs = np.array([r.val_accuracy for r in results]) * 100
    test_accs = np.array([r.test_accuracy for r in results]) * 100
    print(f"\nAcuracia de validacao: {val_accs.mean():.2f}% +- {val_accs.std(ddof=1) if len(val_accs) > 1 else 0.0:.2f}")
    print(f"Acuracia de teste (avaliada uma vez por execucao): "
          f"{test_accs.mean():.2f}% +- {test_accs.std(ddof=1) if len(test_accs) > 1 else 0.0:.2f} "
          f"({len(results)} semente(s))")

    # Figuras da primeira execucao (semente 42).
    first = results[0]
    plot_training_curves(first.history, "p4_training_curves.png")
    plot_confusion_matrix(first.confusion, "p4_confusion_matrix.png")
    plot_sample_predictions(first.trainer.model, data.test_loader(), config.device, "p4_sample_predictions.png")

    print("\nFiguras salvas em:", FIGS_DIR)


if __name__ == "__main__":
    main()
