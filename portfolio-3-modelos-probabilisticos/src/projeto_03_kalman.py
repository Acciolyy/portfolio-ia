from __future__ import annotations

"""
Projeto 3 – Filtro de Kalman: Estimativa de Posição de Drone com GPS Ruidoso
=============================================================================
Disciplina : FGA0221 – Inteligência Artificial
Autor      : Thiago Viriato Accioly
Matrícula  : 231029340
Professor  : Dr. Fabiano Araujo Soares

Descrição
---------
Simula o sistema de navegação de um drone em missão de inspeção de área.
O drone segue uma rota de voo com variações suaves de velocidade (causadas por
vento leve), mas o receptor GPS embarcado introduz ruído de medição considerável.
Um Filtro de Kalman 2D (posição X e Y simultaneamente) é aplicado para estimar
a posição real do drone a partir das leituras ruidosas do GPS, combinando o
modelo dinâmico de voo com as medições do sensor.

Modelo de espaço de estados (2D)
---------------------------------
Estado     : x_k = [pos_x, pos_y, vel_x, vel_y]ᵀ  (posição + velocidade)
Observação : z_k = [pos_x_GPS, pos_y_GPS]ᵀ         (apenas posição, com ruído)

Equações do Filtro de Kalman
-----------------------------
Predição   : x̂_{k|k-1} = F · x̂_{k-1|k-1}
             P_{k|k-1}  = F · P_{k-1|k-1} · Fᵀ + Q

Atualização: K_k = P_{k|k-1} · Hᵀ · (H · P_{k|k-1} · Hᵀ + R)⁻¹
             x̂_k = x̂_{k|k-1} + K_k · (z_k − H · x̂_{k|k-1})
             P_k  = (I − K_k · H) · P_{k|k-1}

Bibliotecas
-----------
- numpy      : álgebra linear para as operações matriciais do filtro.
- matplotlib : geração dos gráficos de trajetória e de erro.
"""

import argparse
import statistics
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------

@dataclass
class KalmanState:
    """Representa o estado interno do filtro em um instante de tempo."""

    x_est: np.ndarray       # Vetor de estado estimado (4,): [px, py, vx, vy]
    P: np.ndarray           # Matriz de covariância do erro de estimativa (4 x 4)
    K: np.ndarray           # Ganho de Kalman calculado (4 x 2)
    innovation: np.ndarray  # Inovação z_k − H·x̂ (2,)


@dataclass
class KalmanResult:
    """Agrega todos os estados estimados ao longo da trajetória."""

    estimated_positions: np.ndarray   # (T, 2) – [pos_x, pos_y] estimados
    covariance_trace: List[float]     # traço de P a cada passo (indicador de incerteza)
    elapsed_ms: float

    def __str__(self) -> str:
        lines = [
            "[Filtro de Kalman 2D] Estimativa de trajetória",
            f"  Passos filtrados  : {len(self.estimated_positions)}",
            f"  Incerteza final   : {self.covariance_trace[-1]:.4f}  (tr(P))",
            f"  Tempo de execução : {self.elapsed_ms:.4f} ms",
        ]
        return "\n".join(lines)


@dataclass
class DroneFlightReport:
    """Relatório completo do experimento de navegação do drone."""

    num_steps: int
    true_trajectory: np.ndarray       # (T, 2)
    noisy_measurements: np.ndarray    # (T, 2)
    kalman_result: KalmanResult | None = None
    rmse_raw: float = 0.0
    rmse_kalman: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Passos de tempo      : {self.num_steps}",
            f"RMSE bruto (GPS)     : {self.rmse_raw:.4f} m",
            f"RMSE Kalman          : {self.rmse_kalman:.4f} m",
            f"Redução de erro      : {(1 - self.rmse_kalman / self.rmse_raw) * 100:.1f} %",
            "",
            str(self.kalman_result),
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Simulação da trajetória
# ---------------------------------------------------------------------------

class DroneTrajectorySimulator:
    """
    Simula a trajetória real de um drone em missão de patrulha.

    O drone mantém uma rota com velocidade base, sofrendo variações suaves
    causadas por vento (ruído de aceleração gaussiano de baixa variância).
    Esse tipo de trajetória é ideal para o modelo de velocidade constante
    do Filtro de Kalman, permitindo demonstrar claramente a redução de erro.
    """

    def __init__(self, dt: float = 0.5, seed: int = 42) -> None:
        self._dt  = dt
        self._rng = np.random.RandomState(seed)

    def generate(
        self,
        num_steps: int,
        gps_noise_std: float = 4.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera trajetória real e medições GPS ruidosas.

        Parâmetros
        ----------
        num_steps     : número de passos de tempo T.
        gps_noise_std : desvio padrão do ruído GPS em metros.

        Retorna
        -------
        Tupla (trajetória_real (T,2), medições_ruidosas (T,2)).
        """
        dt = self._dt
        T  = num_steps

        positions = np.zeros((T, 2))

        # Velocidade inicial de cruzeiro (m/passo)
        vel = np.array([2.0, 0.5])

        for t in range(1, T):
            # Perturbação de vento: aceleração suave e aleatória
            vel = vel + self._rng.normal(scale=0.05, size=2)
            # Atualiza posição: cinemática + ruído de processo residual
            positions[t] = positions[t - 1] + vel * dt + self._rng.normal(scale=0.15, size=2)

        # Medições GPS: posição real + ruído gaussiano independente
        noise      = self._rng.normal(scale=gps_noise_std, size=(T, 2))
        noisy_meas = positions + noise

        return positions, noisy_meas


# ---------------------------------------------------------------------------
# Filtro de Kalman 2D
# ---------------------------------------------------------------------------

class KalmanFilter2D:
    """
    Filtro de Kalman linear para rastreamento de posição 2D com modelo de velocidade constante.

    Estado: x = [pos_x, pos_y, vel_x, vel_y]ᵀ
    Observação: z = [pos_x, pos_y]ᵀ  (GPS mede apenas posição, não velocidade)

    A escolha do modelo de velocidade constante é adequada para o cenário do drone,
    pois o veículo mantém velocidade de cruzeiro com variações lentas, que são
    capturadas pelo ruído de processo Q.
    """

    def __init__(
        self,
        dt: float,
        process_noise_std: float,
        measurement_noise_std: float,
    ) -> None:
        """
        Parâmetros
        ----------
        dt                    : intervalo de tempo entre medições (s).
        process_noise_std     : desvio padrão do ruído de processo (variações de velocidade).
        measurement_noise_std : desvio padrão do ruído do GPS em metros.
        """
        n_state = 4  # dimensão do vetor de estado
        n_obs   = 2  # dimensão do vetor de observação

        # Matriz de transição F: modelo cinemático de velocidade constante
        # pos_k = pos_{k-1} + vel_{k-1} * dt
        # vel_k = vel_{k-1}
        self.F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=float)

        # Matriz de observação H: extrai apenas as componentes de posição do estado
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # Covariância do ruído de processo Q
        # Modela a incerteza na aceleração do drone (vento, rajadas). As acelerações
        # em X e Y são independentes, então Q é bloco-diagonal por eixo: para cada
        # eixo, o modelo discreto de aceleração com G = [dt²/2, dt]ᵀ dá
        # Q_eixo = G·Gᵀ·σ²_a, sem covariância cruzada entre X e Y.
        g = np.array([[dt**2 / 2], [dt]])
        q_axis = (g @ g.T) * (process_noise_std ** 2)
        self.Q = np.zeros((n_state, n_state))
        for axis in (0, 1):
            idx = [axis, axis + 2]               # (posição, velocidade) do eixo
            self.Q[np.ix_(idx, idx)] = q_axis

        # Covariância do ruído de medição R (GPS isotrópico)
        self.R = np.eye(n_obs) * (measurement_noise_std ** 2)

        # Matriz identidade para a forma Joseph da atualização de P
        self._I = np.eye(n_state)

        # Estado e covariância iniciais (serão sobrescritos em initialize())
        self._x: np.ndarray = np.zeros(n_state)
        self._P: np.ndarray = np.eye(n_state) * 100.0

    def initialize(self, first_measurement: np.ndarray) -> None:
        """Inicializa o filtro com a primeira leitura GPS (velocidade inicial = 0)."""
        self._x = np.array([
            first_measurement[0],
            first_measurement[1],
            0.0,
            0.0,
        ])
        self._P = np.eye(4) * 50.0  # incerteza inicial moderada

    def step(self, z: np.ndarray) -> KalmanState:
        """
        Executa um ciclo completo predição → atualização do filtro.

        Parâmetros
        ----------
        z : medição GPS no instante k (shape (2,): [pos_x, pos_y]).

        Retorna
        -------
        KalmanState com o estado estimado atualizado e métricas intermediárias.
        """
        # --- Etapa de Predição (a priori) ---
        x_pred = self.F @ self._x
        P_pred = self.F @ self._P @ self.F.T + self.Q

        # --- Inovação: diferença entre medição e previsão ---
        innovation = z - self.H @ x_pred

        # --- Ganho de Kalman: pondera modelo vs. sensor ---
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # --- Etapa de Atualização (a posteriori) ---
        self._x = x_pred + K @ innovation
        # Forma de Joseph: mais numericamente estável que P = (I-KH)P
        self._P = (self._I - K @ self.H) @ P_pred

        return KalmanState(
            x_est=self._x.copy(),
            P=self._P.copy(),
            K=K,
            innovation=innovation,
        )


# ---------------------------------------------------------------------------
# Execução do filtro sobre a trajetória
# ---------------------------------------------------------------------------

def run_kalman_filter(
    measurements: np.ndarray,
    dt: float = 0.5,
    process_noise_std: float = 0.3,
    measurement_noise_std: float = 4.0,
) -> KalmanResult:
    """
    Aplica o Filtro de Kalman 2D sobre toda a sequência de medições GPS.

    Escolha dos hiperparâmetros
    ---------------------------
    - process_noise_std = 0.3 : confiamos mais no modelo cinemático do que nas
      perturbações de vento, que são lentas. Valor baixo faz o filtro pesar mais
      as predições do modelo.
    - measurement_noise_std   : deve refletir o ruído real do GPS (mesmo valor
      usado na simulação para calibração justa).

    Parâmetros
    ----------
    measurements          : medições GPS (T, 2).
    dt                    : passo de tempo em segundos.
    process_noise_std     : desvio padrão do ruído de processo (aceleração).
    measurement_noise_std : desvio padrão do GPS (igual ao da simulação).

    Retorna
    -------
    KalmanResult com posições estimadas e histórico de covariância.
    """
    T  = len(measurements)
    kf = KalmanFilter2D(
        dt=dt,
        process_noise_std=process_noise_std,
        measurement_noise_std=measurement_noise_std,
    )
    kf.initialize(measurements[0])

    estimated  = np.zeros((T, 2))
    cov_trace  = []

    t0 = time.perf_counter()

    for t in range(T):
        state = kf.step(measurements[t])
        estimated[t, 0] = state.x_est[0]   # pos_x estimado
        estimated[t, 1] = state.x_est[1]   # pos_y estimado
        cov_trace.append(float(np.trace(state.P)))

    elapsed = (time.perf_counter() - t0) * 1000.0

    return KalmanResult(
        estimated_positions=estimated,
        covariance_trace=cov_trace,
        elapsed_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------

class DroneVisualizer:
    """Gera as figuras do experimento do drone."""

    @staticmethod
    def plot_trajectory(
        true_traj: np.ndarray,
        noisy_meas: np.ndarray,
        estimated: np.ndarray,
        output_path: str = "figura_05_trajetoria_drone.png",
    ) -> None:
        """Plota a trajetória real, medições GPS e estimativa do Kalman no plano XY."""

        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor("#F8F8F8")
        ax.set_facecolor("#F4F4F4")

        ax.scatter(
            noisy_meas[:, 0], noisy_meas[:, 1],
            s=6, alpha=0.30, color="#CC4444", label="Medições GPS (ruidosas)",
        )
        ax.plot(
            true_traj[:, 0], true_traj[:, 1],
            color="#1A1A2E", lw=2.0, label="Trajetória real",
        )
        ax.plot(
            estimated[:, 0], estimated[:, 1],
            color="#0078D4", lw=1.8, linestyle="--", label="Estimativa Kalman",
        )
        ax.plot(*true_traj[0],  "go", markersize=9, label="Início")
        ax.plot(*true_traj[-1], "rs", markersize=9, label="Fim")

        ax.set_xlabel("Posição X (m)", fontsize=10)
        ax.set_ylabel("Posição Y (m)", fontsize=10)
        ax.set_title("Estimativa de Trajetória do Drone com Filtro de Kalman 2D", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Figura salva] {output_path}")

    @staticmethod
    def plot_error_and_covariance(
        true_traj: np.ndarray,
        noisy_meas: np.ndarray,
        estimated: np.ndarray,
        cov_trace: List[float],
        output_path: str = "figura_06_erro_covariancia.png",
    ) -> None:
        """Plota o erro euclidiano ao longo do tempo e a convergência de covariância."""

        T    = len(true_traj)
        t_ax = np.arange(T)

        err_gps    = np.linalg.norm(noisy_meas - true_traj, axis=1)
        err_kalman = np.linalg.norm(estimated   - true_traj, axis=1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        fig.patch.set_facecolor("#F8F8F8")

        ax1.plot(t_ax, err_gps,    color="#CC4444", lw=1.0, alpha=0.7, label="Erro GPS bruto")
        ax1.plot(t_ax, err_kalman, color="#0078D4", lw=1.8, label="Erro Kalman")
        ax1.set_ylabel("Erro de posição (m)", fontsize=9)
        ax1.set_title("Erro de Posição ao Longo da Missão", fontsize=11, fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax1.set_axisbelow(True)
        ax1.set_facecolor("#F8F8F8")

        ax2.plot(t_ax, cov_trace, color="#7030A0", lw=1.5, label="tr(P) – incerteza do filtro")
        ax2.set_xlabel("Passo de tempo", fontsize=9)
        ax2.set_ylabel("tr(P)", fontsize=9)
        ax2.set_title("Convergência da Covariância do Filtro de Kalman", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax2.set_axisbelow(True)
        ax2.set_facecolor("#F8F8F8")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Figura salva] {output_path}")


# ---------------------------------------------------------------------------
# Experimento principal
# ---------------------------------------------------------------------------

def run_experiment() -> DroneFlightReport:
    """Executa a simulação completa e retorna o relatório do voo."""

    DT            = 0.5    # segundos por passo
    NUM_STEPS     = 200    # duração total da missão
    GPS_NOISE_STD = 4.0    # desvio padrão do ruído GPS (metros)
    PROC_NOISE    = 0.3    # desvio padrão do ruído de processo (aceleração)

    print("=" * 60)
    print("Projeto 3 – Filtro de Kalman: Posição de Drone")
    print("=" * 60)

    simulator = DroneTrajectorySimulator(dt=DT, seed=42)
    true_traj, noisy_meas = simulator.generate(
        num_steps=NUM_STEPS,
        gps_noise_std=GPS_NOISE_STD,
    )

    report = DroneFlightReport(
        num_steps=NUM_STEPS,
        true_trajectory=true_traj,
        noisy_measurements=noisy_meas,
    )

    kalman_result = run_kalman_filter(
        measurements=noisy_meas,
        dt=DT,
        process_noise_std=PROC_NOISE,
        measurement_noise_std=GPS_NOISE_STD,
    )
    report.kalman_result = kalman_result
    print(kalman_result)
    print()

    rmse_raw    = float(np.sqrt(np.mean(np.sum((noisy_meas - true_traj) ** 2, axis=1))))
    rmse_kalman = float(np.sqrt(np.mean(np.sum((kalman_result.estimated_positions - true_traj) ** 2, axis=1))))
    report.rmse_raw    = rmse_raw
    report.rmse_kalman = rmse_kalman

    print(f"RMSE GPS bruto : {rmse_raw:.4f} m")
    print(f"RMSE Kalman    : {rmse_kalman:.4f} m")
    print(f"Redução de erro: {(1 - rmse_kalman / rmse_raw) * 100:.1f} %")

    return report


# ---------------------------------------------------------------------------
# Geração das figuras
# ---------------------------------------------------------------------------

def generate_figures(report: DroneFlightReport) -> None:
    """Gera e salva todas as figuras do projeto."""

    DroneVisualizer.plot_trajectory(
        true_traj=report.true_trajectory,
        noisy_meas=report.noisy_measurements,
        estimated=report.kalman_result.estimated_positions,
        output_path="figura_05_trajetoria_drone.png",
    )
    DroneVisualizer.plot_error_and_covariance(
        true_traj=report.true_trajectory,
        noisy_meas=report.noisy_measurements,
        estimated=report.kalman_result.estimated_positions,
        cov_trace=report.kalman_result.covariance_trace,
        output_path="figura_06_erro_covariancia.png",
    )


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def evaluate_seeds(n_seeds: int, first_seed: int = 42) -> None:
    """Repete o experimento com várias sementes (trajetória e ruído diferentes)."""
    raw, kal = [], []
    for seed in range(first_seed, first_seed + n_seeds):
        true_traj, noisy = DroneTrajectorySimulator(dt=0.5, seed=seed).generate(
            num_steps=200, gps_noise_std=4.0)
        est = run_kalman_filter(noisy, dt=0.5, process_noise_std=0.3,
                                measurement_noise_std=4.0).estimated_positions
        raw.append(float(np.sqrt(np.mean(np.sum((noisy - true_traj) ** 2, axis=1)))))
        kal.append(float(np.sqrt(np.mean(np.sum((est - true_traj) ** 2, axis=1)))))
    reduction = [(1 - k / r) * 100 for r, k in zip(raw, kal)]
    sd = statistics.stdev if n_seeds > 1 else (lambda _: 0.0)
    print(f"  {n_seeds} execuções (sementes {first_seed}..{first_seed + n_seeds - 1})")
    print(f"  RMSE GPS bruto: {statistics.mean(raw):.3f} ± {sd(raw):.3f} m")
    print(f"  RMSE Kalman   : {statistics.mean(kal):.3f} ± {sd(kal):.3f} m "
          f"(min {min(kal):.3f}, max {max(kal):.3f})")
    print(f"  Redução       : {statistics.mean(reduction):.1f}% ± {sd(reduction):.1f} "
          f"(min {min(reduction):.1f}%, max {max(reduction):.1f}%) | "
          f"Kalman pior que o GPS bruto em {sum(k > r for r, k in zip(raw, kal))} de {n_seeds}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1,
                        help="com N > 1, apenas avalia N execuções e imprime média ± desvio")
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
