"""
Projeto 3: Aprendizado por Reforco - Navegacao Autonoma em Grid World
via Q-Learning.

Disciplina: FGA0221 - Inteligencia Artificial
Portfolio 4 - Aprendizado de Maquina
Autor: Thiago Viriato Accioly

O problema simula um robo de armazem que precisa aprender, por tentativa
e erro, a rota mais curta entre um ponto de partida e uma estacao de
entrega, dentro de um galpao representado como uma grade 8x8 com
obstaculos fixos (prateleiras) e uma zona de risco (area de trafego de
empilhadeiras, que penaliza mas nao bloqueia a passagem). O agente nao
recebe o mapa do ambiente antecipadamente: ele so descobre a estrutura do
galpao por meio das recompensas obtidas a cada movimento, o que
caracteriza um problema de aprendizado por reforco model-free.
"""

from __future__ import annotations

import argparse
import heapq
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from enum import IntEnum

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Pasta onde as figuras geradas por este script sao salvas.
FIGS_DIR = Path("imagens")
FIGS_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


ACTION_DELTAS: dict[Action, tuple[int, int]] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


@dataclass(frozen=True)
class GridWorldConfig:
    """DTO com a configuracao estatica do ambiente (galpao)."""

    n_rows: int = 8
    n_cols: int = 8
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (7, 7)
    obstacles: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    hazard_zone: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    step_penalty: float = -1.0
    hazard_penalty: float = -8.0
    goal_reward: float = 50.0
    max_steps: int = 200


class WarehouseGridWorld:
    """
    Ambiente de grade que expoe uma interface reset()/step() no estilo
    Gymnasium, encapsulando a dinamica de transicao e a funcao de
    recompensa do galpao simulado.
    """

    def __init__(self, config: GridWorldConfig) -> None:
        self._config = config
        self._agent_pos = config.start
        self._steps_taken = 0

    def reset(self) -> tuple[int, int]:
        self._agent_pos = self._config.start
        self._steps_taken = 0
        return self._agent_pos

    def state_index(self, pos: tuple[int, int]) -> int:
        return pos[0] * self._config.n_cols + pos[1]

    @property
    def n_states(self) -> int:
        return self._config.n_rows * self._config.n_cols

    @property
    def n_actions(self) -> int:
        return len(Action)

    def step(self, action: Action) -> tuple[tuple[int, int], float, bool]:
        d_row, d_col = ACTION_DELTAS[action]
        new_row = self._agent_pos[0] + d_row
        new_col = self._agent_pos[1] + d_col
        candidate = (new_row, new_col)

        # Colisao com parede do galpao ou com prateleira: agente nao se move.
        out_of_bounds = not (0 <= new_row < self._config.n_rows and 0 <= new_col < self._config.n_cols)
        blocked = candidate in self._config.obstacles
        if out_of_bounds or blocked:
            candidate = self._agent_pos

        self._agent_pos = candidate
        self._steps_taken += 1

        done = self._agent_pos == self._config.goal
        reward = self._config.step_penalty
        if self._agent_pos in self._config.hazard_zone:
            reward += self._config.hazard_penalty
        if done:
            reward = self._config.goal_reward

        truncated = self._steps_taken >= self._config.max_steps
        return self._agent_pos, reward, (done or truncated)


@dataclass
class EpisodeResult:
    """DTO com o resumo de um episodio de treinamento."""

    episode: int
    total_reward: float
    steps: int
    epsilon: float


@dataclass
class TrainingReport:
    """DTO agregando os resultados de todos os episodios de treinamento."""

    episodes: list[EpisodeResult]
    q_table: np.ndarray

    @property
    def rewards(self) -> np.ndarray:
        return np.array([e.total_reward for e in self.episodes])

    @property
    def steps_per_episode(self) -> np.ndarray:
        return np.array([e.steps for e in self.episodes])


class QLearningAgent:
    """
    Agente tabular Q-Learning (off-policy, TD-control) com politica de
    exploracao epsilon-greedy decrescente.

    A atualizacao segue a equacao classica de Q-Learning:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

    Por ser off-policy, o agente aprende a politica otima (o max sobre
    Q(s', a')) mesmo enquanto explora com acoes subotimas, o que o
    diferencia do SARSA (on-policy), que aprenderia o valor da politica
    efetivamente seguida durante a exploracao.
    """

    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.15,
                 gamma: float = 0.95, epsilon_start: float = 1.0,
                 epsilon_min: float = 0.05, epsilon_decay: float = 0.995,
                 random_state: int = RANDOM_STATE) -> None:
        self._q_table = np.zeros((n_states, n_actions))
        self._n_actions = n_actions
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon_start
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._rng = np.random.default_rng(random_state)

    def select_action(self, state: int) -> Action:
        if self._rng.random() < self._epsilon:
            return Action(self._rng.integers(0, self._n_actions))
        return Action(int(np.argmax(self._q_table[state])))

    def update(self, state: int, action: Action, reward: float, next_state: int, done: bool) -> None:
        best_next = 0.0 if done else np.max(self._q_table[next_state])
        td_target = reward + self._gamma * best_next
        td_error = td_target - self._q_table[state, action]
        self._q_table[state, action] += self._alpha * td_error

    def decay_epsilon(self) -> None:
        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def q_table(self) -> np.ndarray:
        return self._q_table

    def greedy_policy(self) -> np.ndarray:
        return np.argmax(self._q_table, axis=1)


def train(env: WarehouseGridWorld, agent: QLearningAgent, n_episodes: int = 500) -> TrainingReport:
    episodes: list[EpisodeResult] = []
    for ep in range(n_episodes):
        pos = env.reset()
        state = env.state_index(pos)
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_pos, reward, done = env.step(action)
            next_state = env.state_index(next_pos)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
        agent.decay_epsilon()
        episodes.append(EpisodeResult(episode=ep, total_reward=total_reward, steps=steps, epsilon=agent.epsilon))
    return TrainingReport(episodes=episodes, q_table=agent.q_table.copy())


def extract_greedy_path(env: WarehouseGridWorld, agent: QLearningAgent, config: GridWorldConfig,
                         max_steps: int = 100) -> list[tuple[int, int]]:
    pos = env.reset()
    path = [pos]
    for _ in range(max_steps):
        state = env.state_index(pos)
        action = Action(int(np.argmax(agent.q_table[state])))
        pos, _, done = env.step(action)
        path.append(pos)
        if done:
            break
    return path


def plot_learning_curve(report: TrainingReport, filename: str, window: int = 20) -> None:
    rewards = report.rewards
    steps = report.steps_per_episode
    smoothed_rewards = np.convolve(rewards, np.ones(window) / window, mode="valid")
    smoothed_steps = np.convolve(steps, np.ones(window) / window, mode="valid")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4))
    ax1.plot(rewards, alpha=0.25, color="#7B2D26")
    ax1.plot(range(window - 1, len(rewards)), smoothed_rewards, color="#7B2D26", linewidth=2,
              label=f"Media movel ({window} ep.)")
    ax1.set_xlabel("Episodio")
    ax1.set_ylabel("Recompensa Total")
    ax1.set_title("Curva de Aprendizado - Recompensa")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(steps, alpha=0.25, color="#B08968")
    ax2.plot(range(window - 1, len(steps)), smoothed_steps, color="#B08968", linewidth=2,
              label=f"Media movel ({window} ep.)")
    ax2.set_xlabel("Episodio")
    ax2.set_ylabel("Passos ate o Objetivo")
    ax2.set_title("Curva de Aprendizado - Eficiencia")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGS_DIR / filename, dpi=150)
    plt.close(fig)


def plot_policy_and_path(config: GridWorldConfig, path: list[tuple[int, int]], filename: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, config.n_cols)
    ax.set_ylim(0, config.n_rows)
    ax.invert_yaxis()
    ax.set_xticks(range(config.n_cols + 1))
    ax.set_yticks(range(config.n_rows + 1))
    ax.grid(True, alpha=0.4)

    for (r, c) in config.hazard_zone:
        ax.add_patch(Rectangle((c, r), 1, 1, color="#F4C542", alpha=0.6))
    for (r, c) in config.obstacles:
        ax.add_patch(Rectangle((c, r), 1, 1, color="#3D3D3D"))

    start_r, start_c = config.start
    goal_r, goal_c = config.goal
    ax.add_patch(Rectangle((start_c, start_r), 1, 1, color="#4C8C4A", alpha=0.85))
    ax.add_patch(Rectangle((goal_c, goal_r), 1, 1, color="#7B2D26", alpha=0.85))

    path_rows = [p[0] + 0.5 for p in path]
    path_cols = [p[1] + 0.5 for p in path]
    ax.plot(path_cols, path_rows, color="#1E3A5F", linewidth=2.5, marker="o", markersize=4)

    ax.set_title("Trajetoria Aprendida (politica gulosa apos treinamento)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / filename, dpi=150)
    plt.close(fig)


def optimal_reference(config: GridWorldConfig) -> tuple[int, float]:
    """Referencia exata do ambiente: (menor numero de passos, melhor retorno).

    Dijkstra a partir da origem com custo 1 por passo mais |hazard_penalty| ao
    entrar numa celula de risco. O retorno otimo e a recompensa do objetivo
    menos o custo dos passos anteriores a ele.
    """
    best: dict[tuple[int, int], float] = {config.start: 0.0}
    steps: dict[tuple[int, int], int] = {config.start: 0}
    heap = [(0.0, 0, config.start)]
    while heap:
        cost, n_steps, pos = heapq.heappop(heap)
        if cost > best.get(pos, float("inf")):
            continue
        if pos == config.goal:
            break
        for d_row, d_col in ACTION_DELTAS.values():
            nxt = (pos[0] + d_row, pos[1] + d_col)
            if not (0 <= nxt[0] < config.n_rows and 0 <= nxt[1] < config.n_cols) or nxt in config.obstacles:
                continue
            step_cost = -config.step_penalty + (-config.hazard_penalty if nxt in config.hazard_zone else 0.0)
            if nxt == config.goal:
                step_cost = -config.goal_reward
            if cost + step_cost < best.get(nxt, float("inf")):
                best[nxt] = cost + step_cost
                steps[nxt] = n_steps + 1
                heapq.heappush(heap, (cost + step_cost, n_steps + 1, nxt))
    return steps[config.goal], -best[config.goal]


def path_return(config: GridWorldConfig, path: list[tuple[int, int]]) -> float:
    """Retorno nao descontado de um caminho, com as mesmas regras do ambiente."""
    total = 0.0
    for pos in path[1:]:
        if pos == config.goal:
            total += config.goal_reward
        else:
            total += config.step_penalty + (config.hazard_penalty if pos in config.hazard_zone else 0.0)
    return total


def evaluate_seeds(config: GridWorldConfig, n_seeds: int, first_seed: int = RANDOM_STATE) -> None:
    """Treina um agente por semente e compara a politica gulosa final com o otimo."""
    opt_steps, opt_return = optimal_reference(config)
    rows = []
    for seed in range(first_seed, first_seed + n_seeds):
        env = WarehouseGridWorld(config)
        agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions, random_state=seed)
        report = train(env, agent, n_episodes=500)
        path = extract_greedy_path(env, agent, config)
        reached = path[-1] == config.goal
        rows.append({
            "reached": reached,
            "steps": len(path) - 1,
            "return": path_return(config, path) if reached else float("nan"),
            "first20": float(report.rewards[:20].mean()),
            "last20": float(report.rewards[-20:].mean()),
        })

    def fmt(key: str) -> str:
        values = [r[key] for r in rows if r["reached"] or key in ("first20", "last20")]
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return f"{statistics.mean(values):.2f} +- {std:.2f}"

    print(f"=== {n_seeds} sementes (a partir de {first_seed}) | referencia exata: "
          f"{opt_steps} passos, retorno {opt_return:.0f} ===")
    print(f"Politica gulosa chega ao objetivo: {sum(r['reached'] for r in rows)}/{n_seeds}")
    print(f"Politica gulosa com o numero minimo de passos: "
          f"{sum(r['reached'] and r['steps'] == opt_steps for r in rows)}/{n_seeds}")
    print(f"Politica gulosa com o retorno otimo: "
          f"{sum(r['reached'] and r['return'] == opt_return for r in rows)}/{n_seeds}")
    print(f"Passos da politica gulosa: {fmt('steps')}")
    print(f"Retorno da politica gulosa: {fmt('return')}")
    print(f"Recompensa media nos 20 primeiros episodios: {fmt('first20')}")
    print(f"Recompensa media nos 20 ultimos episodios (com exploracao): {fmt('last20')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1,
                        help="com N > 1, apenas treina N agentes (sementes consecutivas) e imprime media +- desvio")
    args = parser.parse_args()

    obstacles = frozenset({
        (1, 2), (2, 2), (3, 2), (4, 2),
        (2, 5), (3, 5), (4, 5), (5, 5),
        (6, 1), (6, 2), (6, 3),
    })
    hazard_zone = frozenset({(1, 6), (2, 6), (3, 6), (1, 7), (2, 7)})

    config = GridWorldConfig(obstacles=obstacles, hazard_zone=hazard_zone)
    if args.seeds > 1:
        evaluate_seeds(config, args.seeds)
        return

    env = WarehouseGridWorld(config)
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)

    report = train(env, agent, n_episodes=500)

    first_20 = report.rewards[:20].mean()
    last_20 = report.rewards[-20:].mean()
    first_steps = report.steps_per_episode[:20].mean()
    last_steps = report.steps_per_episode[-20:].mean()

    print("=== Treinamento Q-Learning: Navegacao em Grid World ===")
    print(f"Recompensa media (20 primeiros episodios): {first_20:.2f}")
    print(f"Recompensa media (20 ultimos episodios):    {last_20:.2f}")
    print(f"Passos medios (20 primeiros episodios): {first_steps:.1f}")
    print(f"Passos medios (20 ultimos episodios):    {last_steps:.1f}")

    optimal_manhattan = abs(config.goal[0] - config.start[0]) + abs(config.goal[1] - config.start[1])
    print(f"\nDistancia de Manhattan minima teorica (sem obstaculos): {optimal_manhattan} passos")

    path = extract_greedy_path(env, agent, config)
    print(f"Caminho encontrado pela politica gulosa final: {len(path) - 1} passos")
    print(f"Chegou ao objetivo: {path[-1] == config.goal}")

    plot_learning_curve(report, "p3_learning_curve.png")
    plot_policy_and_path(config, path, "p3_policy_path.png")

    print("\nFiguras salvas em:", FIGS_DIR)


if __name__ == "__main__":
    main()
