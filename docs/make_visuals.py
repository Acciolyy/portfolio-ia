"""Generate the README animations from the project code and real data.

Every animation replays an algorithm from ``src/`` with a fixed seed; nothing
is drawn from made-up numbers. Each one is written in a light and a dark
version (solid backgrounds, since GIF transparency is 1-bit) to docs/assets/.

Usage (from the repository root):  python docs/make_visuals.py [name ...]
Names: wine tsp kalman qlearning astar
"""
from __future__ import annotations

import heapq
import importlib
import os
import sys
import tempfile
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.datasets import load_wine  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets"
for folder in ("portfolio-2-busca-otimizacao", "portfolio-3-modelos-probabilisticos", "portfolio-4-machine-learning"):
    sys.path.insert(0, str(ROOT / folder / "src"))

# The ML scripts create ``imagens/`` at import time, relative to the current directory.
_cwd = os.getcwd()
os.chdir(tempfile.mkdtemp())
try:
    rl = importlib.import_module("projeto_03_reforco")
finally:
    os.chdir(_cwd)
import informed_search as inf  # noqa: E402
import projeto_03_kalman as kalman  # noqa: E402
import tsp_local_search as tsp  # noqa: E402

MAX_BYTES = 3 * 1024 * 1024
DPI = 90
SIZE = (6.4, 4.2)
THEMES = {
    "light": dict(bg="#ffffff", fg="#1f2328", muted="#656d76", grid="#d0d7de",
                  series=["#0072B2", "#E69F00", "#009E73"], accent="#D55E00", fill="#eaeef2"),
    "dark": dict(bg="#0d1117", fg="#e6edf3", muted="#8b949e", grid="#30363d",
                 series=["#56B4E9", "#E69F00", "#3DDC97"], accent="#FF7B54", fill="#21262d"),
}


# ----------------------------------------------------------------- helpers
def new_fig(theme: dict, size=SIZE, **kw):
    fig = plt.figure(figsize=size, dpi=DPI, facecolor=theme["bg"], **kw)
    return fig


def style(ax, theme: dict, grid: bool = True):
    ax.set_facecolor(theme["bg"])
    for spine in ax.spines.values():
        spine.set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], labelsize=7)
    ax.xaxis.label.set_color(theme["fg"])
    ax.yaxis.label.set_color(theme["fg"])
    ax.title.set_color(theme["fg"])
    if grid:
        ax.grid(True, color=theme["grid"], linewidth=0.5, alpha=0.8)
        ax.set_axisbelow(True)


def grab(fig) -> Image.Image:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(rgba[..., :3].copy())


def save_gif(frames: list[Image.Image], path: Path, duration_ms: int, hold_ms: int = 1500) -> None:
    """Write frames with one shared 256-colour palette (smaller file, no flicker).

    The palette is built from all frames stacked, so colours that only appear
    later in the animation (for example the nodes a search expands) are kept.
    """
    width, height = frames[0].size
    stack = Image.new("RGB", (width, height * len(frames)))
    for i, frame in enumerate(frames):
        stack.paste(frame, (0, i * height))
    base = stack.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    paletted = [f.quantize(palette=base, dither=Image.Dither.NONE) for f in frames]
    durations = [duration_ms] * len(paletted)
    durations[-1] = hold_ms
    paletted[0].save(path, save_all=True, append_images=paletted[1:], duration=durations,
                     loop=0, optimize=True, disposal=1)
    size = path.stat().st_size
    assert size <= MAX_BYTES, f"{path.name} has {size / 1e6:.2f} MB (limit 3 MB)"
    print(f"  {path.relative_to(ROOT)}  {size / 1e6:.2f} MB  {len(frames)} frames")


# --------------------------------------------------- 1. Wine PCA 3D rotating
def wine_frames(theme: dict) -> list[Image.Image]:
    data = load_wine()
    x = StandardScaler().fit_transform(data.data)
    pca = PCA(n_components=3, random_state=42).fit(x)
    z = pca.transform(x)
    var = pca.explained_variance_ratio_ * 100
    frames = []
    for azim in np.arange(0, 360, 6):
        fig = new_fig(theme)
        ax = fig.add_axes([0.0, 0.0, 1.0, 0.9], projection="3d")
        ax.set_facecolor(theme["bg"])
        for i, marker in enumerate(("o", "^", "s")):
            pts = z[data.target == i]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=16, color=theme["series"][i], marker=marker,
                       edgecolor=theme["bg"], linewidth=0.3, label=f"Cultivar {i} (n={len(pts)})", depthshade=True)
        ax.view_init(elev=18 + 6 * np.sin(np.radians(azim)), azim=azim)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0, 0, 0, 0))
            axis._axinfo["grid"]["color"] = theme["grid"]
            axis.line.set_color(theme["grid"])
        ax.tick_params(colors=theme["muted"], labelsize=6)
        ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", color=theme["fg"], fontsize=8)
        ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", color=theme["fg"], fontsize=8)
        ax.set_zlabel(f"PC3 ({var[2]:.1f}%)", color=theme["fg"], fontsize=8)
        leg = ax.legend(loc="upper left", fontsize=7, frameon=False, bbox_to_anchor=(0.0, 0.98))
        for t in leg.get_texts():
            t.set_color(theme["fg"])
        fig.text(0.5, 0.955, f"Wine: 178 amostras, 13 atributos → 3 componentes ({var.sum():.1f}% da variância)",
                 ha="center", color=theme["fg"], fontsize=9)
        frames.append(grab(fig))
    return frames


# ------------------------------------------------------- 2. TSP, 40 cities
def tsp_frames(theme: dict) -> list[Image.Image]:
    cities = tsp.make_random_cities(40, instance_seed=2026)
    budget, seed = 50_000, 42
    checkpoints = sorted(set(np.round(np.geomspace(100, budget, 34)).astype(int).tolist()))
    makers = {
        "HC com reinícios": lambda: tsp.HillClimbingRestarts(cities, seed=seed),
        "SA, resfriamento 0,995": lambda: tsp.SimulatedAnnealing(cities, 800.0, 0.995, seed=seed),
        "SA, resfriamento 0,9998": lambda: tsp.SimulatedAnnealing(cities, 800.0, 0.9998, seed=seed),
    }
    # Re-running optimize(k) with the same seed replays the first k evaluations
    # (the RNG draws and the cooling schedule depend only on the iteration index).
    runs = {name: [make().optimize(k) for k in checkpoints] for name, make in makers.items()}
    curves = {n: np.array([r.best_distance_km for r in rs]) for n, rs in runs.items()}
    for n, c in curves.items():
        assert np.all(np.diff(c) <= 1e-9), f"{n}: best-so-far should not increase"
    lo, hi = min(c.min() for c in curves.values()), max(c.max() for c in curves.values())
    xs, ys = [c.x for c in cities], [c.y for c in cities]

    frames = []
    for i, k in enumerate(checkpoints):
        fig = new_fig(theme)
        ax1 = fig.add_axes([0.02, 0.05, 0.42, 0.78])
        ax2 = fig.add_axes([0.54, 0.16, 0.43, 0.67])
        route = runs["SA, resfriamento 0,9998"][i].best_route
        rx = [c.x for c in route] + [route[0].x]
        ry = [c.y for c in route] + [route[0].y]
        ax1.plot(rx, ry, color=theme["series"][2], linewidth=1.2)
        ax1.scatter(xs, ys, s=14, color=theme["fg"], zorder=3)
        ax1.set_aspect("equal")
        ax1.axis("off")
        ax1.set_title(f"Melhor tour do SA (0,9998): {curves['SA, resfriamento 0,9998'][i]:.1f} km",
                      fontsize=8, color=theme["fg"])
        style(ax2, theme)
        for (name, c), color, ls in zip(curves.items(), theme["series"], ("-", "--", "-")):
            ax2.plot(checkpoints[: i + 1], c[: i + 1], color=color, linewidth=1.8, linestyle=ls, label=name)
        ax2.set_xscale("log")
        ax2.set_xlim(checkpoints[0], budget)
        ax2.set_ylim(lo * 0.98, hi * 1.02)
        ax2.set_xlabel("avaliações da função de custo", fontsize=8)
        ax2.set_ylabel("melhor distância (km)", fontsize=8)
        leg = ax2.legend(fontsize=6.5, frameon=False, loc="upper right")
        for t in leg.get_texts():
            t.set_color(theme["fg"])
        fig.text(0.5, 0.94, "TSP, 40 cidades aleatórias, mesmo orçamento de avaliações (semente 42)",
                 ha="center", color=theme["fg"], fontsize=9)
        fig.text(0.5, 0.02, f"{k:,} avaliações".replace(",", "."), ha="center", color=theme["muted"], fontsize=8)
        frames.append(grab(fig))
    return frames


# --------------------------------------------------------- 3. Kalman tracking
def kalman_frames(theme: dict) -> list[Image.Image]:
    true_traj, noisy = kalman.DroneTrajectorySimulator(dt=0.5, seed=42).generate(200, gps_noise_std=4.0)
    est = kalman.run_kalman_filter(noisy, 0.5, 0.3, 4.0).estimated_positions   # causal: est[t] uses z[:t+1]
    err_raw = np.sum((noisy - true_traj) ** 2, axis=1)
    err_kal = np.sum((est - true_traj) ** 2, axis=1)
    pad = 6
    xlim = (noisy[:, 0].min() - pad, noisy[:, 0].max() + pad)
    ylim = (noisy[:, 1].min() - pad, noisy[:, 1].max() + pad)
    frames = []
    for t in list(range(6, 200, 5)) + [199]:
        fig = new_fig(theme)
        ax = fig.add_axes([0.08, 0.12, 0.89, 0.72])
        style(ax, theme)
        ax.scatter(noisy[: t + 1, 0], noisy[: t + 1, 1], s=7, color=theme["accent"], alpha=0.5, label="GPS ruidoso")
        ax.plot(true_traj[: t + 1, 0], true_traj[: t + 1, 1], color=theme["fg"], linewidth=1.8, label="trajetória real")
        ax.plot(est[: t + 1, 0], est[: t + 1, 1], color=theme["series"][0], linewidth=1.8, linestyle="--",
                label="estimativa do Kalman")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("x (m)", fontsize=8)
        ax.set_ylabel("y (m)", fontsize=8)
        leg = ax.legend(fontsize=7, frameon=False, loc="upper left")
        for tx in leg.get_texts():
            tx.set_color(theme["fg"])
        rmse_raw, rmse_kal = np.sqrt(err_raw[: t + 1].mean()), np.sqrt(err_kal[: t + 1].mean())
        fig.text(0.97, 0.15, f"RMSE até aqui\nGPS {rmse_raw:.2f} m   Kalman {rmse_kal:.2f} m",
                 ha="right", va="bottom", color=theme["fg"], fontsize=8)
        fig.text(0.5, 0.93, "Filtro de Kalman 2D rastreando um drone (semente 42)",
                 ha="center", color=theme["fg"], fontsize=9)
        frames.append(grab(fig))
    return frames


# ----------------------------------------------------------- 4. Q-learning
def qlearning_frames(theme: dict) -> list[Image.Image]:
    obstacles = frozenset({(1, 2), (2, 2), (3, 2), (4, 2), (2, 5), (3, 5), (4, 5), (5, 5), (6, 1), (6, 2), (6, 3)})
    hazard = frozenset({(1, 6), (2, 6), (3, 6), (1, 7), (2, 7)})
    config = rl.GridWorldConfig(obstacles=obstacles, hazard_zone=hazard)
    env = rl.WarehouseGridWorld(config)
    agent = rl.QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)   # seed 42, as in the script
    snapshots = [0, 5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 500]
    rewards, paths, done_eps = [], {}, 0
    for target in snapshots:
        if target > done_eps:
            rewards.extend(rl.train(env, agent, n_episodes=target - done_eps).rewards.tolist())
            done_eps = target
        paths[target] = rl.extract_greedy_path(env, agent, config)
    rewards = np.array(rewards)
    smooth = np.convolve(rewards, np.ones(20) / 20, mode="valid")
    frames = []
    for ep in snapshots:
        path = paths[ep]
        reached = path[-1] == config.goal
        fig = new_fig(theme)
        ax = fig.add_axes([0.03, 0.05, 0.46, 0.80])
        ax.set_xlim(0, 8)
        ax.set_ylim(8, 0)
        ax.set_aspect("equal")
        ax.axis("off")
        for r in range(8):
            for c in range(8):
                color = theme["fill"]
                if (r, c) in obstacles:
                    color = theme["muted"]
                elif (r, c) in hazard:
                    color = theme["accent"]
                ax.add_patch(Rectangle((c, r), 1, 1, facecolor=color, edgecolor=theme["bg"], linewidth=1.5,
                                       alpha=0.55 if (r, c) in hazard else 1.0))
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=theme["series"][2], edgecolor=theme["bg"]))
        ax.add_patch(Rectangle((7, 7), 1, 1, facecolor=theme["series"][0], edgecolor=theme["bg"]))
        ax.plot([p[1] + 0.5 for p in path], [p[0] + 0.5 for p in path], color=theme["fg"], linewidth=2.0,
                marker="o", markersize=3)
        ax.set_title(f"política gulosa: chega ao objetivo em {len(path) - 1} passos" if reached
                     else "política gulosa: não chega ao objetivo", fontsize=8, color=theme["fg"])
        ax2 = fig.add_axes([0.58, 0.16, 0.39, 0.67])
        style(ax2, theme)
        ax2.plot(rewards[:ep], color=theme["series"][1], alpha=0.25, linewidth=0.8)
        shown = max(ep - 19, 0)
        ax2.plot(range(19, 19 + len(smooth))[:shown], smooth[:shown], color=theme["series"][1], linewidth=2)
        ax2.axvline(ep, color=theme["fg"], linewidth=0.8, linestyle=":")
        ax2.set_xlim(0, 500)
        ax2.set_ylim(rewards.min() * 1.05, 60)
        ax2.set_xlabel("episódio", fontsize=8)
        ax2.set_ylabel("recompensa do episódio", fontsize=8)
        fig.text(0.5, 0.94, "Q-Learning num galpão 8×8 (semente 42): política gulosa após N episódios",
                 ha="center", color=theme["fg"], fontsize=9)
        fig.text(0.5, 0.02, f"{ep} episódios treinados", ha="center", color=theme["muted"], fontsize=8)
        frames.append(grab(fig))
    return frames


# ----------------------------------------------------- 5. A* vs Greedy
def _expansion_order(engine, source: str, goal: str):
    """Run the real search while recording the order in which nodes are expanded."""
    order: list[str] = []
    seen: set[str] = set()

    def recording_pop(heap):
        entry = heapq.heappop(heap)
        node = entry.state.current_node
        if node not in seen:
            seen.add(node)
            order.append(node)
        return entry

    original = inf.heapq
    inf.heapq = types.SimpleNamespace(heappush=heapq.heappush, heappop=recording_pop)
    try:
        result = engine.execute_search(source, goal)
    finally:
        inf.heapq = original
    assert order[: result.nodes_expanded] == order and len(order) == result.nodes_expanded
    return result, order


def astar_frames(theme: dict) -> list[Image.Image]:
    source, goal = "Gate", "RU"
    graph = nx.Graph()
    for u, edges in inf.FGA_GRAPH.items():
        for v, w in edges:
            graph.add_edge(u, v, weight=w)
    pos = nx.kamada_kawai_layout(graph, weight="weight")
    engines = {"A*": inf.AStarSearch(inf.FGA_GRAPH, inf.FGA_HEURISTICS),
               "Greedy": inf.GreedySearch(inf.FGA_GRAPH, inf.FGA_HEURISTICS)}
    traces = {name: _expansion_order(e, source, goal) for name, e in engines.items()}
    n_steps = max(len(order) for _, order in traces.values())
    labels = nx.get_edge_attributes(graph, "weight")
    frames = []
    for step in range(n_steps + 2):                       # last frame reveals the final path
        fig = new_fig(theme)
        fig.text(0.5, 0.95, f"{source} → {goal} no campus FGA: A* contra Greedy (layout automático)",
                 ha="center", color=theme["fg"], fontsize=9)
        for col, (name, (result, order)) in enumerate(traces.items()):
            ax = fig.add_axes([0.02 + 0.49 * col, 0.05, 0.47, 0.82])
            ax.axis("off")
            shown = order[: min(step, len(order))]
            nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=theme["grid"], width=1.0)
            if step > n_steps:
                edges = list(zip(result.path, result.path[1:]))
                nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=edges, edge_color=theme["series"][col],
                                       width=3.0)
            nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels={k: f"{v:.0f}" for k, v in labels.items()},
                                         font_size=5.5, font_color=theme["muted"], rotate=False,
                                         bbox=dict(boxstyle="round,pad=0.1", fc=theme["bg"], ec="none"))
            colors = [theme["series"][col] if n in shown else theme["fill"] for n in graph.nodes]
            nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=colors, node_size=330,
                                   edgecolors=[theme["accent"] if n == goal else theme["muted"] for n in graph.nodes],
                                   linewidths=[2.2 if n in (goal, source) else 0.8 for n in graph.nodes])
            nx.draw_networkx_labels(graph, pos, ax=ax, font_size=6, font_color=theme["fg"])
            for rank, node in enumerate(shown, start=1):
                ax.annotate(str(rank), pos[node], xytext=(9, 9), textcoords="offset points", fontsize=7,
                            color=theme["fg"], fontweight="bold")
            if step > n_steps:
                title = f"{name}: {result.total_cost_m:.0f} m, {result.nodes_expanded} nós expandidos"
            else:
                title = f"{name}: {len(shown)} nó(s) expandido(s)"
            ax.set_title(title, fontsize=8.5, color=theme["fg"])
            ax.margins(0.12)
        fig.text(0.5, 0.01, "números = ordem de expansão; contorno laranja = destino; arestas em metros",
                 ha="center", color=theme["muted"], fontsize=6.5)
        frames.append(grab(fig))
    return frames


JOBS = {
    "wine": ("wine-pca-3d", wine_frames, 90),
    "tsp": ("tsp-40-cidades", tsp_frames, 200),
    "kalman": ("kalman-drone", kalman_frames, 110),
    "qlearning": ("qlearning-grid", qlearning_frames, 900),
    "astar": ("astar-vs-greedy", astar_frames, 1100),
}


def main() -> None:
    wanted = sys.argv[1:] or list(JOBS)
    OUT.mkdir(parents=True, exist_ok=True)
    for key in wanted:
        stem, make, duration = JOBS[key]
        print(key)
        for theme_name, theme in THEMES.items():
            save_gif(make(theme), OUT / f"{stem}-{theme_name}.gif", duration)


if __name__ == "__main__":
    main()
