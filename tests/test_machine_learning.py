from collections import deque

import numpy as np
import torch
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import projeto_01_supervisionado as supervised
import projeto_03_reforco as rl
import projeto_04_deep_learning as dl


def _gridworld():
    obstacles = frozenset({(1, 2), (2, 2), (3, 2), (4, 2), (2, 5), (3, 5), (4, 5), (5, 5), (6, 1), (6, 2), (6, 3)})
    return rl.GridWorldConfig(obstacles=obstacles, hazard_zone=frozenset({(1, 6), (2, 6), (3, 6), (1, 7), (2, 7)}))


def _bfs_steps(config):
    seen, queue = {config.start}, deque([(config.start, 0)])
    while queue:
        pos, d = queue.popleft()
        if pos == config.goal:
            return d
        for dr, dc in rl.ACTION_DELTAS.values():
            nxt = (pos[0] + dr, pos[1] + dc)
            if (0 <= nxt[0] < config.n_rows and 0 <= nxt[1] < config.n_cols
                    and nxt not in config.obstacles and nxt not in seen):
                seen.add(nxt)
                queue.append((nxt, d + 1))


def test_q_learning_learns_a_shortest_path_that_avoids_the_hazard_zone():
    config = _gridworld()
    env = rl.WarehouseGridWorld(config)
    agent = rl.QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)
    rl.train(env, agent, n_episodes=500)
    path = rl.extract_greedy_path(env, agent, config)
    assert path[-1] == config.goal
    assert len(path) - 1 == _bfs_steps(config) == rl.optimal_reference(config)[0]
    assert not set(path) & config.hazard_zone


def test_cross_validation_compares_models_on_the_same_folds():
    x, y = load_wine(return_X_y=True)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    result = supervised.run_cross_validation(lr, x, y, "LR", n_splits=5, n_repeats=2)
    assert len(result.fold_scores) == 10
    assert result.mean > 0.9


def test_lenet5_maps_28x28_images_to_ten_logits():
    logits = dl.LeNet5()(torch.zeros(3, 1, 28, 28))
    assert logits.shape == (3, 10)
    assert np.isfinite(logits.detach().numpy()).all()
