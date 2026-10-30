import itertools

import numpy as np
import pytest

import projeto_01_rede_bayesiana as bn
import projeto_02_hmm as hmm
import projeto_03_kalman as kalman


# ----------------------------------------------------------- Bayesian network
def _enumerate_defect(network, evidence):
    """P(DefeitoSabor=1 | evidence) by summing the joint over the other variables."""
    cpds = {c.variable: c.values for c in network._model.get_cpds()}
    names = (network.UMIDADE, network.TEMPERATURA, network.TEMPO)
    num = den = 0.0
    for u, t, tm in itertools.product((0, 1), repeat=3):
        point = dict(zip(names, (u, t, tm)))
        if any(point[k] != v for k, v in evidence.items()):
            continue
        prior = cpds[names[0]][u] * cpds[names[1]][t] * cpds[names[2]][tm]
        num += prior * cpds[network.DEFEITO][1, u, t, tm]
        den += prior
    return num / den


@pytest.fixture(scope="module")
def coffee():
    return bn.CoffeeQualityBayesianNetwork().build()


@pytest.mark.parametrize("evidence", [
    {}, {"TempAlta": 1}, {"TempAlta": 1, "TempoExcessivo": 1},
    {"UmidadeAlta": 0, "TempAlta": 0, "TempoExcessivo": 0}, {"UmidadeAlta": 1, "TempoExcessivo": 0},
])
def test_variable_elimination_matches_enumeration(coffee, evidence):
    result = coffee.query(coffee.DEFEITO, evidence, label="t", inference_type="Causal")
    assert result.prob_true + result.prob_false == pytest.approx(1.0)
    assert result.prob_true == pytest.approx(_enumerate_defect(coffee, evidence))


def test_more_risk_factors_never_lower_the_defect_probability(coffee):
    def p(evidence):
        return coffee.query(coffee.DEFEITO, evidence, label="t", inference_type="Causal").prob_true
    none = p({"UmidadeAlta": 0, "TempAlta": 0, "TempoExcessivo": 0})
    one = p({"UmidadeAlta": 0, "TempAlta": 1, "TempoExcessivo": 0})
    two = p({"UmidadeAlta": 0, "TempAlta": 1, "TempoExcessivo": 1})
    assert none < one < two


# ------------------------------------------------------------------------ HMM
def _toy_hmm():
    pi = np.array([0.6, 0.3, 0.1])
    A = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])
    B = np.array([[0.5, 0.3, 0.1, 0.1], [0.1, 0.2, 0.6, 0.1], [0.2, 0.1, 0.1, 0.6]])
    return hmm.AppUsageHMM(hmm.HMMParameters(pi=pi, A=A, B=B)), pi, A, B


def test_viterbi_returns_the_most_probable_path_found_by_brute_force():
    model, pi, A, B = _toy_hmm()
    obs = [0, 2, 2, 3, 1, 0]

    def log_joint(path):
        lp = np.log(pi[path[0]]) + np.log(B[path[0], obs[0]])
        for t in range(1, len(obs)):
            lp += np.log(A[path[t - 1], path[t]]) + np.log(B[path[t], obs[t]])
        return lp

    best = max(itertools.product(range(3), repeat=len(obs)), key=log_joint)
    result = model.viterbi(obs)
    assert tuple(result.decoded_states) == best
    assert result.log_probability == pytest.approx(log_joint(best), abs=1e-6)


def test_forward_gives_normalised_filtering_distributions():
    model, *_ = _toy_hmm()
    result = model.forward([0, 2, 2, 3, 1, 0])
    assert np.allclose(result.alpha.sum(axis=1), 1.0)
    assert (result.alpha >= 0).all()


# --------------------------------------------------------------------- Kalman
def test_process_noise_has_no_correlation_between_axes():
    q = kalman.KalmanFilter2D(dt=0.5, process_noise_std=0.3, measurement_noise_std=4.0).Q
    x_block, y_block = [0, 2], [1, 3]                  # state = [px, py, vx, vy]
    assert np.allclose(q[np.ix_(x_block, y_block)], 0.0)
    assert np.allclose(q, q.T)
    assert np.linalg.eigvalsh(q).min() >= -1e-12


def test_covariance_stays_symmetric_and_positive_semidefinite():
    _, noisy = kalman.DroneTrajectorySimulator(dt=0.5, seed=1).generate(200, gps_noise_std=4.0)
    kf = kalman.KalmanFilter2D(dt=0.5, process_noise_std=0.3, measurement_noise_std=4.0)
    kf.initialize(noisy[0])
    for z in noisy:
        p = kf.step(z).P
        assert np.allclose(p, p.T, atol=1e-10)
        assert np.linalg.eigvalsh((p + p.T) / 2).min() >= -1e-9


def test_kalman_beats_raw_gps_on_average_over_seeds():
    raw, filtered = [], []
    for seed in range(42, 52):
        truth, noisy = kalman.DroneTrajectorySimulator(dt=0.5, seed=seed).generate(200, gps_noise_std=4.0)
        est = kalman.run_kalman_filter(noisy, 0.5, 0.3, 4.0).estimated_positions
        raw.append(np.sqrt(np.mean(np.sum((noisy - truth) ** 2, axis=1))))
        filtered.append(np.sqrt(np.mean(np.sum((est - truth) ** 2, axis=1))))
    assert np.mean(filtered) < np.mean(raw)
    assert all(f < r for f, r in zip(filtered, raw))
