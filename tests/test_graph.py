import numpy as np
import pytest

from drone_suru_simulatoru import graph


def test_closed_form_algebraic_connectivity():
    n = 9
    assert graph.algebraic_connectivity(graph.ring_adjacency(n)) == pytest.approx(2 - 2 * np.cos(2 * np.pi / n))
    assert graph.algebraic_connectivity(graph.path_adjacency(n)) == pytest.approx(2 - 2 * np.cos(np.pi / n))
    assert graph.algebraic_connectivity(graph.complete_adjacency(n)) == pytest.approx(n)


def test_components_match_laplacian_nullity_on_random_graphs():
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(150):
        n = int(rng.integers(3, 14))
        adj = graph.range_adjacency(rng.uniform(0, 10, (n, 2)), rng.uniform(1.0, 6.0))
        c = graph.n_components(adj)
        seen.add(c)
        assert c == graph.nullity(adj)
        assert (graph.algebraic_connectivity(adj) > 1e-9) == (c == 1)
    assert {1, 2} <= seen          # the random graphs include connected and disconnected ones


def test_components_labels_a_known_graph():
    adj = np.zeros((6, 6), dtype=bool)
    for a, b in ((0, 1), (1, 2), (3, 4)):
        adj[a, b] = adj[b, a] = True
    labels = graph.components(adj)
    assert labels[0] == labels[1] == labels[2] and labels[3] == labels[4]
    assert len(set(labels.tolist())) == 3 and graph.n_components(adj) == 3


def test_laplacian_rows_sum_to_zero_and_is_symmetric():
    adj = graph.knn_adjacency(np.random.default_rng(1).uniform(0, 5, (10, 2)), 3)
    lap = graph.laplacian(adj)
    assert np.allclose(lap.sum(axis=1), 0) and np.allclose(lap, lap.T)
    assert np.linalg.eigvalsh(lap).min() > -1e-10


def test_range_adjacency_threshold_and_dead_drones():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    adj = graph.range_adjacency(pos, 1.5)
    assert adj[0, 1] and not adj[0, 2] and not adj.diagonal().any()
    adj = graph.range_adjacency(pos, 5.0, alive=np.array([True, False, True]))
    assert not adj[1].any() and adj[0, 2]


def test_knn_adjacency_is_symmetric_and_has_at_least_k_neighbours():
    pos = np.random.default_rng(2).uniform(0, 10, (12, 2))
    adj = graph.knn_adjacency(pos, 3)
    assert (adj == adj.T).all() and adj.sum(axis=1).min() >= 3


def test_pinned_laplacian_is_positive_definite_and_rate_formulas():
    adj = graph.path_adjacency(5)
    m = graph.pinned_laplacian(adj, 0)
    assert np.linalg.eigvalsh(m).min() > 0
    # first order: rate = gain * smallest eigenvalue; second order with tau -> 0 approaches it
    first = graph.first_order_rate(m, 0.8)
    assert first == pytest.approx(0.8 * np.linalg.eigvalsh(m).min())
    assert graph.second_order_slowest_rate(m, 0.8, 1e-6) == pytest.approx(first, rel=1e-4)
    # with a lag the slow pole of tau s^2 + s + gain*lambda = 0 moves left (slightly faster decay); compare
    # with the roots found numerically for the smallest eigenvalue
    lam = np.linalg.eigvalsh(m).min()
    root = max(np.roots([0.5, 1.0, 0.8 * lam]).real)
    rate = graph.second_order_slowest_rate(m, 0.8, 0.5)
    assert rate == pytest.approx(-root) and rate > first
    # overdamping with large tau*gain*lambda: complex pair with real part -1/(2 tau)
    big = graph.complete_adjacency(6)
    assert graph.second_order_slowest_rate(big, 10.0, 5.0) == pytest.approx(1.0 / (2 * 5.0))


def test_ring_of_two_and_single_node_edge_cases():
    assert graph.ring_adjacency(2).sum() == 2
    assert graph.algebraic_connectivity(np.zeros((1, 1), dtype=bool)) == 0.0
    assert graph.n_components(np.zeros((0, 0), dtype=bool)) == 0
