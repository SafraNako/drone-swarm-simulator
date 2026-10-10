import numpy as np
import pytest

from drone_suru_simulatoru import assign


def instance(seed, n, m):
    rng = np.random.default_rng(seed)
    return assign.distance_cost(rng.uniform(0, 10, (n, 2)), rng.uniform(0, 10, (m, 2)))


@pytest.mark.parametrize("seed", range(60))
def test_exact_dp_equals_brute_force_and_auction_is_within_n_eps(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 8))
    m = int(rng.integers(1, n + 1))
    cost = instance(seed, n, m)
    brute = assign.total_cost(cost, assign.brute_force(cost))
    assert assign.total_cost(cost, assign.exact(cost)) == pytest.approx(brute, abs=1e-9)
    auction, bids = assign.auction(cost, eps_final=1e-6)
    gap = assign.total_cost(cost, auction) - brute
    assert -1e-9 <= gap <= n * 1e-6 + 1e-12 and bids >= n


def test_auction_is_exact_on_larger_instances_against_the_subset_dp():
    for seed in range(6):
        cost = instance(100 + seed, 11, 11)
        auction, _ = assign.auction(cost, eps_final=1e-7)
        assert assign.total_cost(cost, auction) - assign.total_cost(cost, assign.exact(cost)) <= 11 * 1e-7 + 1e-12


def test_greedy_is_not_optimal_on_a_known_instance():
    # greedy grabs the 1 and is forced into 100; the optimum is 2 + 2
    cost = np.array([[1.0, 2.0], [2.0, 100.0]])
    greedy = assign.greedy(cost)
    assert assign.total_cost(cost, greedy) == 101.0
    assert assign.total_cost(cost, assign.exact(cost)) == 4.0
    assert assign.total_cost(cost, assign.auction(cost)[0]) == pytest.approx(4.0, abs=1e-5)


def test_rectangular_problems_leave_surplus_drones_idle():
    cost = instance(5, 6, 3)
    for chosen in (assign.exact(cost), assign.auction(cost)[0], assign.greedy(cost), assign.brute_force(cost)):
        assert (chosen >= 0).sum() == 3 and sorted(chosen[chosen >= 0].tolist()) == [0, 1, 2]
    # the idle drones are the ones with no cheap target: optimum uses the three closest in matching sense
    assert assign.total_cost(cost, assign.exact(cost)) <= assign.total_cost(cost, assign.greedy(cost)) + 1e-12


def test_single_drone_and_identity_instances():
    assert assign.auction(np.array([[3.0]]))[0].tolist() == [0]
    cost = np.full((4, 4), 10.0)
    np.fill_diagonal(cost, 1.0)
    assert assign.auction(cost)[0].tolist() == [0, 1, 2, 3]
    assert assign.exact(cost).tolist() == [0, 1, 2, 3]


def test_optimal_euclidean_matchings_have_no_crossing_edges():
    from drone_suru_simulatoru.checks import _count_crossings
    rng = np.random.default_rng(9)
    crossed_greedy = 0
    for _ in range(40):
        drones, targets = rng.uniform(0, 10, (7, 2)), rng.uniform(0, 10, (7, 2))
        cost = assign.distance_cost(drones, targets)
        assert _count_crossings(drones, targets, assign.exact(cost)) == 0
        assert _count_crossings(drones, targets, assign.auction(cost)[0]) == 0
        crossed_greedy += _count_crossings(drones, targets, assign.greedy(cost)) > 0
    assert crossed_greedy > 0          # greedy matchings do cross, so the property is not vacuous


def test_errors():
    with pytest.raises(ValueError, match="more targets"):
        assign.auction(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="14"):
        assign.exact(np.zeros((15, 15)))
    with pytest.raises(ValueError, match="8"):
        assign.brute_force(np.zeros((9, 9)))
    with pytest.raises(ValueError, match="twice"):
        assign.total_cost(np.zeros((2, 2)), np.array([0, 0]))


def test_distance_cost_power():
    c1 = assign.distance_cost([[0, 0]], [[3, 4]])
    c2 = assign.distance_cost([[0, 0]], [[3, 4]], power=2)
    assert c1[0, 0] == 5.0 and c2[0, 0] == 25.0
