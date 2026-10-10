import numpy as np
import pytest

from drone_suru_simulatoru import scenarios as sc


def test_takas_with_avoidance_is_collision_free_and_without_it_is_not():
    safe = sc.run_takas(n=12)
    bare = sc.run_takas(n=12, avoid=False)
    assert safe.summary["carpisan_cift"] == 0 and safe.summary["varan"] == 12
    assert safe.summary["en_kucuk_aralik_m"] > 1.0
    assert bare.summary["carpisan_cift"] == 66            # every pair meets in the middle: 12 * 11 / 2
    assert safe.summary["yol_verimi"] > 0.9


def test_takas_dense_swap_with_cruise_reserve_stays_safe_and_is_deterministic():
    a = sc.run_takas(n=24)
    b = sc.run_takas(n=24)
    assert a.summary["carpisan_cift"] == 0 and a.summary == b.summary


def test_takas_works_in_a_steady_wind():
    s = sc.run_takas(n=8, wind=(4.0, -3.0)).summary
    assert s["carpisan_cift"] == 0 and s["varan"] == 8


def test_reciprocal_option_is_reported():
    s = sc.run_takas(n=4, reciprocal=True).summary
    assert s["reciprocal"] is True and s["avoid"] is True


@pytest.mark.parametrize("shape", sc.SHAPES)
def test_formation_offsets_are_distinct_and_at_least_a_spacing_apart(shape):
    off = sc.formation_offsets(shape, 9, spacing=4.0)
    d = np.linalg.norm(off[:, None] - off[None], axis=2) + np.eye(9) * 1e9
    assert off.shape == (9, 2)
    assert d.min() >= 4.0 * 0.78          # the V arms are 0.8 and 0.6 spacings apart per step: |(0.8,0.6)| = 1.0 spacing
    with pytest.raises(ValueError, match="unknown shape"):
        sc.formation_offsets("kare", 5)


def test_formasyon_unknown_topology():
    with pytest.raises(ValueError, match="unknown topology"):
        sc.run_formasyon(topology="yildiz")


def test_formasyon_with_avoidance_is_collision_free_and_keeps_the_rate():
    r = sc.run_formasyon(n=7, shape="v", topology="tam", avoid=True, seed=0)
    assert r.summary["carpisan_cift"] == 0
    assert r.summary["olculen_hiz"] == pytest.approx(r.summary["ongoru_hiz"], rel=2e-3)


def test_kapsama_failure_is_reported_and_costs_recover():
    r = sc.run_kapsama(n=10, density="duzgun", duration=100.0, kill=(40.0, 3), cells=40)
    s = r.summary
    assert s["yok_edilen"] == 3 and len(r.info["killed"]) == 3
    assert s["son_maliyet"] < s["baslangic_maliyet"] / 10
    assert s["son_maliyet"] / s["altigen_alt_sinir"] > 1.0          # never below the rigorous lower bound
    assert s["son_maliyet"] / s["altigen_alt_sinir"] < 1.3
    assert (~r.traj.alive[-1]).sum() == 3 and s["carpisan_cift"] == 0


def test_kapsama_unknown_density():
    with pytest.raises(ValueError, match="unknown density"):
        sc.run_kapsama(density="x")


def test_gorev_methods_ordering_and_arrival():
    sums = {}
    for method in sc.ASSIGNERS:
        s = sc.run_gorev(n=9, method=method, seed=4).summary
        sums[method] = s["atama_maliyeti_m"]
        assert s["varan"] == 9 and s["carpisan_cift"] == 0
    assert sums["ihale"] == pytest.approx(sums["tam"], abs=1e-4)
    assert sums["acgozlu"] >= sums["tam"] - 1e-9


def test_gorev_with_fewer_targets_and_bad_method():
    r = sc.run_gorev(n=8, n_targets=5, method="tam")
    assert (r.info["assignment"] >= 0).sum() == 5
    with pytest.raises(ValueError, match="unknown method"):
        sc.run_gorev(method="rastgele")


def test_spread_points_keeps_the_minimum_distance_and_fails_when_impossible():
    pts = sc._spread_points(np.random.default_rng(0), 20, 0, 30, 4.0)
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2) + np.eye(20) * 1e9
    assert d.min() >= 4.0
    with pytest.raises(ValueError, match="cannot place"):
        sc.run_gorev(n=50, size=10.0)


def test_summaries_are_json_serialisable_numbers():
    import json
    for r in (sc.run_takas(n=4), sc.run_gorev(n=4)):
        json.dumps(r.summary, default=float)
