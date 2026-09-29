"""Claims made in chapters 3 and 4 of the thesis, as assertions.

Each test names the PDF page it is checking.  Where the thesis states an exact
number these are equalities; where it reports the shape of a curve they are
inequalities with a stated tolerance, because the underlying simulations are
stochastic.
"""

from __future__ import annotations

import numpy as np
import pytest

from netmotion.metrics import (
    clustering_coefficient,
    is_connected,
    mean_degree,
    mean_distance,
)
from netmotion.networks import (
    barabasi_albert,
    connectivity_threshold,
    erdos_renyi,
    newman_watts,
    ring_lattice,
    watts_strogatz,
)

pytestmark = pytest.mark.reproduction


# ---------------------------------------------------------------------------
# 3.1 Mean distance (PDF pp. 7-9)
# ---------------------------------------------------------------------------


def test_worked_example_eight_node_ring_gives_10_over_7():
    """PDF p. 9: the hand-computed example.

    Eight nodes, each joined to its four nearest neighbours. Every node reaches
    four others directly and the remaining three in two steps, so

        l = (4*1 + 3*2) / 7 = 1.42
    """
    ring = ring_lattice(8, 4)
    assert mean_distance(ring) == pytest.approx(10 / 7, abs=1e-12)


def test_n50_ring_approaches_the_ordered_theoretical_value():
    """PDF p. 9: for N = 50 with four neighbours the thesis notes that the
    measured mean distance and the ordered-network estimate N/(2k) come closer
    together than they do at N = 8."""

    def relative_gap(n: int) -> float:
        measured = mean_distance(ring_lattice(n, 4))
        theoretical = n / (2 * 4)
        return abs(measured - theoretical) / theoretical

    assert mean_distance(ring_lattice(50, 4)) == pytest.approx(325 / 49, abs=1e-12)
    assert relative_gap(50) < 0.07
    assert relative_gap(50) < relative_gap(8), (
        "the ordered-network estimate applies for N >> 1, so it fits better at "
        "N=50 than at N=8"
    )


# ---------------------------------------------------------------------------
# 3.3 Clustering (PDF pp. 10-11, 16)
# ---------------------------------------------------------------------------


def test_fully_connected_network_has_clustering_one():
    """PDF p. 16: at p = 1 every node is joined to every other, so C_i = 1
    for every node."""
    assert clustering_coefficient(erdos_renyi(10, 1.0, rng=0)) == pytest.approx(1.0)


def test_ring_with_two_neighbours_has_no_triangles():
    assert clustering_coefficient(ring_lattice(20, 2)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4.1 Erdos-Renyi (PDF pp. 12-16)
# ---------------------------------------------------------------------------


def test_connectivity_threshold_for_n10_is_about_0_23():
    """PDF p. 14: 'para la red con N=10 en estudio se obtiene p_c = 0.23'."""
    assert connectivity_threshold(10) == pytest.approx(0.23, abs=0.005)


@pytest.mark.slow
def test_networks_are_connected_above_the_threshold_and_not_below():
    """PDF p. 14, figures 3.a and 3.b: below p_c not every node is joined;
    above it, they are.

    Stated as a probability over realisations, since a single draw at p just
    above the threshold can still be disconnected.
    """
    n, trials = 10, 400
    below = connectivity_threshold(n) * 0.5
    well_above = 0.6

    rng = np.random.default_rng(20110930)
    frac_below = np.mean([is_connected(erdos_renyi(n, below, rng=rng)) for _ in range(trials)])
    frac_above = np.mean([is_connected(erdos_renyi(n, well_above, rng=rng)) for _ in range(trials)])

    assert frac_below < 0.15
    assert frac_above > 0.95


@pytest.mark.slow
def test_mean_degree_matches_the_theoretical_p_times_n_minus_one():
    """The measured mean degree of G(N, p) tracks p(N-1), its expected value."""
    n, trials = 10, 300
    rng = np.random.default_rng(11)
    for p in (0.1, 0.3, 0.5, 0.9):
        measured = np.mean([mean_degree(erdos_renyi(n, p, rng=rng)) for _ in range(trials)])
        assert measured == pytest.approx(p * (n - 1), abs=0.15)


@pytest.mark.slow
def test_clustering_matches_the_theoretical_p():
    """PDF pp. 14, 16, fig. 5.a: for a random graph the clustering coefficient
    equals the connection probability.

    Checked at p >= 0.5, where every node has degree 2 or more.
    """
    n, trials = 10, 300
    rng = np.random.default_rng(5)
    for p in (0.5, 0.7, 0.9, 1.0):
        measured = np.mean(
            [clustering_coefficient(erdos_renyi(n, p, rng=rng)) for _ in range(trials)]
        )
        assert measured == pytest.approx(p, abs=0.06)


# ---------------------------------------------------------------------------
# 4.2 Watts-Strogatz (PDF pp. 16-23)
# ---------------------------------------------------------------------------


def test_ordered_ring_has_a_delta_degree_distribution():
    """PDF p. 18: 'cada nodo esta conectado a 4 nodos luego el grado k de cada
    nodo es 4 y la funcion de distribucion P(k) correspondera a una funcion
    delta con el maximo en el valor 4'."""
    ring = ring_lattice(20, 4)
    assert np.array_equal(ring.degrees(), np.full(20, 4))


def test_p_zero_returns_the_ordered_ring_unchanged():
    ws = watts_strogatz(20, 4, 0.0, rng=0)
    assert np.array_equal(ws.matrix, ring_lattice(20, 4).matrix)


def test_rewiring_conserves_every_link_at_every_p():
    """PDF p. 22, fig. 7.b: the mean degree is a flat line at k.

    The printed algorithm conserves links by construction -- each rewiring
    event writes one pair into M and removes one ring link from the set the
    second sweep restores -- so this is an equality at every p and in every
    single realisation, not an average.
    """
    n, k, trials = 20, 4, 60
    rng = np.random.default_rng(1998)
    for p in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        for _ in range(trials):
            g = watts_strogatz(n, k, p, rng=rng)
            assert g.n_edges == n * k // 2
            assert mean_degree(g) == pytest.approx(float(k))


@pytest.mark.slow
def test_some_nearest_neighbour_links_survive_at_maximum_probability():
    """PDF p. 21: 'aun existe algun nodo que esta conectado a su vecino mas
    proximo, a pesar de que la probabilidad de romper la conexion con el vecino
    mas proximo es maxima'.

    The listing draws the replacement partner once per link examined and does
    not draw again if the guard rejects it, so at p = 1 a minority of the ring
    links are still left in place -- about 11% at N = 20, k = 4.
    """
    n, k, trials = 20, 4, 200
    ring = ring_lattice(n, k).adjacency
    rng = np.random.default_rng(1998)
    kept = [
        float(np.sum((watts_strogatz(n, k, 1.0, rng=rng).adjacency > 0) & (ring > 0)) / 2)
        for _ in range(trials)
    ]
    assert np.mean(kept) == pytest.approx(4.3, abs=1.0)
    assert 0.0 < np.mean(kept) / (n * k / 2) < 0.2


@pytest.mark.slow
def test_degree_distribution_keeps_its_maximum_at_k_and_flattens():
    """PDF p. 21: 'siendo siempre su valor maximo 4'.

    P(k) peaks at k = 4 for every rewiring probability, and spreads out as p
    grows -- from a narrow distribution at p = 0.1 to a broad one at p = 1.
    """
    n, k, trials = 20, 4, 400
    spreads = []
    for p in (0.1, 0.5, 1.0):
        rng = np.random.default_rng(1998)
        counts = np.zeros(n)
        for _ in range(trials):
            for degree in watts_strogatz(n, k, p, rng=rng).degrees().astype(int):
                counts[degree] += 1
        pk = counts / counts.sum()
        assert int(np.argmax(pk)) == k, f"P(k) peaks away from k at p={p}"
        spreads.append(float(pk[k]))

    assert spreads[0] > spreads[1] > spreads[2], "P(k) does not flatten as p grows"


@pytest.mark.slow
def test_clustering_falls_and_distance_falls_as_p_increases():
    """PDF pp. 21-22, figs. 7.a and 7.c: rewiring destroys order, so the mean
    clustering coefficient falls, and the shortcuts it creates shorten paths,
    so the mean distance falls too."""
    n, k, trials = 100, 6, 12
    rng = np.random.default_rng(1998)

    def averaged(p):
        runs = [watts_strogatz(n, k, p, rng=rng) for _ in range(trials)]
        return (
            float(np.mean([clustering_coefficient(g) for g in runs])),
            float(np.mean([mean_distance(g, "largest_component") for g in runs])),
        )

    c0, l0 = averaged(0.0)
    c_mid, l_mid = averaged(0.1)
    c1, l1 = averaged(1.0)

    assert c0 > c_mid > c1
    assert l0 > l_mid > l1


@pytest.mark.slow
def test_small_world_regime_exists_clustering_survives_where_distance_has_collapsed():
    """PDF p. 22, fig. 8: the point of the whole section.

    There is a range of p in which the mean distance has already dropped a long
    way from its ordered value while the clustering coefficient has barely
    moved.  That coexistence is the small-world effect.
    """
    n, k, trials = 100, 6, 12
    rng = np.random.default_rng(1998)

    def averaged(p):
        runs = [watts_strogatz(n, k, p, rng=rng) for _ in range(trials)]
        return (
            float(np.mean([clustering_coefficient(g) for g in runs])),
            float(np.mean([mean_distance(g, "largest_component") for g in runs])),
        )

    c0, l0 = averaged(0.0)
    c, length = averaged(0.05)

    assert length / l0 < 0.75, "mean distance has not collapsed"
    assert c / c0 > 0.60, "clustering has not survived"


@pytest.mark.slow
def test_the_effect_sharpens_with_network_size():
    """PDF p. 23, fig. 9: 'a medida que aumenta el numero de nodos, se ve con
    mas claridad el small world effect'.

    Quantified as the relative drop in mean distance at a small rewiring
    probability, which should be larger for a larger network.
    """
    k, p, trials = 6, 0.05, 10
    drops = []
    for n in (30, 100):
        rng = np.random.default_rng(4321)
        l0 = np.mean(
            [
                mean_distance(watts_strogatz(n, k, 0.0, rng=rng), "largest_component")
                for _ in range(2)
            ]
        )
        lp = np.mean(
            [
                mean_distance(watts_strogatz(n, k, p, rng=rng), "largest_component")
                for _ in range(trials)
            ]
        )
        drops.append((l0 - lp) / l0)

    assert drops[1] > drops[0]


# ---------------------------------------------------------------------------
# 4.3 Newman-Watts (PDF pp. 23-26)
# ---------------------------------------------------------------------------


def test_newman_watts_never_removes_a_ring_link():
    """PDF p. 23: the model exists precisely so that adding a link does not
    require breaking one."""
    n, k = 60, 4
    ring = ring_lattice(n, k)
    nw = newman_watts(n, k, 0.5, rng=99)
    assert np.all(nw.adjacency >= ring.adjacency)
    assert nw.n_edges >= ring.n_edges


@pytest.mark.slow
def test_newman_watts_mean_degree_grows_with_p_while_watts_strogatz_does_not():
    """The observable that separates the two models: Watts-Strogatz conserves
    links (PDF p. 22, fig. 7.b, 'el valor medio es constante'), Newman-Watts
    does not."""
    n, k, trials = 60, 4, 10
    rng = np.random.default_rng(7)

    nw_low = np.mean([mean_degree(newman_watts(n, k, 0.05, rng=rng)) for _ in range(trials)])
    nw_high = np.mean([mean_degree(newman_watts(n, k, 0.8, rng=rng)) for _ in range(trials)])
    assert nw_high > nw_low + 0.5

    ws_low = np.mean([mean_degree(watts_strogatz(n, k, 0.05, rng=rng)) for _ in range(trials)])
    ws_high = np.mean([mean_degree(watts_strogatz(n, k, 0.8, rng=rng)) for _ in range(trials)])
    assert ws_low == pytest.approx(ws_high, abs=1e-9) == pytest.approx(float(k))


# ---------------------------------------------------------------------------
# 4.4 Barabasi-Albert (PDF pp. 26-31)
# ---------------------------------------------------------------------------


def test_growth_produces_hubs():
    """PDF pp. 30-31: the defining feature of a scale-free network is that a
    few nodes have a degree far above the mean."""
    g = barabasi_albert(400, links_per_new_node=2, rng=2001)
    k = g.degrees()
    assert k.max() > 6 * k.mean()


def test_random_completion_gives_exactly_m_links_per_arrival():
    """Opcion 2 of the printed listing guarantees the mean degree."""
    n, m, m0 = 300, 2, 3
    g = barabasi_albert(n, links_per_new_node=m, initial_nodes=m0, fallback="random", rng=17)
    expected = m0 * (m0 - 1) // 2 + m * (n - m0)
    assert g.n_edges == expected


def test_doing_nothing_starves_late_arrivals():
    """Opcion 1: 'no se hace nada', which reproduces rich-gets-richer, since
    only high-degree nodes attract new nodes. An arrival that makes no link
    does not join the network (PDF p. 29: 179 nodes from 400 growth steps)."""
    n = 300
    starved = barabasi_albert(n, links_per_new_node=2, fallback="none", rng=17)
    complete = barabasi_albert(n, links_per_new_node=2, fallback="random", rng=17)

    assert starved.n_edges < complete.n_edges
    assert (starved.degrees() == 0).sum() > 0


def test_the_three_completion_rules_give_different_networks():
    """The thesis presents three alternatives."""
    kw = dict(n_nodes=200, links_per_new_node=2, rng=31)
    edges = {
        rule: barabasi_albert(fallback=rule, **kw).n_edges for rule in ("none", "random", "nearest")
    }
    assert len(set(edges.values())) > 1
