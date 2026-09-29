"""Section 5.5.1.3 of the thesis, the degree k of order n, as assertions.

The worked example gives ``K_1(2) = 4``, ``K_4(2) = 5`` and ``K_5(2) = 8`` on
PDF p. 56; these are equalities.  Where the section makes a qualitative claim
-- that the order-n degree ranks nodes differently from the classical degree,
that the immunisation ranking depends on the order -- the test asserts the
relation.
"""

from __future__ import annotations

import numpy as np
import pytest

from netmotion import Network
from netmotion.metrics.distance import shortest_path_lengths
from netmotion.metrics.higher_order_degree import (
    central_node,
    k_order,
    k_order_all,
    k_order_profile,
    k_order_ranking,
    saturation_order,
)
from netmotion.networks import erdos_renyi
from netmotion.propagation import propagate

pytestmark = pytest.mark.reproduction


# The thesis labels its nodes 1..9; this package indexes from 0.  Every test
# that quotes a printed value maps through THESIS, so the citation and the
# assertion cannot drift apart.
def THESIS(label: int) -> int:
    """Index of the node the thesis calls *label* in figs. 54 and 57."""
    return label - 1


@pytest.fixture
def fig54() -> Network:
    """The nine-node network of fig. 54 (PDF p. 54).

    Read off the rendered page: a double star, ``1-4``, ``2-4``, ``3-4``,
    ``4-5``, ``5-6``, ``6-7``, ``6-8``, ``6-9``.  It is the Estrada-Hatano
    example the whole of section 5.5.1 works through, and it is a tree with
    ``|E| = 8``.
    """
    edges = [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (5, 8)]
    return Network.from_edges(9, edges, name="thesis fig. 54")


# ---------------------------------------------------------------------------
# 5.5.1.3 The three printed values (PDF p. 56)
# ---------------------------------------------------------------------------


def test_the_three_values_printed_on_page_56(fig54):
    """PDF p. 56: 'nodo 1 => K_1(2) = 4, nodo 4 => K_4(2) = 5, nodo 5 =>
    K_5(2) = 8'.

    """
    assert k_order(fig54, THESIS(1), 2) == 4
    assert k_order(fig54, THESIS(4), 2) == 5
    assert k_order(fig54, THESIS(5), 2) == 8


def test_the_network_has_the_eight_links_the_thesis_counts(fig54):
    """PDF p. 56: 'existen un total de 8 ramas'."""
    assert fig54.n_edges == 8


def test_order_one_is_the_classical_degree(fig54):
    """PDF p. 55: 'el grado k de orden 1 coincide con el grado k tal y como ha
    sido hasta ahora definido'."""
    profile = [k_order(fig54, m, 1) for m in range(fig54.n)]
    assert profile == [1, 1, 1, 4, 2, 4, 1, 1, 1]
    assert profile == fig54.degrees().tolist()


def test_full_profile_of_the_figure_54_network(fig54):
    """The order 1..5 table implied by the definition on PDF pp. 55-56.

    """
    assert k_order_profile(fig54, THESIS(1), 5) == [1, 4, 5, 8, 8]
    assert k_order_profile(fig54, THESIS(4), 5) == [4, 5, 8, 8, 8]
    assert k_order_profile(fig54, THESIS(5), 5) == [2, 8, 8, 8, 8]
    assert k_order_profile(fig54, THESIS(6), 5) == [4, 5, 8, 8, 8]
    assert k_order_profile(fig54, THESIS(7), 5) == [1, 4, 5, 8, 8]


def test_profile_agrees_with_repeated_single_order_calls(fig54):
    for m in range(fig54.n):
        assert k_order_profile(fig54, m, 6) == [k_order(fig54, m, n) for n in range(1, 7)]


def test_a_truncated_profile_is_a_prefix_of_a_longer_one(fig54):
    """Asking for fewer orders must not fold the remaining links into the last
    one: K_1(2) is 4 whether or not orders 3 and 4 were also requested."""
    for m in range(fig54.n):
        assert k_order_profile(fig54, m, 2) == k_order_profile(fig54, m, 6)[:2]


def test_k_order_all_is_the_table_of_every_node_and_order(fig54):
    table = k_order_all(fig54, 5)
    assert table.shape == (9, 5)
    assert table[THESIS(5)].tolist() == [2, 8, 8, 8, 8]
    assert table[:, 0].tolist() == fig54.degrees().tolist()


# ---------------------------------------------------------------------------
# Links are counted once
# ---------------------------------------------------------------------------


def test_each_link_is_counted_once(fig54):
    """A sum of degrees counts the link between node 1 and node 4 twice, giving
    5 and 9 where the values are 4 and 5; each link is counted once."""
    degrees = fig54.degrees()

    sum_of_degrees_node_1 = int(degrees[THESIS(1)] + degrees[THESIS(4)])
    assert sum_of_degrees_node_1 == 5
    assert k_order(fig54, THESIS(1), 2) == 4 != sum_of_degrees_node_1

    neighbours_of_4 = [THESIS(1), THESIS(2), THESIS(3), THESIS(5)]
    sum_of_degrees_node_4 = int(degrees[THESIS(4)] + sum(degrees[j] for j in neighbours_of_4))
    assert sum_of_degrees_node_4 == 9
    assert k_order(fig54, THESIS(4), 2) == 5 != sum_of_degrees_node_4


def test_the_count_is_of_links(fig54):
    """The ball of radius 2 around node 1 holds five nodes and its shell three;
    K_1(2) = 4 counts links."""
    dist = shortest_path_lengths(fig54, THESIS(1))
    ball_of_radius_two = int((dist <= 2).sum())
    shell_at_distance_two = int((dist == 2).sum())

    assert ball_of_radius_two == 5
    assert shell_at_distance_two == 3
    assert k_order(fig54, THESIS(1), 2) == 4
    assert k_order(fig54, THESIS(1), 2) not in (ball_of_radius_two, shell_at_distance_two)


def test_nodes_within_n_minus_one_hops_are_not_the_count(fig54):
    """Nodes within n-1 hops of node 1 number 2 at order 2; K_1(2) = 4."""
    dist = shortest_path_lengths(fig54, THESIS(1))
    assert int((dist <= 1).sum()) == 2
    assert k_order(fig54, THESIS(1), 2) == 4


# ---------------------------------------------------------------------------
# 5.5.1.3 Centrality by saturation (PDF pp. 55-56)
# ---------------------------------------------------------------------------


def test_saturation_orders_match_the_reported_propagation_times(fig54):
    """PDF p. 55: 'los nodos 1,2,3,7,8,9 necesitan 4 iteraciones para
    propagarse por toda la red, los nodos 4 y 6 necesitan 3 iteraciones y el
    nodo 5 necesita de 2 iteraciones'.

    On this network the order at which a node knows all eight links equals
    those times.
    """
    assert saturation_order(fig54, THESIS(5)) == 2
    assert saturation_order(fig54, THESIS(4)) == 3
    assert saturation_order(fig54, THESIS(6)) == 3
    for label in (1, 2, 3, 7, 8, 9):
        assert saturation_order(fig54, THESIS(label)) == 4


def test_node_five_is_the_central_node(fig54):
    """PDF p. 56: 'el nodo 5 adquiere el conocimiento de la existencia de las 8
    ramas para el K(2) con lo que este es el nodo central'.

    The same node is the answer of the thermal-bath method and of the methods
    of 5.5.1.1 and 5.5.1.2.
    """
    assert central_node(fig54) == THESIS(5)
    assert k_order(fig54, THESIS(5), 2) == fig54.n_edges
    others = [m for m in range(fig54.n) if m != THESIS(5)]
    assert all(k_order(fig54, m, 2) < fig54.n_edges for m in others)


def test_node_five_leads_at_order_two_but_not_at_order_one(fig54):
    """PDF p. 56: the degree k of order 1 of node 5 is smaller than that of
    node 4, and the degree k of order 2 is larger for node 5."""
    assert k_order(fig54, THESIS(5), 1) == 2
    assert k_order(fig54, THESIS(4), 1) == 4
    assert k_order(fig54, THESIS(5), 1) < k_order(fig54, THESIS(4), 1)
    assert k_order(fig54, THESIS(5), 2) > k_order(fig54, THESIS(4), 2)


# ---------------------------------------------------------------------------
# 5.5.2 Immunisation ranking (PDF pp. 56-57)
# ---------------------------------------------------------------------------


def test_the_immunisation_ranking_depends_on_the_order(fig54):
    """PDF p. 56, qualitative claim: the point of sweeping K(1) ... K(8) as
    immunisation criteria is that they give different orderings, so that one of
    them can be better than the others.

    Under K(1) a degree-4 hub leads; under K(2) node 5, of degree 2, does.
    """
    by_degree = k_order_ranking(fig54, 1)
    by_order_two = k_order_ranking(fig54, 2)

    assert by_degree[0] in (THESIS(4), THESIS(6))
    assert by_order_two[0] == THESIS(5)
    assert by_degree != by_order_two


def test_the_ranking_is_a_permutation_sorted_by_decreasing_k(fig54):
    for n in (1, 2, 3, 8):
        ranking = k_order_ranking(fig54, n)
        assert sorted(ranking) == list(range(fig54.n))
        scores = [k_order(fig54, m, n) for m in ranking]
        assert scores == sorted(scores, reverse=True)


def test_ties_are_broken_by_node_index(fig54):
    """Ties are broken by node index, so the ranking is stable."""
    assert k_order_ranking(fig54, 1)[:2] == [THESIS(4), THESIS(6)]


# ---------------------------------------------------------------------------
# Small graphs where the definition can be checked by hand
# ---------------------------------------------------------------------------


def test_star():
    star = Network.from_edges(4, [(0, 1), (0, 2), (0, 3)])
    assert k_order(star, 0, 1) == 3
    assert k_order(star, 0, 2) == 3
    assert k_order(star, 1, 1) == 1
    assert k_order(star, 1, 2) == 3
    assert k_order(star, 1, 7) == 3


def test_path_of_three_nodes():
    path = Network.from_edges(3, [(0, 1), (1, 2)])
    assert k_order_profile(path, 0, 4) == [1, 2, 2, 2]
    assert k_order_profile(path, 1, 4) == [2, 2, 2, 2]


def test_triangle():
    triangle = Network.from_edges(3, [(0, 1), (1, 2), (2, 0)])
    for m in range(3):
        assert k_order(triangle, m, 1) == 2
        assert k_order(triangle, m, 2) == 3


def test_disconnected_network_never_sees_the_other_component():
    """K_m(n) saturates at the links of m's own component."""
    two_triangles = Network.from_edges(6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    assert k_order(two_triangles, 0, 10) == 3
    assert saturation_order(two_triangles, 0) == float("inf")
    with pytest.raises(ValueError, match="disconnected"):
        central_node(two_triangles)


def test_isolated_node_learns_nothing():
    lonely = Network.from_edges(5, [(1, 2), (2, 3), (3, 4)])
    assert k_order_profile(lonely, 0, 5) == [0, 0, 0, 0, 0]
    assert saturation_order(lonely, 0) == float("inf")


# ---------------------------------------------------------------------------
# Links across the frontier
# ---------------------------------------------------------------------------


def test_five_cycle_pins_the_convention_for_links_across_the_frontier():
    """On the 5-cycle 0-1-2-3-4-0 the link 2-3 has both ends exactly two hops from
    node 0.  rule='incident' counts links with at least one end within n-1
    hops, so K_0(2) = 4; rule='induced' gives 5.
    """
    c5 = Network.from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    assert k_order(c5, 0, 2) == 4
    assert k_order(c5, 0, 2, rule="induced") == 5
    assert k_order(c5, 0, 3) == 5


def test_the_default_rule_gives_the_degree_at_order_one():
    """PDF p. 55: the order-1 degree is the classical degree, which
    rule='incident' returns; rule='induced' scores 3 on a triangle."""
    triangle = Network.from_edges(3, [(0, 1), (1, 2), (2, 0)])
    assert k_order(triangle, 0, 1, rule="induced") == 3
    assert int(triangle.degrees()[0]) == 2
    assert k_order(triangle, 0, 1) == 2


# ---------------------------------------------------------------------------
# Invariants, and agreement with the propagation formulation
# ---------------------------------------------------------------------------


def _k_order_by_state_machine(network: Network, m: int, n: int) -> int:
    """K_m(n) the way PDF p. 56 describes computing it.

    Run the infection of section 5.4.1 from *m*, take the nodes infected up to
    instant *t = n* -- the ball of radius n-1 -- and count the distinct links
    touching them.
    """
    run = propagate(network, m)
    ball = np.flatnonzero(run.infected_at <= n)
    adjacency = network.adjacency
    links = {
        (min(int(v), int(w)), max(int(v), int(w)))
        for v in ball
        for w in np.flatnonzero(adjacency[v])
    }
    return len(links)


def test_agrees_with_the_propagation_formulation_on_random_networks():
    """PDF p. 56 defines K_m(n) through the infection dynamics of 5.4.1; this
    module computes it with one breadth-first sweep instead.

    The two return the same integers.
    """
    rng = np.random.default_rng(20110930)
    for _ in range(30):
        network = erdos_renyi(20, 0.15, rng=rng)
        for m in range(network.n):
            for n in range(1, 7):
                assert k_order(network, m, n) == _k_order_by_state_machine(network, m, n)


def test_monotone_and_saturating_on_random_networks():
    """K_m(n) <= K_m(n+1), and it stops at the links of m's own component."""
    rng = np.random.default_rng(1250058)
    for _ in range(30):
        network = erdos_renyi(25, 0.12, rng=rng)
        adjacency = network.adjacency
        for m in range(network.n):
            profile = k_order_profile(network, m, 10)
            assert profile == sorted(profile)

            dist = shortest_path_lengths(network, m)
            component = np.flatnonzero(np.isfinite(dist))
            rows, cols = np.nonzero(np.triu(adjacency, 1))
            reachable = np.isin(rows, component).sum()
            assert profile[-1] == int(reachable)

            saturation = saturation_order(network, m)
            if np.isfinite(saturation):
                assert k_order(network, m, int(saturation)) == network.n_edges
                if saturation > 1:
                    assert k_order(network, m, int(saturation) - 1) < network.n_edges


def test_central_node_is_a_node_of_minimum_saturation_order():
    rng = np.random.default_rng(7)
    for _ in range(20):
        network = erdos_renyi(15, 0.35, rng=rng)
        orders = [saturation_order(network, m) for m in range(network.n)]
        if not np.isfinite(orders).any():
            continue
        chosen = central_node(network)
        assert orders[chosen] == min(orders)
        assert chosen == min(m for m, o in enumerate(orders) if o == min(orders))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_order_zero_is_undefined(fig54):
    """The order starts at n = 1."""
    with pytest.raises(ValueError, match="starts at 1"):
        k_order(fig54, 0, 0)
    with pytest.raises(ValueError, match="starts at 1"):
        k_order_profile(fig54, 0, 0)


def test_directed_networks_are_rejected():
    """K_m(n) is defined for undirected networks."""
    directed = Network(np.array([[0.0, 1.0], [0.0, 0.0]]), directed=True)
    with pytest.raises(ValueError, match="undirected"):
        k_order(directed, 0, 2)


def test_self_links_are_rejected_by_the_representation():
    """The adjacency convention of section 2.2 has A_ii = 0."""
    with pytest.raises(ValueError, match="self-links"):
        Network(np.array([[1.0, 1.0], [1.0, 0.0]]))


def test_node_index_is_checked(fig54):
    with pytest.raises(IndexError):
        k_order(fig54, 99, 1)


def test_unknown_rule_is_rejected(fig54):
    with pytest.raises(ValueError, match="rule must be"):
        k_order(fig54, 0, 1, rule="ball")
