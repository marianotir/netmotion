"""Section 5.4.2, *Aplicaciones del concepto de propagacion de una infeccion a
traves de la red. El concepto Cluster C de orden n*, as assertions.

The tests follow the two worked figures (PDF pp. 48-49), the identity between
the order-1 coefficient and the section-3.3 coefficient that fig. 47 (PDF p. 50)
shows, and the means printed on fig. 47's panels.  Where the section makes a
qualitative claim -- that orders above 1 find conglomerates the triangle-based
definition does not -- the test asserts the relation.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from netmotion import Network
from netmotion.metrics import local_clustering
from netmotion.networks import barabasi_albert, erdos_renyi, ring_lattice, watts_strogatz
from netmotion.propagation import (
    cluster_coefficient_order1,
    cluster_order_matrix,
    cluster_orders,
    infection_levels,
    local_cluster_coefficient_order1,
    nodes_in_largest_clusters,
)

pytestmark = pytest.mark.reproduction


# ---------------------------------------------------------------------------
# fixtures: the graphs the section is argued on
# ---------------------------------------------------------------------------

Q3_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),  # outer square
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),  # inner square
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),  # spokes
]


def hypercube() -> Network:
    """The *red tipo Hypercubo formada por 8 nodos* of fig. 44 (PDF p. 48)."""
    return Network.from_edges(8, Q3_EDGES, name="Q3")


def cycle(n: int) -> Network:
    return Network.from_edges(n, [(i, (i + 1) % n) for i in range(n)])


def complete(n: int) -> Network:
    return Network.from_edges(n, [(i, j) for i in range(n) for j in range(i + 1, n)])


# ---------------------------------------------------------------------------
# The worked example of fig. 44 (PDF p. 48)
# ---------------------------------------------------------------------------


def test_fig_44_hypercube_gives_no_order_1_three_order_2_and_one_order_3():
    """PDF p. 48, verbatim: 'Por tanto para este ejemplo no tenemos ningun
    cluster de orden 1, 3 clusteres de orden 2 y un cluster de orden 3.'

    The infection is started at the outer left corner, node 3 in the labelling
    of Q3_EDGES.
    """
    assert cluster_orders(hypercube(), 3).as_dict() == {1: 0, 2: 3, 3: 1}


def test_fig_44_propagation_matches_the_four_drawn_panels():
    """PDF p. 48, panels a-d: one red node at t = 0, its three neighbours at
    t = 1, three purple nodes at t = 2, the last node at t = 3."""
    levels = infection_levels(hypercube(), 3)
    assert [level.tolist() for level in levels] == [[3], [0, 2, 7], [1, 4, 6], [5]]


def test_fig_44_all_closures_happen_at_a_node_never_across_a_link():
    """PDF p. 48: every marked circle in fig. 44 turns purple because two
    infection paths meet *in* it. No purple branch appears anywhere in the
    figure, and indeed Q3 is bipartite, so no two nodes at the same distance
    from the seed can be joined."""
    run = cluster_orders(hypercube(), 3)
    assert run.purple_nodes.tolist() == [0, 3, 1]
    assert run.purple_edges.tolist() == [0, 0, 0]


def test_fig_44_third_order_node_has_three_parents_but_counts_as_one_cluster():
    """PDF p. 48: 'se cierra un cluster en el nodo coloreado en morado' --
    singular -- although node 5 is reached from all three of nodes 1, 4 and 6.

    Fig. 44 fixes the multiplicity rule.
    """
    q3 = hypercube()
    levels = infection_levels(q3, 3)
    parents = [v for v in levels[2] if q3.has_edge(int(v), 5)]
    assert sorted(parents) == [1, 4, 6]
    assert cluster_orders(q3, 3)[3] == 1


def test_fig_44_clusters_are_cycles_of_even_length_through_the_seed():
    """PDF p. 48, panels b and d: the red dashed outline is the square
    3-0-1-2-3 and the bold black outline the hexagon 3-0-4-5-6-7-3, i.e.
    cycles of length 2n for n = 2 and n = 3.

    Q3 has cycle rank |E| - |V| + 1 = 5, so the four cycles reported from one
    corner are a proper subset: the method finds the cycles built from two
    shortest paths out of the seed, which are the conglomerates containing m
    that the section sets out to find.
    """
    lengths = cluster_orders(hypercube(), 3).cycle_lengths()
    assert lengths == {4: 3, 6: 1}
    assert all(length % 2 == 0 for length in lengths), "Q3 is bipartite"
    assert sum(lengths.values()) <= len(nx.cycle_basis(hypercube().to_networkx())) == 5


def test_fig_44_result_holds_from_every_corner():
    """The thesis picks one corner and speaks of 'este ejemplo'; Q3 is
    vertex-transitive, so the profile it reports must be the profile of every
    node."""
    q3 = hypercube()
    for seed in range(q3.n):
        assert cluster_orders(q3, seed).as_dict() == {1: 0, 2: 3, 3: 1}


def test_max_order_truncates_the_propagation():
    """Truncating does not change the orders already found."""
    assert cluster_orders(hypercube(), 3, max_order=2).as_dict() == {1: 0, 2: 3}
    assert cluster_orders(hypercube(), 3, max_order=1).as_dict() == {1: 0}


# ---------------------------------------------------------------------------
# The order-1 identity (PDF pp. 49-51)
# ---------------------------------------------------------------------------


def test_order_1_purple_node_count_is_always_zero():
    """PDF p. 49: the order-1 count equals the classical E_i. No purple node
    occurs at order 1: the seed is the only node infected at t = 0, so every
    node infected at t = 1 has one parent."""
    graphs = [
        hypercube(),
        erdos_renyi(60, 0.15, rng=11),
        watts_strogatz(60, 4, 0.2, rng=12),
        barabasi_albert(60, links_per_new_node=2, rng=13),
        ring_lattice(60, 4),
    ]
    for graph in graphs:
        for node in range(graph.n):
            assert int(cluster_orders(graph, node, max_order=1).purple_nodes[0]) == 0


def test_order_1_count_is_the_number_of_links_among_the_neighbours():
    """PDF p. 49: E_i is 'el numero de conglomerados existentes donde se cumple
    M_ij = 1, M_ik = 1, M_jk = 1', and the new method must find the same
    number at order 1."""
    for graph in (hypercube(), erdos_renyi(50, 0.2, rng=7), barabasi_albert(50, links_per_new_node=3, rng=8)):
        adjacency = graph.adjacency
        for node in range(graph.n):
            neighbours = graph.neighbors(node)
            classical = int(adjacency[np.ix_(neighbours, neighbours)].sum() // 2)
            assert cluster_orders(graph, node, max_order=1).as_dict().get(1, 0) == classical


def test_order_1_coefficient_equals_the_section_3_3_coefficient():
    """PDF pp. 49-51, fig. 47: the validation of the whole construction --
    'los resultados del cluster coefficient obtenidos por ambos metodos
    coinciden'. Checked node by node against netmotion's section-3.3
    implementation and, independently, against NetworkX."""
    graphs = [
        erdos_renyi(100, 0.05, rng=1),
        erdos_renyi(100, 0.10, rng=2),
        erdos_renyi(100, 0.30, rng=3),
        watts_strogatz(100, 4, 0.1, rng=4),
        barabasi_albert(100, links_per_new_node=2, rng=5),
        ring_lattice(100, 4),
    ]
    for graph in graphs:
        classical = local_clustering(graph)
        propagated = local_cluster_coefficient_order1(graph)
        assert propagated == pytest.approx(classical, abs=1e-12)

        reference = nx.clustering(graph.to_networkx())
        for node in range(graph.n):
            if graph.degrees()[node] >= 2:
                assert cluster_coefficient_order1(graph, node) == pytest.approx(
                    reference[node], abs=1e-12
                )


def test_fig_47_ordered_network_is_a_flat_line_at_one_half():
    """PDF p. 50, panels 47.c and 47.d: for the 'red ordenada donde todos los
    nodos tienen el mismo numero de ramas' both methods print
    'media del cluster coefficient = 0.5000000000' and draw a flat line, so
    every one of the 100 nodes must score exactly 0.5.

    k = 4, the value of section 4.2, gives 0.5.
    """
    ring = ring_lattice(100, 4)
    curve = local_cluster_coefficient_order1(ring)
    assert curve.tolist() == [0.5] * 100
    assert float(curve.mean()) == pytest.approx(0.5000000000, abs=1e-10)


def test_fig_47_means_agree_between_the_two_methods_whatever_the_network():
    """PDF p. 50: the printed means are identical in every row of the table --
    0.08906582310 (ER), 0.5000000000 (ordered), 0.07130952387 (WS),
    0.1002124542 (SF). The agreement is asserted for each network type."""
    graphs = {
        "ER": erdos_renyi(100, 0.09, rng=21),
        "ordenada": ring_lattice(100, 4),
        "WS": watts_strogatz(100, 4, 0.15, rng=22),
        "SF": barabasi_albert(100, links_per_new_node=2, rng=23),
    }
    for name, graph in graphs.items():
        classical = float(local_clustering(graph).mean())
        propagated = float(local_cluster_coefficient_order1(graph).mean())
        assert propagated == pytest.approx(classical, abs=1e-12), name


def test_the_order_1_coefficient_needs_degree_two():
    """C(1) is undefined for k < 2: ValueError per node, 0 across the network."""
    leafy = Network.from_edges(3, [(0, 1)])
    with pytest.raises(ValueError, match="undefined"):
        cluster_coefficient_order1(leafy, 0)
    with pytest.raises(ValueError, match="undefined"):
        cluster_coefficient_order1(leafy, 2)

    assert local_cluster_coefficient_order1(leafy).tolist() == [0.0, 0.0, 0.0]
    assert np.isnan(local_cluster_coefficient_order1(leafy, low_degree="nan")).all()


# ---------------------------------------------------------------------------
# The two mechanisms in isolation
# ---------------------------------------------------------------------------


def test_triangle_and_complete_graphs_are_pure_order_1():
    """PDF p. 49, fig. 46: the classical conglomerate is a triangle, and it is
    an order-1 cluster. K4 contains three of them at node 0."""
    assert cluster_orders(complete(3), 0).as_dict() == {1: 1}
    assert cluster_coefficient_order1(complete(3), 0) == pytest.approx(1.0)
    assert cluster_orders(complete(4), 0).as_dict() == {1: 3}
    assert cluster_coefficient_order1(complete(4), 0) == pytest.approx(1.0)


def test_square_closes_at_a_node_at_order_2():
    """The minimal purple-node case: a 4-cycle is a cycle of length 2n with
    n = 2, exactly the structure the red dashed outline marks in fig. 44.b."""
    run = cluster_orders(cycle(4), 0)
    assert run.as_dict() == {1: 0, 2: 1}
    assert run.purple_nodes.tolist() == [0, 1]
    assert run.purple_edges.tolist() == [0, 0]


def test_pentagon_closes_across_a_link_at_order_2():
    """PDF p. 49, fig. 45.b: 'los nodos estan infectados, la rama esta
    coloreada en morado y existe un cluster marcado por la linea discontinua'.
    A 5-cycle is a cycle of length 2n + 1 with n = 2, the purple-branch case."""
    run = cluster_orders(cycle(5), 0)
    assert run.as_dict() == {1: 0, 2: 1}
    assert run.purple_nodes.tolist() == [0, 0]
    assert run.purple_edges.tolist() == [0, 1]
    assert run.cycle_lengths() == {5: 1}


def test_hexagon_closes_at_the_opposite_node_at_order_3():
    """The structure fig. 44.d traces in bold: a cycle of length 6 is found at
    order 3, and at no earlier order."""
    assert cluster_orders(cycle(6), 0).as_dict() == {1: 0, 2: 0, 3: 1}
    assert cluster_orders(cycle(6), 0).cycle_lengths() == {6: 1}


def test_a_node_reached_from_three_directions_counts_once():
    """The multiplicity rule of fig. 44 at t = 3, on the smallest graph that
    isolates it: a star whose three leaves all meet again at one further node.
    Counting p - 1 clusters per purple node would give 2 here, not 1."""
    g = Network.from_edges(5, [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)])
    assert cluster_orders(g, 0).as_dict() == {1: 0, 2: 1}


def test_trees_have_no_clusters_of_any_order():
    """A conglomerate is a closed path; an acyclic network has none, at any
    order. Neither mechanism can fire without a cycle."""
    star = Network.from_edges(4, [(0, 1), (0, 2), (0, 3)])
    path = Network.from_edges(3, [(0, 1), (1, 2)])
    assert cluster_orders(star, 0).as_dict() == {1: 0}
    assert cluster_orders(path, 1).as_dict() == {1: 0}
    assert cluster_coefficient_order1(path, 1) == pytest.approx(0.0)
    assert cluster_coefficient_order1(star, 0) == pytest.approx(0.0)


def test_distinct_parents_clause_is_read_as_a_condition_on_the_two_ends():
    """PDF p. 48: a cluster also forms when 'dos nodos n1, n2 son infectados en
    el instante t + 1 a partir de nodos distintos y a su vez existe una rama que
    conecta a los nodos n1 y n2'. distinct_clause='ends' (the default) takes
    'nodos distintos' as the two ends n1, n2, which gives the triangle of fig. 46
    and the agreement of fig. 47; 'parents' is also offered, and this test
    records both.
    """
    assert cluster_orders(complete(3), 0).as_dict() == {1: 1}
    assert cluster_orders(complete(3), 0, distinct_clause="ends").as_dict() == {1: 1}
    assert cluster_orders(complete(3), 0, distinct_clause="parents").as_dict() == {1: 0}

    # Above order 1 the two attachments agree on a cycle: the two ends of the
    # closing branch were reached from different nodes anyway.
    for reading in ("ends", "parents"):
        assert cluster_orders(cycle(5), 0, distinct_clause=reading).as_dict() == {1: 0, 2: 1}

    with pytest.raises(ValueError, match="distinct_clause"):
        cluster_orders(complete(3), 0, distinct_clause="strict")


def test_the_ends_reading_makes_order_1_the_triangle_of_fig_46_everywhere():
    """PDF p. 49, fig. 46: the order-1 cluster is the triangle i, j, k. With the
    clause read as a condition on the two ends, E_m(1) is the number of links
    among the neighbours of m -- one triangle on m per link -- for every node of
    every network tried, which is what fig. 47 requires of it."""
    graphs = [
        complete(3),
        complete(5),
        Network.from_edges(  # Petersen
            10,
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
            + [(5, 7), (7, 9), (9, 6), (6, 8), (8, 5)]
            + [(i, i + 5) for i in range(5)],
        ),
        erdos_renyi(100, 0.1, rng=11),
        ring_lattice(100, 4),
        watts_strogatz(100, 4, 0.1, rng=12),
        barabasi_albert(100, links_per_new_node=2, rng=13),
    ]
    for graph in graphs:
        adjacency = graph.adjacency
        for node in range(graph.n):
            neighbours = graph.neighbors(node)
            links_among_neighbours = int(adjacency[np.ix_(neighbours, neighbours)].sum() // 2)
            run = cluster_orders(graph, node, max_order=1)
            assert run.as_dict().get(1, 0) == links_among_neighbours
            assert int(run.purple_edges[0]) == links_among_neighbours


def test_fig_47_four_network_types_are_identical_node_for_node():
    """PDF pp. 49-50, fig. 47: 'los resultados del cluster coefficient obtenidos
    por ambos métodos coinciden' for ER, red ordenada, WS and SF networks of 100
    nodes. Asserted as exact equality of the two arrays; the ordered network's
    mean is also checked against 0.5.
    """
    graphs = {
        "ER": erdos_renyi(100, 0.1, rng=11),
        "ordenada": ring_lattice(100, 4),
        "WS": watts_strogatz(100, 4, 0.1, rng=12),
        "SF": barabasi_albert(100, links_per_new_node=2, rng=13),
    }
    for name, graph in graphs.items():
        classical = local_clustering(graph)
        propagated = local_cluster_coefficient_order1(graph)
        assert np.array_equal(propagated, classical), name
        assert float(propagated.mean()) == float(classical.mean()), name

    ordered = local_cluster_coefficient_order1(graphs["ordenada"])
    assert float(ordered.mean()) == pytest.approx(0.5000000000, abs=5e-11)


# ---------------------------------------------------------------------------
# The qualitative claim the section is written for (PDF pp. 48, 78)
# ---------------------------------------------------------------------------


def test_orders_above_one_see_conglomerates_the_classical_definition_cannot():
    """PDF p. 48: 'Un cluster de mayor tamano es aquel que contiene mas de tres
    nodos', and finding those with the section-2.3 definition 'seria muy
    costoso'. PDF p. 78: 'se ha desarrollado un nuevo metodo para encontrar
    clusters de mayor tamano dentro de la red'.

    On a triangle-free network the classical coefficient is zero for every node,
    and the order-n method finds conglomerates.
    """
    q3 = hypercube()
    assert local_clustering(q3).tolist() == [0.0] * 8
    assert cluster_order_matrix(q3, 3)[:, 1:].sum() > 0

    ring = ring_lattice(40, 2)  # a plain 40-cycle: triangle-free, one big cycle
    assert local_clustering(ring).tolist() == [0.0] * 40
    profile = cluster_order_matrix(ring, 20)
    assert profile[:, 0].sum() == 0, "no triangles, so no order-1 cluster"
    assert profile[:, 19].sum() == 40, "the 40-cycle closes at order 20 from every node"


def test_ranking_by_order_n_locates_the_nodes_on_the_large_conglomerate():
    """PDF p. 48: the method exists 'para la busqueda de clusters de gran
    tamano dentro de la red'. Qualitative check on a network built to have one
    obvious large conglomerate: a 6-cycle with a pendant tree hanging off it.
    Only the nodes on the cycle may be ranked at order 3."""
    edges = [(i, (i + 1) % 6) for i in range(6)] + [(0, 6), (6, 7), (7, 8)]
    g = Network.from_edges(9, edges)
    assert nodes_in_largest_clusters(g, 3).tolist() == [0, 1, 2, 3, 4, 5]
    assert nodes_in_largest_clusters(g, 1).size == 0


def test_cluster_order_matrix_agrees_with_the_per_node_runs():
    graph = watts_strogatz(40, 4, 0.3, rng=99)
    matrix = cluster_order_matrix(graph, 4)
    for node in range(graph.n):
        counts = cluster_orders(graph, node, max_order=4).counts
        assert matrix[node, : counts.size].tolist() == counts.tolist()
        assert matrix[node, counts.size :].sum() == 0


# ---------------------------------------------------------------------------
# Boundary behaviour
# ---------------------------------------------------------------------------


def test_the_infection_never_leaves_the_seeds_component():
    """On a disconnected network only the seed's own component contributes."""
    two_triangles = Network.from_edges(6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    assert cluster_orders(two_triangles, 0).as_dict() == {1: 1}
    visited = np.concatenate(infection_levels(two_triangles, 0))
    assert sorted(visited.tolist()) == [0, 1, 2]


def test_an_isolated_node_yields_no_orders_at_all():
    assert cluster_orders(Network.empty(4), 2).as_dict() == {}
    assert cluster_orders(Network.empty(4), 2).max_order == 0


def test_results_do_not_depend_on_the_order_the_links_were_added():
    """No randomness enters the method, so the counts must be reproducible and
    invariant under a permutation of the edge list."""
    shuffled = list(reversed(Q3_EDGES))
    first = cluster_orders(hypercube(), 3).as_dict()
    second = cluster_orders(Network.from_edges(8, shuffled), 3).as_dict()
    assert first == second == cluster_orders(hypercube(), 3).as_dict()
