"""Section 5.5.1 of the thesis, as assertions.

The section works its methods on the nine-node network of fig. 54; its
integers are asserted exactly, each test naming the PDF page it checks.  Where
the section makes a qualitative claim -- the methods agree with thermal bath,
information leaves the central node fastest -- the test asserts the relation.

The file also asserts that the minimum-time score of section 5.5.1.2 is the
eccentricity, checked node by node against :mod:`netmotion.metrics.distance` on
other networks.
"""

from __future__ import annotations

import inspect
import re

import numpy as np
import pytest

from netmotion import Network
from netmotion.metrics import (
    CENTRALITY_CRITERIA,
    all_pairs_distances,
    central_node_by_removal,
    central_nodes_by_min_time,
    compare_centrality_criteria,
    degree_order_n_criterion,
    diameter,
    estrada_hatano_network,
    is_connected,
    k_order,
    k_order_ranking,
    min_time_scores,
    node_removal_scores,
    peel_by_minimum_degree,
    rank_nodes,
)
from netmotion.metrics import propagation_centrality as pc
from netmotion.networks import (
    barabasi_albert,
    erdos_renyi,
    newman_watts,
    ring_lattice,
    watts_strogatz,
)

pytestmark = pytest.mark.reproduction


# The thesis numbers the fig. 54 nodes 1..9 and netmotion indexes from 0, so
# every expectation below is written in thesis numbering and shifted once, here.
def thesis_labels(indices) -> list[int]:
    return [int(i) + 1 for i in np.asarray(indices, dtype=int)]


@pytest.fixture
def fig54() -> Network:
    return estrada_hatano_network()


def eccentricities(network: Network) -> np.ndarray:
    """Eccentricity of every node, computed from shortest paths alone."""
    return all_pairs_distances(network).max(axis=1)


# ---------------------------------------------------------------------------
# 5.5.1 The example network (PDF p. 54)
# ---------------------------------------------------------------------------


def test_the_example_network_has_nine_nodes_and_eight_branches(fig54):
    """PDF p. 54, fig. 54, corroborated on PDF p. 56: 'existen un total de 8
    ramas'.

    The edge list is read off the figure.
    """
    assert fig54.n == 9
    assert fig54.n_edges == 8
    assert fig54.degrees().sum() == 16


def test_barabasi_criterion_names_the_two_hubs(fig54):
    """PDF p. 54: 'Segun Barbasi el nodo mas vulnerable deberia ser el 6 y el 4
    puesto que tienen el mayor grado k'.

    """
    assert thesis_labels(rank_nodes(fig54, "degree").most_central) == [4, 6]


# ---------------------------------------------------------------------------
# 5.5.1.1 Node removal (PDF pp. 54-55)
# ---------------------------------------------------------------------------


def test_peeling_reproduces_both_printed_rounds(fig54):
    """PDF pp. 54-55, the two rounds the thesis works through.

    Round 1: 'en el primer paso el grado minimo es 1. Si eliminamos los nodos
    con ese grado queda la red restante formada por los nodos 4,5,6' (fig. 55).
    Round 2: 'Calculamos el grado k minimo con la red restante que vuelve a ser
    1. Eliminando los nodos con ese grado nos queda por ultimo el nodo 5'
    (fig. 56).
    """
    result = peel_by_minimum_degree(fig54)

    assert [thesis_labels(layer) for layer in result.as_lists()] == [[1, 2, 3, 7, 8, 9], [4, 6]]
    assert thesis_labels(result.survivors) == [5]
    assert result.terminated == "single_node"
    assert result.n_rounds == 2


def test_node_removal_names_node_five_as_the_central_node(fig54):
    """PDF p. 55: 'nos queda por ultimo el nodo 5 que coincide con el nodo mas
    vulnerable o nodo central obtenido por thermal bath'.

    """
    assert central_node_by_removal(fig54) == 4  # thesis node 5


def test_node_removal_scores_record_the_layer_each_node_was_stripped_in(fig54):
    """The 'descomposicion de la red en capas' of PDF p. 54, per node: the six
    leaves go in round 0, the two hubs in round 1, node 5 survives both."""
    scores = node_removal_scores(fig54)
    assert scores.tolist() == [0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0]
    assert thesis_labels(rank_nodes(fig54, "node_removal").most_central) == [5]


def test_peeling_is_not_networkx_core_number(fig54):
    """The three printed steps of section 5.5.1.1 recompute degrees between
    batches and find node 5; ``networkx.core_number`` computes a different
    quantity, so it is not used."""
    nx = pytest.importorskip("networkx")
    coreness = nx.core_number(fig54.to_networkx())

    assert set(coreness.values()) == {1}, "core_number gives every node of this tree coreness 1"
    assert central_node_by_removal(fig54) == 4


def test_a_regular_subgraph_is_reported_as_emptied():
    """On a regular subgraph one round would remove every remaining node; the
    peeling stops and reports 'emptied'."""
    triangle = Network.from_edges(3, [(0, 1), (1, 2), (2, 0)])
    result = peel_by_minimum_degree(triangle)
    assert result.terminated == "emptied"
    assert result.as_lists() == []
    assert sorted(result.survivors.tolist()) == [0, 1, 2]

    with pytest.raises(ValueError, match="emptied"):
        central_node_by_removal(triangle)

    pair = Network.from_edges(2, [(0, 1)])
    assert peel_by_minimum_degree(pair).terminated == "emptied"

    cycle = Network.from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    assert peel_by_minimum_degree(cycle).terminated == "emptied"

    # A path of four: the outer pair goes first, then the inner pair ties.
    path4 = Network.from_edges(4, [(0, 1), (1, 2), (2, 3)])
    result = peel_by_minimum_degree(path4)
    assert result.as_lists() == [[0, 3]]
    assert result.terminated == "emptied"
    assert result.survivors.tolist() == [1, 2]


def test_a_single_node_is_already_the_survivor():
    result = peel_by_minimum_degree(Network.empty(1))
    assert result.terminated == "single_node"
    assert result.as_lists() == []
    assert central_node_by_removal(Network.empty(1)) == 0


def test_star_peels_to_its_hub():
    star = Network.from_edges(4, [(0, 1), (0, 2), (0, 3)])
    result = peel_by_minimum_degree(star)
    assert result.as_lists() == [[1, 2, 3]]
    assert central_node_by_removal(star) == 0


# ---------------------------------------------------------------------------
# 5.5.1.2 Minimum propagation time (PDF p. 55)
# ---------------------------------------------------------------------------


def test_minimum_time_scores_reproduce_the_printed_iteration_counts(fig54):
    """PDF p. 55, the only numbers the section reports.

    'los nodos 1,2,3,7,8,9 necesitan 4 iteraciones para propagarse por toda la
    red, los nodos 4 y 6 necesitan 3 iteraciones y el nodo 5 necesita de 2
    iteraciones para llegar a todos los nodos de la red'.
    """
    scores = min_time_scores(fig54)

    assert scores.tolist() == [4.0, 4.0, 4.0, 3.0, 2.0, 3.0, 4.0, 4.0, 4.0]
    assert [thesis_labels(np.flatnonzero(scores == t)) for t in (4.0, 3.0, 2.0)] == [
        [1, 2, 3, 7, 8, 9],
        [4, 6],
        [5],
    ]


def test_minimum_time_names_node_five_as_the_central_node(fig54):
    """PDF p. 55: 'el nodo central obtenido por este metodo vuelve a ser el nodo
    5 que coincide con el obtenido por el metodo thermal bath'."""
    assert thesis_labels(central_nodes_by_min_time(fig54)) == [5]


def test_both_thesis_methods_agree_with_each_other_and_disagree_with_the_degree(fig54):
    """PDF p. 54, the qualitative point of the whole section: 'se describiran
    otros metodos ... cuyo resultado coincidira con el obtenido mediante el
    metodo de thermal bath'.

    The thermal-bath answer, node 5, is Estrada and Hatano's; asserted is the
    relation: the two methods agree on node 5, and the degree picks 4 and 6.
    """
    rankings = compare_centrality_criteria(fig54)
    by_removal = thesis_labels(rankings["node_removal"].most_central)
    by_time = thesis_labels(rankings["min_time"].most_central)
    by_degree = thesis_labels(rankings["degree"].most_central)

    assert by_removal == by_time == [5]
    assert by_degree == [4, 6]
    assert not set(by_time) & set(by_degree)


def test_the_full_orderings_of_the_two_methods_coincide_on_the_example(fig54):
    """On fig. 54 the two methods agree on the whole ordering."""
    rankings = compare_centrality_criteria(fig54, ["node_removal", "min_time"])
    assert rankings["node_removal"].order.tolist() == rankings["min_time"].order.tolist()


def test_on_a_tailed_triangle_the_two_methods_differ():
    """On a triangle with a four-link tail the peeling stops at the triangle,
    and the minimum-time centre is in the middle of the tail."""
    tailed = Network.from_edges(7, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 6)])

    result = peel_by_minimum_degree(tailed)
    assert result.terminated == "emptied"
    assert sorted(result.survivors.tolist()) == [0, 1, 2]
    assert central_nodes_by_min_time(tailed).tolist() == [3, 4]


def test_a_star_infects_everything_in_one_step_from_its_hub():
    """The extreme case of the section's idea that information leaves the centre
    fastest: from the hub the whole network is infected in a single iteration,
    from a leaf it takes two."""
    star = Network.from_edges(4, [(0, 1), (0, 2), (0, 3)])
    assert min_time_scores(star).tolist() == [1.0, 2.0, 2.0, 2.0]
    assert central_nodes_by_min_time(star).tolist() == [0]


def test_every_minimiser_is_returned_when_the_score_ties():
    """A path of four nodes ties at the minimum, and both nodes are returned."""
    path4 = Network.from_edges(4, [(0, 1), (1, 2), (2, 3)])
    assert min_time_scores(path4).tolist() == [3.0, 2.0, 2.0, 3.0]
    assert central_nodes_by_min_time(path4).tolist() == [1, 2]


def test_the_healing_variant_never_covers_the_tree(fig54):
    """With recovery (heals=True) the infected set on this bipartite tree
    alternates between the two parity classes, so no seed covers the network."""
    scores = min_time_scores(fig54, heals=True, max_steps=25)
    assert np.isinf(scores).all()


def test_a_disconnected_network_scores_infinite_and_gets_no_centre():
    """Scores are inf -- the eccentricity -- and no central node is named."""
    split = Network.from_edges(4, [(0, 1), (2, 3)])
    assert np.isinf(min_time_scores(split)).all()

    with pytest.raises(ValueError, match="disconnected"):
        central_nodes_by_min_time(split)


def test_an_immune_node_cannot_seed_and_reshapes_the_scores(fig54):
    """Immunity is defined on PDF p. 54 as S_n(t) = +1 for every t, so an immune
    node is never infected and never transmits.  Scoring a partially immunised
    network is what section 5.5.2 needs.

    Immunising leaf 1 of fig. 54 leaves it unable to seed anything, and shifts
    every other score to its eccentricity in the network that remains.
    """
    scores = min_time_scores(fig54, immune=[0])
    assert np.isinf(scores[0])
    assert scores[1:].tolist() == [4.0, 4.0, 3.0, 2.0, 3.0, 4.0, 4.0, 4.0]
    assert thesis_labels(central_nodes_by_min_time(fig54, immune=[0])) == [5]


# ---------------------------------------------------------------------------
# The minimum time is the eccentricity
# ---------------------------------------------------------------------------


def test_minimum_time_score_is_the_eccentricity_on_the_thesis_graph(fig54):
    """A node at distance d joins the infection at iteration d, so
    t_m = ecc(m); shortest paths check the nine values."""
    assert min_time_scores(fig54).tolist() == eccentricities(fig54).tolist()


def test_the_central_node_is_the_graph_centre_and_the_minimum_is_the_radius(fig54):
    """The same identity, read off the other end: argmin t_m is the Jordan
    centre and min t_m is the radius.  The maximum is the diameter, which
    netmotion.metrics.distance computes by an unrelated route."""
    scores = min_time_scores(fig54)
    ecc = eccentricities(fig54)

    assert central_nodes_by_min_time(fig54).tolist() == np.flatnonzero(ecc == ecc.min()).tolist()
    assert scores.min() == ecc.min() == 2.0  # the radius
    assert scores.max() == diameter(fig54) == 4.0


@pytest.mark.parametrize(
    "network",
    [
        pytest.param(ring_lattice(24, 4), id="ring"),
        pytest.param(erdos_renyi(40, 0.08, rng=20110930), id="erdos-renyi-sparse"),
        pytest.param(erdos_renyi(40, 0.25, rng=20110930), id="erdos-renyi-dense"),
        pytest.param(watts_strogatz(50, 4, 0.1, rng=7), id="watts-strogatz"),
        pytest.param(newman_watts(50, 4, 0.2, rng=11), id="newman-watts"),
        pytest.param(barabasi_albert(60, links_per_new_node=2, rng=2011), id="barabasi-albert"),
    ],
)
def test_minimum_time_score_is_the_eccentricity_on_generated_networks(network):
    """The identity is checked on the four network families of chapter 4,
    disconnected draws included."""
    scores = min_time_scores(network)
    assert scores.tolist() == eccentricities(network).tolist()

    if is_connected(network):
        assert scores.max() == diameter(network)
        assert (
            central_nodes_by_min_time(network).tolist()
            == np.flatnonzero(scores == scores.min()).tolist()
        )


# ---------------------------------------------------------------------------
# The common interface section 5.5.2 needs
# ---------------------------------------------------------------------------


def test_the_three_criteria_of_the_section_are_all_available(fig54):
    """Section 5.5.1 offers three ways of naming the most vulnerable node and
    section 5.5.2 ranks nodes by one of them to immunise.  All three produce the
    same shape, so they can be swapped."""
    assert set(CENTRALITY_CRITERIA) == {"node_removal", "min_time", "degree"}

    for name in CENTRALITY_CRITERIA:
        ranking = rank_nodes(fig54, name)
        assert ranking.criterion == name
        assert ranking.scores.shape == (fig54.n,)
        assert sorted(ranking.order.tolist()) == list(range(fig54.n))
        assert set(ranking.most_central.tolist()) <= set(ranking.order[: fig54.n].tolist())
        assert ranking.order[0] in ranking.most_central


def test_the_order_n_degree_plugs_into_the_same_interface(fig54):
    """Section 5.5.1.3's K_m(n) is the criterion section 5.5.2 immunises by;
    the wrapper gives that module's ranking node for node."""
    for n in (1, 2, 3):
        criterion = degree_order_n_criterion(n)
        ranking = rank_nodes(fig54, criterion)
        assert ranking.order.tolist() == k_order_ranking(fig54, n)
        assert ranking.scores.tolist() == [float(k_order(fig54, m, n)) for m in range(fig54.n)]


def test_the_degree_criterion_is_the_order_one_degree(fig54):
    """PDF p. 55: 'el grado k de orden 1 coincide con el grado k tal y como ha
    sido hasta ahora definido'.  Barabasi's criterion is therefore K_m(1)."""
    plain = rank_nodes(fig54, "degree")
    order_one = rank_nodes(fig54, degree_order_n_criterion(1))
    assert plain.scores.tolist() == order_one.scores.tolist()
    assert plain.order.tolist() == order_one.order.tolist()


def test_the_order_n_criterion_changes_the_winner_between_orders_one_and_two(fig54):
    """PDF p. 56: node 5 has the smaller degree and the larger order-2 degree,
    the answer of 5.5.1.1 and 5.5.1.2."""
    assert thesis_labels(rank_nodes(fig54, degree_order_n_criterion(1)).most_central) == [4, 6]
    assert thesis_labels(rank_nodes(fig54, degree_order_n_criterion(2)).most_central) == [5]


def test_ranking_orders_from_most_to_least_central_and_breaks_ties_by_index(fig54):
    """Both directions have to work: minimum time is best when small, degree and
    peeling depth when large."""
    by_time = rank_nodes(fig54, "min_time")
    assert by_time.higher_is_more_central is False
    assert thesis_labels(by_time.top(3)) == [5, 4, 6]
    assert thesis_labels(by_time.order) == [5, 4, 6, 1, 2, 3, 7, 8, 9]

    by_degree = rank_nodes(fig54, "degree")
    assert by_degree.higher_is_more_central is True
    assert thesis_labels(by_degree.top(3)) == [4, 6, 5]


def test_top_k_is_the_immunisation_prefix_of_a_large_network():
    """What section 5.5.2 consumes: an order over every node, no repeats, so
    that immunising q nodes means taking the first q."""
    network = barabasi_albert(120, links_per_new_node=2, rng=5)
    ranking = rank_nodes(network, "degree")
    order = ranking.order.tolist()

    assert sorted(order) == list(range(network.n))
    for q in (1, 5, 30, network.n):
        assert ranking.top(q).tolist() == order[:q]


def test_an_unknown_criterion_names_the_ones_that_exist(fig54):
    with pytest.raises(KeyError, match="unknown criterion"):
        rank_nodes(fig54, "betweenness")


def test_compare_criteria_can_be_restricted(fig54):
    rankings = compare_centrality_criteria(fig54, ["min_time", degree_order_n_criterion(2)])
    assert list(rankings) == ["min_time", "degree_order_2"]
    assert all(thesis_labels(r.most_central) == [5] for r in rankings.values())


# ---------------------------------------------------------------------------
# Properties both methods must have
# ---------------------------------------------------------------------------


def test_the_answers_follow_a_relabelling_of_the_nodes(fig54):
    """Neither method may depend on the node numbering: permuting the labels of
    fig. 54 must move the central node to wherever node 5 went."""
    perm = np.array([2, 5, 7, 0, 8, 1, 3, 4, 6])
    matrix = np.zeros((9, 9))
    matrix[np.ix_(perm, perm)] = fig54.matrix
    relabelled = Network(matrix)

    assert central_node_by_removal(relabelled) == perm[4]
    assert central_nodes_by_min_time(relabelled).tolist() == [perm[4]]
    assert min_time_scores(relabelled)[perm].tolist() == min_time_scores(fig54).tolist()


def test_nothing_in_the_section_is_stochastic(fig54):
    """PDF p. 43: 'un modelo simple de infeccion en el que no se introduce una
    cierta probabilidad de contagio'.  Neither method has a free parameter, a
    contagion probability or a random draw, so repeated calls must be identical
    and no entry point may take a seed.
    """
    assert min_time_scores(fig54).tolist() == min_time_scores(fig54).tolist()
    assert peel_by_minimum_degree(fig54).as_lists() == peel_by_minimum_degree(fig54).as_lists()

    forbidden = re.compile(r"^(rng|seed|p|probability|random_state)$")
    for name in pc.__all__:
        obj = getattr(pc, name)
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        offenders = [p for p in inspect.signature(obj).parameters if forbidden.match(p)]
        assert not offenders, f"{name} takes {offenders}, but section 5.5.1 has no free parameters"


def test_every_public_object_of_the_module_carries_its_provenance():
    """Every public object says which section it comes from."""
    for name in pc.__all__:
        obj = getattr(pc, name)
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        assert getattr(obj, "source", None) is not None, f"{name} is unsourced"
        assert obj.source.section.startswith("5.5.1"), name
