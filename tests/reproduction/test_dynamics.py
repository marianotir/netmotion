"""Claims made in sections 5.1, 5.2 and 5.3 of the thesis, as assertions.

Each test names the PDF page it checks.  The tests assert the algebra of the
update rules on small hand-computable graphs, the worked examples of the weight
constructions, the two endpoints of the fig. 23/24 sweep, and the qualitative
claims of each section as relations, with the test docstring saying which.
"""

from __future__ import annotations

import numpy as np
import pytest

from netmotion import Network
from netmotion.dynamics import (
    OPINION_RULES,
    THESIS_TYPE_VECTORS,
    assign_opinion_types,
    compare_weighted_unweighted,
    cycle_period,
    degree_extremes,
    equilibrium_sweep,
    equilibrium_time,
    hopfield_step,
    opinion_step,
    random_initial_states,
    run_hopfield,
    run_opinion,
    run_weighted_opinion,
    sgn_states,
    weights_from_degree_rank,
    weights_from_edge_ages,
    weights_from_growth,
)
from netmotion.networks import barabasi_albert, erdos_renyi, newman_watts, watts_strogatz

pytestmark = pytest.mark.reproduction


@pytest.fixture
def k4() -> Network:
    """The complete graph on four nodes; every local field is odd."""
    return Network.from_edges(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])


@pytest.fixture
def triangle_pendant() -> Network:
    """Triangle 0-1-2 with node 3 hanging off node 2; degrees [2, 2, 3, 1]."""
    return Network.from_edges(4, [(0, 1), (0, 2), (1, 2), (2, 3)])


# ---------------------------------------------------------------------------
# 5.1 The base Hopfield rule (PDF pp. 35-37)
# ---------------------------------------------------------------------------


def test_base_rule_is_the_sign_of_the_neighbour_sum(k4):
    """PDF p. 35: S_i(t+1) = sgn(sum_j S_j(t) M_ij).

    On K4 with two nodes up and two down every field is +-1, so the step is
    fixed by the printed equation alone and no convention enters.
    """
    assert hopfield_step(k4, [1, 1, -1, -1]).tolist() == [-1, -1, 1, 1]


def test_the_coupling_matrix_is_the_adjacency_matrix(triangle_pendant):
    """PDF p. 35: 'donde M es la matriz adyacente de la red'.

    The field of a node is the sum of its neighbours' spins.
    """
    states = np.array([1, -1, 1, -1])
    field = triangle_pendant.adjacency @ states.astype(float)
    assert field.tolist() == [0.0, 2.0, -1.0, 1.0]


def test_update_is_simultaneous_not_a_sequential_sweep():
    """PDF p. 35: the rule is written S_i(t+1) = f(S(t)).

    On a single edge a simultaneous update swaps the two states.
    """
    edge = Network.from_edges(2, [(0, 1)])
    assert hopfield_step(edge, [1, -1]).tolist() == [-1, 1]


def test_base_rule_reaches_a_fixed_point(triangle_pendant):
    """PDF p. 35: the attractor is reached 'cuando no existen mas cambios en
    los estados de los nodos'.

    Hand computation: from [+1, +1, +1, -1] the fields are [2, 2, 1, 1], so the
    whole network goes to +1 in one step and stays there.
    """
    run = run_hopfield(triangle_pendant, [1, 1, 1, -1], steps=5)
    assert run.states[1].tolist() == [1, 1, 1, 1]
    assert run.equilibrium_time == 1
    assert run.period == 1
    assert int(run.n_plus[-1]) == 4


def test_synchronous_update_admits_a_two_cycle(k4):
    """HopfieldRun reports a period as well as an equilibrium time.

    On K4 the state [+1, +1, -1, -1] maps to its own negation and back, with
    n_+ constant at 2.
    """
    run = run_hopfield(k4, [1, 1, -1, -1], steps=6)
    assert np.array_equal(run.states[2], run.states[0])
    assert np.array_equal(run.states[3], run.states[1])
    assert not np.array_equal(run.states[1], run.states[0])
    assert run.equilibrium_time is None
    assert run.period == 2
    assert run.n_plus.tolist() == [2] * 7


def test_all_plus_and_all_minus_are_both_fixed_points(triangle_pendant):
    """PDF p. 35: the attractor may be 'todos los nodos ... en estado positivo
    o en estado negativo'. Both consensus states are invariant."""
    for consensus in (1, -1):
        states = np.full(4, consensus, dtype=np.int8)
        assert hopfield_step(triangle_pendant, states).tolist() == states.tolist()


def test_zero_field_conventions_are_all_reachable(triangle_pendant):
    """The zero_rule options for a node whose field is zero.

    Node 0 of the triangle sees one neighbour up and one down, so its field is
    exactly zero. The default is the convention of types 6 and 7 (PDF p. 38):
    keep your state.
    """
    states = np.array([1, -1, 1, -1])
    assert (triangle_pendant.adjacency @ states.astype(float))[0] == 0.0

    assert int(hopfield_step(triangle_pendant, states, zero_rule="hold")[0]) == 1
    assert int(hopfield_step(triangle_pendant, states, zero_rule="positive")[0]) == 1

    flipped = np.array([-1, -1, 1, -1])
    assert (triangle_pendant.adjacency @ flipped.astype(float))[0] == 0.0
    assert int(hopfield_step(triangle_pendant, flipped, zero_rule="hold")[0]) == -1
    assert int(hopfield_step(triangle_pendant, flipped, zero_rule="positive")[0]) == 1
    assert int(hopfield_step(triangle_pendant, flipped, zero_rule="random", rng=0)[0]) in (-1, 1)


def test_isolated_node_never_changes_under_the_hold_convention():
    """An isolated node has field 0 at every instant, so under the hold
    convention it keeps its state (a consequence of the rule on PDF p. 35)."""
    lonely = Network.from_edges(3, [(0, 1)])
    run = run_hopfield(lonely, [1, -1, -1], steps=10)
    assert set(run.states[:, 2].tolist()) == {-1}


def test_the_rule_does_not_conserve_the_magnetisation(triangle_pendant):
    """One step of the rule moves n_+ from 3 to 4 on the triangle with a pendant."""
    run = run_hopfield(triangle_pendant, [1, 1, 1, -1], steps=1)
    assert run.n_plus.tolist() == [3, 4]


def test_initial_condition_is_sixty_forty():
    """PDF pp. 35-36: 'una red ... formada por 100 nodos cuando existen 60
    nodos en el estado +1 y 40 nodos en el estado -1 en el instante inicial'."""
    states = random_initial_states(100, 40, rng=7)
    assert int((states == 1).sum()) == 60
    assert int((states == -1).sum()) == 40


@pytest.mark.parametrize(
    "network",
    [
        erdos_renyi(60, 0.1, rng=1),
        watts_strogatz(60, 4, 0.1, rng=2),
        newman_watts(60, 4, 0.1, rng=3),
        barabasi_albert(60, links_per_new_node=2, rng=4),
    ],
    ids=["ER", "WS", "NW", "SF"],
)
def test_equilibrium_sweep_endpoints_are_topology_independent(network):
    """PDF pp. 36-37, figs. 23 and 24.

    The sweep starts from all +1 and raises the number of seeds at -1. Its two
    endpoints are the only points of those figures determined without knowing
    the realisation: a network entirely at +1 or entirely at -1 is a fixed
    point, so the equilibrium counts there are exactly N and 0.
    """
    n_plus_eq, n_minus_eq = equilibrium_sweep(network, steps=25, rng=0)
    assert n_plus_eq.shape == (network.n + 1,)
    assert float(n_plus_eq[0]) == float(network.n)
    assert float(n_minus_eq[0]) == 0.0
    assert float(n_plus_eq[-1]) == 0.0
    assert float(n_minus_eq[-1]) == float(network.n)


def test_equilibrium_sweep_shows_the_reported_transition():
    """PDF p. 36, fig. 23 and the text below it, QUALITATIVE.

    The thesis reports that the equilibrium count of -1 nodes rises with the
    initial count 'hasta llegar un punto en el cual el numero de nodos en
    estado -1 en equilibrio supera al numero de nodos en estado +1 lo que da
    lugar a una transicion'. What is asserted is that the sweep is dominated
    by +1 at the low end, by -1 at the high end, and crosses over in between.
    """
    network = erdos_renyi(60, 0.1, rng=5)
    n_plus_eq, n_minus_eq = equilibrium_sweep(network, steps=25, rng=5)

    low = n_plus_eq[:15].mean()
    high = n_plus_eq[-15:].mean()
    assert low > high, "the equilibrium must move towards -1 as seeds are added"
    assert low > network.n / 2 > high

    crossings = np.flatnonzero(np.diff(np.sign(n_plus_eq - n_minus_eq)) != 0)
    assert crossings.size >= 1, "the reported transition must actually occur"


def test_equilibrium_time_and_cycle_period_read_a_trace():
    """PDF p. 36, fig. 21: 'a partir de t=8 no existen variaciones'.

    The thesis reads its equilibrium instant off the plotted curve. The same
    reading, applied to a state trace rather than to a count, is what
    equilibrium_time does; cycle_period reports the period of the final orbit.
    """
    settled = np.array([[1, -1], [1, 1], [1, 1], [1, 1]])
    assert equilibrium_time(settled) == 1
    assert cycle_period(settled) == 1

    alternating = np.array([[1, -1], [-1, 1], [1, -1], [-1, 1]])
    assert equilibrium_time(alternating) is None
    assert cycle_period(alternating) == 2


def test_sgn_states_returns_only_the_two_thesis_values():
    """PDF p. 35: the state 'puede tener un valor +-1'."""
    field = np.array([3.0, -2.0, 0.0, 0.5])
    out = sgn_states(field, np.array([1, 1, -1, -1]))
    assert set(out.tolist()) <= {-1, 1}
    assert out.tolist() == [1, -1, -1, 1]


def test_states_outside_plus_minus_one_are_rejected(k4):
    with pytest.raises(ValueError, match=r"\+1 or -1"):
        hopfield_step(k4, [1, 0, -1, 1])


# ---------------------------------------------------------------------------
# 5.2 Opinion dynamics with seven kinds of individual (PDF pp. 37-40)
# ---------------------------------------------------------------------------


def test_there_are_exactly_seven_ways_of_forming_an_opinion():
    """PDF p. 37: 'a continuacion se muestran los 7 tipos de individuos, segun
    su forma de generar opiniones, que existiran en el modelo aqui
    desarrollado'."""
    assert len(OPINION_RULES) == 7


@pytest.mark.parametrize("panel", ["a", "b", "c"])
def test_the_three_thesis_type_vectors_are_transcribed_exactly(panel):
    """PDF p. 40: the vectors behind panels a, b and c of figs. 25-28 are
    [40, 4, 15, 5, 6, 15, 15], [10, 10, 25, 15, 10, 10, 20] and
    [5, 5, 25, 25, 10, 15, 15]."""
    expected = {
        "a": (40, 4, 15, 5, 6, 15, 15),
        "b": (10, 10, 25, 15, 10, 10, 20),
        "c": (5, 5, 25, 25, 10, 15, 15),
    }[panel]
    assert THESIS_TYPE_VECTORS[panel] == expected
    assert sum(THESIS_TYPE_VECTORS[panel]) == 100


@pytest.mark.parametrize("panel", ["a", "b", "c"])
def test_type_assignment_realises_the_percentages_exactly(panel):
    """PDF p. 38: 'un vector de la forma [40, 4, 15, 5, 6, 15, 15] que indica
    que en el modelo habra un 40% de nodos que generara su opinion como lo hace
    el grupo de opinion 1'.

    At the thesis's own N = 100 the percentages are counts, so the assignment
    is exact.
    """
    vector = THESIS_TYPE_VECTORS[panel]
    types = assign_opinion_types(100, vector, rng=0)
    assert [int((types == r).sum()) for r in range(1, 8)] == list(vector)


def test_type_assignment_rejects_a_vector_that_is_not_a_percentage():
    """PDF p. 38: the vector is a percentage split over the seven groups."""
    with pytest.raises(ValueError, match="sum to 100"):
        assign_opinion_types(100, [40, 4, 15, 5, 6, 15, 14], rng=0)
    with pytest.raises(ValueError, match="7"):
        assign_opinion_types(100, [40, 4, 15, 5, 6, 30], rng=0)


def test_type_1_is_exactly_the_section_5_1_rule(triangle_pendant):
    """PDF p. 37, rule 1: 'opinion de la mayoria de tu alrededor', written with
    the same formula as section 5.1.

    A network of nothing but type-1 individuals must therefore be numerically
    identical to the base rule, step for step.
    """
    start = np.array([1, -1, 1, -1])
    opinions = run_opinion(triangle_pendant, [100, 0, 0, 0, 0, 0, 0], states=start, steps=20)
    plain = run_hopfield(triangle_pendant, start, steps=20)
    assert np.array_equal(opinions.states, plain.states)


def test_type_2_is_the_negation_of_type_1_wherever_the_field_is_nonzero(k4):
    """PDF p. 37, rule 2: S_i(t+1) = (-1) sgn(sum_j S_j(t) M_ij).

    On K4 no field vanishes, so the contrarian result is the exact negation of
    the majority result.
    """
    start = np.array([1, 1, -1, -1])
    majority = opinion_step(k4, start, np.full(4, 1))
    contrarian = opinion_step(k4, start, np.full(4, 2))
    assert majority.tolist() == [-1, -1, 1, 1]
    assert contrarian.tolist() == [1, 1, -1, -1]
    assert np.array_equal(contrarian, -majority)


def test_types_3_and_4_copy_the_degree_extremes(triangle_pendant):
    """PDF p. 38, rules 3 and 4: 'opinion del nodo con mayor k' and 'con menor
    k', quantified over d perteneciente a la red R.

    Degrees are [2, 2, 3, 1], so the hub is node 2 and the least connected node
    is node 3; every type-3 node adopts the first's opinion and every type-4
    node the second's, regardless of its own neighbourhood.
    """
    hub_of, sink_of = degree_extremes(triangle_pendant)
    assert hub_of.tolist() == [2, 2, 2, 2]
    assert sink_of.tolist() == [3, 3, 3, 3]

    start = np.array([1, -1, 1, -1])
    assert opinion_step(triangle_pendant, start, np.full(4, 3)).tolist() == [1, 1, 1, 1]
    assert opinion_step(triangle_pendant, start, np.full(4, 4)).tolist() == [-1, -1, -1, -1]


def test_types_3_and_4_have_a_neighbourhood_reading_too(triangle_pendant):
    """scope='neighbourhood' is offered as an option alongside the global
    default.

    Under the local scope the pendant node 3 watches its only neighbour both
    ways, and node 2 -- itself the hub -- watches the least connected node it
    can see.
    """
    hub_of, sink_of = degree_extremes(triangle_pendant, scope="neighbourhood")
    assert hub_of.tolist() == [2, 2, 0, 2]
    assert sink_of.tolist() == [1, 0, 3, 2]


def test_types_6_and_7_read_the_global_majority_and_ignore_the_topology():
    """PDF p. 38, rules 6 and 7: S_i(t+1) = -sgn(sum_j S_j(t)) and +sgn of it.

    Neither rule mentions M, so an empty network must give the same answer as
    a connected one.
    """
    start = np.array([1, 1, 1, -1])
    connected = Network.from_edges(4, [(0, 1), (1, 2), (2, 3)])
    empty = Network.empty(4)
    for network in (connected, empty):
        assert opinion_step(network, start, np.full(4, 6)).tolist() == [-1, -1, -1, -1]
        assert opinion_step(network, start, np.full(4, 7)).tolist() == [1, 1, 1, 1]


def test_types_6_and_7_hold_when_the_network_is_tied(triangle_pendant):
    """PDF p. 38: 'si sum_j S_j(t) = 0 => S_i(t+1) = S_i(t)'.

    The convention is written into the rule, so it is asserted exactly.
    """
    tied = np.array([1, 1, -1, -1])
    assert int(tied.sum()) == 0
    assert opinion_step(triangle_pendant, tied, np.full(4, 6)).tolist() == tied.tolist()
    assert opinion_step(triangle_pendant, tied, np.full(4, 7)).tolist() == tied.tolist()


def test_type_5_is_a_fair_coin_independent_of_the_network():
    """PDF p. 38, rule 5: 'S_i(t+1) = (-1)^b donde b es un numero entero
    aleatorio par o impar'.

    A fair coin per node per instant is implemented, so a large network of type-5 individuals must land near half and half
    whatever its initial opinions, and two seeds must disagree.
    """
    network = Network.empty(2000)
    types = np.full(2000, 5)
    drawn = opinion_step(network, np.ones(2000, dtype=np.int8), types, rng=0)
    assert 0.45 < float((drawn == 1).mean()) < 0.55
    other = opinion_step(network, np.ones(2000, dtype=np.int8), types, rng=1)
    assert not np.array_equal(drawn, other)


def test_the_opinion_update_is_synchronous():
    """PDF p. 37: all seven rules read S(t).

    Two type-1 individuals joined by one edge swap opinions.
    """
    edge = Network.from_edges(2, [(0, 1)])
    assert opinion_step(edge, [1, -1], np.array([1, 1])).tolist() == [-1, 1]


def test_personalities_are_assigned_once_and_never_change(triangle_pendant):
    """PDF p. 38: 'la asociacion de estos tipos de opiniones con los nodos de
    la red se realizara de forma aleatoria' -- once, before the run."""
    run = run_opinion(triangle_pendant, [0, 0, 25, 25, 0, 25, 25], steps=30, n_in_favour=2, rng=0)
    assert sorted(run.types.tolist()) == [3, 4, 6, 7]
    again = assign_opinion_types(4, [0, 0, 25, 25, 0, 25, 25], rng=0)
    assert np.array_equal(run.types, again)


def test_the_run_covers_the_hundred_instants_the_thesis_plots():
    """PDF pp. 38-40, figs. 25-28: the horizon plotted is t = 1..100 at
    N = 100 with 60 opinions in favour at the initial instant."""
    network = erdos_renyi(100, 0.1, rng=0)
    run = run_opinion(network, THESIS_TYPE_VECTORS["a"], rng=0)
    assert run.states.shape == (100, 100)
    assert int(run.n_plus[0]) == 60
    assert int(run.n_minus[0]) == 40


def test_the_opinion_model_does_not_settle():
    """PDF p. 38, fig. 25 caption, QUALITATIVE.

    'En las tres graficas se observa que los estados +-1 oscilan con lo que no
    se termina de alcanzar el equilibrio.' Over the thesis's horizon and with
    its vectors, the state keeps changing and n_+ keeps moving.
    """
    network = barabasi_albert(100, links_per_new_node=2, rng=2)
    for vector in THESIS_TYPE_VECTORS.values():
        run = run_opinion(network, vector, rng=3)
        assert run.equilibrium_time is None, "the thesis reports no equilibrium is reached"
        assert int(np.abs(np.diff(run.n_plus)).sum()) > 0


def test_opposed_tendencies_in_equal_proportion_oscillate_most():
    """PDF p. 40, the author's own table, QUALITATIVE.

    Of the three vectors the thesis singles out the third, [5, 5, 25, 25, 10,
    15, 15], because the opposed ways of forming an opinion appear 'en la red
    en la misma proporcion' so that none can win, against the first, which is
    dominated by the 40% who follow their neighbours, for which 'las posibles
    fluctuaciones ... seran pequenas'. The claim is about relative amplitude only, so
    that is what is asserted, averaged over several networks and seeds.
    """

    def amplitude(vector, seed):
        network = erdos_renyi(100, 0.1, rng=seed)
        run = run_opinion(network, vector, rng=seed)
        # The transient is dropped: the thesis's claim is about the band the
        # curves settle into, not about how they get there.
        return float(run.n_plus[20:].std())

    balanced = np.mean([amplitude(THESIS_TYPE_VECTORS["c"], s) for s in range(6)])
    majority_led = np.mean([amplitude(THESIS_TYPE_VECTORS["a"], s) for s in range(6)])
    assert balanced > majority_led


# ---------------------------------------------------------------------------
# 5.3 Opinion dynamics weighted by the age of the relationship (PDF pp. 40-43)
# ---------------------------------------------------------------------------


def test_weights_are_the_age_of_the_link():
    """PDF p. 41: 'si el elemento de la matriz adyacente adquiere el valor
    M_ij = 1 desde el instante t_n hasta el instante final t_f implica que el
    elemento W_ij de la matriz de pesos tomara el valor W_ij = t_f - t_n
    mientras que W_ij = 0 si M_ij = 0'.

    The last clause of that sentence is why the two links born at t_f carry
    weight zero, as the printed formula gives.
    """
    w = weights_from_edge_ages(4, {(0, 1): 2, (0, 2): 3, (1, 3): 4, (0, 3): 4}, final_time=4)
    assert float(w.matrix[0, 1]) == 2.0
    assert float(w.matrix[0, 2]) == 1.0
    assert float(w.matrix[1, 3]) == 0.0
    assert float(w.matrix[0, 3]) == 0.0
    assert float(w.matrix[2, 3]) == 0.0
    assert np.array_equal(w.matrix, w.matrix.T)
    assert np.all(np.diag(w.matrix) == 0.0)


def test_the_weight_matrix_never_invents_a_link():
    """PDF p. 41: 'W_ij = 0 si M_ij = 0'.

    Checked for both weight constructions of section 5.3.
    """
    for network in (erdos_renyi(40, 0.15, rng=0), watts_strogatz(40, 4, 0.2, rng=1)):
        w = weights_from_degree_rank(network)
        assert np.all(w.matrix[network.adjacency == 0] == 0.0)

    grown = barabasi_albert(40, links_per_new_node=2, initial_nodes=3, rng=2)
    w = weights_from_growth(grown, initial_nodes=3)
    assert np.all(w.matrix[grown.adjacency == 0] == 0.0)


def test_grown_weights_date_a_link_by_its_younger_endpoint():
    """PDF p. 41: the scale-free network is the one with 'una componente
    temporal en su construccion', so its weights are true link ages.

    A link exists only once both its endpoints do, so its age is set by the
    later arrival. The seed network's own links are the oldest in the network
    and the last node to arrive owns nothing but weightless links.
    """
    grown = barabasi_albert(30, links_per_new_node=2, initial_nodes=3, rng=4)
    w = weights_from_growth(grown, initial_nodes=3)
    t_final = 30 - 3

    assert float(w.matrix.max()) == float(t_final)
    assert float(w.matrix[0, 1]) == float(t_final)
    assert float(w.matrix[29].max()) == 0.0

    for i, j, _ in grown.edges():
        arrival = max(max(i - 3 + 1, 0), max(j - 3 + 1, 0))
        assert float(w.matrix[i, j]) == float(t_final - arrival)


def test_degree_rank_weights_follow_the_worked_formula(triangle_pendant):
    """PDF p. 41: 'se ordenaran los nodos de mayor grado k a menor grado k',
    t_i = 1 for the largest and t_j = N for the smallest, and
    W_ij = N - t_i if t_i > t_j, N - t_j if t_i < t_j.

    Degrees are [2, 2, 3, 1], so with the index tie-break the birth instants
    are [2, 3, 1, 4] and the weights follow arithmetically.
    """
    w = weights_from_degree_rank(triangle_pendant)
    assert float(w.matrix[0, 1]) == 4 - max(2, 3)
    assert float(w.matrix[0, 2]) == 4 - max(2, 1)
    assert float(w.matrix[1, 2]) == 4 - max(3, 1)
    assert float(w.matrix[2, 3]) == 4 - max(1, 4)
    assert float(w.matrix[0, 3]) == 0.0
    assert float(w.matrix[1, 3]) == 0.0


def test_the_last_ranked_node_has_zero_weight_links():
    """The last node in the ranking has t = N, and N - max(t_i, t_j) = 0 for
    every link it owns, so it keeps its opinion (a consequence of the rule on
    PDF p. 41).
    """
    network = barabasi_albert(50, links_per_new_node=2, rng=6)
    degrees = network.degrees()
    youngest = int(np.argmin(degrees))
    assert (degrees == degrees[youngest]).sum() >= 1

    w = weights_from_degree_rank(network)
    # With the index tie-break the last node in the ranking is the
    # minimum-degree node of highest index.
    silenced = int(np.flatnonzero(degrees == degrees.min())[-1])
    assert float(np.abs(w.matrix[silenced]).max()) == 0.0

    start = random_initial_states(50, 20, rng=6)
    run = run_weighted_opinion(w, start, steps=20)
    assert set(run.states[:, silenced].tolist()) == {int(start[silenced])}


def test_weighted_rule_worked_example():
    """PDF p. 41: S_i(t+1) = sgn(sum_j S_j(t) W_ij), on the degree-rank weights
    of the triangle-plus-pendant network.

    From [+1, -1, -1, +1] the weighted fields are [-3, 0, 1, 0]: node 0 follows
    its heavily weighted neighbours, node 2 follows its oldest tie, and nodes 1
    and 3 have no information and hold.
    """
    w = Network(
        np.array([[0.0, 1.0, 2.0, 0.0], [1.0, 0.0, 1.0, 0.0], [2.0, 1.0, 0.0, 0.0], [0.0] * 4])
    )
    states = np.array([1, -1, -1, 1])
    assert (w.matrix @ states.astype(float)).tolist() == [-3.0, 0.0, 1.0, 0.0]
    assert run_weighted_opinion(w, states, steps=1).states[1].tolist() == [-1, -1, 1, 1]


def test_the_weighted_rule_is_the_unweighted_rule_when_W_equals_M(triangle_pendant):
    """PDF p. 41: section 5.3 reuses the section 5.1 equation verbatim, with W
    substituted for M.

    So setting W = M must reproduce the section 5.1 trace exactly; W is used
    without normalisation.
    """
    start = np.array([1, -1, 1, -1])
    as_adjacency = Network(triangle_pendant.adjacency)
    weighted = run_weighted_opinion(as_adjacency, start, steps=10)
    plain = run_hopfield(triangle_pendant, start, steps=10)
    assert np.array_equal(weighted.states, plain.states)


def test_weights_change_the_dynamics(triangle_pendant):
    """PDF p. 40: the whole point of the section is that 'las relaciones entre
    nodos seran distintas dependiendo del peso asociado a la rama que une a los
    nodos'.

    Under M the network reaches consensus in one step; under the tie-age
    weights the pendant node is weightless and the state does not move.
    """
    both = compare_weighted_unweighted(
        triangle_pendant, weights_from_degree_rank(triangle_pendant), [1, 1, 1, -1], steps=6
    )
    assert both.without_weights.n_plus.tolist() == [3, 4, 4, 4, 4, 4, 4]
    assert both.with_weights.n_plus.tolist() == [3, 3, 3, 3, 3, 3, 3]


def test_the_comparison_uses_one_initial_condition_for_both_runs():
    """PDF p. 42: 'la evolucion de las opiniones se realizara sobre la red con
    pesos y sin pesos', from 52 nodes at +1 and 48 at -1.

    The two arms must start from the same opinions, otherwise the comparison
    measures the initial condition rather than the weighting.
    """
    network = erdos_renyi(100, 0.08, rng=8)
    start = random_initial_states(100, 48, rng=8)
    both = compare_weighted_unweighted(network, weights_from_degree_rank(network), start, steps=19)
    assert np.array_equal(both.with_weights.states[0], both.without_weights.states[0])
    assert int(both.with_weights.n_plus[0]) == 52
    assert int(both.without_weights.n_minus[0]) == 48
    assert both.with_weights.states.shape == (20, 100)


def test_weights_do_not_systematically_bring_equilibrium_forward():
    """PDF p. 42, the conclusion of section 5.3, QUALITATIVE.

    'De esto se concluye que la introduccion de pesos no provoca que se alcance
    antes el equilibrio entre las opiniones con lo que la evolucion de estas
    depende de los estados iniciales de los nodos.'

    The thesis reports the weighted run settling first for one initial
    condition and the unweighted run settling first for another, on the same
    networks. That is a claim about the absence of a systematic effect, so what
    is asserted is that over a set of initial conditions neither arm wins
    every time.
    """
    verdicts = []
    for seed in range(12):
        network = erdos_renyi(80, 0.06, rng=seed)
        both = compare_weighted_unweighted(
            network,
            weights_from_degree_rank(network),
            random_initial_states(80, 38, rng=100 + seed),
            steps=40,
        )
        weighted = both.with_weights.equilibrium_time
        plain = both.without_weights.equilibrium_time
        if weighted is not None and plain is not None and weighted != plain:
            verdicts.append(weighted < plain)

    assert verdicts, "no pair of runs settled, so the comparison says nothing"
    assert set(verdicts) == {True, False}, (
        "the thesis reports that neither arm settles first consistently: which "
        "one does depends on the initial states"
    )


def test_the_seven_opinion_types_are_not_carried_into_section_5_3():
    """PDF p. 41: section 5.3 writes one rule for every node, the 5.1 base
    rule with W in place of M, so run_weighted_opinion takes no type vector.
    """
    import inspect

    import netmotion.dynamics.weighted_opinion as weighted_module

    parameters = inspect.signature(run_weighted_opinion).parameters
    assert "percentages" not in parameters
    assert "types" not in parameters
    assert not any("opinion_types" in name for name in weighted_module.__all__)
