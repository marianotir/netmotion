"""Section 5.5.2, *Inmunizacion frente a infeccion*, as assertions.

Thesis PDF pp. 56-63.  Each test names the page it is checking.

Three kinds of assertion appear, and each test says which kind it is:

* **exact**, for the hand-checkable example graph of Fig. 54 and for the two
  numbers the thesis prints in words (:math:`\\beta = -0.023` and
  :math:`\\alpha \\approx 0.9`), which are asserted against
  :data:`~netmotion.propagation.immunity.THESIS_SCALE_FREE_100` rather than
  against a simulation;
* **qualitative**, for the claims of the section -- that immunising more nodes shrinks and slows the epidemic, that ranking by
  degree beats ranking at random, that :math:`K(1)` needs the fewest immune
  nodes of the eight orders the thesis sweeps, and that the eight orders give
  values of :math:`\\beta` of one size;
* **convention**, for the two conventions the section's curves use: how
  :math:`\\alpha` is read off each curve, as in section 5.4.1, and the order
  tied nodes are immunised in.

The networks are fixed realisations produced from stated seeds, so every test is
deterministic even though the model is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from netmotion import Network
from netmotion.networks import barabasi_albert, ring_lattice, watts_strogatz
from netmotion.propagation import propagate, propagation_speed
from netmotion.propagation.immunity import (
    CRITERION_ORIGIN,
    FIG_60_HORIZON,
    IMMUNISATION_CRITERIA,
    THESIS_BEST_ORDER_BY_NETWORK,
    THESIS_SCALE_FREE_100,
    _k_order_n,
    beta_factor,
    compare_immunisation_criteria,
    compare_orders,
    immunisation_ranking,
    immunisation_sweep,
    nodes_to_extinction,
)
from netmotion.provenance import Origin

pytestmark = pytest.mark.reproduction


# The nine-node example graph of Fig. 54 (PDF p. 54), re-indexed from zero:
# thesis node k is index k - 1.  Hubs 4 and 6 become indices 3 and 5, and the
# bridge node 5 -- the thesis's central node -- becomes index 4.
FIG_54_EDGES = [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (5, 8)]


@pytest.fixture(scope="module")
def fig_54() -> Network:
    return Network.from_edges(9, FIG_54_EDGES, name="Fig. 54")


@pytest.fixture(scope="module")
def scale_free_60() -> Network:
    """One fixed scale-free realisation, standing in for the thesis's N = 100.

    The thesis works at N = 100 (PDF p. 56).  Sixty nodes keeps the full
    seed-by-immune-count sweep, which is quadratic in N, inside a few seconds
    per criterion while leaving the degree heterogeneity that the whole
    selective-immunisation argument depends on.
    """
    return barabasi_albert(60, 2, rng=5)


@pytest.fixture(scope="module")
def sf60_orders(scale_free_60):
    """The Figs. 59 and 63 experiment: one sweep per order n of K(n), n = 1..8."""
    return compare_orders(scale_free_60, range(1, 9))


@pytest.fixture(scope="module")
def sf60_criteria(scale_free_60):
    return compare_immunisation_criteria(scale_free_60, rng=5)


# ---------------------------------------------------------------------------
# What immunity is (PDF p. 54)
# ---------------------------------------------------------------------------


def test_an_immune_node_neither_catches_nor_passes_on_the_infection():
    """PDF p. 54 defines an immune node as one frozen at S_n(t) = +1.

    Exact.  On a path, immunising the middle node must stop the infection dead
    at the immune node rather than merely delaying it, because a node that is
    never infected can never satisfy the transmission condition for its own
    neighbours.
    """
    path = Network.from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    run = propagate(path, 0, immune=[2])
    assert np.isfinite(run.infected_at).sum() == 2
    assert run.infected_at[:2].tolist() == [1.0, 2.0]
    assert not np.isfinite(run.infected_at[2:]).any()


def test_the_seed_is_never_immunised(fig_54):
    """PDF p. 56: the recipe restarts the infection at node 1 every round.

    Reproduction decision, not a thesis claim.  The origin of the infection is
    skipped when the ranking reaches it, so at least one node is always
    infected however far the sweep runs.  Were it not skipped, the mean number
    of immune nodes needed would be the mean rank of the seed, (N+1)/2 for
    every criterion, and could never separate one criterion from another.
    """
    sweep = immunisation_sweep(fig_54, "degree")
    assert sweep.infected.min() >= 1.0
    assert sweep.infected[-1] == 1.0


def test_extinction_means_the_infection_never_leaves_the_seed(fig_54):
    """PDF p. 57: *"la infeccion se ha frenado ... se reduce a cero"*, *"de
    modo aproximativo"*.

    Reproduction decision.  The seed is never immunised, so the default
    ``threshold=1`` counts the infection as stopped once it never leaves the
    seed; with ``threshold=0`` no seed reaches extinction.
    """
    strict = immunisation_sweep(fig_54, "degree", threshold=0)
    assert np.isinf(strict.extinction).all()
    assert np.isinf(strict.mean_nodes_to_extinction)


# ---------------------------------------------------------------------------
# The rankings (PDF pp. 54-56)
# ---------------------------------------------------------------------------


def test_degree_criterion_picks_the_two_hubs_of_fig_54(fig_54):
    """PDF p. 54: *"el nodo mas vulnerable deberia ser el 6 y el 4 puesto que
    tienen el mayor grado k"* -- Barabasi's criterion, and the selective
    immunisation of [38] that section 5.5.2 starts from.

    Exact.  Indices 3 and 5 are the thesis's nodes 4 and 6.  They have the same
    degree, so the criterion names the pair and the tie order decides which of
    the two is protected first; the pair is what is asserted.
    """
    assert sorted(immunisation_ranking(fig_54, "degree")[:2].tolist()) == [3, 5]


@pytest.mark.parametrize("criterion", ["k_order_n", "min_propagation_time", "node_removal"])
def test_the_other_criteria_immunise_the_central_node_first(fig_54, criterion):
    """PDF pp. 54-56: all three methods of section 5.5.1 name node 5 central.

    Exact.  The minimum-propagation-time score (PDF p. 55) gives node 5 two
    iterations against three for the hubs; the node-elimination peeling
    (PDF pp. 54-55) leaves node 5 as the survivor; and K_5(2) = 8 covers every
    branch of the network (PDF p. 56).  Index 4 is the thesis's node 5.
    """
    assert int(immunisation_ranking(fig_54, criterion, order=2)[0]) == 4


def test_k_of_order_one_is_the_degree(fig_54, scale_free_60):
    """PDF p. 55: K_m(1) is the number of branches leaving m, i.e. its degree.

    Exact.  This is what makes the thesis's K(n) sweep a generalisation of the
    selective immunisation it borrows, rather than a different idea: n = 1 must
    reproduce the degree ranking exactly.
    """
    for network in (fig_54, scale_free_60):
        by_degree = immunisation_ranking(network, "degree")
        by_k1 = immunisation_ranking(network, "k_order_n", order=1)
        assert by_degree.tolist() == by_k1.tolist()


def test_k_of_order_two_matches_the_printed_values(fig_54):
    """PDF p. 56 prints exactly three values: K_1(2)=4, K_4(2)=5, K_5(2)=8.

    Exact.  :mod:`netmotion.metrics.higher_order_degree` counts the distinct
    branches with at least one end within distance n-1 of m.  This checks it
    through the accessor the ranking calls.  Indices 0, 3 and 4 are thesis
    nodes 1, 4 and 5.
    """
    k2 = _k_order_n(fig_54, 2)
    assert (k2[0], k2[3], k2[4]) == (4.0, 5.0, 8.0)
    assert _k_order_n(fig_54, 1).tolist() == fig_54.degrees().tolist()
    # "existen un total de 8 ramas" (PDF p. 56): node 5 is the one that knows
    # them all at order 2, which is the thesis's argument for calling it central.
    assert k2[4] == fig_54.n_edges == 8


def test_criterion_origin_separates_thesis_criteria_from_2026_additions():
    """PDF p. 56: section 5.5.2 ranks on K(n) and on nothing else.

    Provenance.  Random immunisation and the two section-5.5.1 centrality
    scores are offered through the same interface and are marked as
    extensions.
    """
    assert set(CRITERION_ORIGIN) == set(IMMUNISATION_CRITERIA)
    assert CRITERION_ORIGIN["degree"] is Origin.THESIS_2011
    assert CRITERION_ORIGIN["k_order_n"] is Origin.THESIS_2011
    for new in ("random", "min_propagation_time", "node_removal"):
        assert CRITERION_ORIGIN[new] is Origin.EXTENSION


# ---------------------------------------------------------------------------
# The two measured quantities (PDF pp. 57-59)
# ---------------------------------------------------------------------------


def test_infection_size_never_grows_as_more_nodes_are_immunised(scale_free_60):
    """PDF p. 57, Fig. 58: *"decrece la cantidad de nodos infectados a medida
    que aumenta el numero de nodos inmunizados"*.

    Qualitative, and here provably exact: under the monotone rule the infection
    reaches the seed's component, and removing a node can only shrink it.  The
    sweep must therefore be non-increasing at every step, and end with the
    infection confined to the seed.
    """
    sweep = immunisation_sweep(scale_free_60, "degree")
    assert np.all(np.diff(sweep.infected) <= 0.0)
    assert sweep.infected[0] == float(scale_free_60.n)
    assert sweep.infected[-1] == 1.0


def test_propagation_speed_falls_as_more_nodes_are_immunised(scale_free_60):
    """PDF p. 58, Figs. 60-61: *"la velocidad de la propagacion de la infeccion
    se reduce a medida que aumenta el numero de nodos inmunizados"*.

    Qualitative.  Fig. 61 is a clean decay from about 1.25 to a floor, with no
    hump anywhere, and the reproduction decays likewise: it falls at all but a
    few of the fifty-nine steps, and those rise by less than two hundredths, the
    size of the wiggles in the printed Figs. 61-62.
    """
    sweep = immunisation_sweep(scale_free_60, "degree")
    assert sweep.speed[0] > sweep.speed[10] > 0.0
    assert sweep.speed[-1] == 0.0
    assert sweep.beta < 0.0

    steps = np.diff(sweep.speed)
    assert (steps > 0.0).sum() <= 8
    assert steps.max() < 0.02


def test_alpha_is_read_as_in_section_5_4_1(scale_free_60):
    """PDF p. 58: *"se repetiran los calculos antes expuestos"*.

    Convention, and the one the whole beta comparison turns on.  Each curve's
    alpha is the one of section 5.4.1, fitted over the first instants before
    the infection has reached every node, so with nobody immune the sweep
    starts at exactly the section 5.4.1 <alpha> of the network.  Reading every
    curve over the common t = 1..8 axis of Fig. 60 instead starts lower and
    decays about twice as fast.
    """
    sweep = immunisation_sweep(scale_free_60, "degree")
    assert sweep.window == "first"
    assert sweep.horizon is None

    section_5_4_1 = np.mean(
        [propagation_speed(propagate(scale_free_60, s).curve, window="first") for s in range(60)]
    )
    assert sweep.speed[0] == pytest.approx(section_5_4_1)

    common_axis = immunisation_sweep(
        scale_free_60, "degree", horizon=FIG_60_HORIZON, window="to_saturation"
    )
    assert common_axis.speed[0] < sweep.speed[0]
    assert common_axis.beta < 1.5 * sweep.beta


def test_a_ring_ranks_nothing_so_the_tie_order_is_the_whole_ranking():
    """PDF p. 56: *"ordenar los nodos de mayor a menor grado"*.

    Convention.  On a ring lattice every node has the same K(n) at every order,
    so the criterion separates nobody and whatever breaks the tie decides which
    nodes are protected.  The node index is not a neutral choice there: it is
    the ring's own spatial order, so immunising down it protects a contiguous
    arc, which severs the ring at a third fewer immune nodes than a drawn order
    needs.  That saving belongs to the labelling, not to the criterion, which is
    why a drawn tie order is the default.
    """
    ring = ring_lattice(60, 4)
    orders = [1, 4, 8]
    swept = {
        "index": compare_orders(ring, orders, ties="index"),
        "shuffle": compare_orders(ring, orders, ties="shuffle", rng=5),
    }
    by_index = [swept["index"][o].mean_nodes_to_extinction for o in orders]
    drawn = [swept["shuffle"][o].mean_nodes_to_extinction for o in orders]

    # No order can beat another when every node scores alike, either way round.
    assert len(set(np.round(by_index, 6))) == 1
    assert len(set(np.round(drawn, 6))) == 1
    assert max(by_index) < min(drawn)


def test_the_high_orders_cost_more_on_a_small_world_network():
    """PDF pp. 60-61, Figs. 65-66: the WS and NW curves rise with the order n.

    Qualitative, and the visible consequence of the tie order.  A Watts-Strogatz
    network at low rewiring is a ring with a few links moved, so its high orders
    tie almost completely and inherit whatever the tie rule is.  Immunising down
    the node index makes them look like the cheapest criterion of the eight;
    under a drawn tie order they are dearer than K(1), which is the direction
    the thesis's curves run.
    """
    g = watts_strogatz(60, 4, 0.1, rng=5)
    drawn = compare_orders(g, [1, 8], rng=5)
    by_index = compare_orders(g, [1, 8], ties="index")

    assert drawn[8].mean_nodes_to_extinction > drawn[1].mean_nodes_to_extinction
    assert by_index[8].mean_nodes_to_extinction < by_index[1].mean_nodes_to_extinction


def test_beta_factor_recovers_a_known_exponent():
    """PDF pp. 58-59: <alpha>_cn is fitted to B exp(beta cn).

    Exact, on synthetic data: the estimator must return the exponent it was
    given, so that a beta reported for a network is a property of the network
    and not of the fitting code.
    """
    counts = np.arange(1, 41)
    assert beta_factor(counts, 1.5 * np.exp(-0.023 * counts)) == pytest.approx(-0.023)
    with pytest.raises(ValueError, match="fewer than two"):
        beta_factor(np.array([1, 2, 3]), np.array([0.0, 0.0, np.nan]))


def test_nodes_to_extinction_on_a_star_is_the_hand_computable_case():
    """PDF p. 57 defines the Fig. 59 quantity as a mean over seed nodes.

    Exact.  On a five-node star, immunising the hub stops an infection started
    at any of the four leaves at once, while an infection started at the hub is
    only stopped once all four leaves are immune, since the seed is never
    immunised.  The mean is (4x1 + 4)/5 = 1.6.
    """
    star = Network.from_edges(5, [(0, 1), (0, 2), (0, 3), (0, 4)])
    assert nodes_to_extinction(star, "degree") == pytest.approx(1.6)


def test_stopped_seeds_contribute_a_zero_rate_by_default(scale_free_60):
    """PDF p. 58: *"tendremos N factores alpha ... uno por cada nodo n"*.

    Reproduction decision.  Every seed yields a factor, so with
    ``stopped="zero"`` a seed whose infection never moves contributes
    alpha = 0.  ``stopped="skip"`` leaves those seeds out of the average.
    """
    zeroed = immunisation_sweep(scale_free_60, "degree", stopped="zero")
    skipped = immunisation_sweep(scale_free_60, "degree", stopped="skip")
    assert zeroed.beta < skipped.beta
    assert zeroed.speed[-1] == 0.0
    assert not np.isfinite(skipped.speed[-1])


# ---------------------------------------------------------------------------
# What the thesis reported (PDF pp. 57-59, 62-63)
# ---------------------------------------------------------------------------


def test_reported_scale_free_values_are_the_printed_ones():
    """PDF pp. 58-59: the two numbers section 5.5.2 states in words.

    Exact, against the recorded report rather than against a simulation:
    *"se obtiene beta = -0.023"* for the 100-node scale-free network with K(1)
    as the criterion, and *"si inmunizamos la red con los 10 nodos con mayor
    grado K(1) la infeccion se propagara en media de forma exponencial con un
    factor alpha aproximadamente 0.9"*.
    """
    assert THESIS_SCALE_FREE_100.n == 100
    assert THESIS_SCALE_FREE_100.beta[0] == pytest.approx(-0.023)
    assert THESIS_SCALE_FREE_100.alpha_at_ten_immune == pytest.approx(0.9)
    assert len(THESIS_SCALE_FREE_100.beta) == len(THESIS_SCALE_FREE_100.nodes_to_zero) == 8


def test_the_reported_curves_both_pick_K1():
    """PDF pp. 57 and 59: the section's conclusion, stated twice.

    Exact, against the digitised Figs. 59 and 63.  *"K(1) es el criterio
    adecuado para inmunizar la red ya que se necesita inmunizar menos nodos
    para reducir la infeccion a cero"* (p. 57), and *"el grado K(1) es el
    parametro adecuado ... ya que |beta| es mayor"* (p. 59).  The two measures
    agree.
    """
    assert THESIS_SCALE_FREE_100.best_order_by_size == 1
    assert THESIS_SCALE_FREE_100.best_order_by_speed == 1


def test_the_best_order_depends_on_how_ordered_the_network_is():
    """PDF pp. 62-63: the section's closing claim.

    Exact, against the recorded report.  K(1) wins for the four random network
    families at N = 50, 75, 100 (Figs. 64-67), but for the ordered networks of
    Fig. 68, with a defined structure, the thesis reads K(3) instead, and concludes *"el criterio mas adecuado para
    inmunizar una red dependera del grado de ordenamiento de esta"*.
    """
    random_families = ["erdos_renyi", "watts_strogatz", "newman_watts", "scale_free"]
    assert all(THESIS_BEST_ORDER_BY_NETWORK[f] == 1 for f in random_families)
    assert THESIS_BEST_ORDER_BY_NETWORK["ordered"] == 3


# ---------------------------------------------------------------------------
# Reproducing the claims on a fresh network (PDF pp. 57, 62)
# ---------------------------------------------------------------------------


def test_K1_needs_the_fewest_immune_nodes_of_the_eight_orders(sf60_orders):
    """PDF p. 62: *"el criterio mas adecuado para inmunizar las redes ER, WS,
    NW, SF, es el uso del K(1), ya que ... se corta para un menor numero de
    nodos inmunizados que cuando otro K(n) es utilizado como criterio"*.

    Qualitative, reproduced.  On this scale-free realisation the order-1 sweep
    extinguishes the infection with strictly fewer immune nodes than any of
    n = 2..8, which is the thesis's Fig. 59 result.
    """
    needed = {n: sweep.mean_nodes_to_extinction for n, sweep in sf60_orders.items()}
    assert min(needed, key=needed.get) == 1
    assert all(needed[1] < needed[n] for n in range(2, 9))


def test_every_order_slows_the_infection(sf60_orders):
    """PDF p. 59, Figs. 62-63: beta is negative for every K(n) used as a criterion.

    Qualitative, reproduced.  This is the claim the whole beta construction
    rests on -- that immunising by any of these criteria makes the epidemic
    exponentially slower in the number of immune nodes.
    """
    assert all(sweep.beta < 0.0 for sweep in sf60_orders.values())


def test_the_eight_orders_give_betas_of_one_size(sf60_orders):
    """PDF p. 59, Fig. 63: beta for K(1) ... K(8), between -0.019 and -0.023.

    Qualitative, reproduced.  The eight orders slow the infection at rates of
    the same size: in Fig. 63 the largest |beta| is less than a quarter above
    the smallest, and on this realisation less than two fifths.
    """
    betas = np.array([sweep.beta for sweep in sf60_orders.values()])
    assert betas.max() < 0.0
    assert betas.min() / betas.max() < 1.4


def test_selective_immunisation_beats_random(sf60_criteria):
    """PDF p. 56: the section adopts *"el criterio de inmunizacion selectiva
    por el cual se inmunizaran primero los nodos con mayor conectividad dentro
    de la red [38]"*.

    Qualitative.  The section takes this criterion from [38]; here it is
    compared with a random ranking.  Ranking by degree needs about a third
    fewer immune nodes than chance.
    """
    by_degree = sf60_criteria["degree"].mean_nodes_to_extinction
    at_random = sf60_criteria["random"].mean_nodes_to_extinction
    assert by_degree < at_random
    assert by_degree < 0.8 * at_random


def test_the_section_5_5_1_scores_used_as_criteria(sf60_criteria):
    """EXTENSION: the section 5.5.1 scores used as immunisation criteria.

    The minimum-propagation-time score and the node-elimination peeling both
    fall between random immunisation and degree, the criterion section 5.5.2
    uses.
    """
    by_degree = sf60_criteria["degree"].mean_nodes_to_extinction
    at_random = sf60_criteria["random"].mean_nodes_to_extinction
    for criterion in ("min_propagation_time", "node_removal"):
        borrowed = sf60_criteria[criterion].mean_nodes_to_extinction
        assert by_degree < borrowed < at_random
