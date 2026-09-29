"""The growth strategies of thesis section 5.6.4 (PDF pp. 72-77).

Each arriving bank is given one of seven strategies at random, and the strategy
decides how many links it makes and how it picks partners. The point is
heterogeneity: a mix of cautious and reckless rules grows banks that are too
big to fail, and those are the banks whose failure the cascade then follows.
"""

from __future__ import annotations

import numpy as np
import pytest

from netmotion.finance import InterbankNetwork, two_state_cascade
from netmotion.finance.bank_growth import (
    STRATEGY_DESCRIPTIONS,
    Strategy,
    confidence_strategy_growth,
    net_position_from_degrees,
    strategy_growth,
)
from netmotion.networks.communities import community_scale_free

pytestmark = pytest.mark.reproduction


def _networks(n, restricted=False, steps=100, base=20110930):
    return [strategy_growth(rng=base + s, restricted=restricted, steps=steps)[0] for s in range(n)]


# ---------------------------------------------------------------------------
# The seven strategies
# ---------------------------------------------------------------------------


def test_there_are_seven_strategies():
    """PDF pp. 72-73 lists exactly seven, numbered 1 to 7."""
    assert [int(s) for s in Strategy] == [1, 2, 3, 4, 5, 6, 7]
    assert set(STRATEGY_DESCRIPTIONS) == set(Strategy)


def test_every_strategy_is_actually_used():
    """'se le asociara de forma aleatoria una estrategia de entre las 7'."""
    _, assigned = strategy_growth(rng=20110930)
    grown = assigned[assigned > 0]
    assert set(grown.tolist()) == {int(s) for s in Strategy}


def test_the_network_has_the_stated_size():
    """PDF p. 73: same parameters as the plain generator, Cn=10, n0=3, t=100."""
    cn, _ = strategy_growth(rng=20110930)
    assert cn.network.n == 130
    assert cn.n_communities == 10


def test_domestic_banks_make_no_foreign_links():
    """Strategy 7 is 'Union solo interior'."""
    cn, assigned = strategy_growth(rng=5, strategies=(int(Strategy.DOMESTIC),))
    assert cn.exterior_edges() == 10, "only the ten seed links should cross a border"


def test_risky_banks_make_more_links_than_cautious_ones():
    """Strategies 3 and 4 are defined as two links below and above the trend.

    Strategy 4 also inverts the kernel -- it accepts with probability
    1 - <Pi> where strategy 3 accepts with <Pi> -- and since <Pi> = 1/n is
    small, the risky rule accepts nearly always and the cautious one nearly
    never. The gap is therefore large, not marginal.
    """
    cautious = np.mean([cn.density for cn in _cn(Strategy.CAUTIOUS)])
    risky = np.mean([cn.density for cn in _cn(Strategy.RISKY)])
    assert risky > cautious


def _cn(strategy, n=4):
    return [strategy_growth(rng=100 + s, strategies=(int(strategy),))[0] for s in range(n)]


# ---------------------------------------------------------------------------
# Reported numbers
# ---------------------------------------------------------------------------


def test_every_strategy_makes_the_number_of_links_it_prescribes():
    """PDF pp. 72-73 gives each of the seven rules a number of links.

    Strategy 1 joins the largest bank plus two more and one bank abroad, and
    strategy 6 joins two at home and one abroad; strategy 7 is given the two
    uniform interior links of strategy 6 here. The other four derive their count
    from a local mean, bounded (at least 1, at most 4), and each of them makes at
    least the one link its rule guarantees.
    """
    counts = {s: [] for s in Strategy}
    for seed in range(3):
        cn, assigned = strategy_growth(rng=20110930 + seed)
        adjacency = np.asarray(cn.network.adjacency)
        for node in range(30, cn.network.n):
            made = int((adjacency[node, :node] != 0).sum())
            counts[Strategy(int(assigned[node]))].append(made)

    made = {s: np.mean(v) for s, v in counts.items()}

    assert made[Strategy.SAFE] == pytest.approx(4.0)
    assert made[Strategy.RANDOM] == pytest.approx(3.0)
    assert made[Strategy.DOMESTIC] == pytest.approx(2.0)

    # Strategy 4 takes <k> + 2 at home and <KC> + 2 abroad, both capped at 4.
    assert made[Strategy.RISKY] > 6.5

    # None of the seven is left making a single link.
    for strategy, mean in made.items():
        assert min(counts[strategy]) >= 2, f"{strategy.name} made {mean:.2f} links per arrival"


@pytest.mark.slow
def test_strategy_growth_density_is_close_to_the_reported_value():
    """PDF p. 74: the seven-strategy network has d = 2.46 links per node.

    Checked to within a quarter. This implementation reaches the count where
    the rule names an integer and guarantees the rule's floor where the count
    comes from a mean (see the reading note on `strategy_growth`).
    """
    density = np.mean([cn.density for cn in _networks(5)])
    assert density == pytest.approx(2.46, rel=0.25), f"measured {density:.2f}, thesis reports 2.46"


@pytest.mark.slow
def test_strategy_growth_is_denser_than_uniform_growth():
    """PDF p. 74: 2.46 for the seven-strategy network against "para el modelo
    scale free y un tiempo de crecimiento equivalente se obtenia una densidad de
    1.92".

    Two different models measured at the same growth time, and the sentence is
    about the ordering: growing by strategies makes the system denser than
    growing every bank the same way. Both densities are reproduced to within a
    quarter, and the ordering holds on every seed tried.
    """
    strategy = [cn.density for cn in _networks(10)]
    uniform = [community_scale_free(rng=20110930 + s).density for s in range(10)]

    assert np.mean(strategy) == pytest.approx(2.46, rel=0.25)
    assert np.mean(uniform) == pytest.approx(1.92, rel=0.25)
    assert all(a > b for a, b in zip(strategy, uniform, strict=True)), (
        f"strategy {np.mean(strategy):.2f} against uniform {np.mean(uniform):.2f}"
    )


def test_degree_distribution_peaks_where_the_thesis_reports():
    """PDF p. 74, fig. 81.c: p(k) peaks near k = 4-5."""
    cn, _ = strategy_growth(rng=20110930)
    counts = np.bincount(cn.network.degrees())
    assert 3 <= int(np.argmax(counts)) <= 6


@pytest.mark.slow
def test_a_long_run_grows_more_too_big_to_fail_banks():
    """PDF p. 75, fig. 84: a longer run brings the model closer to reality.

    "La cercania de este modelo con la realidad se pone de manifiesto con mas
    claridad si se definen 200 iteraciones ... Observamos en la siguiente grafica la
    existencia de una mayor cantidad de los conocidos nodos too big to fail. Vemos que
    practicamente existe uno para cada comunidad." So the run at t = 200 grows
    more of them than the run at t = 100, and spreads them over more of the ten
    countries -- which is what is checked here, seed by seed.
    """

    def tbtf(steps, seed):
        cn, _ = strategy_growth(rng=seed, steps=steps)
        ib = InterbankNetwork.from_community_network(cn)
        banks = ib.too_big_to_fail(sigma=2.0)
        return len(banks), len(set(cn.community[banks].tolist()))

    short = [tbtf(100, 20110930 + s) for s in range(4)]
    long_ = [tbtf(200, 20110930 + s) for s in range(4)]

    for (n_short, _), (n_long, _) in zip(short, long_, strict=True):
        assert n_long > n_short

    countries = np.mean([c for _, c in long_])
    assert countries >= 5.0, f"{countries:.1f} of 10 countries hold a TBTF bank at t = 200"


# ---------------------------------------------------------------------------
# Restricted growth (PDF pp. 76-77)
# ---------------------------------------------------------------------------


def test_the_restriction_only_lends_abroad_to_solvent_banks():
    """PDF p. 76: an exterior link may be made only to a node with D_i > 0.

    Checked on the mechanism rather than on the finished network, since D is
    recomputed from the degrees as they stand and a bank solvent at the moment
    it was lent to may be in deficit by the end.
    """
    matrix = np.zeros((5, 5))
    for i, j in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2)]:
        matrix[i, j] = matrix[j, i] = 1.0
    d = net_position_from_degrees(matrix, 5)

    k = (matrix != 0).sum(axis=1).astype(float)
    assert np.allclose(d, k * k - (matrix != 0) @ k)
    assert d[0] > 0, "the hub lends more than it owes"


def test_a_refused_attempt_has_two_readings_and_drop_is_the_default():
    """PDF p. 76: what an arrival refused abroad does, `on_refusal`.

    'drop' spends the attempt and goes without the foreign link; 'retry' keeps
    looking for a solvent partner. 'drop' is the default, under which the
    restriction makes the branches between communities hard to grow, as PDF
    p. 76 states.
    """
    for seed in range(3):
        default, _ = strategy_growth(rng=seed, restricted=True)
        explicit, _ = strategy_growth(rng=seed, restricted=True, on_refusal="drop")
        assert np.array_equal(default.network.adjacency, explicit.network.adjacency)

    with pytest.raises(ValueError, match="on_refusal"):
        strategy_growth(on_refusal="give_up")


def test_dropping_the_refused_attempt_starves_the_branches_between_communities():
    """PDF p. 76: "la restriccion impuesta para las conexiones exteriores
    dificulta el crecimiento de las ramas entre comunidades", which is the
    reason given for building the restricted network with t = 1000 rather than
    the t = 100 of the unrestricted network.

    Under 'retry' the same links are made, merely after more candidate draws; under
    'drop' the cross-border count falls several-fold, which is the shortfall a
    longer growth time has to make up. Interior growth is untouched either way,
    so the network still reaches 130 banks.
    """
    free = [strategy_growth(rng=20110930 + s)[0] for s in range(4)]
    retry = [
        strategy_growth(rng=20110930 + s, restricted=True, on_refusal="retry")[0] for s in range(4)
    ]
    drop = [strategy_growth(rng=20110930 + s, restricted=True)[0] for s in range(4)]

    assert all(cn.network.n == 130 for cn in free + retry + drop)

    loose = np.mean([cn.exterior_edges() for cn in free])
    kept = np.mean([cn.exterior_edges() for cn in retry])
    lost = np.mean([cn.exterior_edges() for cn in drop])

    assert kept == pytest.approx(loose, rel=0.15), f"retry {kept:.0f} against {loose:.0f}"
    assert lost < 0.5 * loose, f"drop {lost:.0f} against {loose:.0f}"


@pytest.mark.slow
def test_the_dropping_reading_is_what_a_longer_growth_time_compensates():
    """PDF p. 76 builds the restricted network at t = 1000 rather than t = 100.

    Under the dropping reading that extra time is needed and quantifiable: over
    twelve seeds an unrestricted run carries 149 links across its borders at
    t = 100, and a dropping run reaches that count at about t = 500 -- 126 at
    t = 400, 175 at t = 600 -- because the share of banks holding D_i > 0 at any
    one moment falls as degrees rise. Under the retrying reading no extra time
    is needed at all, at either growth time.
    """
    free = np.mean([strategy_growth(rng=20110930 + s)[0].exterior_edges() for s in range(3)])
    slow = np.mean(
        [
            strategy_growth(rng=20110930 + s, steps=500, restricted=True, on_refusal="drop")[
                0
            ].exterior_edges()
            for s in range(3)
        ]
    )
    assert slow == pytest.approx(free, rel=0.3), (
        f"drop at t = 500 carries {slow:.0f} cross-border links, unrestricted "
        f"t = 100 carries {free:.0f}"
    )


def test_restricted_growth_still_produces_a_valid_network():
    cn, _ = strategy_growth(rng=20110930, restricted=True)
    ib = InterbankNetwork.from_community_network(cn)
    assert cn.network.n == 130
    assert ib.check_conservation()


# ---------------------------------------------------------------------------
# The 2012 article's three strategies
# ---------------------------------------------------------------------------


def test_paper_growth_uses_exactly_three_strategies():
    cn, assigned = confidence_strategy_growth(rng=3)
    assert cn.network.n == 240
    assert set(assigned[assigned > 0].tolist()) <= {1, 2, 3}


def test_strategy_three_is_the_most_connective_of_the_three():
    """Article p. 5: strategy 3 connects to c_min <= alpha*d_in nodes with the
    probability alpha*Pi, strategy 2 to c_min <= d_in/alpha at Pi/alpha.

    "Then the growth of these nodes is favored ... strategy 3 is critical to
    generate big banks as it provides the major amount of edges per node"
    (Fig. 2). So a strategy-3 bank ends up the largest of the three on average,
    and strategy 2 the smallest.
    """
    for seed in range(4):
        cn, assigned = confidence_strategy_growth(rng=seed)
        degrees = cn.network.degrees()
        by_strategy = [float(degrees[assigned == s].mean()) for s in (1, 2, 3)]
        assert by_strategy[2] > by_strategy[0] > by_strategy[1], by_strategy


def test_the_confidence_factor_separates_the_strategies():
    """Article p. 4-5: alpha divides strategy 2's appetite and multiplies
    strategy 3's, so raising it pulls the two further apart.

    "By increasing alpha, the probability and the number of connections is
    reduced" for strategy 2; for strategy 3 "an increment of alpha produces an
    increment in the probability and the number of connections".
    """

    def gap(alpha):
        spread = []
        for seed in range(4):
            cn, assigned = confidence_strategy_growth(rng=seed, confidence=alpha)
            degrees = cn.network.degrees()
            spread.append(float(degrees[assigned == 3].mean() - degrees[assigned == 2].mean()))
        return float(np.mean(spread))

    assert gap(1.0) < gap(2.0) < gap(4.0)


def test_confidence_below_one_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        confidence_strategy_growth(confidence=0.5)


def test_step_four_grows_the_strategy_three_banks():
    """Article p. 5, Step 4: "at each step, one of the nodes n' generated in
    earlier iterations and defined by the strategy 3 is randomly chosen and
    connected to the node added at this iteration with probability Pi_{m'n'}".

    It is the disequilibrium the article puts in the growth loop on purpose,
    and the banks it grows are strategy-3 banks. Checked at the article's own
    parameters, the defaults: C = 10, e = 1, n0 = 4, d_in = 4, d_ex = 2,
    c_min = 1, C_min = 0, alpha = 2, t = 200.
    """
    for seed in range(4):
        cn, assigned = confidence_strategy_growth(rng=seed)
        degrees = cn.network.degrees()
        assert cn.network.n == 240
        # The biggest bank is one that took strategy 3.
        assert int(assigned[int(np.argmax(degrees))]) == 3


def test_the_optional_rewiring_grows_dominant_banks():
    """`too_big_to_fail` is a post-hoc rewiring, not the article's Step 4.

    Step 4 always runs, inside the growth loop; this argument is a separate
    lever that grows the largest banks further once growth has finished.
    """
    plain, _ = confidence_strategy_growth(rng=3)
    with_tbtf, _ = confidence_strategy_growth(rng=3, too_big_to_fail=5)
    assert with_tbtf.network.degrees().max() > plain.network.degrees().max()


def test_the_paper_model_takes_the_restriction_of_its_own_figure_8():
    """Article p. 15: "each node added into the network can only make a
    connection with a node i of another community if D_i > 0"."""
    free = [confidence_strategy_growth(rng=s)[0] for s in range(5)]
    tied = [confidence_strategy_growth(rng=s, restricted=True)[0] for s in range(5)]

    assert all(cn.network.n == 240 for cn in tied)
    assert all(cn.exterior_edges() > 0 for cn in tied)
    # A foreign loan refused to a bank in deficit is not made, so the
    # restriction costs cross-border links rather than adding them.
    assert np.mean([cn.exterior_edges() for cn in tied]) < np.mean(
        [cn.exterior_edges() for cn in free]
    )


def test_seed_20110930_reproduces_the_reported_ordering():
    """PDF p. 74's comparison, at seed 20110930.

    The strategy-grown network is the denser of the two, which is the claim the
    sentence makes. The two densities themselves are checked against 2.46 and
    1.92 in `test_strategy_growth_is_denser_than_uniform_growth`.
    """
    strategy, _ = strategy_growth(rng=np.random.default_rng(20110930))
    uniform = community_scale_free(rng=np.random.default_rng(20110930))

    def density(net):
        return float(np.asarray(net.adjacency).sum() / 2 / net.n)

    d_strategy = density(strategy.network)
    d_uniform = density(uniform.network)

    assert d_strategy > d_uniform, f"{d_strategy:.2f} against {d_uniform:.2f}"
    assert 2.4 < d_strategy < 3.4


def test_restricting_foreign_lending_slows_growth_between_communities():
    """PDF p. 76: the restriction "dificulta el crecimiento de las ramas entre
    comunidades", which is why the restricted network is grown for t = 1000.

    A refused foreign attempt is spent rather than retried (on_refusal="drop"),
    so at the same growth time the restricted network holds far fewer
    cross-border links than the unrestricted one.
    """
    free = [strategy_growth(rng=20110930 + s)[0] for s in range(5)]
    tied = [
        strategy_growth(rng=20110930 + s, restricted=True, on_refusal="drop")[0] for s in range(5)
    ]

    assert all(cn.network.n == 130 for cn in free + tied)
    assert all(cn.exterior_edges() > 0 for cn in tied)

    loose = np.mean([cn.exterior_edges() for cn in free])
    bound = np.mean([cn.exterior_edges() for cn in tied])
    assert bound < 0.6 * loose, f"{bound:.0f} against {loose:.0f}"


def test_the_two_drain_options_on_the_debt_network():
    """drain='once' and drain='every_step' on the community network.

    Started at the bank with the largest net position, draining once stops
    after a few banks while draining at every step takes all 130.
    """
    cn = community_scale_free(rng=np.random.default_rng(20110930))
    ib = InterbankNetwork.from_community_network(cn)
    capital = np.abs(ib.capital)
    seed = int(np.argmax(capital))

    once = two_state_cascade(ib.network, seed, capital.copy(), drain="once")
    every = two_state_cascade(ib.network, seed, capital.copy(), drain="every_step")

    assert ib.network.n == 130
    assert int(once.crashed.size) <= 10
    assert int(every.crashed.size) == 130
    assert once.crashed_fraction < every.crashed_fraction
