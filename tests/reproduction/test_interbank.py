"""The interbank debt network and the crisis cascade.

Thesis section 5.6.4 (PDF pp. 68-77) and the 2012 IJMPC article.

The worked five-bank example is checked by hand against the article's
equations; the network-scale claims are checked as the qualitative relations
the sources state.
"""

from __future__ import annotations

import numpy as np
import pytest

from netmotion import Network
from netmotion.finance import (
    BankState,
    bank_capital,
    debt_matrix,
    four_state_cascade,
    systemic_impact,
    two_state_cascade,
)
from netmotion.finance.interbank import InterbankNetwork
from netmotion.networks.communities import community_scale_free, modularity

pytestmark = pytest.mark.reproduction


# Five banks: 0 and 1 are hubs of degree 4; 2, 3 and 4 have degree 2.
# Worked by hand from article Eq. (1).
FIVE = Network.from_edges(5, [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)])

# Seven banks, arranged so that exactly one bank reacts and has exactly one
# claim to make, and so that its two possible debtors sit at opposite ends of
# the arrival order.  Bank 6 is the biggest (k = 4) and is the one attacked;
# bank 3 (k = 3) survives its crash and becomes affected; banks 0 and 5 are its
# two smaller neighbours, and are identical in every respect except that bank 5
# arrived later, so the link 3-5 is the newer -- the weaker -- of the two.
# Degrees [1, 1, 1, 3, 1, 1, 4]; D = [-2, -3, -3, 3, -3, -2, 10].
RECOVERY = Network.from_edges(7, [(6, 3), (6, 1), (6, 2), (6, 4), (3, 0), (3, 5)])


# ---------------------------------------------------------------------------
# The weight matrix (article Eq. 1, thesis PDF p. 68)
# ---------------------------------------------------------------------------


def test_degrees_of_the_worked_example():
    assert np.array_equal(FIVE.degrees(), [4, 4, 2, 2, 2])
    assert FIVE.n_edges == 7


def test_weights_are_degree_differences():
    """Article Eq. (1): W_ij = k_i - k_j on existing links."""
    w = debt_matrix(FIVE)
    assert w[0, 2] == 2.0
    assert w[0, 1] == 0.0, "equal degrees mean neither owes the other"
    assert w[2, 0] == -2.0
    assert w[0, 3] == w[0, 4] == 2.0


def test_weights_are_zero_where_there_is_no_link():
    w = debt_matrix(FIVE)
    assert w[2, 3] == 0.0
    assert np.array_equal((w != 0), (FIVE.adjacency != 0) & (w != 0))


def test_weight_matrix_is_antisymmetric():
    w = debt_matrix(FIVE)
    assert np.allclose(w, -w.T)


def test_capital_matches_the_hand_computation():
    """D_1 = D_2 = 6, D_3 = D_4 = D_5 = -4 in the thesis's 1-based labels."""
    assert bank_capital(FIVE).tolist() == [6.0, 6.0, -4.0, -4.0, -4.0]


def test_capital_matches_the_closed_form():
    """D_i = k_i^2 - sum of neighbour degrees."""
    d = bank_capital(FIVE)
    k = FIVE.degrees().astype(float)
    closed = k**2 - FIVE.adjacency @ k
    assert np.allclose(d, closed)


def test_the_system_nets_to_zero():
    """Thesis PDF p. 70: sum_i D_i = 0, exact by antisymmetry."""
    assert bank_capital(FIVE).sum() == pytest.approx(0.0, abs=1e-12)


def test_the_thesis_sign_convention_is_the_exact_negative():
    """Thesis PDF p. 68 writes W_ij = k_j - k_i, the debt bank i has with j;
    the article writes W_ij = k_i - k_j. The two capitals are exact negatives.
    """
    paper = bank_capital(FIVE, convention="paper")
    thesis = bank_capital(FIVE, convention="thesis")
    assert np.allclose(thesis, -paper)
    assert set(np.flatnonzero(paper > 0)) == {0, 1}
    assert set(np.flatnonzero(thesis > 0)) == {2, 3, 4}


def test_the_sign_convention_does_not_change_which_banks_fall():
    """The cascade spends the loss ``Theta_ij = |k_i - k_j|`` against the
    capital ``|D_i|``, and neither carries a sign, so the crashed set is the
    same from every starting bank under either convention.
    """
    paper = np.abs(bank_capital(FIVE, convention="paper"))
    thesis = np.abs(bank_capital(FIVE, convention="thesis"))
    assert np.array_equal(paper, thesis)

    for drain in ("once", "every_step"):
        for seed in range(FIVE.n):
            a = two_state_cascade(FIVE, seed, paper.copy(), drain=drain)
            b = two_state_cascade(FIVE, seed, thesis.copy(), drain=drain)
            assert np.array_equal(a.crashed, b.crashed), (
                f"seed {seed}, drain={drain}: the sign convention changed the crashed set"
            )


# ---------------------------------------------------------------------------
# Systemic aggregates (article Eq. 10)
# ---------------------------------------------------------------------------


def test_systemic_impact_ratio():
    """Article Eq. (10) with eta = 100: I_c = 6, R = 18, I_c/R = 33.3%."""
    reserves = np.abs(bank_capital(FIVE))
    impact, rest, ratio = systemic_impact(reserves, 0)
    assert (impact, rest) == (6.0, 18.0)
    assert ratio == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# The 2011 two-state cascade (thesis PDF p. 69)
# ---------------------------------------------------------------------------


def test_equal_degrees_transmit_nothing():
    """The mechanism the thesis describes: capital absorbs the loss.

    Bank 1 has the same degree as bank 0, so Theta = |k_0 - k_1| = 0 and bank
    0's failure costs it nothing directly. Checked on the first step, because
    the cascade can still reach bank 1 later by another route.
    """
    run = two_state_cascade(FIVE, 0, np.abs(bank_capital(FIVE)))
    assert run.capital_history[1][1] == run.capital_history[0][1]
    assert run.state[1] == BankState.UNAFFECTED


def test_a_bank_in_debt_falls_only_once_the_contagion_reaches_it():
    """PDF p. 66: S_i = -1 if D_i < 0 *and* a neighbour has fallen.

    Two separate stars.  The crash of a leaf of the first cannot touch the
    second, so the second's leaves, in debt as they are, stay standing.
    """
    stars = Network.from_edges(7, [(0, 1), (0, 2), (0, 3), (4, 5), (4, 6)])
    capital = bank_capital(stars)
    assert (capital[[5, 6]] < 0).all()
    run = two_state_cascade(stars, 1, capital)
    assert not set(run.crashed.tolist()) & {4, 5, 6}


def test_cascade_records_a_monotone_history():
    run = two_state_cascade(FIVE, 0, np.abs(bank_capital(FIVE)))
    assert np.all(np.diff(run.history) >= 0)
    assert run.history[0] == pytest.approx(1 / 5)


def test_drain_option_changes_the_outcome():
    """``drain``: a crashed bank drains its neighbours once or at every step.

    On this network the two options give different outcomes.
    """
    capital = np.abs(bank_capital(FIVE))
    once = two_state_cascade(FIVE, 0, capital, drain="once")
    every = two_state_cascade(FIVE, 0, capital, drain="every_step")
    assert every.crashed_fraction >= once.crashed_fraction
    assert every.crashed_fraction > once.crashed_fraction


# ---------------------------------------------------------------------------
# The 2012 four-state cascade
# ---------------------------------------------------------------------------


def test_paper_cascade_marks_damaged_but_surviving_banks_as_affected():
    """Article Sec. 2.2 state rules with eta = 100, r = 0, seed = bank 0.

    Banks 2, 3 and 4 each lose Theta = |2 - 4| = 2 from a starting 4, so they
    are damaged but still solvent: state -1, affected, not crashed.
    """
    run = four_state_cascade(FIVE, 0, bank_capital(FIVE), reserves=100.0, reaction_threshold=0.0)
    assert run.state[0] == BankState.CRASHED
    assert run.state[1] == BankState.UNAFFECTED
    assert list(run.state[2:]) == [BankState.AFFECTED] * 3
    assert np.allclose(run.capital[2:], 2.0)


def test_a_reaction_threshold_keeps_damaged_banks_from_reacting():
    """Article Eq. (9): a bank reacts only once its loss exceeds th_i.

    With r = 60 the threshold for banks 2-4 is 4 x 0.6 = 2.4, and their loss is
    2, so they register the damage without reacting to it.
    """
    run = four_state_cascade(FIVE, 0, bank_capital(FIVE), reserves=100.0, reaction_threshold=60.0)
    assert list(run.state[2:]) == [BankState.NOT_REACTED] * 3


def test_lower_reserves_make_the_cascade_worse():
    """eta scales the capital every bank has to absorb losses with."""
    capital = bank_capital(FIVE)
    strong = four_state_cascade(FIVE, 0, capital, reserves=100.0)
    weak = four_state_cascade(FIVE, 0, capital, reserves=20.0)
    assert weak.crashed_fraction >= strong.crashed_fraction


def test_reserves_are_a_haircut_on_the_net_borrowers_only():
    """Article Sec. 2.2, in its own words.

    "It is considered that each bank with a value of D_i < 0 has to leave as
    reserves a certain percentage eta of the entire debt. Then the initial
    proportion of the bank's debt at the time t which is used during the
    contagion process is defined as D_i(t) = |eta D_i|."

    The condition is D_i < 0. Banks 0 and 1 are the creditors of the worked
    example, so eta does not touch them: they enter the cascade holding the
    whole of |D_i| = 6. Banks 2, 3 and 4 are the borrowers and enter at 20% of
    4.
    """
    run = four_state_cascade(FIVE, 0, bank_capital(FIVE), reserves=20.0)
    assert run.initial_capital.tolist() == [6.0, 6.0, 0.8, 0.8, 0.8]


def test_full_reserves_leave_every_bank_at_its_gross_position():
    """At eta = 100 the haircut is empty and borrower and creditor coincide."""
    run = four_state_cascade(FIVE, 0, bank_capital(FIVE), reserves=100.0)
    assert np.array_equal(run.initial_capital, np.abs(bank_capital(FIVE)))


def test_applying_the_haircut_to_every_bank_is_the_harsher_reading():
    """The two are not interchangeable, and the difference is not small.

    ``borrowers=`` is what lets the other reading be stated at all: with every
    bank marked a borrower, eta divides the whole system's capital while the
    loss Theta_ij = |k_i - k_j| is untouched, and the cascade takes banks that
    the printed rule leaves standing. Here it takes banks 0 and 3 as well --
    bank 3 being the one creditor besides the bank attacked, and the only bank
    with room to absorb anything.
    """
    capital = bank_capital(RECOVERY)
    printed = four_state_cascade(RECOVERY, 6, capital, reserves=20.0)
    everyone = four_state_cascade(
        RECOVERY, 6, capital, reserves=20.0, borrowers=np.ones(RECOVERY.n, dtype=bool)
    )
    assert printed.initial_capital == pytest.approx([0.4, 0.6, 0.6, 3.0, 0.6, 0.4, 10.0])
    assert everyone.initial_capital == pytest.approx([0.4, 0.6, 0.6, 0.6, 0.6, 0.4, 2.0])
    assert set(printed.crashed.tolist()) < set(everyone.crashed.tolist())
    assert everyone.crashed.size == RECOVERY.n


def test_capital_without_its_sign_needs_a_borrower_mask():
    """A vector of absolute values cannot say who the borrowers are.

    Silently treating it as all-creditor -- or all-borrower -- would apply the
    rule of Sec. 2.2 to the wrong set, so the call is refused and the caller is
    asked for the signed vector or an explicit mask. At eta = 100 there is
    nothing to decide, so an absolute vector is accepted there.
    """
    absolute = np.abs(bank_capital(FIVE))
    with pytest.raises(ValueError, match="signed net position"):
        four_state_cascade(FIVE, 0, absolute, reserves=20.0)

    four_state_cascade(FIVE, 0, absolute, reserves=100.0)

    named = four_state_cascade(FIVE, 0, absolute, reserves=20.0, borrowers=bank_capital(FIVE) < 0)
    assert named.initial_capital.tolist() == [6.0, 6.0, 0.8, 0.8, 0.8]


def test_the_weakest_connection_is_called_in_first():
    """Article Eqs. 6-7: bank j "will first ask for a repayment to the bank i
    which has the weakest connection to the bank j", where "the fragility of a
    connection is measured by counting the time T_ij that the banks i and j are
    connected since the beginning of the connection during the network growth".

    A short T_ij is a weak connection, so the newest link is called in first.
    Sec. 3.2 says the same from the other side, about the seed links between
    communities: "these connections develop at the beginning of the network,
    then [...] the debt produced in these connections will not be asked for a
    repayment in the first place."

    Bank 6 crashes; bank 3 loses |4 - 3| = 1 and reacts. It can claim
    k_3 - k_i = 2 from either of banks 0 and 5, and needs only 1, so exactly
    one of them pays. Under the default T_ij -- a link dates from the later of
    its endpoints -- that is bank 5, the newer connection.
    """
    run = four_state_cascade(RECOVERY, 6, bank_capital(RECOVERY))
    assert run.capital[5] == 1.0, "the newer link should have been called in"
    assert run.capital[0] == 2.0, "the older link should have been left alone"
    assert run.state[3] == BankState.UNAFFECTED, "bank 3 recovers its whole loss"


def test_connection_age_overrides_the_default_proxy():
    """T_ij is a parameter, and supplying it changes who pays.

    Node index read directly as T_ij makes the low-index link the weakest one,
    and then bank 0 pays instead of bank 5. The check is that the order is
    driven by T_ij and nothing else.
    """
    by_index = np.tile(np.arange(RECOVERY.n, dtype=float), (RECOVERY.n, 1))
    run = four_state_cascade(RECOVERY, 6, bank_capital(RECOVERY), connection_age=by_index)
    assert run.capital[0] == 1.0
    assert run.capital[5] == 2.0


def test_eq_9_can_be_read_against_either_position():
    """Article Eq. 9 writes th_i = D_i(1) r / 100.

    D_i(1) is the money the bank brings to the cascade, which for a borrower is
    already cut by eta: ``threshold_base="initial"``, the default. The option
    ``threshold_base="gross"`` measures the loss against |D_i| instead.

    The capital here is set by hand rather than taken from the network, so that
    one borrower's loss lands between the two thresholds: bank 3 holds
    D = -10, so eta = 80 leaves it D(1) = 8, and its loss of 1 clears
    8 x 0.12 = 0.96 but not 10 x 0.12 = 1.2.
    """
    capital = np.array([-2.0, -3.0, -3.0, -10.0, -3.0, -2.0, 10.0])
    kwargs = dict(reserves=80.0, reaction_threshold=12.0)

    printed = four_state_cascade(RECOVERY, 6, capital, threshold_base="initial", **kwargs)
    gross = four_state_cascade(RECOVERY, 6, capital, threshold_base="gross", **kwargs)

    # Under the printed reading bank 3 reacts, calls in its weakest connection
    # and ends whole again; under the gross reading it never reacts, so it
    # stays a pound short and bank 5 is never asked for anything.
    assert printed.state[3] == BankState.UNAFFECTED
    assert printed.capital[3] == 8.0
    assert gross.state[3] == BankState.NOT_REACTED
    assert gross.capital[3] == 7.0
    assert printed.capital[5] < gross.capital[5] == 1.6

    with pytest.raises(ValueError, match="threshold_base"):
        four_state_cascade(RECOVERY, 6, capital, threshold_base="whichever")


@pytest.mark.slow
def test_draining_once_gives_the_published_r_sweep():
    """Article Fig. 10, PDF p. 17.

    Fig. 10 sweeps r from 0 to 100 (the contagion runs of Sec. 3.2 use
    eta = 20), starting the crash at the
    biggest bank, and its text describes the curve: a maximum across
    0 < r < 20, a fall across 20 < r < 40, and no further change across
    40 < r < 100, ending near 48% of banks crashed against about 68% at the
    left-hand end.

    Draining once gives that shape: around 70% at r = 0, falling to around 48%
    at r = 100, which is why drain="once" is the default. Draining at every
    step gives a flat line above 97% at every r.

    Run on plain scale-free networks.
    """
    from netmotion.networks import barabasi_albert

    rng = np.random.default_rng(20110930)
    nets = [barabasi_albert(200, links_per_new_node=2, rng=rng) for _ in range(4)]
    capital = [bank_capital(g) for g in nets]

    def crashed(drain, r):
        return float(
            np.mean(
                [
                    four_state_cascade(
                        g,
                        int(np.argmax(c)),
                        c,
                        reserves=20.0,
                        reaction_threshold=float(r),
                        drain=drain,
                        # every_step never settles, so it would otherwise run
                        # the full 1000 steps long after the last bank fell.
                        max_steps=200,
                    ).crashed_fraction
                    for g, c in zip(nets, capital, strict=True)
                ]
            )
        )

    left, right = crashed("once", 0), crashed("once", 100)
    assert 0.5 < left < 0.9, f"r = 0 should sit near the published 68%, got {left:.1%}"
    assert right < left - 0.15, (
        f"the published curve falls as r rises; this one goes {left:.1%} -> {right:.1%}"
    )

    flat_left, flat_right = crashed("every_step", 0), crashed("every_step", 100)
    assert min(flat_left, flat_right) > 0.95, (
        f"draining every step should be near-total at every r, got "
        f"{flat_left:.1%} and {flat_right:.1%}"
    )
    assert abs(flat_right - flat_left) < 0.05, (
        f"draining every step should be flat in r: {flat_left:.1%} -> {flat_right:.1%}"
    )


@pytest.mark.slow
def test_the_haircut_on_every_bank_erases_the_dependence_on_structure():
    """The borrower-only rule of Sec. 2.2 against a haircut on every bank.

    Cutting every bank's capital to eta% while leaving the loss term
    Theta_ij = |k_i - k_j| alone puts the model in a regime where the network
    collapses whatever else is varied -- r included -- and so can no longer
    show any structural effect at all. The printed rule leaves banks standing
    and leaves r something to do, which is what Fig. 10 depends on.
    """
    from netmotion.networks import barabasi_albert

    rng = np.random.default_rng(20110930)
    nets = [barabasi_albert(200, links_per_new_node=2, rng=rng) for _ in range(4)]
    capital = [bank_capital(g) for g in nets]
    everyone = [np.ones(c.size, dtype=bool) for c in capital]

    def crashed(masks, r):
        return float(
            np.mean(
                [
                    four_state_cascade(
                        g,
                        int(np.argmax(c)),
                        c,
                        reserves=20.0,
                        borrowers=m,
                        reaction_threshold=float(r),
                    ).crashed_fraction
                    for g, c, m in zip(nets, capital, masks, strict=True)
                ]
            )
        )

    flattened = [crashed(everyone, r) for r in (0, 50, 100)]
    assert min(flattened) > 0.95, f"expected the degenerate regime, got {flattened}"
    assert max(flattened) - min(flattened) < 0.05

    printed = [crashed([None] * len(nets), r) for r in (0, 50, 100)]
    assert max(printed) < 0.9
    assert max(printed) - min(printed) > 0.1, (
        f"under the printed rule r still moves the outcome: {printed}"
    )


# ---------------------------------------------------------------------------
# The grown network (thesis PDF p. 66)
# ---------------------------------------------------------------------------


def test_the_grown_network_has_the_stated_size():
    """PDF p. 66: Cn = 10, n0 = 3, t = 100 gives N = 10*3 + 100 = 130."""
    cn = community_scale_free(rng=20110930)
    assert cn.network.n == 130
    assert cn.n_communities == 10


def test_most_communities_grow():
    """Figure 74.a shows ten populated communities of differing size.

    Read per-community, most communities grow and their sizes spread, which is
    what the figure shows. With ``seed_scope="total"`` only r exterior
    links are placed in all, so most communities start with zero exterior
    degree and stay at their initial three nodes.
    """
    cn = community_scale_free(rng=20110930)
    grew = int((cn.sizes() > 3).sum())
    assert grew >= 7, f"only {grew} of {cn.n_communities} communities grew: {cn.sizes()}"

    frozen = community_scale_free(rng=20110930, seed_scope="total")
    assert (frozen.sizes() == 3).sum() >= 5, (
        "seed_scope='total' should leave most communities at three nodes"
    )


def test_community_structure_is_real():
    cn = community_scale_free(rng=20110930)
    assert cn.interior_edges() > cn.exterior_edges()
    assert modularity(cn) > 0.3


def test_density_is_close_to_the_reported_value():
    """PDF p. 66 reports d = 2.06 links per node.

    Measured with the implementation's default `biased_interior` attachment.
    """
    density = np.mean([community_scale_free(rng=20110930 + s).density for s in range(4)])
    assert density == pytest.approx(2.06, abs=0.30), f"measured {density:.2f}, thesis reports 2.06"


def test_the_attachment_readings_bracket_the_reported_density():
    """The three ``attachment`` options, and where 2.06 falls.

    Whether a rejected draw is retried changes the link count by nearly a
    factor of two; 2.06 lies between the "sweep" and "until_filled" options.
    """

    def density(mode):
        return np.mean(
            [
                community_scale_free(rng=20110930 + s, attachment=mode, exterior_attempts=None).density
                for s in range(3)
            ]
        )

    sweep = density("sweep")
    biased = density("biased_interior")
    filled = density("until_filled")

    assert sweep < biased < filled
    assert sweep < 2.06 < filled, "the reported density is bracketed by the two extremes"


def test_interbank_network_conserves_capital_at_scale():
    """The zero-sum property must survive on a real grown network, not just on
    the hand-worked example."""
    cn = community_scale_free(rng=20110930)
    ib = InterbankNetwork.from_community_network(cn)
    assert ib.check_conservation()
    assert ib.n_banks == 130
    assert ib.creditors().size > 0 and ib.debtors().size > 0


@pytest.mark.slow
def test_no_network_type_absorbs_contagion_better_than_another():
    """Thesis PDF pp. 71-72, fig. 79 and its table.

    The mean percentage of banks surviving the financial virus is reported as
    ER 56.5, WS 58.1, NW 55.4, SF 56.6 -- four numbers within three points of each
    other; the thesis notes that the results do not show clearly which type
    absorbs the contagion better, and that the local arrangement of the nodes has
    a large influence. The conclusion is asserted: the spread
    between network types stays small.
    """
    from netmotion.networks import barabasi_albert, erdos_renyi, newman_watts, watts_strogatz

    n, realisations = 100, 10
    rng = np.random.default_rng(20110930)
    builders = {
        "ER": lambda: erdos_renyi(n, 2 * 2.0 / (n - 1), rng=rng),
        "WS": lambda: watts_strogatz(n, 4, 0.1, rng=rng),
        "NW": lambda: newman_watts(n, 4, 0.1, rng=rng),
        "SF": lambda: barabasi_albert(n, links_per_new_node=2, rng=rng),
    }

    survival = {}
    for label, build in builders.items():
        rates = []
        for _ in range(realisations):
            g = build()
            capital = np.abs(bank_capital(g))
            for seed in range(0, n, 10):
                rates.append(two_state_cascade(g, seed, capital.copy()).survived_fraction)
        survival[label] = float(np.mean(rates))

    values = np.array(list(survival.values()))
    assert values.min() > 0.05, f"cascade destroyed everything: {survival}"
    assert values.max() < 0.999, f"cascade did nothing: {survival}"
    assert values.max() - values.min() < 0.25, (
        f"network type should barely matter: {survival}"
    )
