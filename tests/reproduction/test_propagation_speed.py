"""Propagation speed and its size law.

Thesis section 5.4.1 (PDF pp. 43-48).

The section's headline claim is that the maximum mean propagation speed over
link density obeys

    alpha_max(N) = 0.4974 ln N + 0.012

for every network type. The exponential is fitted up to and including the
first saturated instant, and these tests check the fit and the law.
"""

from __future__ import annotations

import numpy as np
import pytest

from netmotion import Network
from netmotion.networks import (
    barabasi_albert,
    erdos_renyi,
    newman_watts,
    ring_lattice,
    watts_strogatz,
)
from netmotion.propagation import (
    alpha_max_size_law,
    mean_propagation_speed,
    propagate,
    propagation_speed,
    propagation_time,
)

pytestmark = pytest.mark.reproduction


# ---------------------------------------------------------------------------
# The propagation rule (PDF p. 43)
# ---------------------------------------------------------------------------

PATH = Network.from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])


def test_the_seed_is_infected_at_t_equals_one():
    """The thesis indexes from t = 1 with one infected node."""
    run = propagate(PATH, 0)
    assert run.infected_at[0] == 1.0
    assert run.curve[0] == pytest.approx(1 / 5)


def test_the_infection_advances_one_link_per_instant():
    assert propagate(PATH, 0).infected_at.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_the_infected_set_is_the_ball_around_the_seed():
    """I_t = B_{t-1}(seed): a node at distance d is infected at instant d + 1."""
    from netmotion.metrics import shortest_path_lengths

    g = watts_strogatz(40, 4, 0.2, rng=3)
    for seed in (0, 7, 23):
        assert np.array_equal(propagate(g, seed).infected_at - 1.0, shortest_path_lengths(g, seed))


def test_the_healing_variant_lets_the_seed_recover():
    """heals=True is a variant with recovery: a node with no infected neighbour
    returns to +1, so the seed recovers at t = 2. Section 5.4 has no recovery."""
    run = propagate(PATH, 0, heals=True)
    assert run.curve[1] == pytest.approx(1 / 5), "only node 1 is infected at t=2"
    assert run.infected_at[0] == 1.0


def test_the_healing_variant_terminates_on_a_repeated_state():
    """With recovery the state on a path oscillates with period two, so the run
    stops on a repeated state."""
    run = propagate(PATH, 0, heals=True)
    assert run.curve.size < 4 * PATH.n

    run = propagate(watts_strogatz(30, 4, 0.2, rng=5), 0, heals=True)
    assert run.curve.size < 4 * 30

    capped = propagate(PATH, 0, heals=True, max_steps=3)
    assert capped.curve.size <= 4


def test_propagation_time_is_the_eccentricity():
    """Section 5.5.1.2: t_n is the number of iterations to cover the network.

    That is the eccentricity of the seed, so the node minimising it is the
    graph centre and the minimum is the radius.
    """
    from netmotion.metrics import all_pairs_distances

    g = watts_strogatz(50, 4, 0.15, rng=11)
    distances = all_pairs_distances(g)
    for seed in range(0, 50, 7):
        assert propagation_time(g, seed) == pytest.approx(distances[seed].max())

    times = np.array([propagation_time(g, s) for s in range(50)])
    assert times.min() == pytest.approx(distances.max(axis=1).min()), "radius"


def test_immune_nodes_neither_catch_nor_transmit():
    """Section 5.5's definition of immunity (PDF p. 54): frozen at S = +1.

    Cutting the middle of a path stops the infection dead.
    """
    run = propagate(PATH, 0, immune=[2])
    assert set(np.flatnonzero(np.isfinite(run.infected_at))) == {0, 1}


# ---------------------------------------------------------------------------
# The fit window (PDF pp. 46-47)
# ---------------------------------------------------------------------------


def test_the_two_fit_windows_take_different_instants():
    """'to_saturation' fits every instant up to the first saturated one;
    'unsaturated' leaves the saturated instant out."""
    curve = np.array([0.01, 0.1, 0.6, 1.0])
    t = np.arange(1, 5)
    to_saturation = propagation_speed(curve, window="to_saturation")
    unsaturated = propagation_speed(curve, window="unsaturated")

    assert to_saturation == pytest.approx(np.polyfit(t, np.log(curve), 1)[0])
    assert unsaturated == pytest.approx(np.polyfit(t[:3], np.log(curve[:3]), 1)[0])
    assert to_saturation != pytest.approx(unsaturated)


def test_an_undetermined_rate_raises_rather_than_returning_zero():
    """A star's hub covers the network in one step, so there is no rate to fit."""
    star = Network.from_edges(4, [(0, 1), (0, 2), (0, 3)])
    with pytest.raises(ValueError, match="undefined"):
        propagation_speed(propagate(star, 0).curve, window="unsaturated")


# ---------------------------------------------------------------------------
# The size law (PDF p. 47)
# ---------------------------------------------------------------------------


def test_the_reported_law_evaluates_as_stated():
    assert alpha_max_size_law(100) == pytest.approx(0.4974 * np.log(100) + 0.012)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["ER", "WS", "NW", "SF"])
def test_alpha_max_follows_the_reported_size_law(kind):
    """PDF p. 47: alpha_max(N) = 0.4974 ln N + 0.012, for every network type.

    Sweep the link density, take the maximum of the mean speed over every seed,
    and compare with the law. The sizes start at 80 and the density is capped at
    N/4.
    """
    sizes = np.array([80, 160, 320])
    rng = np.random.default_rng(20110930)

    def build(n, d):
        if kind == "ER":
            return erdos_renyi(n, min(1.0, 2 * d / (n - 1)), rng=rng)
        k = max(2, 2 * int(round(d)))
        if kind == "WS":
            return watts_strogatz(n, k, 0.1, rng=rng)
        if kind == "NW":
            return newman_watts(n, k, 0.1, rng=rng)
        return barabasi_albert(n, links_per_new_node=max(1, int(round(d))), rng=rng)

    peaks = []
    for n in sizes:
        best = 0.0
        for d in (4.0, 8.0, 12.0, 18.0, 24.0, 32.0, 48.0):
            if d > n / 4:
                continue
            try:
                best = max(best, mean_propagation_speed(build(int(n), d)))
            except ValueError:
                continue
        peaks.append(best)
    peaks = np.array(peaks)

    expected = alpha_max_size_law(sizes)
    assert np.allclose(peaks, expected, atol=0.05), (
        f"{kind}: measured {peaks.round(3).tolist()}, expected "
        f"{expected.round(3).tolist()}"
    )

    slope = np.polyfit(np.log(sizes), peaks, 1)[0]
    assert slope == pytest.approx(0.4974, abs=0.02), (
        f"{kind}: fitted coefficient of ln N is {slope:.4f}, expected 0.4974"
    )


@pytest.mark.slow
def test_speed_orders_the_network_types_as_structure_predicts():
    """Speed reflects structure.

    At matched size and density an ordered ring must be slowest, because every
    step covers one link, and a scale-free network fastest, because its hubs
    are shortcuts.
    """
    n = 120
    rng = np.random.default_rng(20110930)
    speeds = {
        "ring": mean_propagation_speed(ring_lattice(n, 4)),
        "WS": np.mean(
            [mean_propagation_speed(watts_strogatz(n, 4, 0.1, rng=rng)) for _ in range(3)]
        ),
        "SF": np.mean([mean_propagation_speed(barabasi_albert(n, links_per_new_node=2, rng=rng)) for _ in range(3)]),
    }
    assert speeds["ring"] < speeds["WS"] < speeds["SF"], speeds




def test_newman_watts_carries_more_links_than_its_ring_degree_suggests():
    """Its shortcuts are added to the ring, so the mean degree exceeds k."""
    rng = np.random.default_rng(20110930)
    for k, expected in ((6, 7.0), (4, 4.8)):
        degrees = []
        for _ in range(20):
            g = newman_watts(100, k, 0.1, rng=rng)
            degrees.append(float(np.asarray(g.adjacency).sum(axis=1).mean()))
        mean = float(np.mean(degrees))
        assert mean > k, (k, mean)
        assert mean == pytest.approx(expected, abs=0.35), (k, mean)

    # The rewiring generator, by contrast, keeps the ring degree.
    ws = [
        float(np.asarray(watts_strogatz(100, 6, 0.1, rng=rng).adjacency).sum(axis=1).mean())
        for _ in range(10)
    ]
    assert float(np.mean(ws)) == pytest.approx(6.0, abs=0.15)
