"""How the degree distribution of a grown network depends on its parameters.

Three sweeps: over the size of the network, over how many partners each
arriving node takes when attachment leaves it short, and over the rule used to
choose those partners.

Both parameters swept here -- the size N and the number of partners m -- are
ordered quantities, so each is carried by one sequential ramp running light to
dark.  Five sizes in five unrelated hues would invite the reader to look for a
meaning in the hues that is not there.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..networks import barabasi_albert
from ..provenance import Origin, Source
from ..visualization import INK, SEQUENTIAL
from .registry import FigureContext, figure

THESIS_SEED = 20110930


def _pk(network):
    """Degree distribution as ``(k, P(k))`` over the degrees present."""
    k = network.degrees()
    values, counts = np.unique(k[k > 0], return_counts=True)
    return values, counts / k.size


def _power_law(k, pk):
    """Least-squares straight line through the points in log-log."""
    mask = (k > 0) & (pk > 0)
    slope, intercept = np.polyfit(np.log(k[mask]), np.log(pk[mask]), 1)
    return -slope, np.exp(intercept)


@figure(
    "T17",
    chapter=4,
    title="Degree distribution against network size",
    source=Source(Origin.THESIS_2011, pdf_pages="32", section="4.4.3 Caracteristicas"),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="fig. 17",
)
def scale_free_sizes(ctx: FigureContext):
    """P(k) for five network sizes, with the fitted power law drawn through them.

    Growing the network larger extends the tail without moving the exponent:
    the distribution is scale free, so its shape does not depend on how much of
    it has been built.  The five sizes share one sequential ramp, because N is
    an ordering rather than five names.

    The reference line is the mean of the five fits -- both its slope and its
    height come from those fits, so it lies on the points it summarises instead
    of running parallel to them further down the panel.
    """
    sizes = (750, 1000, 1250, 1500, 1750)
    m = 2
    ctx.parameters(N=list(sizes), m=m, completion="random")
    ctx.realisations(1)

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ramp = (SEQUENTIAL[1], SEQUENTIAL[3], SEQUENTIAL[5], SEQUENTIAL[7], SEQUENTIAL[9])

    exponents = []
    amplitudes = []
    kmax = 0
    for i, n in enumerate(sizes):
        g = barabasi_albert(n, links_per_new_node=m, fallback="random", rng=ctx.rng)
        k, pk = _pk(g)
        ax.loglog(k, pk, marker="o", ms=4.0, lw=0, color=ramp[i], label=f"$N = {n}$")
        gamma, amp = _power_law(k, pk)
        exponents.append(gamma)
        amplitudes.append(amp)
        kmax = max(kmax, int(k.max()))

    ctx.parameters(gamma=[round(float(g), 2) for g in exponents])
    gamma = float(np.mean(exponents))
    amp = float(np.exp(np.mean(np.log(amplitudes))))
    grid = np.logspace(np.log10(2), np.log10(kmax), 60)
    ax.loglog(
        grid,
        amp * grid**-gamma,
        color=INK["muted"],
        lw=1.2,
        ls=(0, (4, 3)),
        label="the mean of the five fits",
    )

    ax.set_xlabel("$k$")
    ax.set_ylabel("$P(k)$")
    ax.set_ylim(3e-4, 1.4)
    ax.set_xlim(1.7, 1.4 * kmax)
    ax.set_xticks([2, 3, 5, 10, 20, 50])
    ax.set_xticklabels(["2", "3", "5", "10", "20", "50"])
    # No title: the caption says a bigger network lengthens the tail without
    # moving the slope.  The parameter line below stays, and the fitted exponent
    # is written against the line it belongs to.
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, which="major", axis="both")
    fig.suptitle(
        f"Grown by preferential attachment, $m = {m}$, one realisation per size",
        fontsize=8.5,
        color=INK["secondary"],
        x=0.0,
        ha="left",
    )
    return fig


def _connection_criterion_figure(ctx, fallback, note, *, emphasise: bool = False, tail: bool = False):
    """P(k) at N = 1000 for m = 2, 3 and 4, each with its own fitted line.

    m is an ordering, so the three series run light to dark along one ramp,
    each fitted line taking the colour of the points it belongs to.  The fits
    are drawn only over the range the data covers rather than extrapolated
    across empty axes.
    """
    n = 1000
    counts = (2, 3, 4)
    ctx.parameters(N=n, m=list(counts), completion=fallback)
    ctx.realisations(1)

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ramp = (SEQUENTIAL[2], SEQUENTIAL[5], SEQUENTIAL[9])
    fitted = []
    kmax = 0

    measured = []
    for m in counts:
        g = barabasi_albert(n, links_per_new_node=m, fallback=fallback, rng=ctx.rng)
        k, pk = _pk(g)
        # The fit uses the body of the distribution, where the points are dense,
        # or, with *tail*, the part beyond k = 6 where P(k) falls away.
        body = (k >= 6) if tail else (k <= max(10, int(np.percentile(k, 90))))
        gamma, amp = _power_law(k[body], pk[body])
        measured.append((k, pk, gamma, amp))
        kmax = max(kmax, int(k.max()))

    # Fig. 19 reads its exponent off one of the three, "the data of the graph
    # represented by a larger symbol", so that series is drawn larger: the one
    # whose exponent is the largest, which is the one the text reports as
    # exceeding the theoretical 3.
    picked = int(np.argmax([gamma for _, _, gamma, _ in measured])) if emphasise else -1
    for i, ((k, pk, gamma, amp), m) in enumerate(zip(measured, counts, strict=True)):
        ax.loglog(
            k,
            pk,
            marker="o",
            ms=7.6 if i == picked else 4.2,
            lw=0,
            color=ramp[i],
            label=f"$m = {m}$ partners",
        )
        fitted.append(gamma)
        grid = np.logspace(np.log10(k.min()), np.log10(k.max()), 40)
        ax.loglog(grid, amp * grid**-gamma, color=ramp[i], lw=1.1, ls=(0, (4, 3)))

    ax.set_xlabel("$k$")
    ax.set_ylabel("$P(k)$")
    ax.set_ylim(4e-4, 1.4)
    ax.set_xlim(1.7, 1.4 * kmax)
    ax.set_xticks([2, 3, 5, 10, 20, 50])
    ax.set_xticklabels(["2", "3", "5", "10", "20", "50"])
    # No title: the caption carries the reading.  The parameter line keeps the
    # fitted exponents, which are measurements and belong on the figure.
    ctx.parameters(gamma=[round(float(g), 2) for g in fitted])
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, which="major", axis="both")
    fig.suptitle(
        note.format(g=", ".join(f"{x:.2f}" for x in fitted)),
        fontsize=8.5,
        color=INK["secondary"],
        x=0.0,
        ha="left",
    )
    return fig


@figure(
    "T18",
    chapter=4,
    title="Degree distribution against the number of partners each arrival takes",
    source=Source(Origin.THESIS_2011, pdf_pages="33", section="4.4.3 Caracteristicas"),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="fig. 18",
)
def connection_criterion_random(ctx: FigureContext):
    """N = 1000, with each arriving node taking 2, 3 or 4 partners at random
    when preferential attachment leaves it short.

    Taking more partners raises the whole curve and thickens the tail, but the
    three distributions stay close to parallel: the exponent is a property of
    the attachment rule, not of how many links it is asked to place.
    """
    return _connection_criterion_figure(
        ctx,
        "random",
        "$N = 1000$, random completion",
    )


@figure(
    "T19",
    chapter=4,
    title="Degree distribution when the arrival attaches to its nearest neighbours",
    source=Source(Origin.THESIS_2011, pdf_pages="34", section="4.4.3 Caracteristicas"),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="fig. 19",
)
def connection_criterion_nearest(ctx: FigureContext):
    """The same sweep, with an arrival that falls short attaching instead to
    the nodes nearest it in the plane.

    The distribution changes character: P(k) varies little for small k and
    then, from a certain k, falls away rapidly.  As the text does, the exponent
    is fitted over the part that falls, k >= 6, and the series whose exponent
    is largest is drawn with the larger symbol.
    """
    return _connection_criterion_figure(
        ctx,
        "nearest",
        "$N = 1000$, nearest-neighbour completion",
        emphasise=True,
        tail=True,
    )
