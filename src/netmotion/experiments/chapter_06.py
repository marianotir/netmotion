"""Figures for chapter 6: propagation.

Thesis section 5.4.  The point of the chapter is that how fast something
crosses a network is a *measurement of the network*, so these figures are about
structure as much as about epidemics.
"""

from __future__ import annotations

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap

from ..networks import barabasi_albert, erdos_renyi, newman_watts, ring_lattice, watts_strogatz
from ..propagation import (
    alpha_max_size_law,
    mean_propagation_speed,
    propagate,
    propagation_speed,
)
from ..provenance import Origin, Source
from ..visualization import INK, SEQUENTIAL, SERIES, draw_network
from ..visualization.charts import histogram, mean_with_band, reference_line
from .registry import FigureContext, figure

THESIS_SEED = 20110930

#: How many networks are built at each point of the density and size sweeps.
#: <alpha> is already averaged over every seed node of a network, so the
#: remaining scatter is the network-to-network one, and three draws per point
#: left enough of it to put visible kinks in curves whose whole content is
#: their shape.  Twenty-five is what it takes for the saturation in F06-03 to
#: read as saturation rather than as a wobble.
SPEED_REALISATIONS = 25

#: Nodes the infection has not reached yet.  A neutral, not a paler blue: the
#: blue ramp is spent on *when* a node was reached, so "not yet" has to sit
#: outside that ramp rather than at the light end of it.
UNREACHED = "#e5e4dc"

#: Where a series is named directly on the plot instead of in a legend, the
#: label is given a thin halo of the page so it stays readable where it crosses
#: a line or a marker.
_HALO = [pe.withStroke(linewidth=2.6, foreground=INK["surface"])]


def _with_density(builder: str, n: int, d: float, rng):
    """Build a network of *n* nodes with about *d* links per node."""
    if builder == "ER":
        return erdos_renyi(n, min(1.0, 2.0 * d / (n - 1)), rng=rng)
    k = max(2, 2 * int(round(d)))
    if builder == "WS":
        return watts_strogatz(n, k, 0.1, rng=rng)
    if builder == "NW":
        return newman_watts(n, k, 0.1, rng=rng)
    if builder == "SF":
        return barabasi_albert(n, links_per_new_node=max(1, int(round(d))), rng=rng)
    raise ValueError(builder)


def _widest_gap(stack: np.ndarray) -> int:
    """Column of *stack* where the series are furthest apart.

    Direct labels beat a legend, but only where the curves are separated enough
    to carry them.  Rather than fixing the label position by hand -- which a
    different seed would invalidate -- the column with the largest minimum gap
    between neighbouring series is found and the names are written there.
    """
    best, best_gap = 0, -np.inf
    for j in range(stack.shape[1]):
        column = np.sort(stack[:, j])
        if not np.all(np.isfinite(column)):
            continue
        gap = float(np.min(np.diff(column)))
        if gap > best_gap:
            best, best_gap = j, gap
    return best


@figure(
    "F06-01",
    chapter=6,
    title="An infection crossing a network",
    source=Source(Origin.THESIS_2011, pdf_pages="43-45", section="5.4.1 Definicion y analisis"),
    seed=THESIS_SEED,
    reproduces_figure="thesis figs. 32-34, PDF pp. 42-44",
)
def propagation_snapshots(ctx: FigureContext):
    """PDF pp. 42-44: the infection advancing, step by step.

    A single node is infected at t = 1 and every neighbour of an infected node
    becomes infected at the next instant. Nodes are shaded by the instant they
    were reached, so the figure shows the distance profile of the seed as much
    as it shows an epidemic.

    The shading is an ordered quantity -- how long the node held out -- so it
    gets the sequential ramp and a key, and nodes not yet reached are drawn in
    a neutral outside that ramp rather than at its light end.
    """
    n, k, p = 60, 4, 0.1
    ctx.parameters(N=n, k=k, p=p, model="Watts-Strogatz")

    g = watts_strogatz(n, k, p, rng=ctx.rng)
    run = propagate(g, 0)
    instants = np.unique(np.linspace(1, run.curve.size, 4).astype(int))

    last = int(run.curve.size)
    ramp = [SEQUENTIAL[i] for i in np.linspace(2, len(SEQUENTIAL) - 1, last).round().astype(int)]
    cmap = ListedColormap(ramp)

    sizes = np.full(n, 26.0)
    sizes[run.first_infected] = 74.0

    rows = 2 if instants.size > 2 else 1
    cols = int(np.ceil(instants.size / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(2.75 * cols, 2.6 * rows))
    flat = np.atleast_1d(axes).ravel()

    for ax, t in zip(flat, instants, strict=False):
        colours = np.array(
            [ramp[int(s) - 1] if np.isfinite(s) and s <= t else UNREACHED for s in run.infected_at]
        )
        draw_network(g, ax, node_color=colours, node_size=sizes, edge_width=0.55)
        ax.set_title(
            f"$t = {t}$   {run.curve[t - 1]:.0%} reached",
            fontsize=10,
            loc="left",
            color=INK["primary"],
        )
    for ax in flat[instants.size :]:
        ax.set_axis_off()

    bar = fig.colorbar(
        ScalarMappable(norm=BoundaryNorm(np.arange(0.5, last + 1.5), last), cmap=cmap),
        ax=flat.tolist(),
        orientation="horizontal",
        fraction=0.035,
        pad=0.015,
        aspect=45,
        ticks=np.arange(1, last + 1),
    )
    bar.set_label("instant the node was reached", fontsize=8.5, color=INK["secondary"])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8, labelcolor=INK["secondary"])

    # No internal title: the printed caption carries it.
    return fig


@figure(
    "F06-02",
    chapter=6,
    title="The infected fraction, from every possible seed",
    source=Source(Origin.THESIS_2011, pdf_pages="45-47", section="5.4.1 Definicion y analisis"),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 35, PDF p. 45",
)
def propagation_curves(ctx: FigureContext):
    """PDF p. 45, fig. 35: one curve per starting node.

    Each curve is p(t), the infected fraction. They differ only in where the
    infection began, and the spread between them is a property of the network:
    a node at the centre reaches everything quickly, one on the rim does not.

    The thesis fits the rising part to an exponential and calls the rate alpha;
    the mean over seeds is what it treats as the network's characteristic.
    """
    n, k, p, realisations = 100, 4, 0.1, 1
    ctx.parameters(N=n, k=k, p=p, model="Watts-Strogatz")
    ctx.realisations(realisations)

    g = watts_strogatz(n, k, p, rng=ctx.rng)
    runs = [propagate(g, s) for s in range(n)]
    longest = max(r.curve.size for r in runs)

    fig, (left, right) = plt.subplots(1, 2, figsize=(6.8, 3.0))

    t = np.arange(1, longest + 1)
    for r in runs:
        padded = np.concatenate([r.curve, np.ones(longest - r.curve.size)])
        left.plot(t, padded, color=SERIES[0], alpha=0.16, lw=1.0)
    mean_curve = np.mean(
        [np.concatenate([r.curve, np.ones(longest - r.curve.size)]) for r in runs], axis=0
    )
    left.plot(t, mean_curve, color=SERIES[1], lw=2.2)

    # One direct label instead of a legend, led in from the empty upper left
    # rather than parked off the end of the axis, which would leave a third of
    # the panel blank to hold it.
    left.annotate(
        f"mean over {n} seeds",
        xy=(t[len(t) // 2], mean_curve[len(t) // 2]),
        xytext=(-8, 44),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=SERIES[1],
        arrowprops=dict(arrowstyle="-", lw=0.9, color=SERIES[1], shrinkA=2, shrinkB=3),
    )
    left.set_xlim(0.7, longest + 0.3)
    left.set_ylim(0, 1.04)
    left.set_xlabel("$t$")
    left.set_ylabel("$p(t)$")
    left.set_title(f"The curve from each\nof the {n} seeds")

    speeds = []
    for r in runs:
        try:
            speeds.append(propagation_speed(r.curve))
        except ValueError:
            continue
    histogram(right, speeds, bins=18)
    reference_line(
        right,
        float(np.mean(speeds)),
        rf"$\langle \alpha \rangle = {np.mean(speeds):.2f}$",
        axis="x",
        label_align="left",
    )
    right.set_xlabel(r"fitted speed $\alpha$")
    right.set_ylabel("seeds")
    # Room between the title and the top of the axes, where the reference line
    # writes its own label.
    right.set_title("and the speed fitted to it", pad=18)

    # No internal title: the printed caption carries it.  The two short titles
    # above identify the panels, which a single caption cannot.
    return fig


@figure(
    "F06-03",
    chapter=6,
    title="Propagation speed as a characteristic of the network",
    source=Source(Origin.THESIS_2011, pdf_pages="47-49", section="5.4.1 Definicion y analisis"),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis figs. 41-43, PDF pp. 47-49",
)
def speed_versus_density(ctx: FigureContext):
    """PDF pp. 47-49: how the mean speed depends on how dense the network is.

    For each network type, the mean propagation speed is measured over a sweep
    of link densities. The curves rise and then flatten: past a certain density
    adding links no longer makes the infection faster, because the network is
    already close to its minimum possible diameter.

    That saturation is what makes the maximum meaningful, and the maximum is
    what the next figure follows as a function of network size.

    Each point is the mean of SPEED_REALISATIONS networks of that type and
    density.  Four curves in four hues share these axes, so they carry no
    spread band -- four overlapping washes would muddy the one thing the panel
    is for, which is the ordering of the four types where they separate.  The
    realisations are still what makes that ordering readable: at three networks
    per point the WS and NW curves swapped places twice on their way up, which
    is a claim the figure was not making.
    """
    n = 120
    realisations = SPEED_REALISATIONS
    densities = np.array([1.5, 2, 3, 4, 6, 8, 11, 14, 18, 24], dtype=float)
    ctx.parameters(N=n, density=densities.tolist())
    ctx.realisations(realisations)

    fig, ax = plt.subplots(figsize=(6.0, 3.4))

    kinds = ["ER", "WS", "NW", "SF"]
    curves = []
    for i, kind in enumerate(kinds):
        samples = np.full((realisations, densities.size), np.nan)
        for j, d in enumerate(densities):
            for r in range(realisations):
                g = _with_density(kind, n, d, ctx.rng)
                try:
                    samples[r, j] = mean_propagation_speed(g)
                except ValueError:
                    continue
        means = mean_with_band(
            ax, densities, samples, color=SERIES[i], band=False, marker="o", ms=4.0
        )
        curves.append(means)

    # Four curves, four names, written on the curves themselves at the density
    # where they are furthest apart.  A legend would make the eye travel.
    # Four curves, four names, written on the curves themselves at the density
    # where they are furthest apart.  A legend would make the eye travel.
    #
    # The name sits at its curve's own height, on the end of a short leader,
    # rather than floating a few points above the point: with four series
    # separated by a tenth of the axis, a label lifted clear of its own curve
    # lands nearer to the one above it, and the reader has to guess.
    stack = np.asarray(curves, dtype=float)
    j = _widest_gap(stack)
    for i, kind in enumerate(kinds):
        ax.annotate(
            kind,
            xy=(densities[j], stack[i, j]),
            xytext=(20, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            weight="semibold",
            color=SERIES[i],
            path_effects=_HALO,
            arrowprops={
                "arrowstyle": "-",
                "color": SERIES[i],
                "lw": 0.8,
                "shrinkA": 3,
                "shrinkB": 2,
            },
        )

    ax.set_xlabel("links per node")
    ax.set_ylabel(r"$\langle \alpha \rangle$")
    # Headroom for the topmost direct label, which would otherwise sit on the
    # top of the axes.
    ax.set_ylim(top=float(np.nanmax(stack)) * 1.07)
    # No internal title: the printed caption carries it, N included.
    return fig


@figure(
    "F06-04",
    chapter=6,
    title="The maximum propagation speed grows as the logarithm of network size",
    source=Source(Origin.THESIS_2011, pdf_pages="49-50", section="5.4.1 Definicion y analisis"),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 44 and its fit, PDF p. 50",
)
def alpha_max_law(ctx: FigureContext):
    """PDF p. 50: the headline quantitative claim of the section.

    Take the maximum of the mean propagation speed over link density, and plot
    it against network size. The thesis reports that the result is the same
    straight line in ln N for all four network types:

        alpha_max(N) = 0.4974 ln N + 0.012

    The sizes are 80, 160 and 320 nodes, and the densities run from 4 to 48
    links per node, up to N/4.

    Because the four types land on the same three points to within the width of
    a marker, colouring them apart would be a promise the plot cannot keep. The
    left panel therefore says only "all four types", in one hue, and the right
    panel carries the per-type identity, where the difference between the types
    is actually resolvable.
    """
    sizes = np.array([80, 160, 320])
    realisations = 3
    ctx.parameters(
        N=sizes.tolist(),
        density="4..48 links/node, capped at N/4 to stay in the three-instant regime",
    )
    ctx.realisations(realisations)

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.0, 3.2), gridspec_kw={"width_ratios": (1.05, 1.0)}
    )

    kinds = ["ER", "WS", "NW", "SF"]
    fitted = {}
    for kind in kinds:
        peaks = []
        for n in sizes:
            # The densities run up to N/4 links per node, the range the figure uses.
            densities = [d for d in (4.0, 8.0, 12.0, 18.0, 24.0, 32.0, 48.0) if d <= n / 4]
            best = 0.0
            for d in densities:
                values = []
                for _ in range(realisations):
                    g = _with_density(kind, int(n), float(d), ctx.rng)
                    try:
                        values.append(mean_propagation_speed(g))
                    except ValueError:
                        continue
                if values:
                    best = max(best, float(np.mean(values)))
            peaks.append(best)
        peaks = np.array(peaks)
        left.plot(sizes, peaks, "o", ms=7, color=SERIES[0], alpha=0.55, zorder=3)
        fitted[kind] = np.polyfit(np.log(sizes), peaks, 1)[0]

    grid = np.linspace(sizes.min() * 0.88, sizes.max() * 1.14, 200)
    left.plot(grid, alpha_max_size_law(grid), color=INK["muted"], lw=1.0, ls=(0, (4, 3)), zorder=1)
    # Anchored growing *left* from a point in the right half of the axes
    # (ha="right", a negative x offset) rather than growing right off the
    # edge: the previous ha="left" placement widened with every narrowing
    # pass, which is exactly the overflow fit_to_print_width's width guard
    # cannot see coming from outside the axes' own extent. "the 2011 law"
    # is dropped too -- the right panel already names 1911's coefficient.
    left.annotate(
        r"$0.4974\,\ln N + 0.012$",
        xy=(grid[145], alpha_max_size_law(grid[145])),
        xytext=(-4, 6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK["secondary"],
    )
    # Matches the docstring's own description of this label ("all four
    # types"), not the four abbreviations spelled out.  Centred over its
    # point rather than hung off one side, which keeps the box's width from
    # growing lopsidedly against either axis edge as the panel narrows.
    left.annotate(
        "all four types,\nwithin marker width",
        xy=(sizes[1], alpha_max_size_law(sizes[1])),
        xytext=(0, 24),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=SERIES[0],
        arrowprops=dict(arrowstyle="-", lw=0.8, color=SERIES[0], shrinkA=1, shrinkB=6),
    )

    left.set_xscale("log")
    left.set_xticks(sizes)
    left.set_xticklabels([str(int(s)) for s in sizes])
    left.set_xticks([], minor=True)
    left.set_xlabel("$N$")
    left.set_ylabel(r"$\alpha_{\max}$")
    left.set_title("Measured maxima")

    # The interesting number is the coefficient, and the four coefficients
    # differ in the third decimal: a bar from zero would draw four identical
    # bars.  Dots on a window around 1/2 show the difference that exists.
    values = np.array([fitted[k] for k in kinds])
    positions = np.arange(len(kinds))[::-1]
    lo = min(float(values.min()), 0.4974)
    hi = max(float(values.max()), 0.5)
    span = max(hi - lo, 0.004)
    right.set_xlim(lo - 0.45 * span, hi + 1.05 * span)

    for v, y in zip(values, positions, strict=True):
        # A hairline leader from the name to the dot: with no ruler under the
        # panel the eye needs something to carry it across.
        right.plot(
            [right.get_xlim()[0], v],
            [y, y],
            color=INK["grid"],
            lw=0.7,
            zorder=1,
            solid_capstyle="butt",
        )
        right.annotate(
            f"{v:.3f}",
            xy=(v, y),
            xytext=(9, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            color=INK["secondary"],
            path_effects=_HALO,
        )
    right.plot(values, positions, "o", ms=7, color=SERIES[0], zorder=4)

    reference_line(right, 0.5, r"exactly $\frac{1}{2}$", axis="x")
    right.axvline(0.4974, color=SERIES[1], lw=0.9, ls=(0, (4, 3)), zorder=2)
    right.annotate(
        "0.4974",
        xy=(0.4974, 0.0),
        xycoords=("data", "axes fraction"),
        xytext=(0, -10),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=7.5,
        color=SERIES[1],
        annotation_clip=False,
    )

    right.set_ylim(-0.6, len(kinds) - 0.4)
    right.set_yticks(positions)
    right.set_yticklabels(kinds)
    right.set_xticks([])
    right.grid(False)
    right.spines["bottom"].set_visible(False)
    # Room above the axes for the reference line's own label.
    right.set_title("Fitted coefficient", pad=24)

    # No internal title: the printed caption carries it.  The two short titles
    # above identify the panels, which a single caption cannot.
    return fig


@figure(
    "F06-05",
    chapter=6,
    title="Propagation speed separates network types",
    source=Source(
        Origin.EXTENSION,
        pdf_pages="47-50",
        section="5.4.1 Definicion y analisis",
        note=(
            "2026 EXTENSION. The thesis measures propagation speed on each "
            "network type separately; putting the four on one axis at matched "
            "size and density is a presentational addition, and it makes the "
            "chapter's own argument -- that speed is a structural measurement -- "
            "directly visible."
        ),
    ),
    seed=THESIS_SEED,
    slow=True,
)
def speed_by_type(ctx: FigureContext):
    """2026 extension: the same size, the same density, four topologies.

    If propagation speed is a characteristic of a network rather than of an
    epidemic, then networks that differ only in topology should differ in
    speed. They do, and the ordering is the one the structure predicts: the
    ordered ring is slowest because everything must be crossed step by step,
    and the scale-free network is fastest because its hubs are shortcuts.

    Drawn in the grammar of ``ranked_bars`` -- ranked, one hue, the value
    written on, no ruler -- but as a range and a median rather than a bar,
    because there are eight realisations behind each row and the width of that
    spread is part of the answer.
    """
    n, realisations = 120, 8
    ctx.parameters(N=n, density="~2 links/node")
    ctx.realisations(realisations)

    builders = {
        "Ring lattice": lambda: ring_lattice(n, 4),
        "Erdos-Renyi": lambda: erdos_renyi(n, 4 / (n - 1), rng=ctx.rng),
        "Watts-Strogatz": lambda: watts_strogatz(n, 4, 0.1, rng=ctx.rng),
        "Newman-Watts": lambda: newman_watts(n, 4, 0.1, rng=ctx.rng),
        "Scale-free": lambda: barabasi_albert(n, links_per_new_node=2, rng=ctx.rng),
    }

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    data, labels = [], []
    for label, build in builders.items():
        values = []
        for _ in range(realisations):
            g = build()
            try:
                values.append(mean_propagation_speed(g))
            except ValueError:
                continue
        data.append(np.asarray(values, dtype=float))
        labels.append(label)

    order = np.argsort([np.median(v) for v in data])
    highest = int(np.argmax([np.median(v) for v in data]))

    for y, i in enumerate(order):
        values = data[i]
        colour = SERIES[1] if i == highest else SERIES[0]
        ax.hlines(y, values.min(), values.max(), color=colour, lw=3.0, alpha=0.28, zorder=2)
        ax.plot(values, np.full(values.size, y), "o", ms=3.6, color=colour, alpha=0.6, zorder=3)
        median = float(np.median(values))
        ax.plot([median], [y], "o", ms=8, color=colour, zorder=4)
        ax.annotate(
            f"{median:.2f}",
            xy=(values.max(), y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK["secondary"],
        )

    slowest = int(order[0])
    ax.annotate(
        "no shortcuts: every step\nof the ring must be walked",
        xy=(data[slowest].max(), 0),
        xytext=(48, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=INK["muted"],
    )
    ax.annotate(
        "hubs are shortcuts",
        xy=(float(np.median(data[highest])), len(data) - 1),
        xytext=(0, 13),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        color=SERIES[1],
    )

    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels([labels[i] for i in order])
    ax.set_ylim(-0.7, len(data) - 0.4)
    ax.set_xlim(0, max(v.max() for v in data) * 1.22)
    ax.set_xlabel(r"$\langle \alpha \rangle$   (dot: median of 8 realisations)")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    # No internal title: the printed caption carries it, N included.
    return fig
