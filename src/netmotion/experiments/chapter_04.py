"""Figures for chapter 4: the four network models.

Every figure here reproduces a result the thesis reports, at the thesis's own
parameters, and says on its face which page it comes from.  Run them with::

    python -m netmotion.experiments.cli --chapter 4

The drawing decisions follow :mod:`netmotion.visualization.charts`: one hue
unless a second one carries information the reader needs, direct labels rather
than legends, and hairline references labelled clear of the data.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from ..metrics import clustering_coefficient, mean_degree, mean_distance
from ..networks import (
    barabasi_albert,
    connectivity_threshold,
    erdos_renyi,
    newman_watts,
    ring_lattice,
    watts_strogatz,
)
from ..provenance import Origin, Source
from ..visualization import INK, SEQUENTIAL, SERIES, circular_layout, draw_network
from ..visualization.charts import panel_title, reference_line
from .registry import FigureContext, figure

THESIS_SEED = 20110930  # the date of submission, used so the seeds mean something


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------


def _direct_label(ax, text, xy, offset, *, color=None, ha="left", va="center", size=8.5):
    """Write a series' name beside the series, instead of into a legend.

    Two or three curves do not need a key: the eye should not have to travel to
    the corner of the figure and back to find out which line it is looking at.
    """
    ax.annotate(
        text,
        xy=xy,
        xytext=offset,
        textcoords="offset points",
        color=color or INK["secondary"],
        fontsize=size,
        ha=ha,
        va=va,
        annotation_clip=False,
    )


def _panel_note(ax, text, *, color=None):
    """A one-line caption under a network drawing."""
    ax.annotate(
        text,
        xy=(0.5, -0.01),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=8,
        color=color or INK["secondary"],
    )


def _draw_split_edges(g, ax, is_marked, *, node_color=SERIES[0], accent=SERIES[1], width=0.9):
    """Draw a network on the circle with one class of link picked out.

    Colour is spent on the distinction the surrounding paragraph is about --
    which links a rewiring has moved off the ring, which chords join second
    neighbours -- and on nothing else.  Returns how many links were marked.
    """
    pos = circular_layout(g.n)
    plain: list = []
    marked: list = []
    for i, j, _ in g.edges():
        (marked if is_marked(i, j) else plain).append((pos[i], pos[j]))
    if plain:
        ax.add_collection(
            LineCollection(plain, colors=INK["muted"], linewidths=width, alpha=0.8, zorder=1)
        )
    if marked:
        ax.add_collection(
            LineCollection(marked, colors=accent, linewidths=width + 0.8, alpha=0.95, zorder=2)
        )
    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        s=max(6.0, 260.0 / max(g.n, 1) ** 0.55),
        c=node_color,
        zorder=3,
        linewidths=0,
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.margins(0.06)
    return len(marked)


def _ring_distance(i: int, j: int, n: int) -> int:
    """How many steps apart two nodes are around the ring."""
    d = abs(i - j) % n
    return min(d, n - d)


# ---------------------------------------------------------------------------
# 4.1 Erdos-Renyi
# ---------------------------------------------------------------------------


@figure(
    "F04-01",
    chapter=4,
    title="Erdos-Renyi: the network fills in as p grows",
    source=Source(Origin.THESIS_2011, pdf_pages="12-13", section="4.1.2 Algoritmo"),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 2, PDF p. 13",
)
def er_growth(ctx: FigureContext):
    """PDF p. 13, fig. 2: N = 10 at four connection probabilities.

    The thesis draws p = 0, 0.1, 0.15 and 0.25 and observes that by p = 0.25
    every node is joined -- consistent with the threshold p_c = ln(N)/N, which
    for N = 10 is about 0.23.

    That claim is about the nodes left out, so the nodes left out are what the
    second colour marks: a grey dot is a node with no link yet, and the count
    under each panel says how many are left.  By p = 0.25 there are none.
    """
    n = 10
    probabilities = (0.0, 0.10, 0.15, 0.25)
    ctx.parameters(N=n, p=list(probabilities))

    # Two by two, not four in a row: a strip of panels scaled to the text
    # width loses the detail that is the whole point of a network drawing.
    fig, axes = plt.subplots(2, 2, figsize=(4.9, 4.7))
    fig.get_layout_engine().set(hspace=0.10, wspace=0.02)
    for ax, p in zip(axes.ravel(), probabilities, strict=True):
        g = erdos_renyi(n, p, rng=ctx.rng)
        alone = g.degrees() == 0
        colours = [INK["axis"] if a else SERIES[0] for a in alone]
        draw_network(g, ax, node_color=colours, edge_width=1.0)
        ax.set_title(f"$p = {p:g}$", fontsize=10.5, loc="center")
        count = int(alone.sum())
        _panel_note(
            ax,
            f"{g.n_edges} links   ·   "
            + ("every node joined" if count == 0 else f"{count} still alone"),
            color=INK["secondary"] if count else SERIES[0],
        )

    # No suptitle: the caption carries the threshold and $N$.  The per-panel
    # $p$ titles stay, because they identify the panels.
    return fig


@figure(
    "F04-02",
    chapter=4,
    title="Erdos-Renyi: clustering and mean degree against theory",
    source=Source(Origin.THESIS_2011, pdf_pages="14-16", section="4.1.3 Caracteristicas"),
    seed=THESIS_SEED,
    reproduces_figure="thesis figs. 5.a and 5.b, PDF p. 16",
)
def er_characteristics(ctx: FigureContext):
    """PDF p. 16, figs. 5.a and 5.b.

    The thesis plots the measured mean clustering coefficient and mean degree
    against p, with the theoretical curves C = p and <k> = p(N-1) drawn over
    them, and concludes that they coincide.

    They do, and one consequence of the chapter's own clustering convention is
    visible at small p. That convention scores a node of degree 0 or 1 as
    C_i = 0 rather than leaving it undefined, and at small p most nodes have
    degree 0 or 1, so the measured curve sits below C = p there. The deviation
    is a property of the definition, not of the sampling.
    """
    n, realisations = 10, 300
    ps = np.linspace(0.0, 1.0, 41)
    ctx.parameters(N=n, p="0..1 in 41 steps")
    ctx.realisations(realisations)

    clustering = np.empty_like(ps)
    degree = np.empty_like(ps)
    for i, p in enumerate(ps):
        runs = [erdos_renyi(n, p, rng=ctx.rng) for _ in range(realisations)]
        clustering[i] = np.mean([clustering_coefficient(g) for g in runs])
        degree[i] = np.mean([mean_degree(g) for g in runs])

    fig, (left, right) = plt.subplots(1, 2, figsize=(6.6, 3.1), sharex=True)

    # One hue for what was measured, a dark hairline for what was predicted:
    # the two are not two categories, they are the data and the reference.
    left.plot(ps, clustering, color=SERIES[0], lw=1.8)
    left.plot(ps, ps, color=INK["primary"], lw=1.0, ls=(0, (4, 3)))
    _direct_label(left, "theory  $C = p$", (0.42, 0.42), (-6, 9), ha="right", va="bottom")
    _direct_label(
        left, "measured", (0.66, clustering[27]), (6, -8), color=SERIES[0], ha="left", va="top"
    )
    left.set_ylabel("$C$")
    left.set_title("Clustering falls short\nat small $p$")
    left.annotate(
        "nodes with $k<2$ are\nscored $C_i=0$, not dropped",
        xy=(0.22, 0.115),
        xytext=(0.30, 0.02),
        fontsize=7.5,
        color=INK["muted"],
        arrowprops=dict(arrowstyle="-", color=INK["muted"], lw=0.6),
    )

    # The two coincide, so both names point at the same line from either side of
    # it: that they land on one curve is the result.
    right.plot(ps, degree, color=SERIES[0], lw=2.8)
    right.plot(ps, ps * (n - 1), color=INK["primary"], lw=1.0, ls=(0, (4, 3)))
    # Both names sit clear of the line, in the two corners it leaves empty.
    right.annotate(
        "measured",
        xy=(0.45, 0.45 * (n - 1)),
        xytext=(0.60, 1.5),
        textcoords="data",
        fontsize=8.5,
        color=SERIES[0],
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=SERIES[0], lw=0.7),
    )
    right.annotate(
        r"theory  $\langle k \rangle = pN$",
        xy=(0.55, 0.55 * (n - 1)),
        xytext=(0.03, 7.7),
        textcoords="data",
        fontsize=8.5,
        color=INK["secondary"],
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=INK["muted"], lw=0.7),
    )
    right.set_ylabel(r"$\langle k \rangle$")
    right.set_title("Mean degree matches\nexactly")

    fig.supxlabel("$p$", fontsize=9.5, color=INK["secondary"])
    return fig


@figure(
    "F04-03",
    chapter=4,
    title="Erdos-Renyi: the connectivity threshold",
    source=Source(Origin.THESIS_2011, pdf_pages="14", section="4.1.3 Caracteristicas"),
    seed=THESIS_SEED,
    reproduces_figure="thesis figs. 3.a and 3.b, PDF p. 14",
)
def er_threshold(ctx: FigureContext):
    """PDF p. 14: below p_c not every node is joined; above it, they are.

    The thesis makes the point with two drawings, one either side of p_c = 0.23.
    Two drawings cannot show a threshold, so the reproduction adds the curve
    they are two points on: the fraction of realisations that come out
    connected, for several network sizes. The 2011 claim is the vertical line.

    N is an ordered quantity, so the three curves take a sequential ramp rather
    than three unrelated hues: dark is the large network, and the reader can see
    the rise marching left as N grows.
    """
    sizes = (10, 25, 50)
    trials = 400
    ps = np.linspace(0.01, 0.8, 60)
    ctx.parameters(N=list(sizes), trials_per_point=trials)
    ctx.realisations(trials)

    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    ramp = (SEQUENTIAL[3], SEQUENTIAL[6], SEQUENTIAL[9])
    curves = []

    for i, n in enumerate(sizes):
        connected = np.array(
            [
                np.mean(
                    [len(_components(erdos_renyi(n, p, rng=ctx.rng))) == 1 for _ in range(trials)]
                )
                for p in ps
            ]
        )
        curves.append(connected)
        ax.plot(ps, connected, color=ramp[i], lw=1.8, zorder=3)

        # Name each curve at a different height up its rise, so that three
        # labels for three curves that run close together never touch.
        j = int(np.argmax(connected >= (0.9, 0.6, 0.3)[i]))
        _direct_label(
            ax, f"$N = {n}$", (ps[j], connected[j]), (-7, 7), color=ramp[i], ha="right", va="bottom"
        )

        # The 2011 claim, as a mark on the curve rather than a rule across it.
        pc = connectivity_threshold(n)
        at_pc = float(np.interp(pc, ps, connected))
        ax.vlines(pc, 0, at_pc, color=SERIES[1], lw=0.8, alpha=0.55, zorder=2)
        ax.plot([pc], [at_pc], "o", ms=5.0, color=SERIES[1], zorder=4)

    pc10 = connectivity_threshold(sizes[0])
    ax.annotate(
        r"$p_c = \ln N / N$",
        xy=(pc10, float(np.interp(pc10, ps, curves[0]))),
        xytext=(26, -14),
        textcoords="offset points",
        fontsize=8.5,
        color=SERIES[1],
        arrowprops=dict(arrowstyle="-", color=SERIES[1], lw=0.7),
    )

    ax.set_xlabel("$p$")
    ax.set_ylabel("realisations that come out connected")
    ax.set_ylim(-0.03, 1.06)
    ax.set_xlim(0.0, 0.82)
    # No title: one panel, and the caption carries it.
    return fig


def _components(network):
    from ..metrics import connected_components

    return connected_components(network)


# ---------------------------------------------------------------------------
# 4.2 Watts-Strogatz
# ---------------------------------------------------------------------------


@figure(
    "F04-04",
    chapter=4,
    title="Watts-Strogatz: from ring to random",
    source=Source(Origin.THESIS_2011, pdf_pages="17-21", section="4.2 Small World Network"),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 6, PDF pp. 20-21",
)
def ws_rewiring(ctx: FigureContext):
    """PDF pp. 20-21, fig. 6: N = 20, k = 4, at p = 0, 0.1, 0.5 and 1.

    The ordered ring is on the left. As p grows, links leave the ring and land
    anywhere, and the degree distribution spreads out around its constant mean.

    Which links have left is the whole content of the sequence, so that is what
    the second colour says: grey links still join near neighbours around the
    ring, coloured ones have been moved somewhere else.
    """
    n, k = 20, 4
    probabilities = (0.0, 0.1, 0.5, 1.0)
    ctx.parameters(N=n, k=k, p=list(probabilities))

    fig, axes = plt.subplots(2, 2, figsize=(5.0, 5.0))
    fig.get_layout_engine().set(hspace=0.10, wspace=0.02)
    for ax, p in zip(axes.ravel(), probabilities, strict=True):
        g = ring_lattice(n, k) if p == 0 else watts_strogatz(n, k, p, rng=ctx.rng)
        moved = _draw_split_edges(g, ax, lambda i, j: _ring_distance(i, j, n) > k // 2, width=0.9)
        ax.set_title(f"$p = {p:g}$", fontsize=10.5, loc="center")
        if moved:
            _panel_note(ax, f"{moved} of {g.n_edges} links have left the ring", color=SERIES[1])
        else:
            _panel_note(ax, "the ordered ring, every link still local")

    # No suptitle: the caption says what the rewiring does, and carries $N$
    # and $k$.  The per-panel $p$ titles stay.
    return fig


@figure(
    "F04-05",
    chapter=4,
    title="The small-world effect",
    source=Source(Origin.THESIS_2011, pdf_pages="22-23", section="4.2.3 Caracteristicas"),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 9, PDF p. 23",
)
def ws_small_world_effect(ctx: FigureContext):
    """PDF p. 23, fig. 9: the central result of the chapter.

    Normalised mean distance and clustering against the rewiring probability,
    for N = 30, 50 and 100. Between the ordered and the random extremes there
    is a range where the distance has already collapsed and the clustering has
    not -- and the thesis's point is that the gap opens wider as N grows,
    because a long ring has much more distance to lose.

    Six curves on one axes need a six-line key and are read one pair at a time
    anyway, so the sizes become the columns of a small multiple.  The shaded
    band between the two curves *is* the small-world window, and it is the thing
    the reader is being asked to compare across the row.
    """
    sizes = (30, 50, 100)
    k, realisations = 6, 20
    ps = np.logspace(-3.0, 0.0, 22)
    ctx.parameters(N=list(sizes), k=k, p="1e-3..1 log-spaced, 22 points")
    ctx.realisations(realisations)

    measured = []
    for n in sizes:
        ring = ring_lattice(n, k)
        l0 = mean_distance(ring, "largest_component")
        c0 = clustering_coefficient(ring)

        lengths, clusters = np.empty(ps.size), np.empty(ps.size)
        for j, p in enumerate(ps):
            runs = [watts_strogatz(n, k, p, rng=ctx.rng) for _ in range(realisations)]
            lengths[j] = np.mean([mean_distance(g, "largest_component") for g in runs])
            clusters[j] = np.mean([clustering_coefficient(g) for g in runs])
        measured.append((lengths / l0, clusters / c0))

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9), sharex=True, sharey=True)
    fig.get_layout_engine().set(wspace=0.06)

    for ax, n, (length, cluster) in zip(axes, sizes, measured, strict=True):
        ax.fill_between(ps, length, cluster, color=SERIES[1], alpha=0.15, lw=0, zorder=1)
        ax.plot(ps, cluster, color=SERIES[1], lw=1.8, zorder=3)
        ax.plot(ps, length, color=SERIES[0], lw=1.8, zorder=3)
        ax.set_title(f"$N = {n}$", fontsize=10, loc="center")
        reference_line(ax, 1.0, "the ordered ring" if n == sizes[-1] else "")

    # The two quantities are named once, in the panel where the band between
    # them is widest and the names cannot land on top of one another.
    last = axes[-1]
    length, cluster = measured[-1]
    _direct_label(
        last,
        r"$C(p)/C(0)$",
        (ps[10], cluster[10]),
        (0, 8),
        color=SERIES[1],
        ha="center",
        va="bottom",
    )
    _direct_label(
        last,
        r"$\ell(p)/\ell(0)$",
        (ps[10], length[10]),
        (0, -9),
        color=SERIES[0],
        ha="center",
        va="top",
    )

    axes[0].set_xscale("log")
    axes[0].set_ylim(0.0, 1.16)
    axes[0].set_ylabel("value, relative to the ring")
    fig.supxlabel("$p$", fontsize=9.5, color=INK["secondary"])
    # No figure title: the book's caption carries it.  The panel titles stay,
    # because the caption cannot say which column is which.
    return fig


# ---------------------------------------------------------------------------
# 4.3 Newman-Watts
# ---------------------------------------------------------------------------


@figure(
    "F04-06",
    chapter=4,
    title="Newman-Watts: adding shortcuts instead of moving links",
    source=Source(Origin.THESIS_2011, pdf_pages="23-26", section="4.3 Newman-Watts"),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis section 4.3.3, PDF pp. 24-26",
)
def nw_versus_ws(ctx: FigureContext):
    """PDF pp. 23-26: the one-line difference, and what it does.

    Newman-Watts deletes the instruction that breaks the ring link. So the
    ring survives, links only accumulate, and the mean degree rises with p --
    where Watts-Strogatz holds it fixed. Both shorten paths; only one of them
    does it without ever taking a link away.
    """
    n, k, realisations = 100, 4, 20
    ps = np.logspace(-2.5, 0.0, 18)
    ctx.parameters(N=n, k=k, p="1e-2.5..1 log-spaced, 18 points")
    ctx.realisations(realisations)

    ws_len, ws_deg, nw_len, nw_deg = (np.empty(ps.size) for _ in range(4))
    for j, p in enumerate(ps):
        ws = [watts_strogatz(n, k, p, rng=ctx.rng, variant="standard") for _ in range(realisations)]
        nw = [newman_watts(n, k, p, rng=ctx.rng) for _ in range(realisations)]
        ws_len[j] = np.mean([mean_distance(g, "largest_component") for g in ws])
        nw_len[j] = np.mean([mean_distance(g, "largest_component") for g in nw])
        ws_deg[j] = np.mean([mean_degree(g) for g in ws])
        nw_deg[j] = np.mean([mean_degree(g) for g in nw])

    fig, (left, right) = plt.subplots(1, 2, figsize=(6.8, 3.0), sharex=True)

    # Each model is named at the end of its own curve, in both panels: two
    # names beside two lines, rather than the same key printed twice.
    left.plot(ps, ws_len, color=SERIES[0], lw=1.8)
    left.plot(ps, nw_len, color=SERIES[1], lw=1.8)
    _direct_label(
        left, "Watts-Strogatz", (ps[-1], ws_len[-1]), (-4, 9), color=SERIES[0], ha="right"
    )
    # At the end of the curve this label fell below the axis and printed on top
    # of the tick labels, so it is set further back along the line, where there
    # is room for it under the curve and inside the axes.
    _direct_label(
        left,
        "Newman-Watts",
        (ps[-5], nw_len[-5]),
        (-2, -9),
        color=SERIES[1],
        ha="right",
        va="top",
    )
    left.set_xscale("log")
    left.set_ylabel(r"$\ell$")
    # Two panel titles on a six-inch figure have three inches each; the longer
    # forms ran into one another across the gutter.
    left.set_title("Both shorten the paths", fontsize=9.5)

    right.plot(ps, ws_deg, color=SERIES[0], lw=1.8)
    right.plot(ps, nw_deg, color=SERIES[1], lw=1.8)
    _direct_label(
        right,
        "Newman-Watts",
        (ps[-2], nw_deg[-2]),
        (-6, 2),
        color=SERIES[1],
        ha="right",
        va="bottom",
    )
    _direct_label(
        right, "Watts-Strogatz", (ps[-1], ws_deg[-1]), (-4, 9), color=SERIES[0], ha="right"
    )
    right.set_xscale("log")
    right.set_ylabel(r"$\langle k \rangle$")
    right.set_title(r"Only rewiring conserves $\langle k \rangle$", fontsize=9.5)

    fig.supxlabel("$p$", fontsize=9.5, color=INK["secondary"])
    # No figure title: the caption carries it, and the two panel titles say what
    # the caption cannot, which of the two panels is which.
    return fig


# ---------------------------------------------------------------------------
# 4.4 Barabasi-Albert
# ---------------------------------------------------------------------------


@figure(
    "F04-07",
    chapter=4,
    title="A grown scale-free network",
    source=Source(Origin.THESIS_2011, pdf_pages="26-31", section="4.4 Barabasi model"),
    seed=THESIS_SEED,
    reproduces_figure="thesis figs. on PDF pp. 30-31",
)
def ba_network(ctx: FigureContext):
    """PDF pp. 30-31: what preferential attachment builds.

    Nodes are sized by degree, so the hubs are visible as hubs. This is the
    structural difference from every model before it in the chapter: the
    Erdos-Renyi and small-world networks have a characteristic degree, and this
    one does not.

    The three largest nodes carry the accent colour, because the sentence the
    figure exists to support is about them and not about the other hundred and
    forty-seven.
    """
    n, m = 150, 2
    ctx.parameters(N=n, m=m, m0=m + 1, fallback="random")

    g = barabasi_albert(n, links_per_new_node=m, rng=ctx.rng)
    k = g.degrees()
    hubs = set(np.argsort(k)[-3:].tolist())
    colours = [SERIES[1] if i in hubs else SERIES[0] for i in range(n)]

    fig, ax = plt.subplots(figsize=(4.9, 4.3))
    pos = draw_network(
        g,
        ax,
        layout="spring",
        rng=ctx.rng,
        node_color=colours,
        scale_by_degree=True,
        edge_width=0.45,
    )

    biggest = int(np.argmax(k))
    top_three = ", ".join(str(int(v)) for v in np.sort(k[list(hubs)])[::-1])
    ax.annotate(
        f"the three largest hubs,\n$k = {top_three}$",
        xy=pos[biggest],
        xycoords="data",
        xytext=(0.0, 0.97),
        textcoords="axes fraction",
        fontsize=8.5,
        color=SERIES[1],
        ha="left",
        va="top",
        arrowprops=dict(arrowstyle="-", color=SERIES[1], lw=0.7, alpha=0.8),
    )
    _panel_note(
        ax,
        f"$N = {n}$, $m = {m}$;  mean degree {k.mean():.1f}\n"
        f"half the nodes have {int(np.median(k))} or fewer",
    )
    # No figure title: the caption carries it.  The note under the drawing keeps
    # the parameters and the two degrees, which are readings, not a title.
    return fig


@figure(
    "F04-09",
    chapter=4,
    title="Three completion rules give three different networks",
    source=Source(Origin.THESIS_2011, pdf_pages="27-29", section="4.4.2 Algoritmo"),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis Opcion 1, 2 and 3, PDF pp. 28-29",
)
def ba_three_options(ctx: FigureContext):
    """PDF pp. 28-29: Opcion 1, Opcion 2 and Opcion 3.

    When one sweep of rejection sampling leaves a new node with fewer than m
    partners, the printed listing offers three ways to finish: leave it short
    ('no se hace nada'), pick at random, or pick the nearest node in the plane.
    Which of the three produced its figures is not recorded.

    They are not interchangeable. Leaving the node short is the strongest
    rich-get-richer rule and leaves many nodes with no links at all; the
    geometric rule makes the model spatial. All three are therefore carried
    separately here rather than one being assumed.

    Opcions 2 and 3 fall almost on top of each other, so no arrangement of
    three curves on one axes can be labelled without ambiguity.  One panel per
    rule solves it: each panel names its own rule, draws the other two behind
    it in the page's own grey, and the reader compares the coloured curve with
    the pale ones.  What the figure has to show -- that the first rule departs
    and the other two agree -- is then visible three times over.
    """
    n, m, realisations = 600, 2, 12
    ctx.parameters(N=n, m=m, rules=["none", "random", "nearest"])
    ctx.realisations(realisations)

    rules = [
        # Broken over two lines here rather than left to be wrapped at draw
        # time.  Three panels on a seven-inch figure have a little over two
        # inches each, "Option 2: pick at random" set semibold does not fit in
        # that, and measuring it while drawing does not help: the layout engine
        # narrows the axes afterwards, so the measurement is of a panel wider
        # than the one the title ends up in.
        ("Option 1\nleave it short", "none"),
        ("Option 2\npick at random", "random"),
        ("Option 3\npick the nearest", "nearest"),
    ]

    curves = []
    for _label, rule in rules:
        counts = np.zeros(n)
        for _ in range(realisations):
            k = barabasi_albert(n, links_per_new_node=m, fallback=rule, rng=ctx.rng).degrees()
            counts += np.bincount(k, minlength=n)
        pk = counts / (realisations * n)
        k = np.flatnonzero(pk)
        curves.append((k, pk[k]))

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharex=True, sharey=True)
    fig.get_layout_engine().set(wspace=0.08)

    for i, (ax, (label, _rule)) in enumerate(zip(axes, rules, strict=True)):
        for j, (k, pk) in enumerate(curves):
            if j != i:
                ax.plot(k, pk, color=INK["axis"], lw=1.2, zorder=2)
        k, pk = curves[i]
        # Only the rule the panel is about takes a colour, and the accent goes
        # to the one rule whose curve is somewhere else.
        ax.plot(k, pk, color=SERIES[1] if i == 0 else SERIES[0], lw=1.8, zorder=3)
        panel_title(ax, label, fontsize=9, loc="center")
        ax.grid(True, which="major", axis="both")

    axes[0].set_xscale("symlog", linthresh=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("$P(k)$")
    _direct_label(
        axes[0],
        "the other two rules",
        (curves[1][0][6], curves[1][1][6]),
        (6, 10),
        color=INK["secondary"],
        ha="left",
        size=7.5,
    )
    fig.supxlabel("$k$", fontsize=9.5, color=INK["secondary"])
    # No figure title: the caption carries the reading and the parameters.  The
    # panel titles stay, since they name the rule each panel is about.
    return fig
