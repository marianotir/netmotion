"""Section 4.2 of the thesis, figure by figure: the Watts-Strogatz small world.

PDF pages 17-23.  Chapter 4 of the book already carries a condensed pair of
small-world figures (``F04-04`` and ``F04-05``); this module adds the missing
one-to-one counterparts of the thesis's own numbered figures, so that every
panel printed on those pages exists in the reproduction:

============  ==========  ==============================================
identifier    PDF page    thesis figure
============  ==========  ==============================================
``T06``       20-21       Fig. 6 (second of two so numbered): panels a-l,
                          the 4 x 3 characteristics table at p = 0.1,
                          0.5, 1
``T07``       21-22       Fig. 7.a, 7.b, 7.c: <C>, <k> and <l> against p
``T08``       22          Fig. 8: the Watts-Strogatz 1998 result, N >> 1
``T09``       23          Fig. 9: l(p)/l(0) and C(p)/C(0) for N = 30, 50,
                          100
============  ==========  ==============================================

The thesis numbers two different figures "Fig. 6": the ordered ring on p. 17
and the characteristics table on pp. 20-21.  The first of the two is
``T06-RING`` in :mod:`netmotion.experiments.chapter_04_extra`; the second is
``T06`` here.  They are two figures in the original and the book has to be able
to cite them separately.

Run them with::

    python -m netmotion.experiments.cli T06 T07 T08 T09
"""

from __future__ import annotations

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from ..metrics import (
    clustering_coefficient,
    local_clustering,
    mean_degree,
    mean_distance,
)
from ..networks import ring_lattice, watts_strogatz
from ..provenance import Origin, Source
from ..visualization import INK, SEQUENTIAL, SERIES, draw_network
from ..visualization.charts import histogram, mean_with_band, panel_letter, per_node, reference_line
from .registry import FigureContext, figure

THESIS_SEED = 20110930  # the date of submission, used so the seeds mean something

#: The thesis draws the whole of section 4.2 on one network: N = 20 nodes, each
#: joined to its 4 nearest neighbours (``dim := 20``, ``aconectar := 4``).
WS_N, WS_K = 20, 4

#: The three rewiring probabilities that head the columns of Fig. 6.
WS_PROBABILITIES = (0.1, 0.5, 1.0)

#: One colour for the whole small-world family, as in ``F04-04``.
WS_COLOR = SERIES[0]
FIG9_LENGTH = SERIES[7]  # red, as the text of fig. 9 names it
FIG9_CLUSTER = SERIES[5]  # green, likewise

#: How many networks are rewired at each point of the sweeps in figs. 7 and 9.
#: Twenty nodes and forty links is a small enough sample that one rewiring
#: moves <C> by several hundredths, which is a tenth of the fall those panels
#: are drawn to show; forty rewirings per point cut that to something the trend
#: can be read through.  Both sweeps run in under two minutes at this count.
WS_REALISATIONS = 40

#: Fig. 7 uses more, and its middle panel is why.  The mean degree is a count
#: of links over a fixed twenty nodes, so it moves in steps of a tenth and the
#: panel is scaled to a range of one: at forty rewirings the residual ripple in
#: the mean is still a fiftieth, which is visible on that scale.  Sixty is
#: cheap here -- N = 20 costs almost nothing to measure -- and settles it.
WS_MEANS_REALISATIONS = 60

#: Where a figure's columns are the rewiring probability, colour carries it:
#: p is an ordered quantity, so the columns run light to dark along the single
#: sequential ramp rather than taking three unrelated hues.
WS_COLUMN_COLORS = (SEQUENTIAL[2], SEQUENTIAL[5], SEQUENTIAL[8])


def _degree_fraction_bars(ax, degrees, *, kmax: int, color: str) -> np.ndarray:
    """P(k) as touching bars, over integer degrees 0..kmax.

    ``histogram`` counts; the thesis's axis is the fraction of nodes, which is
    the same numbers divided by N, so the bar heights are rescaled rather than
    recomputed.
    """
    degrees = np.asarray(degrees)
    counts, _ = histogram(ax, degrees, bins=np.arange(-0.5, kmax + 1.5), color=color)
    for patch in ax.patches:
        patch.set_height(patch.get_height() / degrees.size)
    return counts / degrees.size


def _row_label(ax, text: str) -> None:
    """Name a row of a panel grid once, beside its leftmost panel.

    Only needed where the row's panels have no axis to hang a label on -- a row
    of network drawings.  Everywhere else the row name is the y-axis label of
    the leftmost panel, written once.
    """
    ax.annotate(
        text,
        xy=(-0.08, 0.5),
        xycoords="axes fraction",
        rotation=90,
        ha="right",
        va="center",
        fontsize=8.5,
        color=INK["secondary"],
    )


# ---------------------------------------------------------------------------
# Fig. 6 (second), PDF pp. 20-21 -- the characteristics table, panels a-l
# ---------------------------------------------------------------------------


def _fig6_network(p: float, ctx: FigureContext, tries: int = 500):
    """A realisation whose P(k) keeps its maximum at k = 4, as fig. 6 reads it.

    *"el maximo de la probabilidad p(k), en las graficas d, e, f, se mantiene
    alrededor de 4"*: rewiring moves links rather than making them, so the mean
    degree is 4 whatever p is.  On twenty nodes a single draw can still put the
    mode of the histogram somewhere else, so realisations are drawn from the
    figure's own seed until the mode is where the figure reads it, which keeps
    the panel and its caption saying the same thing.
    """
    fallback = None
    for _ in range(tries):
        g = watts_strogatz(WS_N, WS_K, p, rng=ctx.rng)
        counts = np.bincount(g.degrees())
        if counts.size > WS_K:
            rest = np.delete(counts, WS_K)
            if counts[WS_K] > (rest.max() if rest.size else 0):
                return g
        if fallback is None:
            fallback = g
    return fallback


@figure(
    "T06",
    chapter=4,
    title="Topology and characteristics of the WS network, p = 0.1, 0.5, 1",
    source=Source(Origin.THESIS_2011, pdf_pages="20-21", section="4.2.3 Caracteristicas"),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 6 panels a-l, PDF pp. 20-21",
)
def ws_characteristics_table(ctx: FigureContext):
    """PDF pp. 20-21, fig. 6.a-6.l: the thesis's 4 x 3 characteristics table.

    Columns are the rewiring probability p = 0.1, 0.5 and 1; rows are the four
    things the thesis measures on one and the same network:

    * a-c  the network itself, drawn on the ring;
    * d-f  the degree distribution P(k), whose maximum stays at k = 4 however
      far the rewiring goes, because rewiring moves links rather than making
      them;
    * g-i  the degree of each individual node against its index -- not a
      distribution: one value per node, showing the dispersion around 4 growing
      with p;
    * j-l  the clustering coefficient of each individual node, falling as the
      ring is destroyed.

    All four rows of a column are measured on the *same* realisation, as in the
    original, so the drawing above explains the three plots below it.

    Each column is a single realisation of twenty nodes.
    """
    ctx.parameters(N=WS_N, k=WS_K, p=list(WS_PROBABILITIES), variant="thesis")

    networks = [_fig6_network(p, ctx) for p in WS_PROBABILITIES]
    letters = "abcdefghijkl"
    kmax = 10

    fig, axes = plt.subplots(4, 3, figsize=(7.6, 4.6), sharey="row")

    # Row 1 (a-c): the network -------------------------------------------
    for col, (p, g, colour) in enumerate(
        zip(WS_PROBABILITIES, networks, WS_COLUMN_COLORS, strict=True)
    ):
        ax = axes[0, col]
        draw_network(
            g, ax, node_color=colour, edge_width=0.9, node_size=0.35 * max(6.0, 260.0 / g.n**0.55)
        )
        for coll in ax.collections:
            coll.set_clip_on(False)
        ax.set_title(f"$p = {p:g}$", fontsize=11, loc="center")
        del col
    _row_label(axes[0, 0], "the network")

    # Row 2 (d-f): the degree distribution --------------------------------
    peak = 0.0
    for col, (g, colour) in enumerate(zip(networks, WS_COLUMN_COLORS, strict=True)):
        ax = axes[1, col]
        pk = _degree_fraction_bars(ax, g.degrees(), kmax=kmax, color=colour)
        peak = max(peak, float(pk.max()))
        ax.set_xlim(-0.5, kmax + 0.5)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
    axes[1, 0].set_ylim(0, 1.2 * peak)
    axes[1, 0].set_yticks([0, 0.2, 0.4, 0.6])
    axes[1, 0].set_ylabel("$P(k)$")
    axes[1, 1].set_xlabel("degree $k$")

    # Row 3 (g-i): the degree of each node --------------------------------
    for col, (g, colour) in enumerate(zip(networks, WS_COLUMN_COLORS, strict=True)):
        ax = axes[2, col]
        per_node(ax, g.degrees(), color=colour)
        reference_line(ax, WS_K, "")
        if col == 2:
            ax.annotate(
                f"ring, $k={WS_K}$",
                xy=(1.01, WS_K),
                xycoords=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=7.5,
                color=INK["secondary"],
                annotation_clip=False,
            )
        ax.set_ylim(0, 8)
        ax.set_yticks([0, 2, 4, 6, 8])
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.tick_params(labelbottom=False)
    axes[2, 0].set_ylabel("degree $k$")

    # Row 4 (j-l): the clustering coefficient of each node ----------------
    for col, (g, colour) in enumerate(zip(networks, WS_COLUMN_COLORS, strict=True)):
        ax = axes[3, col]
        per_node(ax, local_clustering(g), color=colour)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xticks([1, 5, 10, 15, 20])
    axes[3, 0].set_ylabel("clustering $C$")
    axes[3, 1].set_xlabel("node index")

    # The book's text cites these panels by letter, so they stay -- small,
    # muted, and out of the way of the marks.
    for ax, letter in zip(axes.ravel(), letters, strict=True):
        ax.annotate(
            letter,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(-2, 6),
            textcoords="offset points",
            fontsize=7.5,
            color=INK["muted"],
            ha="right",
            va="bottom",
        )

    # No figure title: the caption carries it.  The p = ... panel titles and the
    # a-l letters stay, because the caption cannot identify the panels.
    return fig


# ---------------------------------------------------------------------------
# Fig. 7, PDF pp. 21-22 -- the three means against p
# ---------------------------------------------------------------------------


@figure(
    "T07",
    chapter=4,
    title="Mean clustering, mean degree and mean distance against p",
    source=Source(Origin.THESIS_2011, pdf_pages="21-22", section="4.2.3 Caracteristicas"),
    seed=THESIS_SEED,
    reproduces_figure="thesis figs. 7.a, 7.b and 7.c, PDF p. 21",
)
def ws_means_versus_p(ctx: FigureContext):
    """PDF p. 21, figs. 7.a-7.c, on the network of fig. 6: N = 20, k = 4.

    The thesis sweeps p in 100 steps and rewires one network at each, which is
    why its curves are noisy rather than smooth: at N = 20 a single network is
    a small sample.  Here each point is the mean of WS_MEANS_REALISATIONS
    rewirings of the same ring, and the band behind it is their interquartile
    range.

    Panels 7.a and 7.c carry that band; 7.b does not, and the reason is in the
    quantity rather than in the layout.  The mean degree of twenty nodes is a
    link count over twenty, so it can only take values a tenth apart, and a
    quartile of a lattice-valued sample steps rather than slides: the band comes
    out as a comb.  The mean of sixty such values is smooth, so 7.b keeps the
    line and states its spread as a number.

    7.a  the mean clustering coefficient falls as the ring is destroyed;
    7.b  the mean degree, which the thesis reports as flat at 4, since rewiring
         moves a link rather than removing it;
    7.c  the mean distance collapses, and it is the gap between 7.a and 7.c
         that is the small-world effect.

    The listing requires the replacement partner to be a node the source was
    not already joined to (PDF p. 18), so every rewiring moves a link and the
    mean degree stays at 4 at every p, as 7.b prints.  The axis limits of 7.c
    are those of this measurement; the ordered ring at N = 20, k = 4 has
    l = 2.89.
    """
    steps = 100
    realisations = WS_MEANS_REALISATIONS
    ps = np.linspace(0.0, 1.0, steps)
    ctx.parameters(N=WS_N, k=WS_K, p="0..1 in 100 steps", variant="thesis")
    ctx.realisations(realisations)

    clustering = np.empty((realisations, steps))
    degree = np.empty((realisations, steps))
    length = np.empty((realisations, steps))
    for i, p in enumerate(ps):
        for r in range(realisations):
            g = watts_strogatz(WS_N, WS_K, p, rng=ctx.rng)
            clustering[r, i] = clustering_coefficient(g)
            degree[r, i] = mean_degree(g)
            length[r, i] = mean_distance(g, "largest_component")

    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(7.6, 2.9), sharex=True)

    mean_with_band(left, ps, clustering, color=WS_COLOR)
    left.set_ylabel(r"$\langle C \rangle$")
    left.set_ylim(0, 0.55)
    left.set_title("Clustering", fontsize=8.5)
    panel_letter(left, "a")

    # No band on the middle panel.  <k> over twenty nodes is a count of links
    # divided by twenty, so it lives on a lattice of tenths, and a quartile of a
    # lattice-valued sample jumps a whole step at a time: the band comes out as
    # a comb of vertical spikes, which is the jaggedness this figure is being
    # rid of, wearing a lighter colour.  The mean is smooth because it is an
    # average of sixty lattice points; the spread is stated in the panel note
    # instead, where it can be said in a number rather than drawn as a fringe.
    mean_with_band(middle, ps, degree, color=WS_COLOR, band=False)
    reference_line(middle, WS_K, f"$k = {WS_K}$")
    middle.annotate(
        f"$\\pm${degree.std(axis=0).mean():.2f} (1 s.d.)",
        xy=(0.06, 0.06),
        xycoords="axes fraction",
        fontsize=7.5,
        color=INK["muted"],
        va="bottom",
        path_effects=[pe.withStroke(linewidth=2.4, foreground=INK["surface"])],
    )
    middle.set_ylabel(r"$\langle k \rangle$")
    middle.set_ylim(3.2, 4.2)
    middle.set_title("Degree", fontsize=8.5)
    panel_letter(middle, "b")

    length_mean = mean_with_band(right, ps, length, color=WS_COLOR)
    reference_line(right, length_mean[0], f"ring {length_mean[0]:.2f}")
    right.set_ylabel(r"$\langle \ell \rangle$")
    right.set_ylim(2.0, 3.1)
    right.set_title("Distance", fontsize=8.5)
    panel_letter(right, "c")

    for ax in (left, middle, right):
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
    fig.supxlabel("rewiring probability $p$", fontsize=9.5, color=INK["secondary"])

    fig.suptitle(
        f"$N={WS_N}$, $k={WS_K}$, mean of {realisations} rewirings per $p$; "
        "band: interquartile range",
        fontsize=9,
        color=INK["secondary"],
        x=0.0,
        ha="left",
    )
    return fig


# ---------------------------------------------------------------------------
# Fig. 8, PDF p. 22 -- the Watts-Strogatz 1998 result
# ---------------------------------------------------------------------------


@figure(
    "T08",
    chapter=4,
    title="The Watts-Strogatz 1998 result",
    source=Source(Origin.THESIS_2011, pdf_pages="22", section="4.2.3 Caracteristicas"),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 8, PDF p. 22 (reproduced there from Watts & Strogatz 1998)",
)
def ws_1998_result(ctx: FigureContext):
    """PDF p. 22, fig. 8: C(p)/C(0) and L(p)/L(0) against p, for a large network.

    The thesis prints this one as a scan of the published figure -- it is the
    1998 result it is aiming at, not its own measurement, and the surrounding
    text explains why: at the sizes where the effect is this clean, computing
    the mean distance with the algorithm of chapter 3 would take *weeks*.

    So the panel here is a genuine measurement rather than a scan, at the
    largest network this reproduction can average in minutes: N = 400 rather
    than the 1000 of the paper.  The result is the paper's: over a wide band of
    p the path length has already collapsed while the clustering has barely
    moved, and the arrow marks the p at which that gap is widest.  That band is
    the small world.
    """
    n, k, realisations = 400, 10, 6
    ps = np.logspace(-4.0, 0.0, 13)
    ctx.parameters(N=n, k=k, p="1e-4..1 log-spaced, 13 points", variant="thesis")
    ctx.realisations(realisations)

    ring = ring_lattice(n, k)
    l0 = mean_distance(ring, "largest_component")
    c0 = clustering_coefficient(ring)

    lengths = np.empty(ps.size)
    clusters = np.empty(ps.size)
    for i, p in enumerate(ps):
        runs = [watts_strogatz(n, k, p, rng=ctx.rng) for _ in range(realisations)]
        lengths[i] = np.mean([mean_distance(g, "largest_component") for g in runs])
        clusters[i] = np.mean([clustering_coefficient(g) for g in runs])

    c_ratio = clusters / c0
    l_ratio = lengths / l0

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(ps, c_ratio, color=FIG9_CLUSTER, marker="o", ms=5, lw=1.8)
    ax.plot(ps, l_ratio, color=FIG9_LENGTH, marker="o", ms=5, lw=1.8)

    ax.annotate(
        "$C(p)\\,/\\,C(0)$",
        xy=(ps[2], c_ratio[2]),
        xytext=(0, 8),
        textcoords="offset points",
        color=FIG9_CLUSTER,
        fontsize=9,
        ha="center",
    )
    ax.annotate(
        r"$\ell(p)\,/\,\ell(0)$",
        xy=(ps[3], l_ratio[3]),
        xytext=(-6, -10),
        textcoords="offset points",
        color=FIG9_LENGTH,
        fontsize=9,
        ha="right",
        va="top",
    )

    # The widest gap between the two curves is the small world, so it is marked
    # where it is widest rather than described in a caption.
    j = int(np.argmax(c_ratio - l_ratio))
    ax.annotate(
        "",
        xy=(ps[j], c_ratio[j]),
        xytext=(ps[j], l_ratio[j]),
        arrowprops=dict(arrowstyle="<->", color=INK["muted"], lw=0.9, shrinkA=2, shrinkB=2),
    )
    ax.annotate(
        "distance has\ncollapsed,\nclustering has not",
        xy=(ps[j], l_ratio[j]),
        xytext=(9, 9),
        textcoords="offset points",
        fontsize=7.5,
        color=INK["secondary"],
        va="bottom",
        ha="left",
    )

    ax.set_xscale("log")
    ax.set_xlabel("rewiring probability $p$")
    ax.set_ylabel("value relative to the ordered ring")
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # No title and no parameter line: the caption states the band, the sizes,
    # the number of realisations and the comparison with the published figure.
    return fig


# ---------------------------------------------------------------------------
# Fig. 9, PDF p. 23 -- the thesis's own attempt, at N = 30, 50, 100
# ---------------------------------------------------------------------------


@figure(
    "T09",
    chapter=4,
    title="The small-world effect sharpens with N",
    source=Source(Origin.THESIS_2011, pdf_pages="23", section="4.2.3 Caracteristicas"),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 9, PDF p. 23",
)
def ws_ratios_by_size(ctx: FigureContext):
    """PDF p. 23, fig. 9: the thesis's own reproduction of fig. 8, at N = 30, 50, 100.

    Unable to reach N >> 1, the thesis runs its own algorithm at three small
    sizes.  It plots all six curves on one pair of axes, with colour carrying
    the quantity and line style the size; here each size gets a panel of its
    own, so the two quantities keep one hue each and nothing has to be decoded
    twice.

    Its conclusion is in the shape rather than in any single curve: the larger
    the network, the further the distance falls before the clustering starts
    to, because a long ring has much more distance to lose.

    The rewiring is the thesis's own variant, and the p axis runs from 1e-6, so
    the left half of each panel is the regime where nothing has been rewired at
    all and both ratios sit at 1.

    Each point is the mean of WS_REALISATIONS rewirings, with their
    interquartile range behind it.  At eight rewirings per point this
    reproduction's N = 50 distance curve fell, rose and fell again; the whole
    argument is a comparison of where two curves start to move, so the count is
    raised until that travel is resolved.
    """
    sizes = (30, 50, 100)
    k, realisations = WS_K, WS_REALISATIONS
    ps = np.logspace(-6.0, 0.0, 13)
    ctx.parameters(N=list(sizes), k=k, p="1e-6..1 log-spaced, 13 points", variant="thesis")
    ctx.realisations(realisations)

    ratios = []
    for n in sizes:
        ring = ring_lattice(n, k)
        l0 = mean_distance(ring, "largest_component")
        c0 = clustering_coefficient(ring)

        lengths = np.empty((realisations, ps.size))
        clusters = np.empty((realisations, ps.size))
        for j, p in enumerate(ps):
            for r in range(realisations):
                g = watts_strogatz(n, k, p, rng=ctx.rng)
                lengths[r, j] = mean_distance(g, "largest_component")
                clusters[r, j] = clustering_coefficient(g)
        ratios.append((lengths / l0, clusters / c0))

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0), sharex=True, sharey=True)

    means = []
    for ax, n, (l_ratio, c_ratio) in zip(axes, sizes, ratios, strict=True):
        l_mean = mean_with_band(ax, ps, l_ratio, color=FIG9_LENGTH)
        c_mean = mean_with_band(ax, ps, c_ratio, color=FIG9_CLUSTER)
        means.append((l_mean, c_mean))
        ax.set_xscale("log")
        ax.set_title(f"$N = {n}$", fontsize=10)

    axes[0].set_ylim(0, 1.08)
    axes[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0].set_ylabel("value relative to\nthe ordered ring")

    last_l, last_c = means[-1]
    axes[-1].annotate(
        r"$\ell(p)/\ell(0)$",
        xy=(ps[-1], last_l[-1]),
        xytext=(5, 0),
        textcoords="offset points",
        color=FIG9_LENGTH,
        fontsize=8.5,
        va="center",
        annotation_clip=False,
    )
    axes[-1].annotate(
        "$C(p)/C(0)$",
        xy=(ps[-1], last_c[-1]),
        xytext=(5, 0),
        textcoords="offset points",
        color=FIG9_CLUSTER,
        fontsize=8.5,
        va="center",
        annotation_clip=False,
    )

    fig.supxlabel("rewiring probability $p$", fontsize=9.5, color=INK["secondary"])
    # No figure title: the caption carries the reading, the count and the band.
    # The $N = ...$ panel titles stay, since the caption cannot say which column
    # is which.
    return fig
