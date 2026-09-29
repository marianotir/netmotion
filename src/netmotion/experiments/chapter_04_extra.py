"""Chapter 4, section 4.1: the Erdos-Renyi figures of the thesis, one for one.

The figures in :mod:`~netmotion.experiments.chapter_04` distil section 4.1 into
three summary plots.  This module does the other job: it reproduces the printed
figures of PDF pages 12-17 *as the thesis lays them out*, panel for panel, so
that a reader with the 2011 document open can put the two side by side.

Laying a figure out the same way is not the same as drawing it the same way.
The 2011 originals are thin red polylines on a default grid; what is
reproduced here is what they measure, drawn to the conventions of
:mod:`netmotion.visualization.charts` -- a per-node series as stems rather than
as a barcode, a distribution as bars that touch, a mean as a labelled hairline,
and colour spent only where it tells the reader something.

Identifiers are the thesis's own figure numbers -- ``T02`` is Fig. 2, ``T04`` is
Fig. 4 with its twelve sub-panels a-l -- rather than the book's inventory codes.

Run them with::

    python -m netmotion.experiments.cli T02 T03 T04 T05
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..metrics import connected_components, is_connected, local_clustering
from ..networks import connectivity_threshold, erdos_renyi, ring_lattice
from ..provenance import Origin, Source
from ..visualization import INK, SEQUENTIAL, SERIES, draw_network
from ..visualization.charts import histogram, per_node, reference_line
from .chapter_04 import THESIS_SEED, _draw_split_edges, _panel_note, _ring_distance
from .registry import FigureContext, figure

# The thesis draws its networks red on blue.  The book's palette is fixed, so
# the mapping is by *role*, not by hue: one colour for the drawn network and
# everything measured from it, one accent for whatever the paragraph beside the
# figure is actually about, and the ink for a theoretical curve, which is a
# reference rather than a second series.
NETWORK = SERIES[0]
MEASURED = SERIES[0]
ACCENT = SERIES[1]
THEORY = INK["primary"]
FIG5_MEASURED = SERIES[7]  # red, as the caption of fig. 5 names it
FIG5_THEORY = SERIES[5]  # green, likewise
#: p is ordered: light to dark, as WS_COLUMN_COLORS / NW_COLUMN_COLORS
ER_COLUMN_COLORS = (SEQUENTIAL[2], SEQUENTIAL[5], SEQUENTIAL[8])


def _letter(ax, letter: str) -> None:
    """Print the thesis's sub-panel letter clear of the panel's contents.

    The letters earn their place only because the surrounding text cites them
    ('panel g shows eight nodes at k = 0'), so they are set small, in the muted
    ink, and outside the frame where they cannot land on the data.
    """
    ax.annotate(
        letter,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(-3, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK["muted"],
    )


def _isolated_colours(g) -> list[str]:
    """One colour per node, marking the nodes that have no link yet."""
    return [INK["axis"] if k == 0 else NETWORK for k in g.degrees()]


# ---------------------------------------------------------------------------
# PDF p. 12: the ten nodes, before any link
# ---------------------------------------------------------------------------


@figure(
    "T02",
    chapter=4,
    title="Building an Erdos-Renyi network, p = 0 to 0.25",
    source=Source(Origin.THESIS_2011, pdf_pages="12-13", section="4.1.2 Algoritmo"),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 2 a-d, PDF pp. 12-13",
)
def er_construction_sequence(ctx: FigureContext):
    """PDF pp. 12-13, fig. 2.a-d: N = 10 at p = 0, 0.1, 0.15 and 0.25.

    The thesis prints panel a at the foot of p. 12 and panels b, c and d down
    the right-hand column of p. 13, each headed by the Maple string
    ``probabilidad de conexion = ...``.  Reproduced here as the strip it is,
    with the node positions held fixed so that the sequence reads as one
    network filling in rather than four unrelated draws.

    The count under each panel is what the caption claims in words: at p = 0
    the nodes are all separate, and by p = 0.25 -- just above
    p_c = ln(N)/N = 0.23 -- no node is left without a link.  A grey node is one
    of those still on its own, so the claim can be checked by eye as well as
    read off the count.
    """
    n = 10
    probabilities = (0.0, 0.10, 0.15, 0.25)
    ctx.parameters(N=n, p=list(probabilities), p_c=round(connectivity_threshold(n), 3))

    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.4))
    for ax, p, letter in zip(axes, probabilities, "abcd", strict=True):
        g = erdos_renyi(n, p, rng=ctx.rng)
        draw_network(g, ax, node_color=_isolated_colours(g), edge_width=1.0)
        ax.margins(0.12)
        isolated = int((g.degrees() == 0).sum())
        ax.set_title(f"$p = {p:g}$", fontsize=10, loc="center")
        _panel_note(
            ax,
            f"{g.n_edges} links\n"
            + ("none left alone" if isolated == 0 else f"{isolated} still alone"),
            color=NETWORK if isolated == 0 else INK["secondary"],
        )
        _letter(ax, letter)

    # No suptitle: the caption carries the threshold and $N$.
    return fig


# ---------------------------------------------------------------------------
# PDF p. 14, Fig. 3: either side of the threshold
# ---------------------------------------------------------------------------


@figure(
    "T03",
    chapter=4,
    title="Either side of the critical probability",
    source=Source(Origin.THESIS_2011, pdf_pages="14", section="4.1.2 Algoritmo"),
    seed=THESIS_SEED,
    reproduces_figure="thesis figs. 3.a and 3.b, PDF p. 14",
)
def er_threshold_pair(ctx: FigureContext):
    """PDF p. 14, figs. 3.a and 3.b: N = 10 at p = 0.22 and p = 0.24.

    The thesis verifies Erdos and Renyi's p_c = ln(N)/N by drawing one network
    just below it and one just above, and observing that the first has nodes
    left out and the second does not.

    At N = 10 the two probabilities straddle p_c = 0.23, and neither of them
    settles connectivity on its own: at p = 0.22 a G(10, p) comes out connected
    about a fifth of the time, and at p = 0.24 about a quarter.  The two panels
    show each side of the threshold, so
    this reproduction draws from ctx.rng until it finds the same pair, and
    records how many draws that took.  F04-03 plots the curve these two panels are two points on.

    The one thing the reader has to see is which nodes belong together, so that
    is the one thing colour is spent on: in panel a the piece that is cut off
    carries the accent, and in panel b there is nothing to pick out.
    """
    n = 10
    p_low, p_high = 0.22, 0.24
    ctx.parameters(N=n, p=[p_low, p_high], p_c=round(connectivity_threshold(n), 3))

    def draw_until(p: float, want_connected: bool, limit: int = 500):
        for attempt in range(1, limit + 1):
            g = erdos_renyi(n, p, rng=ctx.rng)
            if is_connected(g) is want_connected:
                return g, attempt
        raise RuntimeError(f"no G({n}, {p}) with connected={want_connected} in {limit} draws")

    disconnected, tries_low = draw_until(p_low, False)
    connected, tries_high = draw_until(p_high, True)
    ctx.parameters(draws_until_disconnected=tries_low, draws_until_connected=tries_high)

    fig, (left, right) = plt.subplots(1, 2, figsize=(4.9, 2.9))

    pieces = connected_components(disconnected)
    largest = set(pieces[0].tolist())
    colours = [NETWORK if i in largest else ACCENT for i in range(n)]
    draw_network(disconnected, left, node_color=colours, edge_width=1.0)
    left.set_title(f"$p = {p_low} < p_c = 0.23$", fontsize=10, loc="center")
    _panel_note(
        left,
        f"{disconnected.n_edges} links, {len(pieces)} separate pieces:\n"
        f"not every node is joined\n"
        "",
    )
    _letter(left, "a")

    draw_network(connected, right, node_color=NETWORK, edge_width=1.0)
    right.set_title(f"$p = {p_high} > p_c = 0.23$", fontsize=10, loc="center")
    _panel_note(
        right,
        f"{connected.n_edges} links, one piece:\n"
        f"every node is reachable\n"
        "",
    )
    _letter(right, "b")

    # No suptitle: the caption makes the point, and the number of draws each
    # drawing took is now recorded under the panel it belongs to.
    return fig


# ---------------------------------------------------------------------------
# PDF p. 15, Fig. 4: the twelve-panel table
# ---------------------------------------------------------------------------


@figure(
    "T04",
    chapter=4,
    title="Topology and characteristics of the ER network, p = 0.1, 0.5, 1",
    source=Source(Origin.THESIS_2011, pdf_pages="15", section="4.1.3 Caracteristicas"),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 4.a-l, PDF p. 15",
)
def er_characteristics_table(ctx: FigureContext):
    """PDF p. 15, fig. 4.a-l: the whole of section 4.1.3 in one table.

    Four rows by three columns, exactly as the thesis prints it.  The columns
    are the connection probability p = 0.1, 0.5 and 1 at N = 10; the rows are
    the network drawn (a-c), its degree distribution P(k) (d-f), the degree of
    each individual node (g-i), and the clustering coefficient of each
    individual node (j-l).

    The two lower rows are the ones that carry the argument and the ones a
    summary plot always loses.  They are *per node*, not distributions: the
    horizontal axis is the node index 1..10, so panel g shows eight nodes at
    k = 0 and a couple with a link or two, and panel k shows the clustering
    coefficient swinging between 0 and 0.7 from one node to the next.  At p = 1
    both rows go flat -- every node has k = 9 and C = 1 -- which is the thesis's
    closing observation on p. 16.

    All twelve panels come from one realisation per column, as in 2011: rows 2
    to 4 describe exactly the network drawn above them, not an ensemble.

    A grid of twelve panels is where the book's own drawing rules do most work,
    so they are stated here.  The panels share their axes by row, so the
    quantity is named once down the left and the probability once along the
    top; the per-node rows are stems on a common baseline rather than ten
    detached markers joined by a line, because nothing happens *between* node 4
    and node 5; and each of those rows carries its own mean as a hairline, which
    is what makes the columns comparable at a glance.
    """
    n = 10
    probabilities = (0.1, 0.5, 1.0)
    ctx.parameters(N=n, p=list(probabilities), realisation="one network per column")
    ctx.realisations(1)

    networks = [erdos_renyi(n, p, rng=ctx.rng) for p in probabilities]
    letters = iter("abcdefghijkl")

    fig, axes = plt.subplots(4, 3, figsize=(7.2, 4.6), sharey="row")
    fig.get_layout_engine().set(hspace=0.06, wspace=0.17)

    # row 1: the network itself -------------------------------------------
    for col, (ax, p, g, colour) in enumerate(
        zip(axes[0], probabilities, networks, ER_COLUMN_COLORS, strict=True)
    ):
        draw_network(
            g, ax, node_color=colour, edge_width=0.9, node_size=0.6 * max(6.0, 260.0 / g.n**0.55)
        )
        for coll in ax.collections:
            coll.set_clip_on(False)
        ax.set_title(f"$p = {p:g}$", fontsize=11, loc="center")
        _letter(ax, next(letters))
        if col == 0:
            ax.annotate(
                "the network",
                xy=(-0.08, 0.5),
                xycoords="axes fraction",
                rotation=90,
                ha="right",
                va="center",
                fontsize=8.5,
                color=INK["secondary"],
            )

    # row 2: the degree distribution P(k) ---------------------------------
    # Unit-wide bins on a density histogram are exactly P(k), and the bars
    # touch: k is a count, and a gap between k = 3 and k = 4 would suggest
    # there is something in between.
    for col, (ax, g, colour) in enumerate(zip(axes[1], networks, ER_COLUMN_COLORS, strict=True)):
        histogram(ax, g.degrees(), bins=np.arange(-0.5, n + 0.5), color=colour, density=True)
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_xticks([0, 3, 6, 9])
        _letter(ax, next(letters))
        if col == 0:
            ax.set_ylabel("$P(k)$")
        if col == 1:
            ax.set_xlabel("degree $k$", labelpad=1.0)

    # row 3: the degree of each node --------------------------------------
    for col, (ax, g, colour) in enumerate(zip(axes[2], networks, ER_COLUMN_COLORS, strict=True)):
        k = g.degrees()
        per_node(ax, k, color=colour)
        reference_line(ax, float(k.mean()), "")
        ax.annotate(
            f"mean {k.mean():.1f}",
            xy=(0.98, 0.98),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=7.5,
            color=INK["secondary"],
        )
        _letter(ax, next(letters))
        if col == 0:
            ax.set_ylabel("degree $k$")

    # row 4: the clustering coefficient of each node ----------------------
    for col, (ax, g, colour) in enumerate(zip(axes[3], networks, ER_COLUMN_COLORS, strict=True)):
        c = local_clustering(g)
        per_node(ax, c, color=colour)
        reference_line(ax, float(c.mean()), "")
        ax.annotate(
            f"mean {c.mean():.2f}",
            xy=(0.98, 0.98),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=7.5,
            color=INK["secondary"],
        )
        if c.max() == 0.0:
            # Nothing to draw is itself the result; say so, or the panel reads
            # as a figure that failed to render.
            ax.annotate(
                "every node\nscores $C_i = 0$",
                xy=(0.5, 0.36),
                xycoords="axes fraction",
                ha="center",
                va="center",
                fontsize=7.5,
                color=INK["muted"],
            )
        _letter(ax, next(letters))
        if col == 0:
            ax.set_ylabel("clustering $C$")
        if col == 1:
            ax.set_xlabel("node index", labelpad=1.0)

    # Shared scales, set once per row; the node axis is shared down the two
    # per-node rows, so its ticks are printed once, at the foot of the grid.
    axes[1][0].set_ylim(0, 1.05)
    axes[1][0].set_yticks([0, 0.5, 1])
    axes[2][0].set_ylim(0, 11.5)
    axes[2][0].set_yticks([0, 3, 6, 9])
    axes[3][0].set_ylim(0, 1.25)
    axes[3][0].set_yticks([0, 0.5, 1])
    for col in range(3):
        axes[2][col].sharex(axes[3][col])
        axes[2][col].tick_params(labelbottom=False)
        axes[3][col].set_xticks([1, 5, 10])
    for row in (1, 2, 3):
        for col in (1, 2):
            axes[row][col].tick_params(labelleft=False)

    # No suptitle: the caption carries it.  The column titles and the row
    # labels stay, because they identify the twelve panels.
    return fig


# ---------------------------------------------------------------------------
# PDF p. 16, Fig. 5: measured against theory, one network per p
# ---------------------------------------------------------------------------


@figure(
    "T05",
    chapter=4,
    title="Mean clustering and mean degree against p",
    source=Source(Origin.THESIS_2011, pdf_pages="16", section="4.1.3 Caracteristicas"),
    seed=THESIS_SEED,
    reproduces_figure="thesis figs. 5.a and 5.b, PDF p. 16",
)
def er_theory_sweep(ctx: FigureContext):
    """PDF p. 16, figs. 5.a and 5.b: <C> and <k> against p, with the theory.

    The thesis sweeps ``probabilidad := r/countmax`` for r = 0..100, building
    one N = 10 network per value and measuring it once, and this figure follows
    the same sweep.  The travel of the red curve about the green line is
    therefore the spread of a single ten-node draw; F04-02 measures the same two
    quantities as an average over 300 draws per p, which is the other way of
    reading the same sweep and is kept as a figure of its own.

    The theoretical line drawn in fig. 5.b runs from 0 to 9, i.e. p(N-1), and
    that is the line used here.
    """
    n = 10
    steps = 100  # countmax, as printed
    ps = np.arange(steps + 1) / steps
    ctx.parameters(N=n, p="r/countmax, r = 0..100", countmax=steps)
    ctx.realisations(1)

    clustering = np.empty(ps.size)
    degree = np.empty(ps.size)
    for i, p in enumerate(ps):
        g = erdos_renyi(n, p, rng=ctx.rng)
        clustering[i] = float(local_clustering(g).mean())
        degree[i] = float(g.degrees().mean())

    fig, (left, right) = plt.subplots(1, 2, figsize=(6.4, 3.4), sharex=True)

    def label_pair(ax, theory_end, theory_text, measured_xy):
        """Name both lines without a leader crossing the scatter.

        A diagonal leaves two corners empty and the two lines meet in a third,
        so each name goes in a corner of its own: the theory above the
        diagonal, the measured curve below it.  Named at the end of its line
        instead, the theory was struck through by the curve it is drawn for.
        """
        del theory_end
        ax.annotate(
            theory_text,
            xy=(0.04, 0.94),
            xycoords="axes fraction",
            fontsize=8.5,
            color=FIG5_THEORY,
            ha="left",
            va="top",
        )
        ax.annotate(
            "measured",
            xy=measured_xy,
            xycoords="data",
            fontsize=8.5,
            color=FIG5_MEASURED,
            ha="left",
            va="center",
        )

    left.plot(ps, clustering, color=FIG5_MEASURED, lw=1.4)
    left.plot(ps, ps, color=FIG5_THEORY, lw=1.3, ls=(0, (4, 3)))
    label_pair(left, (1.0, 1.0), "$C = p$", (0.46, 0.13))
    left.set_ylabel(r"$\langle C \rangle$")
    left.set_yticks([0, 0.5, 1])
    left.set_ylim(-0.04, 1.05)
    left.set_xticks([0, 0.5, 1.0])
    left.set_title("Clustering")
    _letter(left, "a")

    right.plot(ps, degree, color=FIG5_MEASURED, lw=1.4)
    right.plot(ps, ps * (n - 1), color=FIG5_THEORY, lw=1.3, ls=(0, (4, 3)))
    label_pair(right, (1.0, n - 1), r"$p(N-1)$", (0.46, 1.2))
    right.set_ylabel(r"$\langle k \rangle$")
    right.set_yticks([0, 3, 6, 9])
    right.set_ylim(-0.4, 9.6)
    right.set_title("Mean degree")
    _letter(right, "b")

    # Each panel names its own axis: over two panels a figure-centred label
    # lands at the inner end of panel a and reads as that panel's alone.
    left.set_xlabel("$p$")
    right.set_xlabel("$p$")
    # No suptitle: the caption carries it.  The two panel titles stay.
    return fig


# ---------------------------------------------------------------------------
# PDF p. 17, Fig. 6 (the first of two figures the thesis numbers 6)
# ---------------------------------------------------------------------------


@figure(
    "T06-RING",
    chapter=4,
    title="The ordered network, N = 20, four nearest neighbours",
    source=Source(Origin.THESIS_2011, pdf_pages="17", section="4.2.2 Algoritmo"),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 6 (the ordered ring), PDF p. 17",
)
def ordered_ring(ctx: FigureContext):
    """PDF p. 17, fig. 6: the starting point of the small-world model.

    ``dim := 20``, ``aconectar := 4``, so ``nod := 2`` links per direction: each
    node joins the two nodes before it and the two after it, wrapping round.
    Every node has k = 4 exactly, so P(k) is a delta at 4, and the drawing shows
    the characteristic double band -- the polygon of first neighbours with the
    chords of second neighbours inside it.

    The two bands are the content of the figure, so the second colour separates
    them: the polygon is one class of link and the chords inside it are the
    other.  Undifferentiated, they read as a slightly thick circle.

    The identifier carries a suffix because the thesis prints two different
    figures numbered 6: this ordered ring on p. 17, and the Watts-Strogatz
    characteristics table on pp. 20-21.
    """
    n, k = 20, 4
    ctx.parameters(N=n, aconectar=k, nod=k // 2)

    g = ring_lattice(n, k)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    _draw_split_edges(g, ax, lambda i, j: _ring_distance(i, j, n) == 2, width=1.2)

    # Name the two bands where they are furthest apart, on the right of the
    # circle, rather than in a key.
    pos_x = np.cos(2.0 * np.pi * np.arange(1, n + 1) / n)
    pos_y = np.sin(2.0 * np.pi * np.arange(1, n + 1) / n)
    first = ((pos_x[19] + pos_x[0]) / 2, (pos_y[19] + pos_y[0]) / 2)
    second = ((pos_x[19] + pos_x[1]) / 2, (pos_y[19] + pos_y[1]) / 2)
    ax.annotate(
        "first neighbours",
        xy=first,
        xytext=(1.24, -0.16),
        textcoords="data",
        fontsize=8.5,
        color=INK["secondary"],
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=INK["muted"], lw=0.7),
        annotation_clip=False,
    )
    ax.annotate(
        "second neighbours",
        xy=second,
        xytext=(1.24, 0.62),
        textcoords="data",
        fontsize=8.5,
        color=ACCENT,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.7),
        annotation_clip=False,
    )
    # No title and no note: $N$, $k$ and the two bands are all in the caption,
    # and the two bands are named on the drawing itself.
    return fig
