"""Chapters 2 and 3 of the thesis, figure for figure (PDF pp. 6-11).

``chapter_03.py`` already carries three figures drawn from these pages, but it
paraphrases them: it shows an eight-node ring where section 2.2 shows a
six-node one, and it shows the *finished* distance matrix where section 3.1
prints the intermediate matrices of an iteration worked out by hand.  This
module adds the missing counterparts so that every visual element of PDF
pp. 6-11 has one.

Identifiers follow the thesis, not the book: ``T01`` is the thesis's Fig. 1.
Sections 2.2, 3.1, 3.2 and 3.3 also print matrices, worked examples and Maple
listings that carry no figure number, so those are keyed by PDF page instead --
``TU07`` is the unnumbered material of PDF p. 7.

What is on those six pages:

======  ====================================================================
p. 6    Fig. 1, the six-node network; the symbolic matrix; the numeric one
p. 7    the Dijkstra listing; the eight-node ring; its adjacency matrix M
p. 8    the listing continued; L(1); L(2); the accumulated sum of the L(n)
p. 9    the listing closed; the shortest-path matrix Ld; the degree listing
p. 10   the degree listing continued
p. 11   the clustering listing
======  ====================================================================

The listings themselves are code, not figures, so what is drawn here is what
each listing *computes* when it is run on the thesis's own example -- which is
also the only way to check that the reconstruction in :mod:`netmotion.metrics`
agrees with the numbers the thesis printed.  It does: :func:`thesis_dijkstra`
below is a literal transcription of the printed Maple, and it reproduces
L(1), L(2), the accumulated sum and Ld entry for entry.

Run them with::

    python -m netmotion.experiments.cli T01 TU07 TU08 TU09 TU10 TU11
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from ..metrics import local_clustering, mean_distance, neighbour_links
from ..networks import erdos_renyi, ring_lattice, watts_strogatz
from ..provenance import Origin, Source
from ..visualization import INK, SEQUENTIAL, SERIES
from ..visualization.charts import histogram, panel_title, per_node, reference_line
from ..visualization.draw import circular_layout
from .registry import FigureContext, figure

THESIS_SEED = 20110930

#: A node and a link, drawn the way every other network in this book is drawn:
#: one hue for the nodes, a recessive grey for the links, since nothing in these
#: drawings is being told apart from anything else.
NODE = SERIES[0]
LINK = INK["muted"]

#: Fig. 1 and the drawing of the matrix M, in the same style; the paragraph of
#: section 2.2 names these colours.
FIG1_NODE = SERIES[0]  # blue, as every network of the book
FIG1_LINK = INK["muted"]  # grey, as every network of the book


# ---------------------------------------------------------------------------
# drawing a matrix the way the thesis prints one
# ---------------------------------------------------------------------------


def _bracket_matrix(
    ax,
    entries,
    *,
    label: str = "",
    colors=None,
    fontsize: float = 7.5,
    label_fontsize: float = 10.0,
    title: str | None = None,
):
    """Write *entries* as a bracketed matrix, as the thesis prints them.

    ``entries`` is a list of rows of strings; ``colors`` an optional matching
    grid of colours, so that the entries an algorithm has just written can be
    picked out from the zeros it left alone.
    """
    rows, cols = len(entries), len(entries[0])
    ax.set_axis_off()

    pad = 0.55
    left = -pad - (2.3 if label else 0.0)
    ax.set_xlim(left - 0.15, cols - 1 + pad + 0.15)
    ax.set_ylim(-(rows - 1) - 0.75, 0.75)
    ax.set_aspect("equal")
    # Equal aspect shrinks the axes box, and a shrunk box drops its title with
    # it. Anchoring north keeps the titles of matrices of different cell widths
    # on one line.
    ax.set_anchor("N")

    for i, row in enumerate(entries):
        for j, text in enumerate(row):
            color = INK["primary"] if colors is None else colors[i][j]
            ax.text(
                j,
                -i,
                text,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=color,
            )

    top, bottom = 0.62, -(rows - 1) - 0.62
    tick = 0.16
    for x, direction in ((-pad, 1.0), (cols - 1 + pad, -1.0)):
        ax.plot([x, x], [bottom, top], color=INK["primary"], lw=0.9, solid_capstyle="butt")
        ax.plot(
            [x, x + direction * tick],
            [top, top],
            color=INK["primary"],
            lw=0.9,
            solid_capstyle="butt",
        )
        ax.plot(
            [x, x + direction * tick],
            [bottom, bottom],
            color=INK["primary"],
            lw=0.9,
            solid_capstyle="butt",
        )

    if label:
        ax.text(
            -pad - 0.35,
            -(rows - 1) / 2.0,
            label,
            ha="right",
            va="center",
            fontsize=label_fontsize,
            color=INK["primary"],
        )
    if title:
        panel_title(ax, title, fontsize=9)
    return ax


def _ints(matrix) -> list[list[str]]:
    return [[f"{int(v)}" for v in row] for row in np.asarray(matrix)]


# ---------------------------------------------------------------------------
# 2.2 Matriz Adyacente -- PDF p. 6
# ---------------------------------------------------------------------------

#: The hand placement of Fig. 1, read off the page.  The thesis scatters its six
#: nodes in the plane rather than on a circle, and the reproduction keeps that:
#: the point of the figure is that a network is an abstract object which any
#: placement encodes equally well.
FIG1_POSITIONS = np.array(
    [
        [0.06, 0.40],  # 1
        [0.27, 0.05],  # 2
        [0.51, 0.57],  # 3
        [0.72, 0.21],  # 4
        [0.73, 0.94],  # 5
        [0.97, 0.73],  # 6
    ]
)

#: The edge set the caption of Fig. 1 states: {1,3}, {5,6}, {4,6}.
FIG1_EDGES = ((0, 2), (4, 5), (3, 5))


@figure(
    "T01",
    chapter=2,
    title="Six nodes, three links",
    source=Source(Origin.THESIS_2011, pdf_pages="6", section="2.2 Matriz Adyacente"),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 1, PDF p. 6",
)
def fig1_six_node_network(ctx: FigureContext):
    """PDF p. 6, fig. 1.

    The thesis's opening example: six nodes with {1,3}, {5,6} and {4,6} joined
    and node 2 left isolated, drawn by hand in the plane.  The thesis then
    writes it twice as a matrix -- first as the symbolic template of entries
    n_{i,j}, then filled in with the zeros and ones of this particular network
    -- and the book sets both matrices in its text, so the figure is the
    drawing alone.

    ``F02-01`` shows a network beside its adjacency matrix, but it shows the
    eight-node ring of section 3.1 rather than this one, so the thesis's own
    first example had no counterpart. It is worth having: node 2 is the reason
    the thesis says "se considera como nula la interaccion de cada nodo consigo
    mismo" in the same breath as it shows an all-zero row, and a reader meeting
    the adjacency matrix for the first time needs the small case.
    """
    n = 6
    ctx.parameters(N=n, edges=[(1, 3), (5, 6), (4, 6)], isolated_node=2)

    fig = plt.figure(figsize=(2.93, 3.15))

    # The drawing, at the thesis's own hand placement ------------------------
    left = fig.add_subplot(1, 1, 1)
    pos = FIG1_POSITIONS
    for i, j in FIG1_EDGES:
        left.plot(
            [pos[i, 0], pos[j, 0]],
            [pos[i, 1], pos[j, 1]],
            color=FIG1_LINK,
            lw=1.2,
            zorder=1,
            solid_capstyle="round",
        )
    left.scatter(pos[:, 0], pos[:, 1], s=90, c=FIG1_NODE, zorder=2, linewidths=0)
    # The number is pushed away from the centre of the drawing rather than a
    # fixed distance up and left, so that it never lands on a link.
    centre = pos.mean(axis=0)
    for i, (x, y) in enumerate(pos):
        away = np.array([x, y]) - centre
        norm = np.hypot(*away) or 1.0
        left.annotate(
            f"{i + 1}",
            xy=(x, y),
            xytext=tuple(13 * away / norm),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9,
            color=INK["secondary"],
        )
    left.annotate(
        "node 2 is\njoined to nothing",
        xy=(pos[1, 0], pos[1, 1]),
        xytext=(10, -14),
        textcoords="offset points",
        fontsize=8,
        color=INK["muted"],
    )
    left.set_xlim(-0.12, 1.18)
    left.set_ylim(-0.14, 1.12)
    left.set_aspect("equal")
    left.set_anchor("N")
    left.set_axis_off()

    # No title: the caption carries it.  The two matrices are set in the text
    # below the figure, as in the thesis, so they are not repeated here.
    return fig


@figure(
    "TM03",
    chapter=3,
    title="The network of the matrix M",
    source=Source(Origin.THESIS_2011, pdf_pages="8", section="3.1 Distancia media"),
    seed=THESIS_SEED,
    reproduces_figure="the drawing under the matrix M, PDF p. 8",
)
def matrix_m_network(ctx: FigureContext):
    """PDF p. 8: the eight nodes of the matrix M, each joined to its four nearest.

    Drawn from the matrix itself -- the ring lattice of eight nodes and degree
    four -- in the colours of fig. 1, with each node numbered as its row of M,
    so that the iterations worked through below can be followed on it.
    """
    n, k = 8, 4
    ctx.parameters(N=n, k=k)
    g = ring_lattice(n, k)
    # Node 1 at the top, the numbers running clockwise.
    angles = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    pos = np.column_stack([np.cos(angles), np.sin(angles)])

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    for i, j, _ in g.edges():
        ax.plot(
            [pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
            color=FIG1_LINK, lw=1.1, zorder=1, solid_capstyle="round",
        )
    ax.scatter(pos[:, 0], pos[:, 1], s=70, c=FIG1_NODE, zorder=2, linewidths=0)
    for i, (x, y) in enumerate(pos):
        ax.annotate(
            f"{i + 1}", xy=(x, y), xytext=(13 * x, 13 * y), textcoords="offset points",
            ha="center", va="center", fontsize=9, color=INK["secondary"],
        )
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig


# ---------------------------------------------------------------------------
# 3.1 Distancia media -- the worked Dijkstra example, PDF pp. 7-9
# ---------------------------------------------------------------------------

#: The sentinel the printed listing initialises Ld with: ``Ld := Matrix(N,N,100)``.
DIJKSTRA_SENTINEL = 100.0


def thesis_dijkstra(matrix: np.ndarray, source: int) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """The printed Maple of PDF pp. 7-9, transcribed line for line.

    Returns the row of ``Ld`` for *source* and the dictionary of iteration
    matrices ``L(n)``.  This is *not* the implementation the rest of the
    package uses -- :func:`netmotion.metrics.shortest_path_lengths` is, and it
    is a clean breadth-first search.  This one exists so that the thesis's
    hand-worked intermediate matrices can be drawn as it printed them, and so
    that the two can be checked against each other.

    Three properties of the printed listing shape the intermediate matrices,
    and are worth naming so that they can be read:

    * ``sumaAcumulada`` is the *global* minimum found so far across every
      stored ``L(m)``, rather than the distance of the node being expanded, so
      an entry written late in the run can carry a value larger than the
      distance -- the 10 in the accumulated matrix on PDF p. 8 is one;
    * only the column of the node just left is closed (``C[x,d] := 0``), not the
      row, so a node can be expanded more than once, which is why several rows
      of that same accumulated matrix hold two contributions added together;
    * the closing block takes the minimum over every ``L(n)`` entry in the
      column, which is what fixes the result.
    """
    n_nodes = matrix.shape[0]
    closed = matrix.copy()  # the listing's Co / C
    row = np.full(n_nodes, DIJKSTRA_SENTINEL)
    iterations: dict[int, np.ndarray] = {}

    n, w, i, nl, accumulated = 1, 0, source, 1, 0.0
    while w < 1:
        iterations[n] = np.zeros((n_nodes, n_nodes))
        for j in range(n_nodes):
            if closed[i, j] != 0 and matrix[i, j] != 0:
                iterations[n][i, j] = matrix[i, j] + accumulated

        smallest, target, origin = np.inf, -1, -1
        for m in range(1, nl + 1):
            for a in range(n_nodes):
                for b in range(n_nodes):
                    value = iterations[m][a, b]
                    if 0 < value < smallest and closed[a, b] != 0:
                        smallest, target, origin = value, b, a
        if not np.isfinite(smallest):
            # 'if minimo = infinity then w := 10' -- every node has been reached
            break

        accumulated = smallest
        closed[origin, target] = 0.0
        closed[:, origin] = 0.0
        i = target
        n += 1
        nl = n

    for j in range(n_nodes):
        for i2 in range(n_nodes):
            for m in range(1, nl):
                value = iterations[m][i2, j]
                if 0 < value < row[j]:
                    row[j] = value
    row[row == DIJKSTRA_SENTINEL] = 0.0
    return row, iterations


def thesis_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    """``Ld`` for every source, by running :func:`thesis_dijkstra` N times."""
    return np.vstack([thesis_dijkstra(matrix, s)[0] for s in range(matrix.shape[0])])


def _draw_numbered_ring(ax, network, *, highlight=None, title=None):
    """The ring of PDF p. 7: nodes on an octagon, numbered 1..N, chords inside."""
    pos = circular_layout(network.n)
    for i, j, _ in network.edges():
        ax.plot(
            [pos[i, 0], pos[j, 0]],
            [pos[i, 1], pos[j, 1]],
            color=LINK,
            lw=0.8,
            zorder=1,
            solid_capstyle="round",
        )
    colors = [NODE] * network.n
    if highlight is not None:
        for node, color in highlight.items():
            colors[node] = color
    ax.scatter(pos[:, 0], pos[:, 1], s=80, c=colors, zorder=2, linewidths=0)
    for i, (x, y) in enumerate(pos):
        ax.annotate(
            f"{i + 1}",
            xy=(x, y),
            xytext=(15 * x, 15 * y),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8.5,
            color=INK["secondary"],
        )
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.margins(0.22)
    if title:
        ax.set_title(title, fontsize=9)
    return pos


@figure(
    "TU07",
    chapter=3,
    title="The network the Dijkstra listing is worked on, and its matrix A",
    source=Source(Origin.THESIS_2011, pdf_pages="7", section="3.1 Distancia media"),
    seed=THESIS_SEED,
    reproduces_figure="thesis worked example, right column of PDF p. 7 (unnumbered)",
)
def dijkstra_example_network(ctx: FigureContext):
    """PDF p. 7, right-hand column: the example network and the matrix M.

    Eight nodes on a ring, each joined to its four nearest neighbours -- the two
    either side. The thesis draws it as an octagon with the second-neighbour
    chords crossing inside, which is why the drawing looks like a star, and
    prints the adjacency matrix beside it with four ones in every row.

    Every number in the two pages that follow is read off this matrix, so it is
    the one figure of section 3.1 that has to be shown exactly as printed.
    """
    n, k = 8, 4
    ctx.parameters(N=n, k=k)
    g = ring_lattice(n, k)
    a = g.adjacency

    fig, (left, right) = plt.subplots(1, 2, figsize=(6.4, 3.2))
    _draw_numbered_ring(left, g, title="The ordered network")
    colors = [[SERIES[0] if a[i, j] else INK["muted"] for j in range(n)] for i in range(n)]
    _bracket_matrix(
        right,
        _ints(a),
        label="$A =$",
        colors=colors,
        fontsize=8.0,
        title="Its adjacency matrix",
    )
    right.annotate(
        f"four ones in every row: $k = {k}$",
        xy=(0.5, -0.04),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=7.5,
        color=INK["muted"],
    )

    # No suptitle: the caption carries the title, N and k included.
    return fig


@figure(
    "TU08",
    chapter=3,
    title="Dijkstra by hand: the iteration matrices L(1), L(2) and their sum",
    source=Source(Origin.THESIS_2011, pdf_pages="8", section="3.1 Distancia media"),
    seed=THESIS_SEED,
    reproduces_figure="thesis worked example, right column of PDF p. 8 (unnumbered)",
)
def dijkstra_iteration_matrices(ctx: FigureContext):
    """PDF p. 8: the three matrices the thesis writes out by hand.

    Starting at node 1, ``L(1)`` holds the direct distances -- one row, four
    ones. The algorithm then takes the smallest of them, which is a tie, breaks
    it by index and moves to node 2, and ``L(2)`` holds what is reachable from
    there with ``sumaAcumulada = 1`` added on. Summing every ``L(n)`` gives the
    third matrix, which the thesis prints as the record of the whole run.

    The three panels here are produced by :func:`thesis_dijkstra`, a literal
    transcription of the printed Maple, and they come out identical to the
    printed matrices entry for entry, including the intermediate entries that
    exceed any path length in an eight-node ring -- the 10 in row 5 and the 8
    in row 8. Those arise because the listing carries a single running
    ``sumaAcumulada`` and closes columns only, so a node can be expanded twice,
    the second time with a larger offset; the closing block then takes a
    minimum down each column, and they do not enter the result. Naming them is
    what lets a reader follow the printed matrices through this transcription.
    """
    n, k = 8, 4
    ctx.parameters(N=n, k=k, source_node=1, sentinel=DIJKSTRA_SENTINEL)
    g = ring_lattice(n, k)

    _, iterations = thesis_dijkstra(g.adjacency, 0)
    l1, l2 = iterations[1], iterations[2]
    total = sum(iterations.values())

    # Height trimmed to the matrices: with the suptitle gone the old 2.9 left a
    # blank band under the three panels.
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))

    def shade(matrix, written_color):
        return [
            [written_color if matrix[i, j] else INK["grid"] for j in range(n)] for i in range(n)
        ]

    # L(1) and L(2) are successive steps of one run, not two categories, so they
    # take two stops of the sequential ramp rather than two unrelated hues.
    _bracket_matrix(
        axes[0],
        _ints(l1),
        label="$L(1) =$",
        colors=shade(l1, SEQUENTIAL[3]),
        fontsize=7.5,
        title="From node 1",
    )
    _bracket_matrix(
        axes[1],
        _ints(l2),
        label="$L(2) =$",
        colors=shade(l2, SEQUENTIAL[8]),
        fontsize=7.5,
        title="On to node 2",
    )
    big = total > 2
    colors = [
        [
            SERIES[7] if big[i, j] else (INK["primary"] if total[i, j] else INK["grid"])
            for j in range(n)
        ]
        for i in range(n)
    ]
    _bracket_matrix(
        axes[2],
        _ints(total),
        label=r"$\sum_n L(n) =$",
        colors=colors,
        fontsize=7.5,
        title="Every $L(n)$, summed",
    )
    axes[2].annotate(
        "red: an intermediate entry,\nclosed out of the result",
        xy=(0.5, -0.02),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=7.5,
        color=SERIES[7],
    )

    # No suptitle: the caption carries the title.
    return fig


@figure(
    "TU09",
    chapter=3,
    title="The shortest-path matrix D, and the mean distance it gives",
    source=Source(Origin.THESIS_2011, pdf_pages="9", section="3.1 Distancia media"),
    seed=THESIS_SEED,
    reproduces_figure="thesis Ld matrix and the Lenght/Lteorico comparison, PDF p. 9",
)
def dijkstra_result_and_theory(ctx: FigureContext):
    """PDF p. 9: the matrix the algorithm ends on, and the number it yields.

    Every row of ``Ld`` is a permutation of the same counts -- four nodes one
    step away, three nodes two steps away -- because a ring is the same seen
    from any of its nodes. So

        l = (4x1 + 3x2)/7 = 10/7 = 1.4286

    and the thesis prints 1.42.

    It then compares that with the value for an ordered network,
    ``Lteorico = N/(2 k)``, which for N = 8 and k = 4 is 1, notes that the two
    disagree, and explains why: the theoretical expression holds for N >> 1. It
    checks the explanation at N = 50, where it reports Lteorico = 6.25 against
    Lenght = 6.63. The right-hand panel is that check carried across every size
    in between, and the thesis's two data points sit on it.
    """
    n, k = 8, 4
    sizes = np.arange(8, 121, 2)
    ctx.parameters(N=n, k=k, comparison_sizes="8..120 even, k = 4")

    g = ring_lattice(n, k)
    ld = thesis_distance_matrix(g.adjacency)
    ell = ld.sum() / (n * (n - 1))

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.2), width_ratios=[1.0, 1.05])

    colors = [
        [
            SERIES[0] if ld[i, j] == 1 else (SERIES[1] if ld[i, j] == 2 else INK["grid"])
            for j in range(n)
        ]
        for i in range(n)
    ]
    _bracket_matrix(
        left,
        _ints(ld),
        label="$D =$",
        colors=colors,
        fontsize=8.0,
    )
    left.annotate(
        "blue: one step      orange: two steps",
        xy=(0.5, -0.06),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=7.5,
        color=INK["muted"],
    )

    measured = np.array([mean_distance(ring_lattice(int(m), k)) for m in sizes])
    theory = sizes / (2.0 * k)

    # The measured curve is the series; the closed form is context behind it,
    # so it is a hairline rather than a second full-weight line.  Both are named
    # at their right-hand end, which is where the eye leaves them.
    right.plot(sizes, measured, color=SERIES[0], lw=1.8)
    right.plot(sizes, theory, color=INK["muted"], lw=0.9, ls=(0, (4, 3)))
    for values, note, at, offset, va, color in (
        (measured, r"$\ell$, the algorithm", 0.62, (-6, 12), "bottom", SERIES[0]),
        (theory, r"$\ell_{\mathrm{theory}} = N/(2\,k)$", 0.86, (6, -12), "top", INK["secondary"]),
    ):
        idx = int(at * (sizes.size - 1))
        right.annotate(
            note,
            xy=(sizes[idx], values[idx]),
            xytext=offset,
            textcoords="offset points",
            ha="right" if offset[0] < 0 else "left",
            va=va,
            fontsize=8,
            color=color,
        )
    for size, note, offset in (
        (8, f"$N=8$:  {ell:.2f} vs 1.00", (16, -22)),
        (50, "$N=50$:  6.63 vs 6.25", (10, -18)),
    ):
        idx = int(np.argmin(np.abs(sizes - size)))
        right.plot([sizes[idx]], [measured[idx]], marker="o", ms=5, color=SERIES[1], zorder=3)
        right.annotate(
            note,
            xy=(sizes[idx], measured[idx]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.5,
            color=SERIES[1],
            arrowprops=dict(arrowstyle="-", color=SERIES[1], lw=0.6, shrinkA=0, shrinkB=3),
        )
    # Room under the curves for the N = 8 label, without a tick below zero.
    right.set_ylim(-2.2, float(measured.max()) * 1.06)
    right.set_yticks(np.arange(0, measured.max() + 1, 2.0))
    right.set_xlabel("$N$")
    right.set_ylabel(r"$\ell$")
    # Wrapped by hand, not measured by panel_title(): this figure is drawn at
    # a canvas wider than it prints at, and save_figure() only narrows it to
    # the KDP print width *after* this function returns, so a width measured
    # in here is the wrong one -- panel_title would see the title fit the
    # wide draft and leave it on one line, which then runs into the panel
    # beside it once the canvas is actually narrowed.  The break keeps every
    # word of the original title; only the line ending moves.
    left.set_title("Four ones and three twos,\nevery row", fontsize=9)
    right.set_title(f"The gap closes as $N$ grows\n($k = {k}$)", fontsize=9.5)

    # No suptitle: the caption carries the title and the value of ell.
    return fig


# ---------------------------------------------------------------------------
# 3.2 Degree Distribution -- PDF pp. 9-10
# ---------------------------------------------------------------------------


def thesis_degree_distribution(network) -> tuple[np.ndarray, np.ndarray]:
    """The listing of PDF pp. 9-10, transcribed: ``k`` per node and ``p``.

    Returns ``(k, p)`` where ``k`` is the per-node degree vector the first loop
    fills and ``p`` is indexed by degree, ``p[j]`` being the value the listing
    stores in ``p_j``.

    The normalisation is the listing's own.  The inner loop accumulates
    ``p_n := p_n + k_i`` -- the *degree*, not one -- and then divides by
    ``ksum``, the total degree.  So what comes out is

        p_n = n x (number of nodes of degree n) / sum_i k_i

    the degree-weighted distribution ``n P(n) / <k>``, where
    ``P(n) = (number of nodes of degree n) / N`` is the unweighted form.  The
    two agree when every node has the same degree, which is the case the
    listing's last branch handles explicitly.  Both are drawn in
    :func:`degree_algorithm`.
    """
    k = network.degrees()
    ksum = int(k.sum())
    positive = k[k > 0]
    mink = int(positive.min()) if positive.size else 0
    maxk = int(k.max()) if k.size else 0

    p = np.zeros(maxk + 1)
    if ksum:
        for i in range(network.n):
            for n in range(mink, maxk + 1):
                if n == k[i]:
                    p[n] += k[i]
        p /= ksum
    if mink == maxk:
        # 'if mink = maxk then p[mink] := 1' -- every node has the same degree
        p[:] = 0.0
        p[mink] = 1.0
    return k, p


@figure(
    "TU10",
    chapter=3,
    title="What the degree-distribution listing computes",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="9-10",
        section="3.2 Degree Distribution - Distribucion de grado",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis degree-distribution algorithm box, PDF pp. 9-10 (unnumbered)",
)
def degree_algorithm(ctx: FigureContext):
    """PDF pp. 9-10: the listing, run on the chapter's own network and on one
    that is not regular.

    Section 3.2 prints two pages of Maple and no figure, so what is drawn here
    is the listing's output. It builds two things: a vector ``k`` holding the
    degree of every node -- the left column, one point per node, which is the
    quantity the thesis plots as *grado por nodo* throughout chapter 4 -- and a
    vector ``p`` indexed by degree.

    Top row: the eight-node ring of section 3.1, where every node has k = 4.
    ``mink = maxk``, the listing takes its last branch, and p is a spike of
    height 1 at k = 4.

    Bottom row: an Erdos-Renyi network, so that the general branch runs. Here
    the listing's ``p`` is a degree-weighted distribution rather than the plain
    P(k): it accumulates the *degree* into each bin and divides by the total
    degree, giving ``n P(n)/<k>``. It sums to 1 in that weighted sense, and it
    sits higher at large k than P(k) does -- visibly, in the right-hand panel.
    The two forms differ by a factor of k, so on logarithmic axes their slopes
    differ by one, which is why this panel names which of the two it is
    drawing. The vector the thesis plots throughout chapter 4 is the left
    column, the per-node degree k, not this one.
    """
    ring_n, ring_k = 8, 4
    er_n, er_p = 50, 0.10
    ctx.parameters(ring=dict(N=ring_n, k=ring_k), erdos_renyi=dict(N=er_n, p=er_p))

    cases = [
        (
            f"ordered ring\n$N={ring_n}$,  $k={ring_k}$",
            ring_lattice(ring_n, ring_k),
        ),
        (
            f"Erdos-Renyi\n$N={er_n}$,  $p={er_p}$",
            erdos_renyi(er_n, er_p, rng=ctx.rng),
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.6))

    # The two rows are different networks at different sizes, so the axes cannot
    # be shared; what is shared is the *meaning* of each column, and that is
    # written once along the top rather than six times round the panels.
    columns = (
        "Degree per node",
        "Nodes per degree",
        "The listing's $p_k$ vs $P(k)$",
    )
    for ax, title in zip(axes[0], columns, strict=True):
        ax.set_title(title, fontsize=9)

    for row, (label, g) in enumerate(cases):
        k, p = thesis_degree_distribution(g)
        bottom = row == len(cases) - 1

        # (a) the vector k: degree against node index, one value per node
        ax = axes[row, 0]
        per_node(ax, k, mean_label=rf"$\langle k \rangle = {k.mean():.2f}$")
        ax.set_ylim(0, max(3.0, k.max() + 1.4))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.set_ylabel("$k_i$")

        # (b) the counts the histogram loop builds.  The bins run from zero so
        # that the regular network reads as a spike rather than as one wide bar.
        ax = axes[row, 1]
        counts = np.bincount(k, minlength=int(k.max()) + 1)
        histogram(ax, k, bins=np.arange(-0.5, int(k.max()) + 1.5))
        ax.set_ylabel("nodes")

        # (c) the thesis's p against the ordinary P(k) ----------------------
        # P(k) is the reference and carries the plain hue; the listing's own
        # normalisation is the subject of the panel, so it is the one picked out.
        ax = axes[row, 2]
        standard = counts / g.n
        support = np.arange(standard.size)
        support_p = np.arange(p.size)
        ax.bar(support, standard, width=1.0, color=SERIES[0], edgecolor="white", lw=0.8, zorder=2)
        ax.step(support_p, p, where="mid", color=SERIES[1], lw=1.8, zorder=3)
        ax.set_ylabel("probability")
        ax.set_ylim(0, max(p.max(), standard.max()) * 1.30)
        ax.set_xlim(-0.6, max(support[-1], support_p[-1]) + 0.6)

        if bottom:
            # Two series, so two direct labels rather than a legend: P(k) on its
            # peak, the listing's p_k out on the tail it has fattened.
            shoulder = int(np.argmax(p[: standard.size] - standard))
            ax.annotate(
                "the listing's $p_k$",
                xy=(support_p[shoulder], p[shoulder]),
                xytext=(4, 9),
                textcoords="offset points",
                ha="left",
                fontsize=7.5,
                color=SERIES[1],
            )
            top_s = int(np.argmax(standard))
            ax.annotate(
                r"$P(k)$",
                xy=(support[top_s], standard[top_s]),
                xytext=(-5, 5),
                textcoords="offset points",
                ha="right",
                fontsize=7.5,
                color=SERIES[0],
            )
        else:
            ax.annotate(
                "one degree only, so\nthe two coincide",
                xy=(0.04, 0.86),
                xycoords="axes fraction",
                fontsize=7.5,
                color=INK["secondary"],
            )

        # Axis names once per column, on the bottom row only.
        if bottom:
            for column, name in zip(axes[row], ("node $i$", "$k$", "$k$"), strict=True):
                column.set_xlabel(name)

        # The row named once, on the left.
        axes[row, 0].annotate(
            label,
            xy=(-0.75, 0.5),
            xycoords="axes fraction",
            rotation=90,
            ha="center",
            va="center",
            fontsize=9,
            weight="semibold",
            color=INK["primary"],
        )

    # No suptitle: the caption carries the title.  The panel titles inside the
    # grid stay, because they identify the panels and a caption cannot.
    return fig


# ---------------------------------------------------------------------------
# 3.3 Clustering -- PDF p. 11
# ---------------------------------------------------------------------------


@figure(
    "TU11",
    chapter=3,
    title="What the clustering listing computes: k, E and C, node by node",
    source=Source(Origin.THESIS_2011, pdf_pages="10-11", section="3.3 Clustering - Conglomerado"),
    seed=THESIS_SEED,
    reproduces_figure="thesis clustering algorithm box, PDF p. 11 (unnumbered)",
)
def clustering_algorithm(ctx: FigureContext):
    """PDF pp. 10-11: the definition, and the listing that evaluates it.

    Section 3.3 defines C_i as the fraction of the links that *could* exist
    among the k_i neighbours of node i and actually do:

        C_i = 2 E_i / [k_i (k_i - 1)]

    and gives a listing that fills three vectors -- k, E and C -- and averages
    the last one. It prints no figure, so this draws the three quantities.

    Left: the definition on the section's own network. Node 1 of the eight-node
    ring has four neighbours; of the six pairs they form, three are joined, so
    C_1 = 2x3/(4x3) = 1/2. Every node of the ring gives the same, which is why
    an ordered network has high clustering.

    Middle and right: the same three vectors for a Watts-Strogatz network at
    the parameters of the thesis's fig. 6 (N = 20, k = 4, p = 0.5). Rewiring
    breaks triangles, so E falls where k does not, and C is no longer the same
    at every node -- which is the observation the whole of chapter 4 turns on.
    """
    ring_n, ring_k = 8, 4
    ws_n, ws_k, ws_p = 20, 4, 0.5
    ctx.parameters(ring=dict(N=ring_n, k=ring_k), watts_strogatz=dict(N=ws_n, k=ws_k, p=ws_p))

    ring = ring_lattice(ring_n, ring_k)
    ws = watts_strogatz(ws_n, ws_k, ws_p, rng=ctx.rng)

    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.9), width_ratios=[0.85, 1.05, 1.05])

    # (a) the definition, drawn ------------------------------------------
    # Two hues, keyed to the two symbols in the formula: blue is k_1, the links
    # leaving node 1, and orange is E_1, the links among the nodes it reaches.
    ax = axes[0]
    focus = 0
    neighbours = list(ring.neighbors(focus))
    pos = circular_layout(ring.n)

    # every pair of neighbours that could be joined, dashed
    for a_idx, a in enumerate(neighbours):
        for b in neighbours[a_idx + 1 :]:
            joined = ring.has_edge(a, b)
            ax.plot(
                [pos[a, 0], pos[b, 0]],
                [pos[a, 1], pos[b, 1]],
                color=SERIES[1] if joined else INK["grid"],
                lw=2.4 if joined else 1.0,
                ls="-" if joined else (0, (2, 2)),
                zorder=1,
            )
    for i, j, _ in ring.edges():
        touches_focus = focus in (i, j)
        ax.plot(
            [pos[i, 0], pos[j, 0]],
            [pos[i, 1], pos[j, 1]],
            color=SERIES[0] if touches_focus else INK["grid"],
            lw=1.6 if touches_focus else 0.7,
            zorder=0,
        )
    colors = [INK["grid"]] * ring.n
    for node in neighbours:
        colors[node] = SERIES[0]
    colors[focus] = INK["primary"]
    ax.scatter(pos[:, 0], pos[:, 1], s=44, c=colors, zorder=3, linewidths=0)
    for i, (x, y) in enumerate(pos):
        ax.annotate(
            f"{i + 1}",
            xy=(x, y),
            xytext=(9 * x, 9 * y),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8,
            color=INK["primary"] if i == focus else INK["secondary"],
            weight="semibold" if i == focus else "normal",
        )
    k_focus = int(ring.degrees()[focus])
    e_focus = int(neighbour_links(ring)[focus])
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.margins(0.28)
    ax.set_title("Counting $E$\nat node 1", fontsize=9)
    for text, height, color in (
        (rf"$k_1={k_focus}$ links out", -0.03, SERIES[0]),
        (rf"$E_1={e_focus}$ among them", -0.14, SERIES[1]),
        (
            rf"$C_1 = \frac{{2\cdot{e_focus}}}{{{k_focus}\cdot{k_focus - 1}}} = "
            rf"{2 * e_focus / (k_focus * (k_focus - 1)):.1f}$",
            -0.28,
            INK["primary"],
        ),
    ):
        ax.annotate(
            text,
            xy=(0.5, height),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=9,
            color=color,
        )

    # (b) k and E per node ------------------------------------------------
    k = ws.degrees()
    e = neighbour_links(ws)
    c = local_clustering(ws)
    nodes = np.arange(1, ws.n + 1)

    ax = axes[1]
    per_node(ax, k)
    ax.plot(nodes, e, ls="none", marker="o", ms=4.5, color=SERIES[1], zorder=4)
    ax.set_ylim(-0.4, max(k.max(), e.max()) + 1.6)
    # Two series, so two direct labels at the right-hand end rather than a legend.
    for values, name, color in ((k, "$k_i$", SERIES[0]), (e, "$E_i$", SERIES[1])):
        ax.annotate(
            name,
            xy=(nodes[-1], values[-1]),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=color,
            annotation_clip=False,
        )
    ax.set_xlabel("node $i$")
    ax.set_ylabel("count")
    ax.set_title("$E$ drops more than $k$", fontsize=9)

    # (c) C per node ------------------------------------------------------
    ax = axes[2]
    per_node(ax, c, mean_label=rf"$C = {c.mean():.2f}$")
    ring_c = float(local_clustering(ring).mean())
    reference_line(ax, ring_c, f"ring, {ring_c:.1f}", color=SERIES[1])
    ax.set_xlabel("node $i$")
    ax.set_ylabel("$C_i$")
    ax.set_ylim(0, 0.6)
    ax.set_title("Clustering falls unevenly", fontsize=9)

    # No suptitle: the definition and the Watts-Strogatz parameters are both
    # stated in the caption.  The three panel titles stay.
    return fig
