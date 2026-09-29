"""Figures for chapter 3: characterising a network.

Mean distance, degree distribution and clustering -- the three numbers the rest
of the book is measured with.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..metrics import (
    all_pairs_distances,
    local_clustering,
    mean_distance,
)
from ..networks import ring_lattice, watts_strogatz
from ..provenance import Origin, Source
from ..visualization import INK, SEQUENTIAL, SERIES, draw_adjacency, draw_network
from ..visualization.charts import histogram, per_node, reference_line
from .registry import FigureContext, figure

THESIS_SEED = 20110930


@figure(
    "F02-01",
    chapter=2,
    title="A network and its adjacency matrix",
    source=Source(Origin.THESIS_2011, pdf_pages="6", section="2.2 Matriz Adyacente"),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 1 and the example matrices, PDF p. 6",
)
def adjacency_matrix(ctx: FigureContext):
    """PDF p. 6: the same network drawn twice.

    On the left as nodes and links, on the right as the matrix. The thesis
    treats the matrix as the primary object -- every algorithm it writes reads
    and writes this array -- so it is worth seeing the two side by side once.
    """
    n, k = 8, 4
    ctx.parameters(N=n, k=k)
    g = ring_lattice(n, k)

    fig, (left, right) = plt.subplots(1, 2, figsize=(6.2, 3.0))
    draw_network(g, left, node_color=SERIES[0])
    left.set_title("The network", fontsize=9.5)

    draw_adjacency(g, right, title="The same links, as $A$: four ones a row")
    right.set_xticks(range(n))
    right.set_yticks(range(n))
    right.set_xticklabels(range(1, n + 1), fontsize=7.5)
    right.set_yticklabels(range(1, n + 1), fontsize=7.5)

    # No suptitle: the caption carries the title, and N and k are stated there.
    return fig


@figure(
    "F03-01",
    chapter=3,
    title="The worked example: mean distance of an eight-node ring",
    source=Source(Origin.THESIS_2011, pdf_pages="7-9", section="3.1 Distancia media"),
    seed=THESIS_SEED,
    reproduces_figure="thesis worked example, PDF pp. 7-9",
)
def mean_distance_example(ctx: FigureContext):
    """PDF pp. 7-9: the example the thesis computes by hand.

    Eight nodes on a ring, each joined to its four nearest neighbours. From any
    node, four others are one step away and the remaining three are two steps
    away, so

        l = (4x1 + 3x2)/7 = 10/7 = 1.4286

    and the thesis prints 1.42. The right panel is the distance matrix its
    algorithm builds; every row is a permutation of the same counts, because
    the ring is vertex-transitive.
    """
    n, k = 8, 4
    ctx.parameters(N=n, k=k)
    g = ring_lattice(n, k)
    dist = all_pairs_distances(g)
    ctx.parameters(mean_distance=round(float(mean_distance(g)), 4))

    fig, (left, right) = plt.subplots(1, 2, figsize=(6.4, 3.1))

    hops = dist[0]
    draw_network(g, left, node_color=hops, scale_by_degree=False)
    # Zero hops is the palest stop of the ramp, so the node the distances are
    # measured from would otherwise be the one you cannot see.
    source_x, source_y = _circle(n)[0]
    left.plot([source_x], [source_y], "o", ms=8, color=SERIES[1], zorder=5)
    for i, (x, y) in enumerate(_circle(n)):
        left.annotate(
            f"{int(hops[i])}" if i else "0, node 1",
            xy=(x, y),
            xytext=(15 * x, 15 * y),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8,
            color=SERIES[1] if i == 0 else INK["secondary"],
            weight="semibold" if i == 0 else "normal",
        )
    left.margins(0.24)
    left.set_title("Hops from node 1", fontsize=9.5)

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("nm-seq", ["#ffffff", SEQUENTIAL[4]])
    right.imshow(dist, cmap=cmap, interpolation="nearest")
    for i in range(n):
        for j in range(n):
            right.text(
                j,
                i,
                f"{int(dist[i, j])}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=INK["primary"] if dist[i, j] < 2 else "#ffffff",
            )
    right.set_xticks(range(n))
    right.set_yticks(range(n))
    right.set_xticklabels(range(1, n + 1), fontsize=7.5)
    right.set_yticklabels(range(1, n + 1), fontsize=7.5)
    right.grid(False)
    right.set_title(r"Every row of $D_{ij}$ holds the same counts", fontsize=9.5)

    # No suptitle: the caption carries the title and the value of ell.
    return fig


def _circle(n: int):
    angles = 2.0 * np.pi * np.arange(1, n + 1) / n
    return list(zip(np.cos(angles), np.sin(angles), strict=True))


@figure(
    "F03-02",
    chapter=3,
    title="The three characteristics on one network",
    source=Source(Origin.THESIS_2011, pdf_pages="7-11", section="3. Caracteristicas"),
    seed=THESIS_SEED,
    reproduces_figure="thesis figs. 4.d-l, PDF p. 15",
)
def three_characteristics(ctx: FigureContext):
    """PDF p. 15: the thesis tabulates degree, P(k) and C per node.

    Shown here for one Watts-Strogatz network so the three can be read against
    each other: the degree distribution is narrow, the clustering is not
    uniform, and the two are only loosely related.
    """
    n, k, p = 60, 6, 0.15
    ctx.parameters(N=n, k=k, p=p)
    g = watts_strogatz(n, k, p, rng=ctx.rng)

    degrees = g.degrees()
    clustering = local_clustering(g)

    # The busiest node is picked out in both per-node panels, because the point
    # of showing the three together is that the node with the most neighbours is
    # not the node whose neighbours know each other.
    busiest = int(np.argmax(degrees))

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7))

    per_node(
        axes[0],
        degrees,
        mean_label=rf"$\langle k \rangle = {degrees.mean():.1f}$",
        highlight=[busiest],
    )
    axes[0].set_xlabel("node $i$")
    axes[0].set_ylabel("$k_i$")
    axes[0].set_title("Degree stays near $k$", fontsize=9.5)

    # Unit-width integer bins, so density is exactly (count of nodes) / N.
    histogram(axes[1], degrees, density=True)
    reference_line(axes[1], float(degrees.mean()), r"$\langle k \rangle$", axis="x")
    axes[1].set_xlabel("$k$")
    axes[1].set_ylabel("$P(k)$")
    axes[1].set_title("The spread is narrow", fontsize=9.5, pad=16)

    per_node(
        axes[2],
        clustering,
        mean_label=rf"$C = {clustering.mean():.2f}$",
        highlight=[busiest],
    )
    axes[2].set_ylim(0, 1.0)
    axes[2].set_xlabel("node $i$")
    axes[2].set_ylabel("$C_i$")
    axes[2].set_title("Clustering is far less even", fontsize=9.5)

    # The same node named in both per-node panels, so the reader can see that
    # the busiest node is nothing special once its neighbours are counted.
    for ax, value in ((axes[0], degrees[busiest]), (axes[2], clustering[busiest])):
        ax.annotate(
            f"node {busiest + 1}",
            xy=(busiest + 1, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=SERIES[1],
            bbox=dict(boxstyle="round,pad=0.15", facecolor=INK["surface"], edgecolor="none"),
        )

    # No suptitle: the parameters have been moved into the caption.  The three
    # panel titles stay, because they identify the panels.
    return fig
