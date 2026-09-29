"""Chapter 4, sections 4.3 and 4.4: the figures of PDF pp. 23-31.

One figure here per numbered figure of the thesis, named after it: ``T10`` is
the thesis's fig. 10, ``T16`` its fig. 16.  Where the thesis prints a table of
panels (fig. 10 has twelve, lettered a-l) the reproduction builds the same
table as one figure, rows = quantity, columns = the parameter, as printed.

Covered here
------------
==========  =========  ====================================================
Thesis      PDF page   What it shows
==========  =========  ====================================================
fig. 10     24-25      NW at N = 10: network, P(k), k per node, C per node
fig. 11     25         NW averages <C>, <k>, <l> against p
fig. 12     26         NW normalised l(p)/l(0) and C(p)/C(0) against p
fig. 13     30         Scale free, Opcion 1: network and k per node
fig. 14     30         Scale free, Opcion 2: network and k per node
fig. 15     31         Scale free, Opcion 3: network and k per node
fig. 16     31         The scale-free P(k) and its 2m^2 data collapse
==========  =========  ====================================================

Run them with::

    python -m netmotion.experiments.cli T10 T11 T12 T13 T14 T15 T16
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from ..metrics import (
    clustering_coefficient,
    local_clustering,
    mean_degree,
    mean_distance,
)
from ..networks import barabasi_albert, newman_watts, ring_lattice
from ..provenance import Origin, Source
from ..visualization import INK, SEQUENTIAL, SERIES, draw_network
from ..visualization.charts import histogram, mean_with_band, panel_letter, per_node, reference_line
from .registry import FigureContext, figure

THESIS_SEED = 20110930  # the date of submission, as everywhere else in the book


def _preferential_degrees(n: int, m: int, rng) -> np.ndarray:
    """The degrees of a preferential-attachment network of *n* nodes.

    Fig. 16 is drawn at 300,000 nodes, and
    :func:`~netmotion.networks.barabasi_albert` keeps a dense ``n x n``
    adjacency matrix, which at that size no machine will hold.  A degree
    distribution needs the degrees and nothing else, and preferential
    attachment hands them over directly: in an array holding every endpoint
    ever written, each node appears exactly as often as its degree, so drawing
    from that array uniformly *is* drawing a node with probability
    :math:`k_i / \\sum_j k_j`.  The seed is the complete network on
    :math:`m + 1` nodes, as everywhere else here.
    """
    m0 = m + 1
    seed_edges = m0 * (m0 - 1) // 2
    endpoints = np.empty(2 * (seed_edges + m * (n - m0)), dtype=np.int64)
    at = 0
    for i in range(m0):
        for j in range(i + 1, m0):
            endpoints[at], endpoints[at + 1] = i, j
            at += 2

    for new in range(m0, n):
        # m *distinct* targets, as the rule asks; repeats are redrawn rather
        # than allowed to collapse an arrival into fewer than m links.
        targets = np.unique(endpoints[rng.integers(0, at, size=m)])
        while targets.size < m:
            more = endpoints[rng.integers(0, at, size=m - targets.size)]
            targets = np.unique(np.concatenate((targets, more)))
        endpoints[at : at + 2 * m : 2] = new
        endpoints[at + 1 : at + 2 * m : 2] = targets
        at += 2 * m

    return np.bincount(endpoints[:at], minlength=n)


#: The three connection probabilities the thesis puts in the columns of fig. 10.
NW_PROBABILITIES = (0.1, 0.5, 1.0)

#: The Newman-Watts network of section 4.3.3 is stated in the text: "una red
#: con N = 10 nodos", on the ordered ring of degree 4 that p = 0 leaves behind.
NW_N, NW_K = 10, 4

#: How many networks are drawn at each p of the fig. 11 and fig. 12 sweeps.
#: Ten nodes is small enough that a single draw carries a standard deviation of
#: about a tenth on <C>, which is the size of the effect being swept; forty
#: draws divide that by six and turn the curve back into a curve.  The whole of
#: T11 and T12 still runs in a few seconds at this count, so there is nothing
#: to buy by going lower.
NW_REALISATIONS = 40

#: p is an ordered quantity, so the columns of the fig. 10 table run light to
#: dark along the one sequential ramp rather than taking three unrelated hues.
NW_COLUMN_COLORS = (SEQUENTIAL[2], SEQUENTIAL[5], SEQUENTIAL[8])


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


# ---------------------------------------------------------------------------
# 4.3.3 Caracteristicas -- Newman-Watts
# ---------------------------------------------------------------------------


@figure(
    "T10",
    chapter=4,
    title="Newman-Watts at N = 10: the twelve panels",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="24-25",
        section="4.3.3 Caracteristicas (Newman-Watts)",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 10.a-l, PDF pp. 24-25",
)
def nw_characteristics_table(ctx: FigureContext):
    """PDF pp. 24-25, fig. 10.a-l: N = 10 at p = 0.1, 0.5 and 1.

    The thesis lays this out as a table: the columns are the connection
    probability, the rows are the four things it measures -- the network, the
    degree distribution P(k), the degree of each node and the clustering
    coefficient of each node.  The reproduction keeps that layout, because the
    argument is made by reading down a column and across a row.

    Two of these rows are per-node plots, one value per node against the node
    index, not distributions: fig. 10.g-i is the degree of node 1, node 2, ...
    node 10, and fig. 10.j-l the same for the clustering coefficient.  At
    p = 0.1 both are flat -- the ring has barely been touched, every node still
    has k = 4 and C = 0.5 -- and the thesis's point is that they lift and
    roughen together as p grows, because Newman-Watts only ever *adds* links.

    The realisation drawn here reproduces the printed one closely: k = 4 for
    eight of ten nodes at p = 0.1, degrees spread over 5-8 at p = 0.5, and
    P(k) peaking at k = 7 exactly as printed in fig. 10.e.
    """
    ctx.parameters(N=NW_N, k=NW_K, p=list(NW_PROBABILITIES), realisation="single")

    # One realisation per column, drawn in order from the figure's own stream --
    # the thesis shows one network per probability, not an average.
    networks = [newman_watts(NW_N, NW_K, p, rng=ctx.rng) for p in NW_PROBABILITIES]

    fig, axes = plt.subplots(4, 3, figsize=(7.4, 4.6), sharey="row")
    letters = "abcdefghijkl"
    kmax = NW_N - 1

    # Row 1 -- the network (fig. 10.a-c)
    for ax, g, p, colour in zip(axes[0], networks, NW_PROBABILITIES, NW_COLUMN_COLORS, strict=True):
        draw_network(
            g, ax, node_color=colour, edge_width=1.0, node_size=0.6 * max(6.0, 260.0 / g.n**0.55)
        )
        for coll in ax.collections:
            coll.set_clip_on(False)
        ax.set_title(f"$p = {p:g}$", fontsize=11, loc="center")
    axes[0, 0].annotate(
        "the network",
        xy=(-0.08, 0.5),
        xycoords="axes fraction",
        rotation=90,
        ha="right",
        va="center",
        fontsize=8.5,
        color=INK["secondary"],
    )

    # Row 2 -- P(k) (fig. 10.d-f)
    peak = 0.0
    for ax, g, colour in zip(axes[1], networks, NW_COLUMN_COLORS, strict=True):
        pk = _degree_fraction_bars(ax, g.degrees(), kmax=kmax, color=colour)
        peak = max(peak, float(pk.max()))
        ax.set_xlim(-0.5, kmax + 0.5)
        ax.set_xticks([0, 2, 4, 6, 8])
    axes[1, 0].set_ylim(0, 1.15 * peak)
    axes[1, 0].set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    axes[1, 0].set_ylabel("$P(k)$")
    axes[1, 1].set_xlabel("degree $k$")

    # Row 3 -- the degree of each node (fig. 10.g-i)
    for col, (ax, g, colour) in enumerate(zip(axes[2], networks, NW_COLUMN_COLORS, strict=True)):
        per_node(ax, g.degrees(), color=colour)
        reference_line(ax, NW_K, "")
        if col == 2:
            ax.annotate(
                f"ring, $k={NW_K}$",
                xy=(1.01, NW_K),
                xycoords=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=7.5,
                color=INK["secondary"],
                annotation_clip=False,
            )
        ax.set_ylim(0, 9.6)
        ax.set_yticks([0, 2, 4, 6, 8])
        ax.set_xticks([1, 5, 10])
        ax.tick_params(labelbottom=False)
    axes[2, 0].set_ylabel("degree $k$")

    # Row 4 -- the clustering coefficient of each node (fig. 10.j-l)
    for ax, g, colour in zip(axes[3], networks, NW_COLUMN_COLORS, strict=True):
        per_node(ax, local_clustering(g), color=colour)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xticks([1, 5, 10])
    axes[3, 0].set_ylabel("clustering $C$")
    axes[3, 1].set_xlabel("node index")

    # The book's text cites these panels by letter, so they stay -- small,
    # muted, and clear of the marks.
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


@figure(
    "T11",
    chapter=4,
    title="Newman-Watts: mean clustering, mean degree and mean distance against p",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="25-26",
        section="4.3.3 Caracteristicas (Newman-Watts)",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 11.a-c, PDF p. 25",
)
def nw_mean_characteristics(ctx: FigureContext):
    """PDF p. 25, fig. 11.a-c: the three averages of the N = 10 network.

    The thesis sweeps p and plots, from a single realisation at each p, the
    mean clustering coefficient, the mean degree and the mean distance.  At
    N = 10 a single draw moves <C> by about a tenth between neighbouring values
    of p, which is the size of the trend itself, so each point here is an
    average instead.

    Each point here is therefore the mean of NW_REALISATIONS draws at that p,
    and the band behind it is their interquartile range.  Both are worth
    having: the line is the estimate, and the band puts a width on any one
    ten-node measurement -- the same spread the printed single-realisation
    curve traces out point by point, drawn as a band instead of a jag.

    <C> and <k> both rise, and for the same reason: the model adds links and
    never removes one, so every node ends the sweep with more neighbours than
    it started with.  <l> falls.  Rising clustering and falling distance
    together are the small-world effect, and section 4.3.3 closes on exactly
    that.

    The ring parameters for panel c are not printed, and both values are given
    here.  This reproduction plots the ring the text specifies, N = 10 and
    k = 4 -- the network fig. 10.a draws, and the one whose C = 0.5 fig. 11.a
    starts from -- which has <l> = 1.67; the printed panel starts near 3, the
    mean distance of a plain 10-cycle (k = 2, <l> = 2.78).
    """
    realisations = NW_REALISATIONS
    ps = np.linspace(0.01, 1.0, 100)
    ctx.parameters(N=NW_N, k=NW_K, p="0.01..1 in 100 linear steps")
    ctx.realisations(realisations)

    clustering = np.empty((realisations, ps.size))
    degree = np.empty((realisations, ps.size))
    distance = np.empty((realisations, ps.size))
    for i, p in enumerate(ps):
        for r in range(realisations):
            g = newman_watts(NW_N, NW_K, p, rng=ctx.rng)
            clustering[r, i] = clustering_coefficient(g)
            degree[r, i] = mean_degree(g)
            distance[r, i] = mean_distance(g, "largest_component")

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), sharex=True)
    panels = (
        (clustering, r"$\langle C \rangle$", "More triangles"),
        (degree, r"$\langle k \rangle$", "More neighbours"),
        (distance, r"$\langle \ell \rangle$", "Shorter distances"),
    )
    for letter, (ax, (values, ylabel, title)) in zip(
        "abc", zip(axes, panels, strict=True), strict=True
    ):
        panel_letter(ax, letter)
        mean_with_band(ax, ps, values, color=SERIES[0])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlim(0, 1)
        # Three ticks, not five: at the narrow per-panel width five labels
        # ("0.00" through "1.00") ran together into an unreadable smear.
        ax.set_xticks([0, 0.5, 1.0])

    ring = ring_lattice(NW_N, NW_K)
    reference_line(axes[2], mean_distance(ring), "ring 1.67")

    fig.supxlabel("probability of connection $p$", fontsize=9.5, color=INK["secondary"])
    # The same form as the Watts-Strogatz figure (T07): the caption does not
    # give N, k, the realisation count or the band, so this line does, with the
    # band spelled out.  "Newman-Watts" is left off because with it the line is
    # wider than the 5.05 in print column (the section heading names the model).
    fig.suptitle(
        f"$N = {NW_N}$, $k = {NW_K}$, mean of {realisations} networks per $p$; band: interquartile range",
        fontsize=9,
        color=INK["secondary"],
        x=0.0,
        ha="left",
    )
    return fig


@figure(
    "T12",
    chapter=4,
    title="Newman-Watts: the clustering ratio rises above one while the distance falls",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="26",
        section="4.3.3 Caracteristicas (Newman-Watts)",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 12, PDF p. 26",
)
def nw_normalised_length_and_clustering(ctx: FigureContext):
    """PDF p. 26, fig. 12: *Variacion de <l>, <C> con p para red NW*.

    The same normalisation as the Watts-Strogatz figure it is meant to be read
    against (fig. 9, PDF p. 23, reproduced as F04-05): both quantities divided
    by their value on the ordered ring, p on a logarithmic axis.

    The difference is the whole of section 4.3.  Under Watts-Strogatz the
    clustering can only fall, because a rewired link is a triangle destroyed.
    Under Newman-Watts nothing is destroyed, so C(p)/C(0) climbs *above* one --
    to about 1.5 here, and to about 1.65 in the printed figure -- while
    l(p)/l(0) falls.  The thesis's conclusion is that this makes the small-world
    effect easier to obtain, and the two curves parting above and below 1 is
    the reason.

    That parting is a few tenths wide, and at N = 10 a single draw per p moves
    each ratio by about as much.  Each point is therefore the mean of
    NW_REALISATIONS networks, with the interquartile range of those networks
    behind it in the same hue.  The bands separate at about the p where the two
    means part, so the parting is something the reader can see for himself.
    """
    realisations = NW_REALISATIONS
    ps = np.logspace(-2.0, 0.0, 90)
    ctx.parameters(N=NW_N, k=NW_K, p="0.01..1 log-spaced, 90 points")
    ctx.realisations(realisations)

    ring = ring_lattice(NW_N, NW_K)
    l0 = mean_distance(ring, "largest_component")
    c0 = clustering_coefficient(ring)

    lengths = np.empty((realisations, ps.size))
    clusters = np.empty((realisations, ps.size))
    for i, p in enumerate(ps):
        for r in range(realisations):
            g = newman_watts(NW_N, NW_K, p, rng=ctx.rng)
            lengths[r, i] = mean_distance(g, "largest_component")
            clusters[r, i] = clustering_coefficient(g)

    l_ratio = lengths / l0
    c_ratio = clusters / c0

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    l_mean = mean_with_band(ax, ps, l_ratio, color=SERIES[7])
    c_mean = mean_with_band(ax, ps, c_ratio, color=SERIES[5])
    reference_line(ax, 1.0, "")

    # Not at the end of its curve: the reference line is named at that corner,
    # and the two labels were printed on top of each other.  This one is put
    # where the curve has clear air above it, with a leader down to the point.
    c_at = int(0.72 * (ps.size - 1))
    ax.annotate(
        "$C(p)/C(0)$",
        xy=(ps[c_at], c_mean[c_at]),
        xytext=(-10, 26),
        textcoords="offset points",
        color=SERIES[5],
        fontsize=8.5,
        ha="right",
        va="bottom",
        arrowprops={"arrowstyle": "-", "color": SERIES[5], "lw": 0.7},
    )
    ax.annotate(
        r"$\ell(p)/\ell(0)$",
        xy=(ps[-1], l_mean[-1]),
        xytext=(-6, -5),
        textcoords="offset points",
        color=SERIES[7],
        fontsize=8.5,
        ha="right",
        va="top",
    )
    ax.annotate(
        "adding links can only\nmake more triangles",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        fontsize=7.5,
        color=INK["secondary"],
        ha="left",
        va="top",
    )

    ax.set_xscale("log")
    ax.set_ylim(0.64, 1.62)
    ax.set_xlabel("probability of connection $p$")
    ax.set_ylabel("value relative to the ordered ring")
    # No title: the caption says the clustering climbs and the distance falls.
    fig.suptitle(
        f"$N = {NW_N}$, $k = {NW_K}$, "
        f"mean of {realisations} networks per $p$; bands: interquartile range",
        fontsize=9,
        color=INK["secondary"],
        x=0.0,
        ha="left",
    )
    return fig


# ---------------------------------------------------------------------------
# 4.4.3 Caracteristicas -- the three completion rules, one figure each
# ---------------------------------------------------------------------------


def _option_figure(ctx: FigureContext, *, fallback: str, steps: int, label: str):
    """The layout the thesis uses for figs. 13, 14 and 15.

    Left: the grown network drawn at the coordinates the algorithm itself
    assigns each arriving node (``rand(-dim-z..dim+z)`` in the printed Maple),
    not at a layout invented afterwards.  Right: the degree of node 1, node 2,
    ... node N against the arrival index, which is the plot that makes
    preferential attachment visible as an ordering in time rather than as a
    distribution.  It is one value per node, so it is drawn as stems on a common
    baseline: the profile is the point, and the first arrival to take everything
    is picked out.
    """
    network = barabasi_albert(steps, links_per_new_node=2, fallback=fallback, rng=ctx.rng)
    positions = network.positions

    # Under "Opcion 1" an arriving node that fails the probability test is left
    # unconnected, so a growth run of `steps` steps yields fewer nodes than
    # steps: the thesis reports 179 nodes from 400 steps for exactly this
    # reason. Nodes that never received a link are not part of the network.
    keep = np.flatnonzero(network.degrees() > 0)
    network = network.subgraph(keep)
    positions = positions[keep]

    n = network.n
    k = network.degrees()
    # How long a link is, in the plane the algorithm draws its nodes in. It is
    # the one number that separates Opcion 3 from Opcion 2, which are otherwise
    # the same rule.
    lengths = np.array([np.linalg.norm(positions[i] - positions[j]) for i, j, _ in network.edges()])
    # As a fraction of the field the last arrival is placed in, so the number
    # means the same thing across figures built from different step counts.
    reach = float(np.median(lengths)) / (2.0 * steps)
    ctx.parameters(
        growth_steps=steps - 3,
        nodos_a_conectar=2,
        fallback=fallback,
        nodes_formed=int(n),
        k_max=int(k.max()),
        median_link_reach=round(reach, 3),
    )

    return _draw_option(network, positions, k, label)


def _draw_option(network, positions, k, label, *, node_size=None, note=None):
    """The two panels of figs. 13, 14 and 15: the network above, k node by node below."""
    k = np.asarray(k)
    fig, (left, right) = plt.subplots(
        2, 1, figsize=(5.05, 6.0), gridspec_kw={"height_ratios": [1.6, 1.0]}, layout="constrained"
    )
    draw_network(
        network,
        left,
        layout=positions,
        node_color=SERIES[0],
        node_size=node_size,
        scale_by_degree=node_size is None,
        edge_width=0.35,
    )
    # Just the name.  The caption already says what the panel shows, and the
    # longer form ran out of the panel and under the axis of the one beside it.
    left.set_title(label, fontsize=9.5)
    panel_letter(left, "a")
    panel_letter(right, "b")
    if note:
        left.text(
            0.5,
            -0.02,
            note,
            transform=left.transAxes,
            ha="center",
            va="top",
            fontsize=7,
            color=INK["muted"],
        )

    busiest = int(np.argmax(k))
    per_node(right, k, highlight=[busiest], mean_label=f"mean {k.mean():.1f}")
    right.annotate(
        f"node {busiest + 1}: $k = {k[busiest]}$",
        xy=(busiest + 1, k[busiest]),
        xytext=(10, -2),
        textcoords="offset points",
        fontsize=8,
        color=SERIES[1],
        ha="left",
        va="top",
    )
    right.set_xlabel("node, in order of arrival")
    right.set_ylabel("$k$")
    right.set_title("Degree by order of arrival", fontsize=9.5)
    right.set_ylim(0, k.max() * 1.12)
    right.yaxis.set_major_locator(MaxNLocator(integer=True))  # k counts links
    return fig


@figure(
    "T13",
    chapter=4,
    title="Scale free, Option 1: a few nodes take everything",
    source=Source(Origin.THESIS_2011, pdf_pages="30", section="4.4.3 Caracteristicas"),
    seed=THESIS_SEED,
    reproduces_figure="thesis figs. 13.a and 13.b, PDF p. 30",
)
def sf_option_1(ctx: FigureContext):
    """PDF p. 30, figs. 13.a-b: *Opcion 1 -- no se hace nada*.

    When one sweep of rejection sampling leaves the arriving node short of its
    two links, Opcion 1 does nothing about it, and the node joins the network
    only if the sweep gave it a link at all; 400 growth steps are run, as in
    the thesis, and the number of nodes formed is recorded.  The per-node plot
    is the evidence the section reads: a handful of early arrivals take the
    links, and the later ones settle onto a flat floor.
    """
    return _option_figure(ctx, fallback="none", steps=400, label="Option 1")


@figure(
    "T14",
    chapter=4,
    title="Scale free, Option 2: a continuous range of degrees",
    source=Source(Origin.THESIS_2011, pdf_pages="30", section="4.4.3 Caracteristicas"),
    seed=THESIS_SEED,
    reproduces_figure="thesis figs. 14.a and 14.b, PDF p. 30",
)
def sf_option_2(ctx: FigureContext):
    """PDF p. 30, figs. 14.a-b: *Opcion 2* -- finish at random.

    A node that fails the probability test is joined to uniformly random
    partners until it has its two links, so every one of the 200 nodes enters
    the network and the minimum degree really is 2.

    The section's reading of the panel is *el grado k esta distribuido a
    lo largo de la red ... existe un rango de valores de k entre los valores
    maximos y valores minimos que no se daba en la estructura anterior*.  The
    per-node profile decays instead of collapsing -- large and noisy for the
    first sixty arrivals, settling onto the floor of 2 near the end.  The
    largest degree here is the same 18 the printed panel reaches.
    """
    return _option_figure(ctx, fallback="random", steps=200, label="Option 2")


@figure(
    "T15",
    chapter=4,
    title="Scale free, Option 3: a thinner tail, made spatial",
    source=Source(Origin.THESIS_2011, pdf_pages="31", section="4.4.3 Caracteristicas"),
    seed=THESIS_SEED,
    reproduces_figure="thesis figs. 15.a and 15.b, PDF p. 31",
)
def sf_option_3(ctx: FigureContext):
    """PDF p. 31, figs. 15.a-b: *Opcion 3* -- finish with the nearest neighbour.

    Identical to Opcion 2 except that the make-up links go to the arriving
    node's nearest neighbours *in the plane* rather than to random nodes.  The
    thesis's comment is that the degree decays gradually, as in Opcion 2, and
    it does.

    The drawing carries one further consequence of the rule: the make-up
    links are short.  Measured in the plane the algorithm places
    its nodes in, the median link here reaches 6% of the width of the field
    against 17% under Opcion 2 -- the same network, built by the same
    preferential attachment, with a third of the reach.  Every link that is
    not short is a preferential one, which is why the long chords survive at
    all.  This is the option that turns a
    topological model into a geometric one, and it is why the three options are
    carried separately through the reproduction instead of being collapsed into
    one generator.
    """
    return _option_figure(ctx, fallback="nearest", steps=200, label="Option 3")


@figure(
    "T16",
    chapter=4,
    title="The scale-free degree distribution and its data collapse",
    source=Source(Origin.THESIS_2011, pdf_pages="31", section="4.4.3 Caracteristicas"),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 16, PDF p. 31 (Barabasi's published result)",
)
def sf_degree_distribution_collapse(ctx: FigureContext):
    """PDF p. 31, fig. 16: the published result the thesis measures itself against.

    Fig. 16 is not the author's own plot -- it is a scan of Barabasi's figure,
    reproduced so that the p(k) curves the thesis goes on to measure (figs. 17
    to 19) have something to be compared with.  So the counterpart here is that
    same figure remade from this repository's own generator: P(k) on log-log
    axes for several values of m, and the inset that made the original
    persuasive, P(k)/2m^2, on which every m falls onto one line.

    Two claims are being checked at once.  The main panel shows the exponent
    does not depend on m -- the curves are parallel.  The inset shows the
    prefactor is 2m^2, which is what the mean-field solution predicts:
    P(k) = 2m^2/k^3.  The dashed line is that prediction, not a fit.

    m is an ordered quantity, so it is carried by one sequential ramp rather
    than by four unrelated hues, with the marker shape as the second cue --
    the open squares and triangles the text names.

    The networks are of 300,000 nodes, the size the original was drawn at, one
    network per value of m; see :func:`_preferential_degrees` for how they are
    grown without a dense adjacency matrix.
    """
    n, realisations = 300_000, 1
    ms = (1, 2, 3, 5)
    ctx.parameters(N=n, m=list(ms), fallback="random", reference=r"P(k) = 2 m^2 / k^3")
    ctx.realisations(realisations)

    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    # Top right is the one corner a decaying power law leaves empty, and the
    # inset is opaque so nothing shows through it.
    inset = ax.inset_axes((0.66, 0.68, 0.33, 0.29))
    inset.set_facecolor(INK["surface"])
    inset.patch.set_alpha(1.0)
    markers = ("s", "o", "^", "D")
    ramp = (SEQUENTIAL[2], SEQUENTIAL[4], SEQUENTIAL[6], SEQUENTIAL[9])

    for i, m in enumerate(ms):
        counts = np.zeros(n + 1)
        for _ in range(realisations):
            k = _preferential_degrees(n, m, ctx.rng)
            counts += np.bincount(k, minlength=n + 1)
        pk = counts / (realisations * n)
        # A degree held by one or two nodes out of 300,000 is the estimator
        # talking, not the distribution: every such degree lands at the same
        # height, 1/N, and together they draw a flat carpet along the foot of
        # the axes that has nothing to do with the power law.  Five nodes is
        # the usual floor, and it is applied to every m alike.
        k = np.flatnonzero(counts >= 5 * realisations)
        ax.plot(
            k,
            pk[k],
            color=ramp[i],
            marker=markers[i],
            ms=3.6,
            lw=0.0,
            mfc="none",
            mew=0.8,
            label=f"$m = {m}$",
        )
        inset.plot(
            k,
            pk[k] / (2 * m**2),
            color=ramp[i],
            marker=markers[i],
            ms=2.2,
            lw=0.0,
            mfc="none",
            mew=0.8,
        )

    # The mean-field prediction, drawn over the range the data actually covers
    # rather than extrapolated into empty axes: one line for m = 2 in the main
    # panel, and the same line for every m at once in the collapsed inset.
    kk = np.logspace(0.0, np.log10(300.0), 80)
    ax.plot(
        kk,
        2 * 2**2 * kk**-3.0,
        color=INK["muted"],
        lw=1.1,
        ls=(0, (4, 3)),
        label=r"$2m^2k^{-3}$ at $m = 2$, the mean field",
    )
    inset.plot(kk, kk**-3.0, color=INK["muted"], lw=1.0, ls=(0, (4, 3)))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$k$")
    ax.set_ylabel("$P(k)$")
    ax.set_xlim(0.8, 400.0)
    ax.set_ylim(1.2e-5, 6.0)
    # No title: the caption carries the reading and the size.  The inset keeps
    # its own one-line title, which names what the inset does.
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, which="major", axis="both")

    inset.set_xscale("log")
    inset.set_yscale("log")
    inset.set_ylim(2e-7, 5.0)
    inset.set_yticks([1e-6, 1e-3, 1.0])
    inset.set_xlabel("$k$", fontsize=7.5, labelpad=1)
    inset.set_ylabel("$P(k)/2m^2$", fontsize=7.5, labelpad=1)
    inset.tick_params(labelsize=7.5)
    inset.grid(False)
    inset.set_title("divided by $2m^2$, every $m$ is one line", fontsize=7.5, pad=3, fontweight="normal")

    # The inset's own patch only fills its plot rectangle, not the margin
    # where its tick labels, axis labels and title sit -- so cover that
    # margin too, sized to what actually rendered rather than a guessed
    # padding, or a main-panel marker can still show through a tick label.
    fig.canvas.draw()
    pad = 0.015
    bbox = inset.get_tightbbox(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
    ax.add_patch(
        plt.Rectangle(
            (bbox.x0 - pad, bbox.y0 - pad),
            bbox.width + 2 * pad,
            bbox.height + 2 * pad,
            transform=ax.transAxes,
            facecolor=INK["surface"],
            edgecolor="none",
            zorder=2.9,
        )
    )
    return fig
