"""Chapter 7, figure for figure: the immunisation section of the thesis.

Thesis section 5.5, *Inmunidad frente a infecciones* (PDF pp. 54-63).  Where
:mod:`netmotion.experiments.chapter_07` distils that section into five summary
figures, this module reproduces its **fifteen printed figures one for one**, so
that the book and the thesis can be read side by side.  Each identifier names
the thesis figure it stands for: ``T54`` is Fig. 54, ``T63`` is Fig. 63.

The section falls into two halves.

**5.5.1, which node is the most vulnerable** (Figs. 54-57).  A nine-node tree
borrowed from Estrada and Hatano is worked through by hand: the degree names
nodes 4 and 6, the thermal-bath method names node 5, and the thesis's own two
methods -- peeling the minimum-degree nodes away, and the degree :math:`K_m(n)`
of order *n* -- agree with thermal bath.  Figs. 55 and 56 are the two rounds of
the peeling; Fig. 57 is the colour-coded calculation of :math:`K_m(2)`.

**5.5.2, immunisation** (Figs. 58-68).  The nodes are ranked by :math:`K(n)`,
the top-ranked ones are frozen at :math:`S = +1`, and two things are measured
as the immune set grows: how large the avalanche still is (Figs. 58-60) and how
fast it still travels (Figs. 61-62), the latter through the decay factor
:math:`\\beta` of :math:`\\langle\\alpha\\rangle = B e^{\\beta\\,cn}` (Fig. 63).
Figs. 64-67 repeat the whole sweep for four network families at three sizes,
and Fig. 68 repeats it for three *ordered* networks, where the thesis's answer
changes from :math:`K(1)` to :math:`K(3)`.

Nothing here re-implements a model or a metric: the peeling, the higher-order
degree, the infection and the immunisation sweep all live in the package and
are only driven from here.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from ..graph import Network
from ..metrics.higher_order_degree import k_order_all
from ..metrics.propagation_centrality import (
    estrada_hatano_network,
    peel_by_minimum_degree,
)
from ..networks import barabasi_albert, erdos_renyi, newman_watts, watts_strogatz
from ..propagation import propagate
from ..propagation.immunity import (
    beta_fit_range,
    compare_orders,
    immunisation_ranking,
    immunisation_sweep,
)
from ..provenance import Origin, Source
from ..visualization import INK, SEQUENTIAL, SERIES, draw_network
from ..visualization.charts import panel_letter
from . import _thesis
from .registry import FigureContext, figure

THESIS_SEED = 20110930

#: Section 5.5.2 sweeps the order n of K(n) from 1 to 8.
ORDERS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)

#: The three sizes of Figs. 64-67.
SIZES: tuple[int, ...] = (50, 75, 100)

# The thesis colour-codes Figs. 55-57 in red, blue and green.  Those hues are
# not kept: what the colours actually stand for is *how far from the node under
# study* something is, which is an ordered quantity, so it is drawn with the
# sequential ramp -- dark near, pale far -- and the accent is spent on the one
# node each panel is about.
_SUBJECT = SERIES[7]  # red -- the node under study
_HOP1 = SERIES[0]  # blue -- what it reaches in one step
_HOP2 = SERIES[5]  # green -- what it reaches in two
_OFF = INK["grid"]  # the part of the network the count leaves out
_GREY = INK["muted"]

#: The house ramp as a continuous colormap, for the occasions when the ordered
#: quantity has a hundred values rather than eight (Fig. 60).
_RAMP = LinearSegmentedColormap.from_list("netmotion_sequential", list(SEQUENTIAL))
#: Fig. 60 is a family of red curves, and the immune count darkens them.
_RED_RAMP = LinearSegmentedColormap.from_list(
    "netmotion_red", ["#f6b6b5", "#e34948", "#a51f1e", "#5e0f0e"]
)

#: Node coordinates of Fig. 54, read off the printed drawing (PDF p. 54): two
#: three-leaf stars joined through the bridge node 5.  Thesis node k is index
#: k-1, as everywhere in :mod:`netmotion`.
FIG54_POSITIONS = np.array(
    [
        [-2.0, 1.0],  # 1
        [-2.0, 0.0],  # 2
        [-2.0, -1.0],  # 3
        [-1.0, 0.0],  # 4
        [0.0, 0.0],  # 5
        [1.0, 0.0],  # 6
        [2.0, 1.0],  # 7
        [2.0, 0.0],  # 8
        [2.0, -1.0],  # 9
    ]
)


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------


def _draw_fig54(
    ax,
    node_colours,
    edge_colours: dict[tuple[int, int], str] | None = None,
    *,
    label_colours=None,
):
    """Draw the Fig. 54 network at its printed geometry, numbered 1..9."""
    g = estrada_hatano_network()
    draw_network(
        g,
        ax,
        layout=FIG54_POSITIONS,
        node_color=np.asarray(node_colours),
        node_size=95.0,
        edge_color=INK["grid"],
        edge_width=1.1,
    )
    if edge_colours:
        segments, colours = [], []
        for (i, j), colour in edge_colours.items():
            segments.append((FIG54_POSITIONS[i], FIG54_POSITIONS[j]))
            colours.append(colour)
        ax.add_collection(LineCollection(segments, colors=colours, linewidths=1.8, zorder=1.5))
    for i, (x, y) in enumerate(FIG54_POSITIONS):
        colour = _GREY if label_colours is None else label_colours[i]
        ax.annotate(
            str(i + 1),
            xy=(x, y),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=colour,
        )
    ax.margins(0.16)
    return g


def _order_colours(orders) -> list[str]:
    """The sequential ramp across the order *n*, dark at n = 1.

    The eight series of Figs. 58 and 62 are one ordered quantity, not eight
    categories, so they take a single-hue ramp rather than eight arbitrary
    colours.  It runs dark to pale because the low orders are the ones the
    section concludes for, and the ramp then reads directly as "the further the
    criterion looks, the paler it gets".
    """
    ramp = list(SEQUENTIAL)
    positions = np.linspace(len(ramp) - 1, 2, len(list(orders)))
    return [ramp[int(round(p))] for p in positions]


def _ramp_key(ax, colours, low: str, high: str, *, x=0.55, y=0.90, width=0.30) -> None:
    """A row of swatches naming only the two ends of a ramp.

    A ramp of eight needs no eight-entry legend: the swatches show the order,
    and only the ends have to be named.
    """
    height = 0.035
    step = width / len(colours)
    for i, colour in enumerate(colours):
        ax.add_patch(
            Rectangle(
                (x + i * step, y),
                step,
                height,
                transform=ax.transAxes,
                facecolor=colour,
                edgecolor="none",
                zorder=5,
                clip_on=False,
            )
        )
    for label, at, ha, dx in ((low, x, "right", -5), (high, x + width, "left", 5)):
        ax.annotate(
            label,
            xy=(at, y + height / 2),
            xycoords="axes fraction",
            xytext=(dx, 0),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=7.5,
            color=INK["secondary"],
            annotation_clip=False,
        )


def _mark_best(ax, x, values, *, formatter="$K({})$", below: bool = False) -> int:
    """Ring the lowest point of a curve.

    The ring marks where the curve is lowest; which order that is, and what it
    means, the text says.
    """
    values = np.asarray(values, dtype=float)
    best = int(np.nanargmin(values))
    ax.plot(
        x[best],
        values[best],
        "o",
        ms=9,
        mfc="none",
        mec=SERIES[1],
        mew=1.5,
        zorder=5,
        clip_on=False,
    )
    return best


def _beta_or_nan(sweep) -> float:
    try:
        return sweep.beta
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# 5.5.1  Analisis del nodo de maxima vulnerabilidad
# ---------------------------------------------------------------------------


@figure(
    "T54",
    chapter=7,
    title="The network the vulnerability analysis is done on",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="54",
        section="5.5.1 Analisis del nodo de maxima vulnerabilidad",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 54, PDF p. 54",
)
def vulnerability_network(ctx: FigureContext):
    """PDF p. 54, fig. 54: nine nodes, eight links, and two rival answers.

    The thesis draws this network -- Estrada and Hatano's -- because the two
    ways of naming its most vulnerable node disagree.  Barabasi's criterion is
    the degree, and picks the two hubs, nodes 4 and 6, each of degree four.
    The thermal-bath method picks node 5 instead, the bridge of degree two,
    on the grounds that removing it is what actually breaks the network apart.

    The drawing is the thesis's own: two three-leaf stars facing each other
    across the bridge.  Nodes are marked here by which criterion names them,
    which is the disagreement the rest of section 5.5.1 sets out to settle.
    """
    g = estrada_hatano_network()
    degrees = g.degrees()
    ctx.parameters(
        N=g.n,
        E=g.n_edges,
        degrees=degrees.tolist(),
        barabasi_answer="nodes 4 and 6 (largest k)",
        thermal_bath_answer="node 5 (the bridge)",
    )

    hubs = {3, 5}
    bridge = 4
    colours = np.array(
        [SERIES[0] if i in hubs else (_SUBJECT if i == bridge else _GREY) for i in range(g.n)]
    )
    edges = {(i, j): INK["muted"] for i, j, _ in g.edges()}

    fig, ax = plt.subplots(figsize=(4.9, 2.35))
    _draw_fig54(ax, colours, edges, label_colours=[INK["secondary"]] * g.n)

    for hub_x in (-1.0, 1.0):  # nodes 4 and 6, the two hubs of degree 4
        ax.annotate(
            "",
            xy=(hub_x, -0.22),
            xytext=(0.0, -1.3),
            arrowprops={"arrowstyle": "-", "lw": 0.7, "color": SERIES[0], "shrinkA": 0},
        )
    ax.text(
        0.0,
        -1.35,
        "largest $k$: Barab\u00e1si\u2019s answer",
        ha="center",
        va="top",
        fontsize=8,
        color=SERIES[0],
    )
    ax.annotate(
        "thermal bath\u2019s answer",
        xy=(0.0, 0.47),
        xytext=(0.0, 1.35),
        ha="center",
        va="bottom",
        fontsize=8,
        color=_SUBJECT,
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": _SUBJECT},
    )

    return fig


@figure(
    "T55",
    chapter=7,
    title="Node removal, first round: the leaves go",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="54-55",
        section="5.5.1.1 Metodo por elminacion de nodos",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 55, PDF p. 54",
)
def peeling_round_one(ctx: FigureContext):
    """PDF p. 54, fig. 55: strip every node of minimum degree, all at once.

    The recipe of section 5.5.1.1 is three lines long: compute every degree,
    delete every node attaining the minimum, repeat.  On this network the
    minimum degree is 1, so the six leaves go together and *"queda la red
    restante formada por los nodos 4, 5, 6"*.

    Greyed nodes and links are the ones the round removed; what is left is
    drawn in the live colours.
    """
    g = estrada_hatano_network()
    result = peel_by_minimum_degree(g)
    removed = set(int(v) for v in result.layers[0])
    survivors = [i for i in range(g.n) if i not in removed]
    ctx.parameters(
        N=g.n,
        min_degree_removed=1,
        removed=[v + 1 for v in sorted(removed)],
        remaining=[v + 1 for v in survivors],
    )

    colours = np.array([_GREY if i in removed else _SUBJECT for i in range(g.n)])
    edges = {(i, j): (_OFF if i in removed or j in removed else _SUBJECT) for i, j, _ in g.edges()}

    fig, ax = plt.subplots(figsize=(4.9, 1.95))
    _draw_fig54(
        ax,
        colours,
        edges,
        label_colours=[_GREY if i in removed else INK["secondary"] for i in range(g.n)],
    )
    return fig


@figure(
    "T56",
    chapter=7,
    title="Node removal, second round: only node 5 survives",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="55",
        section="5.5.1.1 Metodo por elminacion de nodos",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 56, PDF p. 55",
)
def peeling_round_two(ctx: FigureContext):
    """PDF p. 55, fig. 56: the survivor is the central node.

    On the path 4-5-6 that round one left behind, the minimum degree is again
    1, so nodes 4 and 6 go and node 5 is left alone.  *"Nos queda por ultimo el
    nodo 5 que coincide con el nodo mas vulnerable o nodo central obtenido por
    thermal bath"* -- which is the whole point of the method: a purely local,
    repeated deletion rule arrives at the answer the degree could not give.
    """
    g = estrada_hatano_network()
    result = peel_by_minimum_degree(g)
    survivor = int(result.survivors[0])
    ctx.parameters(
        N=g.n,
        rounds=result.n_rounds,
        layers=[[v + 1 for v in layer] for layer in result.as_lists()],
        survivor=survivor + 1,
        terminated=result.terminated,
    )

    colours = np.array([_SUBJECT if i == survivor else _GREY for i in range(g.n)])
    edges = {(i, j): _OFF for i, j, _ in g.edges()}

    fig, ax = plt.subplots(figsize=(4.9, 1.95))
    _draw_fig54(
        ax,
        colours,
        edges,
        label_colours=[INK["secondary"] if i == survivor else _GREY for i in range(g.n)],
    )
    return fig


@figure(
    "T57",
    chapter=7,
    title="How the degree k of order 2 is counted",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="55-56",
        section="5.5.1.3 El metodo del grado k de orden n del nodo m",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 57.a-c, PDF pp. 55-56",
)
def k_order_two_by_colour(ctx: FigureContext):
    """PDF pp. 55-56, fig. 57.a-c: the same network, three nodes, one rule.

    :math:`K_m(2)` counts *links*, not nodes: the links hanging off *m* plus
    the links hanging off *m*'s neighbours.  The thesis draws it as a colouring
    and then simply adds up the coloured links.  What the colours stand for is
    a distance in hops, which is ordered, so they are two steps of the one
    sequential ramp -- dark for the first hop, pale for the second -- with the
    accent reserved for *m* itself and the uncounted remainder left grey.

    The three panels are the thesis's own choice of nodes, and they are chosen
    to make the point.  Node 4 has the largest degree, :math:`K_4(1) = 4`, but
    reaches only one more link at order 2.  Node 5 has :math:`K_5(1) = 2` and
    reaches the *entire* network at order 2.  Looking one step further turns
    the ranking over.
    """
    g = estrada_hatano_network()
    nodes = (0, 3, 4)  # the thesis's nodes 1, 4 and 5
    profiles = k_order_all(g, 2)
    values = {m: int(profiles[m, 1]) for m in nodes}
    ctx.parameters(
        N=g.n,
        E=g.n_edges,
        nodes=[m + 1 for m in nodes],
        K_order_2={f"K_{m + 1}(2)": values[m] for m in nodes},
        K_order_1={f"K_{m + 1}(1)": int(profiles[m, 0]) for m in nodes},
    )

    adjacency = g.adjacency
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3))

    for ax, m in zip(axes, nodes, strict=True):
        first = {int(v) for v in np.flatnonzero(adjacency[m])}
        second = set()
        for v in first:
            second |= {int(w) for w in np.flatnonzero(adjacency[v])}
        second -= first | {m}

        colours, labels = [], []
        for i in range(g.n):
            if i == m:
                colours.append(_SUBJECT)
            elif i in first:
                colours.append(_HOP1)
            elif i in second:
                colours.append(_HOP2)
            else:
                colours.append(_GREY)
            labels.append(_GREY if colours[-1] == _GREY else INK["secondary"])

        edges = {}
        for i, j, _ in g.edges():
            if m in (i, j):
                edges[(i, j)] = _HOP1
            elif i in first or j in first:
                edges[(i, j)] = _HOP2
            else:
                edges[(i, j)] = _OFF

        _draw_fig54(ax, np.array(colours), edges, label_colours=labels)
        ax.set_title(f"node {m + 1}", fontsize=8.0, loc="center")
        panel_letter(ax, "abc"[nodes.index(m)])

    return fig


# ---------------------------------------------------------------------------
# 5.5.2  Inmunizacion frente a infeccion -- the scale-free worked case
# ---------------------------------------------------------------------------


def _thesis_scale_free(ctx: FigureContext, n: int = 100) -> Network:
    """The 100-node scale-free network section 5.5.2 works its example on."""
    return barabasi_albert(n, links_per_new_node=2, rng=ctx.rng)


@figure(
    "T58",
    chapter=7,
    title="How the avalanche shrinks, for every order of K(n)",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="57",
        section="5.5.2 Inmunizacion frente a infeccion",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 58, PDF p. 57",
)
def infected_versus_immunised(ctx: FigureContext):
    """PDF p. 57, fig. 58: infected nodes against immune nodes, eight criteria.

    The infection always starts at node 1, and the immune set grows one node at
    a time down the ranking that :math:`K(n)` gives.  Each of the eight series
    is one choice of *n*.

    Two features of the printed figure survive reproduction.  All eight curves
    fall from nearly the whole network to nothing, so *any* degree-like
    criterion works; and they do not fall together -- the low orders reach zero
    first, which is the observation Fig. 59 turns into a number.
    """
    n = 100
    counts = np.arange(0, n)
    ctx.parameters(
        N=n,
        model="scale-free (Barabasi-Albert, m=2)",
        seed_node="node 1",
        orders=list(ORDERS),
        immune_counts=f"0..{n - 1}",
    )
    g = _thesis_scale_free(ctx, n)
    ctx.realisations(1)

    colours = _order_colours(ORDERS)
    curves = {}

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for i, order in enumerate(ORDERS):
        sweep = immunisation_sweep(
            g, "k_order_n", order=order, start_nodes=[0], counts=counts, rng=ctx.rng
        )
        curves[order] = np.asarray(sweep.infected, dtype=float)
        ax.plot(
            sweep.counts,
            sweep.infected,
            color=colours[i],
            lw=1.5,
            ls=(0, (1.6, 1.6)),
            zorder=2 + i,
        )

    # Name the two ends of the ramp where they are furthest apart, which is
    # also the place the figure is read: how many immune nodes each criterion
    # has spent to get the infection down.
    gap = int(np.nanargmax(curves[ORDERS[-1]] - curves[ORDERS[0]]))
    for order, dx, dy, ha in ((ORDERS[0], -62, -4, "left"), (ORDERS[-1], 10, 14, "left")):
        ax.annotate(
            f"$K({order})$",
            xy=(counts[gap], curves[order][gap]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=8.5,
            color=colours[0] if order == ORDERS[0] else INK["secondary"],
            bbox={"facecolor": INK["surface"], "edgecolor": "none", "pad": 1.5},
            arrowprops={"arrowstyle": "-", "color": INK["muted"], "lw": 0.7},
        )

    ax.set_xlabel("immune nodes")
    ax.set_ylabel("nodes still infected")
    ax.set_xticks(np.arange(0, n + 1, 10))
    ax.set_ylim(bottom=0)
    _ramp_key(ax, colours, "$K(1)$", "$K(8)$", x=0.62, y=0.90, width=0.26)
    return fig


@figure(
    "T59",
    chapter=7,
    title="How many nodes each criterion has to immunise",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="57",
        section="5.5.2 Inmunizacion frente a infeccion",
    ),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 59, PDF p. 57",
)
def nodes_to_zero_by_order(ctx: FigureContext):
    """PDF p. 57, fig. 59: the immune nodes needed to reduce the infection to zero.

    Fig. 58 is read once per criterion and reduced to a single number: the
    point at which the curve hits zero, averaged over every possible seed
    rather than over node 1 alone.  Plotted against the order *n*, that number
    is the thesis's verdict on which :math:`K(n)` to immunise by -- and its
    answer is the lowest point of the curve, :math:`K(1)`, plain degree.

    The printed curve rises from about 53 nodes at :math:`K(1)` to a maximum
    near :math:`K(4)` and then flattens, which is what higher orders do: past
    the diameter of the network every node's :math:`K_m(n)` counts every link,
    the score stops discriminating, and the ranking stops improving.
    """
    n = 100
    counts = np.arange(0, n)
    ctx.parameters(
        N=n,
        model="scale-free (Barabasi-Albert, m=2)",
        orders=list(ORDERS),
        seeds="every node",
        immune_counts=f"0..{n - 1}",
        extinction="final infected count of 1 (the infection never left the seed)",
    )
    g = _thesis_scale_free(ctx, n)
    ctx.realisations(n)

    sweeps = compare_orders(g, list(ORDERS), counts=counts, rng=ctx.rng)
    needed = np.array([sweeps[o].mean_nodes_to_extinction for o in ORDERS])

    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    ax.plot(ORDERS, needed, color=SERIES[0], lw=1.8, marker="o", ms=5, zorder=3)
    _mark_best(ax, np.asarray(ORDERS), needed, below=True)
    ax.set_xlabel("order $n$ of the $K(n)$ used to rank the nodes")
    ax.set_ylabel("immune nodes needed")
    ax.set_xticks(list(ORDERS))
    ax.margins(y=0.32)

    return fig


@figure(
    "T60",
    chapter=7,
    title="One propagation curve per number of immune nodes",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="58",
        section="5.5.2 Inmunizacion frente a infeccion",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 60, PDF p. 58",
)
def propagation_curves_under_immunity(ctx: FigureContext):
    """PDF p. 58, fig. 60: the whole family of avalanches, drawn on top of each other.

    One curve per number of immune nodes, all seeded at node 1, all immunising
    down the :math:`K(1)` ranking.  With nobody immune the infection saturates
    the network within five or six steps; with enough immune nodes it barely
    leaves the seed.

    The figure is worth more than the summary numbers that follow it, because
    it shows *two* effects at once.  The curves do not merely end lower as
    immunity grows -- they also rise more slowly, and it is that slope, not the
    final height, that Figs. 61-63 measure.
    """
    n = 100
    ctx.parameters(
        N=n,
        model="scale-free (Barabasi-Albert, m=2)",
        criterion="K(1)",
        seed_node="node 1",
        curves=f"one per immune count 0..{n - 1}",
    )
    g = _thesis_scale_free(ctx, n)
    ctx.realisations(1)

    ranking = immunisation_ranking(g, "k_order_n", order=1, rng=ctx.rng)
    ranking = ranking[ranking != 0]

    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    longest = 1
    for count in range(n):
        run = propagate(g, 0, immune=ranking[:count])
        longest = max(longest, run.curve.size)
        # The immune count is an ordered quantity, so it is what the ramp
        # carries: the curves darken as the immune set grows, and the family
        # sinks into the dark instead of being a hundred identical red lines.
        ax.plot(
            np.arange(1, run.curve.size + 1),
            run.curve,
            color=_RED_RAMP(0.15 + 0.85 * count / (n - 1)),
            lw=1.0,
            alpha=0.75,
            zorder=2 + count / n,
        )

    ax.set_xlabel("$t$")
    ax.set_ylabel("fraction infected, $p(t)$")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(1, longest + 1))
    _ramp_key(
        ax,
        [_RED_RAMP(t) for t in np.linspace(0.15, 1.0, 12)],
        "nobody immune",
        f"{n - 1} immune",
        x=0.60,
        y=0.05,
        width=0.24,
    )
    return fig


def _order_sweeps(ctx: FigureContext, n: int = 100):
    """The immunisation sweep of section 5.5.2 for every order, on the network of fig. 58."""
    g = _thesis_scale_free(ctx, n)
    return compare_orders(g, list(ORDERS), counts=np.arange(0, n), rng=ctx.rng)


def _decay_fit(sweep) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The points <alpha> against immune nodes, and the fit B e^(beta cn) through them.

    The fit is the one of :func:`beta_factor`, over the counts where <alpha> is
    at least BETA_FIT_FLOOR, and is drawn over those counts only.
    """
    usable = np.isfinite(sweep.speed) & (sweep.speed > 0) & (sweep.counts >= 1)
    fitted_on = beta_fit_range(sweep.counts, sweep.speed)
    beta, log_b = np.polyfit(sweep.counts[fitted_on], np.log(sweep.speed[fitted_on]), 1)
    grid = np.linspace(1, sweep.counts[fitted_on].max(), 200)
    return usable, grid, np.exp(log_b + beta * grid)


@figure(
    "T61",
    chapter=7,
    title="Propagation speed falls as immunity grows",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="58",
        section="5.5.2 Inmunizacion frente a infeccion",
    ),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 61, PDF p. 58",
)
def alpha_versus_immunised(ctx: FigureContext):
    """PDF p. 58, fig. 61: <alpha> against the number of immunised nodes, K(1).

    Every curve of fig. 60 is reduced to its factor alpha, and averaging over
    every seed node gives one <alpha> per number of immune nodes.  The decay is
    fitted to B e^(beta cn), drawn dashed; its beta is recorded, with <alpha>
    at ten immune nodes, the value the text reads off the figure.
    """
    n = 100
    sweep = immunisation_sweep(
        _thesis_scale_free(ctx, n), "k_order_n", order=1, counts=np.arange(0, n), rng=ctx.rng
    )
    usable, grid, fitted = _decay_fit(sweep)
    at_ten = float(sweep.speed[list(sweep.counts).index(10)])
    ctx.parameters(
        N=n, model="scale free (m=2)", criterion="K(1)", seeds="every node",
        beta=round(float(sweep.beta), 4), alpha_at_10=round(at_ten, 2),
    )
    fig, ax = plt.subplots(figsize=(5.0, 3.2), layout="constrained")
    ax.plot(sweep.counts[usable], sweep.speed[usable], ls="none", marker="o", ms=3.2,
            color=SERIES[0], zorder=3)
    ax.plot(grid, fitted, color=SERIES[1], lw=1.5, ls=(0, (5, 2)), zorder=4)
    ax.annotate(
        rf"$B e^{{\beta\, cn}}$, $\beta = {sweep.beta:.3f}$",
        xy=(grid[40], fitted[40]), xytext=(10, 12), textcoords="offset points",
        fontsize=8.5, color=SERIES[1],
    )
    ax.set_xlabel("immune nodes")
    ax.set_ylabel(r"$\langle \alpha \rangle$")
    ax.set_xlim(0, n + 2)
    ax.set_xticks(np.arange(0, n + 1, 10))
    return fig


@figure(
    "T62",
    chapter=7,
    title="The speed decay, for all eight orders",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="59",
        section="5.5.2 Inmunizacion frente a infeccion",
    ),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 62, PDF p. 59",
)
def alpha_decay_by_order(ctx: FigureContext):
    """PDF p. 59, fig. 62: <alpha> against the immune count, for K(1) ... K(8).

    The measurement of fig. 61 repeated for every order.  Eight clouds of
    points drawn over one another make one cloud, so each order has a panel of
    its own on shared axes, with its fit B e^(beta cn) (the beta of fig. 63),
    named in a key above the panels.
    """
    sweeps = _order_sweeps(ctx)
    ctx.parameters(N=100, orders=list(ORDERS), seeds="every node")
    fig, axes = plt.subplots(2, 4, figsize=(5.05, 3.4), sharex=True, sharey=True,
                             layout="constrained")
    for ax, order in zip(axes.flat, ORDERS, strict=True):
        sweep = sweeps[order]
        usable, grid_o, fitted = _decay_fit(sweep)
        ax.plot(sweep.counts[usable], sweep.speed[usable], ls="none", marker="o", ms=1.6,
                color=SERIES[0], zorder=2)
        ax.plot(grid_o, fitted, color=SERIES[1], lw=1.1, zorder=3, label=r"fit $B e^{\beta\, cn}$")
        ax.set_title(f"$K({order})$", fontsize=9)
        ax.set_xticks((0, 50, 100))
    fig.supxlabel("immune nodes", fontsize=9, color=INK["secondary"])
    fig.supylabel(r"$\langle \alpha \rangle$", fontsize=9, color=INK["secondary"])
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside upper center",
               frameon=False, fontsize=8)
    return fig


@figure(
    "T63",
    chapter=7,
    title="The decay factor beta, order by order",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="59",
        section="5.5.2 Inmunizacion frente a infeccion",
    ),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 63, PDF p. 59",
)
def beta_by_order(ctx: FigureContext):
    """PDF p. 59, fig. 63: beta for each order n of the K(n) used to immunise.

    The beta of each panel of fig. 62.
    """
    sweeps = _order_sweeps(ctx)
    betas = np.array([_beta_or_nan(sweeps[o]) for o in ORDERS])
    ctx.parameters(N=100, orders=list(ORDERS), beta=[round(float(b), 4) for b in betas])
    fig, ax = plt.subplots(figsize=(5.0, 3.2), layout="constrained")
    ax.plot(ORDERS, betas, color=SERIES[0], lw=1.8, marker="o", ms=5, zorder=3)
    ax.set_xlabel("order $n$ of the $K(n)$ used to rank the nodes")
    ax.set_ylabel(r"decay factor $\beta$")
    ax.set_xticks(list(ORDERS))
    ax.margins(y=0.22)
    return fig


# ---------------------------------------------------------------------------
# 5.5.2  Figs. 64-67: the same sweep across families and sizes
# ---------------------------------------------------------------------------


def _family_table(ctx: FigureContext, builders: dict[int, object], family: str, label: str,
                  rngs: dict[int, np.random.Generator] | None = None):
    """The 2 x 3 table of Figs. 64-67: rows are quantities, columns are N.

    Six panels of the same two quantities need two axis labels and one set of
    tick labels, not twelve of each: the columns share the order axis and are
    named once along the top, the rows are named once down the left, and every
    panel is read for the single number it is drawn for -- the order that wins,
    ringed on the curve.
    """
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 2.4), sharex=True)
    top_row: list[int] = []
    bottom_row: list[int] = []

    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for n in SIZES:
        g = builders[n]()
        rng = (rngs or {}).get(n, ctx.rng)
        sweeps = compare_orders(g, list(ORDERS), counts=np.arange(0, n), rng=rng)
        needed = np.array([sweeps[o].mean_nodes_to_extinction for o in ORDERS])
        betas = np.array([_beta_or_nan(sweeps[o]) for o in ORDERS])
        results[n] = (needed, betas)

    orders = np.asarray(ORDERS)
    for column, n in enumerate(SIZES):
        needed, betas = results[n]

        ax = axes[0, column]
        ax.plot(ORDERS, needed, color=SERIES[0], lw=1.8, marker="o", ms=4, zorder=3)
        top_row.append(ORDERS[_mark_best(ax, orders, needed, below=True)])
        ax.set_title(f"$N = {n}$", fontsize=9.5, loc="center")
        panel_letter(ax, "abc"[column])
        ax.margins(y=0.34)

        ax = axes[1, column]
        ax.plot(ORDERS, betas, color=SERIES[0], lw=1.8, marker="o", ms=4, zorder=3)
        bottom_row.append(ORDERS[_mark_best(ax, orders, betas, below=True)])
        panel_letter(ax, "def"[column])
        ax.margins(y=0.34)
        ax.set_xticks(list(ORDERS))

    axes[0, 0].set_ylabel("immune nodes\nto extinction")
    axes[1, 0].set_ylabel("decay factor\n" + r"$\beta$")
    fig.supxlabel(
        "order $n$ of the $K(n)$ used to rank the nodes", fontsize=9.5, color=INK["secondary"]
    )

    ctx.parameters(
        family=family, sizes=list(SIZES), orders=list(ORDERS),
        best_by_nodes_needed=[int(o) for o in top_row],
        best_by_beta=[int(o) for o in bottom_row],
        **{f"nodes_needed_N{n}": [round(float(v), 2) for v in results[n][0]] for n in SIZES},
        **{f"beta_N{n}": [round(float(v), 4) for v in results[n][1]] for n in SIZES},
        **{
            f"correlation_N{n}": round(float(np.corrcoef(results[n][0], results[n][1])[0, 1]), 2)
            for n in SIZES
        },
    )
    return fig


@figure(
    "T64",
    chapter=7,
    title="Immunisation on Erdos-Renyi networks, at three sizes",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="60",
        section="5.5.2 Inmunizacion frente a infeccion",
    ),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 64.a-f, PDF p. 60",
)
def er_table(ctx: FigureContext):
    """PDF p. 60, fig. 64: the whole sweep repeated on ER networks at N = 50, 75, 100.

    The thesis draws this as a ruled table -- three columns, one per size, and
    two rows, one per measure -- and the grid is kept here because it is what
    makes the reading possible: the shape of a row should not change as the
    column changes, and if it does, the conclusion depends on the size of the
    network rather than on its kind.

    For the random graph it does not change.  Both rows have the same shape at
    all three sizes, which is the section's argument that :math:`K(1)` does not
    depend on any one particular network.
    """
    ctx.parameters(
        model="Erdos-Renyi",
        mean_degree=4,
        p="4 / (N - 1)",
        seeds="every node",
        immune_counts="0..N-1",
    )
    builders = {n: (lambda n=n: erdos_renyi(n, 4.0 / (n - 1), rng=ctx.rng)) for n in SIZES}
    return _family_table(ctx, builders, "erdos_renyi", "Erdos-Renyi")


@figure(
    "T65",
    chapter=7,
    title="Immunisation on Watts-Strogatz networks, at three sizes",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="60-61",
        section="5.5.2 Inmunizacion frente a infeccion",
    ),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 65.a-f, PDF pp. 60-61",
)
def ws_table(ctx: FigureContext):
    """PDF pp. 60-61, fig. 65: the same table for the rewired ring.

    Fig. 65 straddles a page break in the 2011 PDF: the three size panels sit at
    the foot of p. 60 and the three :math:`\\beta` panels at the head of p. 61.
    It is one figure, and is drawn here as one.

    The small-world network is the interesting middle case.  Its degrees are
    nearly uniform -- rewiring a ring at *p* = 0.1 moves a few links but leaves
    every node with roughly *k* neighbours -- so degree has much less to rank
    with here than in a scale-free network, and the advantage of :math:`K(1)`
    is correspondingly thinner.
    """
    ctx.parameters(
        model="Watts-Strogatz",
        k=4,
        p=0.1,
        seeds="every node",
        immune_counts="0..N-1",
    )
    builders = {n: (lambda n=n: watts_strogatz(n, 4, 0.1, rng=ctx.rng)) for n in SIZES}
    return _family_table(ctx, builders, "watts_strogatz", "Watts-Strogatz")


@figure(
    "T66",
    chapter=7,
    title="Immunisation on Newman-Watts networks, at three sizes",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="61",
        section="5.5.2 Inmunizacion frente a infeccion",
    ),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 66.a-f, PDF p. 61",
)
def nw_table(ctx: FigureContext):
    """PDF p. 61, fig. 66: the same table for the shortcut model.

    Newman-Watts adds shortcuts instead of moving links, so the ring is never
    broken and the network cannot fall apart.  That shows in the top row: the
    number of immune nodes needed is the highest of any family, because there
    is no hub whose removal disconnects anything -- immunity has to be bought
    node by node.
    """
    ctx.parameters(
        model="Newman-Watts",
        k=4,
        p=0.1,
        seeds="every node",
        immune_counts="0..N-1",
    )
    builders = {n: (lambda n=n: newman_watts(n, 4, 0.1, rng=ctx.rng)) for n in SIZES}
    return _family_table(ctx, builders, "newman_watts", "Newman-Watts")


@figure(
    "T67",
    chapter=7,
    title="Immunisation on scale-free networks, at three sizes",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="62",
        section="5.5.2 Inmunizacion frente a infeccion",
    ),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 67.a-f, PDF p. 62",
)
def scale_free_table(ctx: FigureContext):
    """PDF p. 62, fig. 67: the same table for the grown network.

    This is the family selective immunisation was invented for.  A scale-free
    network has hubs, the hubs are what the infection travels through, and
    protecting them buys more than protecting anything else -- which is why the
    top row here sits lower than in any other family: fewer immune nodes are
    needed for the same result.
    """
    ctx.parameters(
        model="scale-free (Barabasi-Albert)",
        m=2,
        seeds="every node",
        immune_counts="0..N-1",
    )
    builders = {n: (lambda n=n: barabasi_albert(n, links_per_new_node=2, rng=ctx.rng)) for n in SIZES}
    # N = 100 is the network of Figs. 58-63, as in the thesis (fig. 67.c, f = figs. 59, 63).
    worked = np.random.default_rng(THESIS_SEED)
    builders[100] = lambda: barabasi_albert(100, links_per_new_node=2, rng=worked)
    return _family_table(ctx, builders, "scale_free", "Scale-free", rngs={100: worked})


# ---------------------------------------------------------------------------
# 5.5.2  Fig. 68: the ordered networks, where the answer changes
# ---------------------------------------------------------------------------


def _fullerene_patch() -> tuple[Network, np.ndarray]:
    """The polyhedral network of Fig. 68.a: three shells, every node of degree 3.

    Read off the printed drawing: an outer nine-sided ring, a ring of fifteen
    carrying the pentagons, and a small inner ring, with one spoke from every
    middle node -- nine of them outward and six inward.  Thirty nodes,
    forty-five links, faces of four, five and six sides, and a three-fold
    symmetry, which is the symmetry the printed picture has.

    The shell sizes are deliberately unequal.  A network in which every node
    looks the same -- the dodecahedron, say -- is useless here, because then
    :math:`K_m(n)` gives every node the same score at every order and no
    criterion can rank anything: the resulting :math:`\\beta` is flat and the
    figure says nothing.  Unequal shells give the nodes genuinely different
    neighbourhoods while keeping the network exactly ordered, which is the
    condition Fig. 68 is about.
    """
    outer, middle, inner = 9, 15, 6
    n = outer + middle + inner
    o = lambda j: j  # noqa: E731 - index helpers, local and one-line
    m = lambda j: outer + j  # noqa: E731
    i_ = lambda j: outer + middle + j  # noqa: E731

    # Per group of five middle nodes: out, in, out, in, out.
    outward = [5 * g + k for g in range(3) for k in (0, 2, 4)]
    inward = [5 * g + k for g in range(3) for k in (1, 3)]

    edges: list[tuple[int, int]] = []
    for j in range(outer):
        edges.append((o(j), o((j + 1) % outer)))
        edges.append((o(j), m(outward[j])))
    for j in range(middle):
        edges.append((m(j), m((j + 1) % middle)))
    for j in range(inner):
        edges.append((i_(j), i_((j + 1) % inner)))
        edges.append((i_(j), m(inward[j])))

    pos = np.zeros((n, 2))
    for j in range(outer):
        a = np.radians(90 + 360 * outward[j] / middle)
        pos[o(j)] = 1.00 * np.array([np.cos(a), np.sin(a)])
    for j in range(middle):
        a = np.radians(90 + 360 * j / middle)
        pos[m(j)] = 0.62 * np.array([np.cos(a), np.sin(a)])
    for j in range(inner):
        a = np.radians(90 + 360 * inward[j] / middle)
        pos[i_(j)] = 0.26 * np.array([np.cos(a), np.sin(a)])
    return Network.from_edges(n, edges, name="polyhedral patch (fig. 68.a)"), pos


def _pentagon_with_clusters() -> tuple[Network, np.ndarray]:
    """A pentagonal frame with a small grid hung at each of its five corners.

    The second ordered network of Fig. 68: a five-fold symmetric skeleton --
    inner pentagon, five junction nodes, an outer ten-ring -- with a two-by-
    three grid cluster attached at every junction.
    """
    inner = list(range(5))
    hub = [5 + i for i in range(5)]
    outer = [10 + j for j in range(10)]
    cluster = [[20 + 6 * i + t for t in range(6)] for i in range(5)]
    n = 50

    edges: list[tuple[int, int]] = []
    for i in range(5):
        edges.append((inner[i], inner[(i + 1) % 5]))
        edges.append((inner[i], hub[i]))
        edges.append((hub[i], outer[2 * i]))
        edges.append((hub[i], outer[2 * i + 1]))
        c = cluster[i]
        for row in range(2):
            for col in range(2):
                edges.append((c[3 * row + col], c[3 * row + col + 1]))
        for col in range(3):
            edges.append((c[col], c[3 + col]))
        edges.append((hub[i], c[0]))
        edges.append((c[5], outer[(2 * i + 1) % 10]))
    for j in range(10):
        edges.append((outer[j], outer[(j + 1) % 10]))

    pos = np.zeros((n, 2))
    for i in range(5):
        a = np.radians(90 + 72 * i)
        u = np.array([np.cos(a), np.sin(a)])
        t = np.array([-np.sin(a), np.cos(a)])
        pos[inner[i]] = 0.28 * u
        pos[hub[i]] = 0.52 * u
        pos[outer[2 * i]] = 1.05 * np.array(
            [np.cos(np.radians(90 + 72 * i - 20)), np.sin(np.radians(90 + 72 * i - 20))]
        )
        pos[outer[2 * i + 1]] = 1.05 * np.array(
            [np.cos(np.radians(90 + 72 * i + 20)), np.sin(np.radians(90 + 72 * i + 20))]
        )
        centre = 0.80 * u
        for row in range(2):
            for col in range(3):
                pos[cluster[i][3 * row + col]] = (
                    centre + (col - 1) * 0.15 * t + (0.5 - row) * 0.17 * u
                )
    return Network.from_edges(n, edges, name="pentagon with clusters (fig. 68.b)"), pos


def _square_lattice(rows: int = 7, cols: int = 8) -> tuple[Network, np.ndarray]:
    """The plain square grid of Fig. 68.c: 8 columns by 7 rows."""
    n = rows * cols
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((r * cols + c, r * cols + c + 1))
            if r + 1 < rows:
                edges.append((r * cols + c, (r + 1) * cols + c))
    pos = np.array([[c / (cols - 1), -r / (rows - 1)] for r in range(rows) for c in range(cols)])
    return Network.from_edges(n, edges, name="square lattice (fig. 68.c)"), pos


@figure(
    "T68",
    chapter=7,
    title="Which order immunises the ordered networks best",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="63",
        section="5.5.2 Inmunizacion frente a infeccion",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis figs. 68.a.1-c.2, PDF p. 63",
)
def ordered_networks(ctx: FigureContext):
    """PDF p. 63, fig. 68: three ordered networks, and beta for each order of K(n).

    The thesis's own three networks -- the first two cubic, the third a 7 x 8
    grid -- node for node and edge for edge as its figure draws them, read by
    ``book/tools/fig68_data.py``; on each, the immunisation sweep of section
    5.5.2 is run for every order n, and the beta it gives is drawn, the largest
    |beta| ringed.
    """
    data = _thesis.data("68")
    fig, axes = plt.subplots(3, 2, figsize=(5.05, 5.6), gridspec_kw={"width_ratios": [1.0, 1.25]},
                             layout="constrained")
    best_beta, best_needed = [], []
    for (left, right), net, row in zip(axes, data["networks"], "abc", strict=True):
        pos = np.asarray(net["positions"], dtype=float)
        g = Network.from_edges(len(pos), [tuple(e) for e in net["edges"]])
        draw_network(g, left, layout=pos, node_color=SERIES[0], edge_width=0.8)
        panel_letter(left, f"{row}.1")

        sweeps = compare_orders(g, list(ORDERS), counts=np.arange(0, g.n), rng=ctx.rng)
        betas = np.array([_beta_or_nan(sweeps[o]) for o in ORDERS])
        needed = np.array([sweeps[o].mean_nodes_to_extinction for o in ORDERS])
        right.plot(ORDERS, betas, color=SERIES[0], lw=1.8, marker="o", ms=4.5, zorder=3)
        best_beta.append(ORDERS[_mark_best(right, np.asarray(ORDERS), betas, below=True)])
        best_needed.append(int(ORDERS[int(np.argmin(needed))]))
        right.set_xticks(list(ORDERS))
        right.margins(y=0.3)
        if row == "c":
            lo, hi = right.get_ylim()
            right.set_ylim(lo, hi + 0.12 * (hi - lo))
        right.set_ylabel(r"$\beta$")
        panel_letter(right, f"{row}.2")
    axes[-1, 1].set_xlabel("order $n$ of the criterion $K(n)$")
    ctx.parameters(networks=["a.1", "b.1", "c.1"], best_by_beta=best_beta,
                   best_by_nodes_needed=best_needed)
    return fig
