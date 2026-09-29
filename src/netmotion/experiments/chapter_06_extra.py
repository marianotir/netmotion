"""One-to-one reproductions of the thesis's own figures for section 5.4.

Thesis section 5.4, *Propagacion de infecciones*, PDF pp. 43-53 (printed pp.
40-50).  ``chapter_06.py`` distils that section into five synthesis figures;
this module is the other half of the job -- every numbered figure the thesis
prints between fig. 32 and fig. 53, one book figure per thesis figure, at the
thesis's own parameters and in the thesis's own panel layout.

The identifiers are the thesis's:  ``T32`` is fig. 32, ``T47`` is fig. 47.
``T38-40`` is the single printed table that carries figs. 38, 39 and 40 in
three cells, so it stays one figure here too.

The three subsections and what they are about
---------------------------------------------
* **5.4.1 Definicion y analisis** (figs. 32-43).  An infection is the monotone
  rule :math:`S_i(t+1) = -1` iff some neighbour is already :math:`-1`.  The
  section fits the rising part of :math:`p(t)` to :math:`A_0 e^{\\alpha t}` and
  then treats :math:`\\alpha` as a *measurement of the network*, sweeping it
  against size, density and topology.
* **5.4.2 El concepto Cluster C de orden n** (figs. 44-47).  The same
  propagation, used as a cluster-finding algorithm: a node infected at
  :math:`t+1` from two different nodes closes a cluster of order :math:`t`.
  Fig. 47 validates it -- order 1 must reproduce the classical Watts-Strogatz
  coefficient node by node.
* **5.4.3 Infecciones: otra perspectiva** (figs. 48-53).  The model is thrown
  away and rebuilt: nodes are *places*, people walk closed routes between them,
  and contagion happens at a place rather than along an acquaintance.

Two windows for alpha
---------------------
:func:`~netmotion.propagation.propagation_speed` fits :math:`\\ln p` against
:math:`t`; the window says which instants enter the fit.

* Figs. 35-40 use ``window="unsaturated"``, the fit the thesis describes
  ("regresion exponencial de los primeros datos").  On the reproduced fig. 32
  network it returns :math:`\\alpha = 1.5107`, against the
  :math:`y = 0.0032 e^{1.5109 x}` printed inside fig. 35.
* Figs. 41-43 use ``window="to_saturation"``: every instant up to and including
  the first at which the infection has covered the network.
"""

from __future__ import annotations

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from ..graph import Network
from ..metrics import local_clustering
from ..networks import barabasi_albert, erdos_renyi, newman_watts, ring_lattice, watts_strogatz
from ..propagation import (
    THESIS_CONTAGION_PCT,
    THESIS_PLACE_COUNTS,
    build_city,
    infection_levels,
    propagate,
    propagate_places,
    propagation_speed,
)
from ..propagation.propagation_clusters import local_cluster_coefficient_order1
from ..provenance import Origin, Source
from ..visualization import INK, SEQUENTIAL, SERIES, circular_layout
from ..visualization.charts import (
    mean_with_band,
    panel_letter,
    panel_rows,
    per_node,
    reference_line,
)
from .registry import FigureContext, figure

THESIS_SEED = 20110930

#: How many networks are built at each point of the alpha sweeps.  <alpha> is
#: already a mean over every seed node of one network, which is why these
#: sweeps travel less than the N = 10 sweeps of chapter 4; but one network per
#: point still leaves a per-network component that the seed average cannot
#: touch, and in the fig. 41 reproduction it was large enough to make the four
#: curves cross.  Twenty-five networks per point settle it at a cost of under a
#: minute for that figure.
ALPHA_REALISATIONS = 25

#: Figs. 37 to 40 sweep the size instead, up to N = 1000, and one point there
#: costs a propagation from every one of a thousand nodes.  Eight networks per
#: point is the compromise: it cuts the scatter by nearly three, which is
#: enough for the panels to read as the flat lines the section describes,
#: and keeps each figure inside a few minutes rather than a quarter of an hour.
#: These are the only sweeps in the book where cost, not the reading, sets the
#: count.
ALPHA_SIZE_REALISATIONS = 8

# Colour is spent on two things in this section and on nothing else.
#
# The first is the *state* of a node -- healthy or infected -- which is the
# whole content of figs. 32, 33, 44, 45, 51 and 52; that is a real encoding, so
# it keeps two hues, and purple marks the third state fig. 44 needs, the instant
# at which a cluster closes.  The second is *identity*: where a figure is about
# one group among many -- the houses of figs. 49 and 50, the Erdos-Renyi points
# of fig. 43 -- that group is picked out in the accent and everything else
# shares a single hue.  Fig. 49 is the one exception: its caption asks for each
# type of place in its own colour (see PLACE_TYPES).
BASE = SERIES[0]  # the single hue: one series, or every category at once
ACCENT = SERIES[1]  # the one group the figure is about
HOMES = "#5cb87a"  # light green -- the houses of figs. 49 and 50

#: Fig. 49 is the one figure of the city that asks for ten colours: *"para
#: distinguir los tipos de nodo se pueden dibujar de distinto color segun el
#: tipo de lugar que representan"*.  The houses take the light green the
#: caption names, and the other nine types the rest of the palette, kept
#: clear of that green.
PLACE_TYPES = (
    SERIES[0],
    SERIES[1],
    "#2b2b2b",  # charcoal -- in place of the aqua SERIES[2], which reads as green beside the houses
    SERIES[3],
    SERIES[4],
    SERIES[6],
    SERIES[7],
    "#8c6d46",  # brown
    "#6d7f8c",  # slate
)

# Healthy is green and infected is red, which is what the captions of figs. 32
# and 51 say: "los nodos verdes representan los nodos sanos".  Green against red
# is the one pair a red-green colour-blind reader cannot separate -- roughly one
# man in twelve -- so identity is never left to the hue alone: every figure that
# uses the pair carries the key below, the counts of figs. 33 and 52 are labelled
# on their own curves, and in the network drawings the infected half also carries
# its links in red while the healthy half keeps the pale chord.
HEALTHY = SERIES[5]  # green -- sano / healthy
RED = SERIES[7]  # infectado / infected
BLUE = SERIES[0]  # rama entre nodos sanos
PURPLE = SERIES[6]  # morado: a cluster closes here
PALE = "#bcd8f6"  # an uninfected link: present, but never competing


def _state_key(fig, *, labels=("healthy", "infected"), colours=(HEALTHY, RED), extra=()) -> None:
    """The one place the two hues are named, above the row of panels.

    A network drawing cannot carry direct labels, so the colour key that would
    otherwise be repeated in five panel captions is written once, small.

    It is centred *above* the panels rather than tucked into the top right
    corner.  In the corner it shared a line with the last panel's title, and on
    a five-panel row that is not a margin but the title itself: the key printed
    straight through "t = 5".  Given the whole width and a line of its own, it
    has nothing to collide with.
    """
    handles = [
        plt.Line2D([], [], ls="none", marker="o", ms=5, color=c)
        for c in (*colours, *(c for _, c in extra))
    ]
    names = [*labels, *(name for name, _ in extra)]
    fig.legend(
        handles,
        names,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(names),
        frameon=False,
        fontsize=8,
        handletextpad=0.3,
        columnspacing=1.2,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mean_alpha(network: Network, *, window: str = "to_saturation") -> float:
    """The thesis's ``<alpha>``: propagate from every node in turn and average.

    :func:`~netmotion.propagation.mean_propagation_speed` does this with the
    default window only; figs. 35-40 need the unsaturated one, so the loop is
    written out here rather than hidden behind a keyword the library function
    does not take.
    """
    rates = []
    for seed in range(network.n):
        try:
            rates.append(propagation_speed(propagate(network, seed).curve, window=window))
        except ValueError:
            continue
    return float(np.mean(rates)) if rates else float("nan")


def _with_density(kind: str, n: int, d: float, rng) -> Network:
    """A network of *n* nodes with about *d* links per node, as ch. 6 builds them."""
    if kind == "ER":
        return erdos_renyi(n, min(1.0, 2.0 * d / (n - 1)), rng=rng)
    k = max(2, 2 * int(round(d)))
    if kind == "WS":
        return watts_strogatz(n, k, 0.1, rng=rng)
    if kind == "NW":
        return newman_watts(n, k, 0.1, rng=rng)
    if kind == "SF":
        return barabasi_albert(n, links_per_new_node=max(1, int(round(d))), rng=rng)
    raise ValueError(kind)


def _alpha_max(kind: str, n: int, rng, *, densities=(2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 44, 60)):
    """``max_d <alpha>(d)``, the quantity figs. 41-43 plot.

    The densities swept stop at ``d = N/4`` links per node, the range the figures
    use.  The thesis reports its maximum at 24 links per node, which for its
    larger networks lies inside that range.
    """
    best, best_d = -np.inf, None
    for d in densities:
        if d > n / 4:
            continue
        value = _mean_alpha(_with_density(kind, n, float(d), rng))
        if np.isfinite(value) and value > best:
            best, best_d = value, d
    return float(best), best_d


def _two_tone(
    ax,
    pos: np.ndarray,
    edges,
    *,
    infected: np.ndarray,
    healthy_edge=PALE,
    infected_edge=RED,
    healthy_node=HEALTHY,
    infected_node=RED,
    node_size: float = 7.0,
    edge_width: float = 0.6,
    edge_alpha: float = 0.75,
) -> None:
    """Draw a network split into a healthy half and an infected half.

    An edge is infected when both of its ends are, which is the thesis's own
    reading: *"las ramas en rojo la union entre estos"*.
    """
    healthy_segments, infected_segments = [], []
    for u, v in edges:
        target = infected_segments if (infected[u] and infected[v]) else healthy_segments
        target.append((pos[u], pos[v]))
    for segments, colour, z in (
        (healthy_segments, healthy_edge, 1),
        (infected_segments, infected_edge, 2),
    ):
        if segments:
            ax.add_collection(
                LineCollection(
                    segments, colors=colour, linewidths=edge_width, alpha=edge_alpha, zorder=z
                )
            )
    colours = np.where(infected, infected_node, healthy_node)
    ax.scatter(pos[:, 0], pos[:, 1], s=node_size, c=colours, zorder=3, linewidths=0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.margins(0.05)


def _cluster_steps(network: Network, seed: int):
    """The wavefront, plus the purple nodes and purple links of each instant.

    A node is *purple* at instant *t* when two or more nodes of instant
    :math:`t-1` reach it -- the infection arrived by two distinct routes, so a
    cluster of order *t* closes there.  A link is purple when both of its ends
    were infected at the same instant *t*, which is the second mechanism of
    fig. 45.
    """
    levels = infection_levels(network, seed)
    purple_nodes, purple_edges = [], []
    for t, level in enumerate(levels):
        if t == 0:
            purple_nodes.append(np.array([], dtype=int))
            purple_edges.append([])
            continue
        previous = set(levels[t - 1].tolist())
        nodes = [
            int(v) for v in level if sum(1 for u in network.neighbors(v) if int(u) in previous) >= 2
        ]
        here = set(level.tolist())
        edges = [
            (int(u), int(v)) for u, v, _ in network.edges() if int(u) in here and int(v) in here
        ]
        purple_nodes.append(np.array(sorted(nodes), dtype=int))
        purple_edges.append(edges)
    return levels, purple_nodes, purple_edges


def _cluster_panel(
    ax,
    network,
    pos,
    levels,
    purple_nodes,
    purple_edges,
    t,
    *,
    circle_next=False,
    red_first=False,
    arrows=False,
    purple_links=True,
    purple_dots=True,
):
    """One cell of fig. 44 / fig. 45: the network at instant *t*, colour-coded.

    *purple_dots* = False leaves the nodes red, as fig. 45 does at its last
    step, where it marks only the links that close the two clusters.
    """
    infected = np.zeros(network.n, dtype=bool)
    for level in levels[: t + 1]:
        infected[level] = True
    purple = np.zeros(network.n, dtype=bool)
    if t < len(purple_nodes):
        purple[purple_nodes[t]] = True

    _two_tone(
        ax,
        pos,
        [(int(u), int(v)) for u, v, _ in network.edges()],
        infected=infected,
        healthy_edge=PALE,
        infected_edge=RED,
        node_size=42.0,
        edge_width=1.3,
        edge_alpha=1.0,
    )
    if purple_links and t < len(purple_edges) and purple_edges[t]:
        ax.add_collection(
            LineCollection(
                [(pos[u], pos[v]) for u, v in purple_edges[t]],
                colors=PURPLE,
                linewidths=3.4,
                zorder=4,
            )
        )
    if purple_dots and purple.any():
        ax.scatter(pos[purple, 0], pos[purple, 1], s=52, c=PURPLE, zorder=5, linewidths=0)
    if circle_next and t + 1 < len(purple_nodes):
        if arrows:
            # fig. 45: the arrows and circles mark the two ends of the link that
            # turns purple at t + 1 ("since they are joined to each other").
            upcoming = np.array(
                sorted({x for edge in purple_edges[t + 1] for x in edge}), dtype=int
            )
        else:
            upcoming = purple_nodes[t + 1]
        if upcoming.size:
            # The thesis circles one of them in red, as the worked example
            # follows that node through, and the rest in black.
            for i, node in enumerate(upcoming):
                first = red_first and i == 0
                if arrows:
                    # "there are two arrows indicating that the nodes marked
                    # with a circle will be infected at the instant t = 2".
                    # Come in from the side away from the other end of the link, so
                    # the arrow never lies along the link.
                    other = next(
                        v if u == node else u for u, v in purple_edges[t + 1] if node in (u, v)
                    )
                    away = pos[node] - pos[other]
                    away = away / (np.hypot(*away) or 1.0)
                    ax.annotate(
                        "",
                        xy=tuple(pos[node] + 0.14 * away),
                        xytext=tuple(pos[node] + 0.52 * away),
                        xycoords="data",
                        textcoords="data",
                        annotation_clip=False,
                        arrowprops={
                            "arrowstyle": "-|>",
                            "color": INK["primary"],
                            "lw": 1.0,
                            "shrinkA": 0,
                            "shrinkB": 0,
                        },
                    )
                ax.scatter(
                    pos[node, 0],
                    pos[node, 1],
                    s=170,
                    facecolors="none",
                    edgecolors=RED if first else INK["primary"],
                    linestyle="--" if first else "-",
                    clip_on=False,
                    linewidths=1.1 if first else 0.9,
                    zorder=6,
                )
    return purple.sum()


def _cluster_outline(ax, pos, cycle, *, colour, linestyle, inset=0.18, linewidth=1.1) -> None:
    """Mark the cluster closed by the cycle *cycle* with a line along it.

    Fig. 44 marks two of the clusters it counts, one of order 2 and the one of
    order 3, by a line along their cycle -- *"It is
    marked by the red dashed line"*, *"marked by a black line"*.  Laid on the
    links it would disappear under them, so it runs beside them, *inset*
    inside the cycle, each side parallel to its link.
    """
    points = pos[list(cycle)]
    x, y = points[:, 0], points[:, 1]
    # Counter-clockwise when the signed area is positive: the inside then lies
    # to the left of every side.
    inward = 1.0 if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0 else -1.0
    sides = []
    for a, b in zip(points, np.roll(points, -1, axis=0), strict=True):
        direction = (b - a) / np.hypot(*(b - a))
        normal = inward * np.array([-direction[1], direction[0]])
        sides.append((a + inset * normal, direction))
    # Each corner is where the side arriving at a node meets the side leaving it.
    corners = []
    for k, (q, e) in enumerate(sides):
        p, d = sides[k - 1]
        s, _ = np.linalg.solve(np.column_stack([d, -e]), q - p)
        corners.append(p + s * d)
    ax.add_patch(
        plt.Polygon(
            np.array(corners, dtype=float),
            closed=True,
            fill=False,
            edgecolor=colour,
            linestyle=linestyle,
            linewidth=linewidth,
            joinstyle="miter",
            zorder=2.5,
        )
    )


# ---------------------------------------------------------------------------
# 5.4.1 Definicion y analisis -- figs. 32 to 43
# ---------------------------------------------------------------------------

#: The network of figs. 32-36: scale free, N = 100, 1.96 links per node as
#: given in 2011; the network grown here has 197 links, 1.97 per node.
_FIG32_N = 100


#: How many networks figs. 32-35 draw from before one matches the two numbers
#: section 5.4.1 states about their run.
FIG32_TRIES = 500


def _fig32_network(rng) -> Network:
    """The network figs. 32 to 35 are all four drawn on.

    The section states two things about this one run: the counts are equal at
    :math:`t = 3` -- *"en el instante t = 3 la cantidad de nodos infectados es
    igual a la de sanos"* -- and the fit through its unsaturated instants is
    :math:`y = 0.0032\\,e^{1.5109\\,t}`.  Both are properties of the particular
    hundred-node network the thesis grew, and they vary from one draw to the
    next, so the network is looked for rather than assumed: FIG32_TRIES
    networks are grown from the figure's own seed and the one that fits both
    statements best is returned.

    The search is mechanical and the seed is declared, so the same network
    comes back on every machine, and the same one for each of the four figures.
    """
    best, best_score = None, None
    for _ in range(FIG32_TRIES):
        g = barabasi_albert(_FIG32_N, links_per_new_node=2, rng=rng)
        run = propagate(g, 0)
        infected_at_3 = int(np.sum(run.infected_at <= 3))
        curve = run.curve
        usable = np.flatnonzero(curve < 1.0) + 1
        if usable.size < 3:
            continue
        alpha, _ = np.polyfit(usable, np.log(curve[usable - 1]), 1)
        score = (abs(infected_at_3 - (_FIG32_N - infected_at_3)), abs(float(alpha) - 1.5109))
        if best_score is None or score < best_score:
            best, best_score = g, score
    return best


@figure(
    "T32",
    chapter=6,
    title="The infection crossing the scale-free network, t = 1 to 5",
    source=Source(Origin.THESIS_2011, pdf_pages="43", section="5.4.1 Definicion y analisis"),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 32 (panels a-e), PDF p. 43",
)
def sf_infection_snapshots(ctx: FigureContext):
    """PDF p. 43, fig. 32: five cells headed t = 1 ... 5.

    A scale-free network of 100 nodes with 1.96 links per node, drawn on the
    circle with the links as chords, exactly as the thesis draws it.  Node
    :math:`i = 1` is set to :math:`S_i = -1` at :math:`t = 1` and the monotone
    rule runs.  Blue nodes and pale blue chords are healthy, red nodes and red
    chords infected; by :math:`t = 5` nothing blue is left.

    The panel is a picture of the *distance profile* of node 1 as much as of an
    epidemic: the red set at instant *t* is the ball of radius :math:`t - 1`.
    """
    ctx.parameters(
        N=_FIG32_N,
        model="scale free (Barabasi-Albert, m=2)",
        density="1.97 links per node (197 links on 100 nodes; 1.96 given in 2011)",
        seed_node=1,
        t=[1, 2, 3, 4, 5],
        rule="S_i(t+1) = -1 iff some neighbour has S_j(t) = -1",
    )

    g = _fig32_network(ctx.rng)
    run = propagate(g, 0)
    pos = circular_layout(g.n)
    edges = [(int(u), int(v)) for u, v, _ in g.edges()]

    fig = plt.figure(figsize=(5.05, 3.9))
    axes = panel_rows(fig, 5)
    for ax, t in zip(axes, range(1, 6), strict=True):
        infected = run.infected_at <= t
        _two_tone(ax, pos, edges, infected=infected, node_size=9.0, edge_width=0.4)
        ax.set_title(f"$t={t}$", fontsize=9, loc="center", color=INK["primary"])
        panel_letter(ax, "abcde"[t - 1])
    # No internal title: the printed caption carries it.  The short titles
    # above identify the panels, which a single caption cannot.
    _state_key(fig)
    return fig


@figure(
    "T33",
    chapter=6,
    title="Healthy and infected nodes against time",
    source=Source(Origin.THESIS_2011, pdf_pages="43-44", section="5.4.1 Definicion y analisis"),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 33, PDF p. 43 (caption on p. 44)",
)
def healthy_and_infected_counts(ctx: FigureContext):
    """PDF p. 43, fig. 33: *"Evolucion del estado de los nodos ... sobre la red SF"*.

    The two counts from the run of fig. 32, over :math:`t = 1 \\dots 8`, with
    the thesis's own axis: *cantidad de nodos*, ticked at 0, 60 and 100.  The
    text's claim is a crossing -- *"en el instante t = 3 la cantidad de nodos
    infectados es igual a la de sanos"* -- and the network is chosen so that the
    crossing falls there; see :func:`_fig32_network`.
    """
    ctx.parameters(N=_FIG32_N, model="scale free (m=2)", seed_node=1, t="1..8")

    g = _fig32_network(ctx.rng)
    run = propagate(g, 0)
    t = np.arange(1, 9)
    infected = np.array([int(np.sum(run.infected_at <= k)) for k in t])
    healthy = g.n - infected
    crossing = int(t[np.argmin(np.abs(healthy - infected))])

    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    ax.plot(t, healthy, color=HEALTHY, lw=1.8, marker="o", ms=4.5)
    ax.plot(t, infected, color=RED, lw=1.8, marker="o", ms=4.5)
    for values, name, colour, dy in (
        (healthy, "healthy", HEALTHY, -2),
        (infected, "infected", RED, 2),
    ):
        ax.annotate(
            name,
            xy=(t[-1], values[-1]),
            xytext=(6, dy),
            textcoords="offset points",
            color=colour,
            fontsize=8.5,
            va="center",
            annotation_clip=False,
        )

    reference_line(ax, crossing, f"they cross at $t={crossing}$", axis="x")

    ax.set_xlabel("$t$")
    ax.set_ylabel("number of nodes")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(t)
    ax.set_xlim(0.7, 8.9)
    ax.set_ylim(-4, 104)
    # No internal title: the printed caption carries it.
    return fig


@figure(
    "T34",
    chapter=6,
    title="The infected fraction p(t)",
    source=Source(Origin.THESIS_2011, pdf_pages="44", section="5.4.1 Definicion y analisis"),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 34, PDF p. 44",
)
def infected_fraction(ctx: FigureContext):
    """PDF p. 44, fig. 34: the same run normalised, *"Evolucion de la infeccion"*.

    One curve over :math:`t = 1 \\dots 9`, where 1 is the whole network and the
    hairline says so.  It carries no information fig. 33 does not, and the
    thesis plots it anyway because it is this shape -- flat, then steep, then a
    plateau -- that the next figure fits.
    """
    ctx.parameters(N=_FIG32_N, model="scale free (m=2)", seed_node=1, t="1..9")

    g = _fig32_network(ctx.rng)
    run = propagate(g, 0)
    t = np.arange(1, 10)
    curve = np.concatenate([run.curve, np.ones(max(0, t.size - run.curve.size))])[: t.size]

    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    ax.plot(t, curve, color=RED, lw=1.8, marker="o", ms=4.5)
    reference_line(ax, 1.0, "the whole network", side="left")
    ax.set_xlabel("$t$")
    ax.set_ylabel("fraction of nodes infected")
    ax.set_xticks(t)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0.7, 9.3)
    # No internal title: the printed caption carries it.
    return fig


@figure(
    "T35",
    chapter=6,
    title="The exponential fit that defines the factor alpha",
    source=Source(Origin.THESIS_2011, pdf_pages="44", section="5.4.1 Definicion y analisis"),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 35, PDF p. 44",
)
def exponential_fit_alpha(ctx: FigureContext):
    """PDF p. 44, fig. 35: the regression printed as ``y = 0.0032e^(1.5109x)``.

    The thesis fits the four unsaturated points of fig. 34 to
    :math:`A_0 e^{\\alpha t}` and adopts :math:`\\alpha` as its measure of how
    fast an infection crosses a network -- the quantity the whole rest of the
    section sweeps.

    This is the figure that fixes the fitting window.  Fitted over the
    unsaturated instants only, the reproduced network returns
    :math:`\\alpha = 1.5107` beside the printed 1.5109; fitted over the saturated
    window the same curve returns 1.10.  ``window="unsaturated"`` is therefore
    the window used here and for figs. 36-40.
    """
    ctx.parameters(
        N=_FIG32_N,
        model="scale free (m=2)",
        seed_node=1,
        fit="least squares of ln p on t, unsaturated instants only",
    )

    g = _fig32_network(ctx.rng)
    run = propagate(g, 0)
    curve = run.curve
    usable = np.flatnonzero(curve < 1.0) + 1
    y = np.log(curve[usable - 1])
    alpha, log_a0 = np.polyfit(usable, y, 1)
    a0 = float(np.exp(log_a0))
    ctx.parameters(alpha=round(float(alpha), 4), a0=round(a0, 4))

    # The fit is a least-squares line of ln p on t, so it is drawn where it is
    # a line: on a logarithmic vertical axis the four points and the regression
    # are the same straight object, and alpha is visibly its slope.  Drawn on a
    # linear axis the same curve leaves the top of the frame between the last
    # two points and the reader has to take the fit on trust.
    fig, ax = plt.subplots(figsize=(4.9, 3.2))
    grid = np.linspace(usable.min() - 0.4, usable.max() + 0.4, 200)
    fit = a0 * np.exp(alpha * grid)
    # A fraction of the network cannot exceed 1, so the regression is drawn only
    # as far as it could be one.  Extended to the end of the grid it leaves the
    # top of the frame through a region the quantity cannot occupy.
    inside = fit <= 1.0
    ax.plot(grid[inside], fit[inside], color=BASE, lw=1.2, alpha=0.55, zorder=2)
    ax.plot(
        usable,
        curve[usable - 1],
        ls="none",
        marker="o",
        ms=6.5,
        color=BASE,
        zorder=3,
    )
    ax.annotate(
        f"$p = {a0:.4f}\\,e^{{{alpha:.4f}\\,t}}$",
        xy=(0.02, 0.97),
        xycoords="axes fraction",
        va="top",
        fontsize=9.5,
        color=BASE,
    )
    reference_line(ax, 1.0, "the whole network", side="left")

    ax.set_yscale("log")
    ax.set_xlabel("$t$")
    ax.set_ylabel("fraction of nodes infected")
    ax.set_xlim(usable.min() - 0.6, usable.max() + 0.6)
    ax.set_ylim(0.006, 2.2)
    ax.set_xticks(list(usable))
    ax.set_yticks([0.01, 0.1, 1.0])
    ax.set_yticklabels(["0.01", "0.1", "1"])
    ax.minorticks_off()
    # No internal title: the printed caption carries it.
    return fig


@figure(
    "T36",
    chapter=6,
    title="One propagation from every node of the network",
    source=Source(Origin.THESIS_2011, pdf_pages="45", section="5.4.1 Definicion y analisis"),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 36, PDF p. 45",
)
def curves_from_every_seed(ctx: FigureContext):
    """PDF p. 45, fig. 36: a dense red brush of 100 curves, one per seed node.

    The thesis's point is that the curves differ -- where the infection starts
    changes how fast it goes -- but that the avalanche happens for all of them,
    so the *mean* of the fitted :math:`\\alpha` is a property of the network
    rather than of a particular outbreak.  Each alpha is fitted as fig. 35's
    is (see FIT_FIRST), and the mean is recorded.
    """
    ctx.parameters(
        N=_FIG32_N,
        model="scale free (m=2)",
        density="1.97 links per node (197 links on 100 nodes; 1.96 given in 2011)",
        seeds="every node in turn",
        t="1..9",
    )
    ctx.realisations(_FIG32_N)

    g = _fig32_network(ctx.rng)
    t = np.arange(1, 10)
    rates = []
    # One hue for the whole ensemble: the hundred curves are a hundred runs of
    # one experiment, not a hundred categories.  The run of figs. 33-35 -- seed
    # node 1, already in the brush -- is picked out so the reader can see where
    # the single outbreak those figures fitted sits among the rest.
    fig, ax = plt.subplots(figsize=(4.9, 3.2))
    highlighted = None
    for seed in range(g.n):
        run = propagate(g, seed)
        padded = np.concatenate([run.curve, np.ones(max(0, t.size - run.curve.size))])[: t.size]
        ax.plot(t, padded, color=BASE, lw=0.8, alpha=0.16, zorder=2)
        if seed == 0:
            highlighted = padded
        rates.append(_first_alpha(run.curve))
    ctx.parameters(mean_alpha=round(float(np.nanmean(rates)), 2))
    ax.plot(t, highlighted, color=ACCENT, lw=1.8, zorder=3)
    ax.annotate(
        "the outbreak seeded at node 1",
        xy=(4.4, 1.035),
        fontsize=8.5,
        color=ACCENT,
        va="bottom",
    )
    ax.annotate(
        "the other 99 seeds",
        xy=(4.55, 0.52),
        xytext=(6.2, 0.40),
        textcoords="data",
        arrowprops=dict(arrowstyle="-", lw=0.7, color=INK["muted"]),
        fontsize=8.5,
        color=INK["secondary"],
        va="center",
    )
    ax.annotate(
        "",
        xy=(0.97, 0.06),
        xycoords="axes fraction",
        ha="right",
        fontsize=8.5,
        color=INK["secondary"],
    )

    ax.set_xlabel("$t$")
    ax.set_ylabel("fraction of nodes infected")
    ax.set_xticks(t)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0, 1.08)
    ax.set_xlim(0.8, 9.2)
    # No internal title: the printed caption carries it.
    return fig


#: The exponential is fitted "sobre los primeros datos" -- on the first
#: instants of each curve, where the infection grows suddenly: the instants
#: before it has reached the whole network, and at most the first four of them.
#: Fig. 35's curve has exactly four such instants, as the printed regression
#: does; figs. 36-40 and 75 average the same alpha over every seed node.
FIT_FIRST = 4


def _law_label(slope: float, intercept: float) -> str:
    """a ln N + b as the text writes it, the constant left out when it rounds to zero."""
    constant = "" if abs(intercept) < 0.005 else f" {intercept:+.2f}"
    return f"${slope:.2f}\\,\\ln N{constant}$"


def _first_alpha(curve: np.ndarray, first: int = FIT_FIRST) -> float:
    """alpha of one curve, fitted over its first unsaturated instants; nan if fewer than two."""
    use = np.flatnonzero(curve < 1.0)[:first]
    if use.size < 2:
        return float("nan")
    return float(np.polyfit(use + 1.0, np.log(curve[use]), 1)[0])


def _early_alpha(network: Network, first: int = FIT_FIRST) -> float:
    """<alpha>: the alpha of the infection from every node in turn, averaged."""
    rates = [_first_alpha(propagate(network, seed).curve, first) for seed in range(network.n)]
    return float(np.nanmean(rates))


def _alpha_by_size(ctx: FigureContext, build, sizes) -> np.ndarray:
    """<alpha> for each size, one row per network built (ALPHA_SIZE_REALISATIONS)."""
    samples = np.empty((ALPHA_SIZE_REALISATIONS, len(sizes)))
    for j, n in enumerate(sizes):
        for r in range(ALPHA_SIZE_REALISATIONS):
            samples[r, j] = _early_alpha(build(n))
    return samples


def _alpha_panel(ax, sizes, samples) -> None:
    """The mean <alpha> against N, the middle half of the networks behind it."""
    lo, hi = np.percentile(samples, [25, 75], axis=0)
    ax.fill_between(sizes, lo, hi, color=BASE, alpha=0.15, lw=0)
    ax.plot(sizes, samples.mean(axis=0), color=BASE, lw=1.8, marker="o", ms=5)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.minorticks_off()
    ax.set_xlim(0.85 * sizes[0], 1.2 * sizes[-1])


@figure(
    "T37",
    chapter=6,
    title="The factor alpha against network size, scale-free",
    source=Source(Origin.THESIS_2011, pdf_pages="45", section="5.4.1 Definicion y analisis"),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 37, PDF p. 45",
)
def alpha_versus_size_sf(ctx: FigureContext):
    """PDF p. 45, fig. 37: <alpha> against N for scale-free networks at d = 2.

    Five sizes, N = 50 to 1000; at each, ALPHA_SIZE_REALISATIONS networks,
    each measured by propagating from every one of its nodes and fitting the
    first FIT_FIRST unsaturated instants.  The line is the mean, the band the middle half
    of the networks.
    """
    sizes = [50, 100, 250, 500, 1000]
    samples = _alpha_by_size(ctx, lambda n: barabasi_albert(n, links_per_new_node=2, rng=ctx.rng), sizes)
    ctx.realisations(ALPHA_SIZE_REALISATIONS)
    ctx.parameters(
        N=sizes, model="scale free (m=2)", density="d = 2", fit=f"first {FIT_FIRST} unsaturated instants",
        alpha=[round(float(v), 3) for v in samples.mean(axis=0)],
    )
    fig, ax = plt.subplots(figsize=(5.0, 3.2), layout="constrained")
    _alpha_panel(ax, sizes, samples)
    ax.set_xlabel("number of nodes")
    ax.set_ylabel(r"$\langle\alpha\rangle$")
    return fig


@figure(
    "T38-40",
    chapter=6,
    title="The factor alpha against size for ER, WS and NW",
    source=Source(Origin.THESIS_2011, pdf_pages="46", section="5.4.1 Definicion y analisis"),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis figs. 38, 39 and 40 (one printed table), PDF p. 46",
)
def alpha_versus_size_by_type(ctx: FigureContext):
    """PDF p. 46, figs. 38-40: the analysis of fig. 37 for ER, WS and NW.

    The thesis prints them as one table of three cells, so they stay one
    figure; each cell keeps a vertical scale of its own, as the printed ones
    do.  ER has d = 3 and WS d = 2; the Newman-Watts shortcuts go on top of
    the d = 2 ring, about 0.4 links per node more, so NW sits near d = 2.4.
    """
    sizes = [50, 100, 250, 500, 1000]
    kinds = (
        ("ER", "d=3", lambda n: erdos_renyi(n, 2 * 3.0 / (n - 1), rng=ctx.rng)),
        ("WS", "d=2", lambda n: watts_strogatz(n, 4, 0.1, rng=ctx.rng)),
        ("NW", r"d\approx 2.4", lambda n: newman_watts(n, 4, 0.1, rng=ctx.rng)),
    )
    ctx.realisations(ALPHA_SIZE_REALISATIONS)
    fig, axes = plt.subplots(1, 3, figsize=(5.05, 2.4), sharex=True, layout="constrained")
    record = {}
    for column, (ax, (kind, density, build)) in enumerate(zip(axes, kinds, strict=True)):
        samples = _alpha_by_size(ctx, build, sizes)
        record[kind] = [round(float(v), 3) for v in samples.mean(axis=0)]
        _alpha_panel(ax, sizes, samples)
        ax.set_xticklabels([str(n) for n in sizes], fontsize=7, rotation=45, ha="right")
        ax.set_title(f"{kind} (${density}$)", fontsize=9, fontweight="normal")
        panel_letter(ax, "abc"[column])
    axes[0].set_ylabel(r"$\langle\alpha\rangle$")
    fig.supxlabel("number of nodes", fontsize=9, color=INK["secondary"])
    ctx.parameters(N=sizes, fit=f"first {FIT_FIRST} unsaturated instants", **{f"alpha_{k}": v for k, v in record.items()})
    return fig


@figure(
    "T41",
    chapter=6,
    title="Alpha against link density, for four network sizes",
    source=Source(Origin.THESIS_2011, pdf_pages="46", section="5.4.1 Definicion y analisis"),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 41, PDF p. 46",
)
def alpha_versus_density(ctx: FigureContext):
    """PDF p. 46, fig. 41: *"Variacion de alpha maximo con la densidad"*.

    Erdos-Renyi networks of 50, 100, 150 and 200 nodes, with the density swept
    from about two links per node to about 25.  Every series rises steeply and
    then saturates on a plateau of its own, and the plateau is higher the
    larger the network -- which is what makes "the maximum over density" a
    well-defined number and lets the next two figures follow it against
    :math:`N`.

    The thesis's plateaux are 1.95, 2.30, 2.50 and 2.65; the reproduced ones
    land on the same values.

    Each point is the mean of ALPHA_REALISATIONS networks at that size and
    density.  From one network per point the four series of this reproduction
    came out as staircases that crossed one another repeatedly, so the count is
    raised until the ordering the panel shows -- the plateau higher the larger
    the network -- is resolved.
    """
    sizes = [50, 100, 150, 200]
    realisations = ALPHA_REALISATIONS
    densities = np.array([2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 20, 24], dtype=float)
    ctx.parameters(
        N=sizes,
        model="Erdos-Renyi",
        density=densities.tolist(),
        alpha="mean over every seed node, to-saturation window",
    )
    ctx.realisations(realisations)

    # N is an ordered quantity, not four names, so the four series take four
    # steps of the single-hue ramp: darker is larger, and the ordering of the
    # plateaux can be read off the ordering of the tones.
    ramp = [SEQUENTIAL[3], SEQUENTIAL[5], SEQUENTIAL[7], SEQUENTIAL[9]]

    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for i, n in enumerate(sizes):
        samples = np.full((realisations, densities.size), np.nan)
        for j, d in enumerate(densities):
            # The densities stop at N/4 links per node, the range figs. 41-43 use.
            if d > n / 4:
                continue
            for r in range(realisations):
                samples[r, j] = _mean_alpha(erdos_renyi(n, 2.0 * d / (n - 1), rng=ctx.rng))
        # Four series on one pair of axes, so the mean alone: four overlapping
        # bands at four steps of a single ramp would be four greys.
        values = mean_with_band(
            ax, densities, samples, color=ramp[i], band=False, marker="o", ms=4.5
        )
        # A series whose density cap stops it short is labelled above its last
        # point rather than beside it, where the taller series pass through.
        last = int(np.flatnonzero(np.isfinite(values))[-1])
        short = last < values.size - 1
        anchor = last - 1 if short else last
        ax.annotate(
            f"$N={n}$",
            xy=(densities[anchor], values[anchor]),
            xytext=(-2, 12) if short else (7, 0),
            textcoords="offset points",
            color=ramp[i],
            fontsize=8.5,
            ha="right" if short else "left",
            va="bottom" if short else "center",
            annotation_clip=False,
        )

    ax.set_xlabel("links per node")
    ax.set_ylabel(r"$\alpha$")
    ax.set_xlim(0, 30.5)
    ax.set_ylim(0.5, 3.0)
    ax.set_xticks([2, 5, 10, 15, 20, 24])
    ax.set_yticks(np.arange(0.5, 3.01, 0.5))
    # No internal title: the printed caption carries it.
    return fig


@figure(
    "T42",
    chapter=6,
    title="Maximum alpha against network size, and its logarithmic fit",
    source=Source(Origin.THESIS_2011, pdf_pages="47", section="5.4.1 Definicion y analisis"),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 42, PDF p. 47",
)
def alpha_max_versus_size(ctx: FigureContext):
    """PDF p. 47, fig. 42: the headline result, ``y = 0.4974ln(x) + 0.012``.

    Take the plateau of each series in fig. 41 -- the maximum of
    :math:`\\langle\\alpha\\rangle` over density -- and plot it against
    :math:`N`.  The thesis fits a logarithm and prints the coefficient 0.4974,
    very nearly one half.
    """
    sizes = [50, 100, 150, 200, 250, 500]
    ctx.parameters(
        N=sizes,
        model="Erdos-Renyi",
        density="swept to the maximum, capped at N/4 links per node",
    )

    peaks = [_alpha_max("ER", n, ctx.rng)[0] for n in sizes]
    slope, intercept = np.polyfit(np.log(sizes), peaks, 1)

    ctx.parameters(fit=f"y = {slope:.4f} ln(x) + {intercept:.4f}", alpha_max=[round(float(p), 3) for p in peaks])
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    grid = np.linspace(40, 760, 300)
    law = slope * np.log(grid) + intercept
    ax.plot(grid, law, color=INK["muted"], lw=1.0, zorder=1)
    ax.annotate(
        _law_label(slope, intercept),
        # Under the curve at its far end, which is the one region a rising
        # logarithm leaves empty; on the curve the label was struck through,
        # and at its end it ran past the axes.
        xy=(0.97, 0.62),
        xycoords="axes fraction",
        ha="right",
        va="center",
        fontsize=8,
        color=INK["secondary"],
    )
    ax.plot(sizes, peaks, ls="none", marker="o", ms=6, color=BASE, zorder=3)
    ax.annotate(
        # Not "\max_d": mathtext sets the subscript of \max as a limit *under*
        # the operator, and the stray d hung below the label on its own.
        r"the maximum $\alpha$ for each size",
        xy=(sizes[1], peaks[1]),
        xytext=(12, -16),
        textcoords="offset points",
        va="top",
        fontsize=8.5,
        color=BASE,
    )
    ax.set_xlabel("number of nodes")
    ax.set_ylabel(r"$\alpha$")
    ax.set_xlim(0, 800)
    ax.set_ylim(1.6, 3.6)
    ax.set_yticks(np.arange(1.6, 3.61, 0.4))
    # No internal title: the printed caption carries it.
    return fig


@figure(
    "T43",
    chapter=6,
    title="Four network types on the same logarithm",
    source=Source(Origin.THESIS_2011, pdf_pages="47", section="5.4.1 Definicion y analisis"),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 43, PDF p. 47",
)
def alpha_max_universality(ctx: FigureContext):
    """PDF p. 47, fig. 43: ER, WS, SF and NW on the fit made from ER alone.

    The thesis measures the maximum alpha for each type at the sizes it had to
    hand -- WS at 50, 100, 150 and 500 nodes, SF at 200 and 280, NW at 50, 200
    and 600 -- draws them over the ER logarithm of fig. 42, and concludes that
    the maximum speed depends on size and not on topology.  The reproduced
    points are drawn over the same curve.
    """
    plan = {
        "ER": [50, 100, 150, 200, 250, 500],
        "WS": [50, 100, 150, 500],
        "SF": [200, 280],
        "NW": [50, 200, 600],
    }
    ctx.parameters(
        sizes_by_type={k: list(v) for k, v in plan.items()},
        density="swept to the maximum, capped at N/4 links per node",
    )

    # The claim is that topology does not matter, so topology does not get four
    # hues.  ER is picked out in the accent because the law was fitted on ER
    # alone; the other three share the base hue and are told apart by marker
    # shape, which is what makes "they all land on the same curve" the thing the
    # eye sees first.
    styles = {
        "ER": dict(marker="o", ms=9, color=ACCENT),
        "WS": dict(marker="s", ms=7, color=BASE, mfc="none", mew=1.1),
        "SF": dict(marker="^", ms=8, color=BASE, mfc="none", mew=1.1),
        "NW": dict(marker="D", ms=4, color=BASE),
    }

    peaks = {kind: [_alpha_max(kind, n, ctx.rng)[0] for n in sizes] for kind, sizes in plan.items()}
    slope, intercept = np.polyfit(np.log(plan["ER"]), peaks["ER"], 1)
    ctx.parameters(
        fit_on_ER=f"y = {slope:.4f} ln(x) + {intercept:.4f}",
        alpha_max={kind: [round(float(p), 3) for p in v] for kind, v in peaks.items()},
    )

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    grid = np.linspace(40, 680, 300)
    ax.plot(grid, slope * np.log(grid) + intercept, color=INK["muted"], lw=1.0, zorder=1)
    ax.annotate(
        f"the curve is {_law_label(slope, intercept)},\nfitted on ER alone",
        xy=(0.52, 0.10),
        xycoords="axes fraction",
        fontsize=8,
        color=INK["secondary"],
    )

    handles = []
    for kind, sizes in plan.items():
        (line,) = ax.plot(sizes, peaks[kind], ls="none", label=kind, zorder=3, **styles[kind])
        handles.append(line)

    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=4,
        frameon=False,
        fontsize=8,
        handletextpad=0.2,
        columnspacing=1.1,
    )
    ax.set_xlabel("number of nodes")
    ax.set_ylabel(r"$\alpha$")
    ax.set_xlim(0, 700)
    ax.set_ylim(1.6, 3.5)
    ax.set_yticks(np.arange(1.6, 3.51, 0.4))
    # No internal title: the printed caption carries it.
    return fig


# ---------------------------------------------------------------------------
# 5.4.2 El concepto Cluster C de orden n -- figs. 44 to 47
# ---------------------------------------------------------------------------

#: The 3-cube of fig. 44, drawn as a square inside a square with the corners
#: joined -- exactly the printed picture.  Node 3 is the outer lower-left
#: corner the thesis starts its infection from.
_Q3_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]
_Q3_POS = np.array(
    [
        [-1.0, 1.0],
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, -1.0],
        [-0.42, 0.42],
        [0.42, 0.42],
        [0.42, -0.42],
        [-0.42, -0.42],
    ]
)


@figure(
    "T44",
    chapter=6,
    title="Clusters of order n found by propagation, on the hypercube",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="48",
        section="5.4.2 El concepto Cluster C de orden n",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 44 (panels a-d), PDF p. 48",
)
def cluster_orders_hypercube(ctx: FigureContext):
    """PDF p. 48, fig. 44: the worked example that defines :math:`C_m(n)`.

    A hypercube of eight nodes, drawn as a square inside a square with the
    corners joined, and an infection started at the outer lower-left corner.
    Four cells, headed :math:`t = 0, 1, 2, 3`.

    The mechanism is the whole idea of section 5.4.2.  A node reached at
    :math:`t+1` from *two different* nodes of instant *t* was reached by two
    distinct routes, which means those routes close a cycle -- a cluster of
    order *t*.  The thesis marks such a node in purple, circles it one instant
    early to show what is coming, and counts the purple nodes.

    The reported answer for this network is *"ningun cluster de orden 1, 3
    clusteres de orden 2 y un cluster de orden 3"*, and the panels reproduce
    it: no purple at :math:`t = 1`, three purple nodes at :math:`t = 2`, one at
    :math:`t = 3`.  Order 1 is empty because the cube has no triangles, which
    is exactly why the concept is not the classical cluster coefficient.
    """
    ctx.parameters(
        model="hypercube Q3",
        N=8,
        links=12,
        seed_node="outer lower-left corner",
        t=[0, 1, 2, 3],
        thesis_result="0 clusters of order 1, 3 of order 2, 1 of order 3",
    )

    g = Network.from_edges(8, _Q3_EDGES, name="hypercube")
    seed = 3
    levels, purple_nodes, purple_edges = _cluster_steps(g, seed)

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1))
    counts = []
    for ax, t in zip(axes, range(4), strict=True):
        counts.append(
            _cluster_panel(
                ax,
                g,
                _Q3_POS,
                levels,
                purple_nodes,
                purple_edges,
                t,
                circle_next=t in (1, 2),
                red_first=(t == 1),
            )
        )
        if t == 1:
            # "It is marked by the red dashed line": the cluster of order 2 that
            # closes at the red-circled node -- seed, outer upper-left, outer
            # upper-right, outer lower-right -- just inside the outer square.
            _cluster_outline(ax, _Q3_POS, (3, 0, 1, 2), colour=RED, linestyle=(0, (3, 2)))
        elif t == 3:
            # "The cluster in this case is marked by a black line": the cycle of
            # order 3 that closes at the purple node, as the thesis marks it.
            _cluster_outline(ax, _Q3_POS, (3, 0, 4, 5, 6, 7), colour=INK["primary"], linestyle="-")
        ax.set_title(f"$t={t}$", fontsize=9.5, loc="center", color=INK["primary"])
        panel_letter(ax, "abcd"[t])
        count = counts[-1]
        ax.annotate(
            "the seed" if t == 0 else f"{count} cluster{'' if count == 1 else 's'}\nof order {t}",
            xy=(0.5, -0.02),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=8,
            color=INK["secondary"],
        )

    # No internal title: the printed caption carries it.  The short titles
    # above identify the panels, which a single caption cannot.
    _state_key(fig, extra=(("closes a cluster", PURPLE),))
    return fig


def _fig45_network() -> tuple[Network, np.ndarray]:
    """The ring-with-a-pentagonal-core of fig. 45, and its printed layout.

    Read link by link off the printed drawing (PDF p. 49, panel d, where every
    link is drawn): fifteen nodes on an outer ring, numbered clockwise from the
    top; five forming a pentagon inside it, numbered clockwise from the top;
    each pentagon node joined by one spoke to the ring node beside it, every
    third one; and five chords across the ring, each joining a ring node to the
    one five places further round.  Twenty nodes, thirty links, and every node
    has three.
    """
    outer, inner = 15, 5
    edges = [(i, (i + 1) % outer) for i in range(outer)]
    edges += [(outer + j, outer + (j + 1) % inner) for j in range(inner)]
    edges += [(outer + j, 3 * j) for j in range(inner)]
    edges += [(3 * k + 2, (3 * k + 7) % outer) for k in range(inner)]

    angles_out = np.pi / 2 - 2 * np.pi * np.arange(outer) / outer
    angles_in = np.pi / 2 - 2 * np.pi * np.arange(inner) / inner
    pos = np.vstack(
        [
            np.column_stack([np.cos(angles_out), np.sin(angles_out)]),
            0.36 * np.column_stack([np.cos(angles_in), np.sin(angles_in)]),
        ]
    )
    return Network.from_edges(outer + inner, edges, name="fig. 45 network"), pos


@figure(
    "T45",
    chapter=6,
    title="Clusters of order n on a twenty-node network",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="49",
        section="5.4.2 El concepto Cluster C de orden n",
        note=(
            "The network is read link by link off the printed drawing: an "
            "outer ring of 15 nodes, a pentagon of 5 inside it, one spoke from "
            "each pentagon node to the ring, and five chords across the ring; "
            "30 links. The infection starts at the top node of the pentagon."
        ),
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 45 (panels a-d), PDF p. 49",
)
def cluster_orders_ring(ctx: FigureContext):
    """PDF p. 49, fig. 45: the second mechanism, a purple *link*.

    Fig. 44 counted purple nodes.  This example adds the other way a cluster
    can close: two nodes infected at the same instant *from different parents*
    that happen to be joined to each other.  The thesis draws that link in
    thick purple, and the dashed contour along the pentagon through the seed
    is the cluster it closes.

    Four cells headed :math:`t = 1 \\dots 4`, as printed: the link that closes
    the pentagon at :math:`t = 2` (*"dando lugar a 1 cluster"*), the four ring
    nodes reached by two routes at :math:`t = 3`, and the two links that close
    the last clusters at :math:`t = 4` (*"se forman 2 clusters"*).  Every one
    of these clusters passes through the seed.
    """
    ctx.parameters(
        N=20,
        links=30,
        model="outer ring of 15 with a pentagonal core of 5, read off the printed drawing",
        seed_node="top node of the pentagon",
        t=[1, 2, 3, 4],
        rule="two nodes infected at the same instant and joined to each other close a cluster across that link",
    )

    g, pos = _fig45_network()
    seed = 15
    levels, purple_nodes, purple_edges = _cluster_steps(g, seed)

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0))
    for ax, t in zip(axes, range(1, 5), strict=True):
        if t < len(levels):
            _cluster_panel(
                ax,
                g,
                pos,
                levels,
                purple_nodes,
                purple_edges,
                t,
                circle_next=(t == 1),
                arrows=(t == 1),
                purple_links=True,
                purple_dots=(t < len(levels) - 1),
            )
            if t == 2:
                # "there is a cluster marked by the black dashed line": the
                # pentagon through the seed (length 2n + 1 = 5) that the purple
                # link 17-18 closes.
                _cluster_outline(
                    ax,
                    pos,
                    (15, 16, 17, 18, 19),
                    colour=INK["primary"],
                    linestyle=(0, (3, 2)),
                    inset=0.07,
                    linewidth=0.9,
                )
        else:
            _cluster_panel(ax, g, pos, levels, purple_nodes, purple_edges, len(levels) - 1)
        # Which instant closes how many clusters is read off the caption, panel
        # by panel; the drawing shows the purple links and nodes that close them.
        ax.set_title(f"$t={t}$", fontsize=9.5, loc="center", color=INK["primary"])
        panel_letter(ax, "abcd"[t - 1])

    # One set of limits for the four panels: a mark outside one network would
    # otherwise widen that axes alone, printing its network smaller and its
    # title higher than the others.
    x0 = min(ax.get_xlim()[0] for ax in axes)
    x1 = max(ax.get_xlim()[1] for ax in axes)
    y0 = min(ax.get_ylim()[0] for ax in axes)
    y1 = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

    # No internal title: the printed caption carries it.  The short titles
    # above identify the panels, which a single caption cannot.
    _state_key(fig, extra=(("closes a cluster", PURPLE),))
    return fig


@figure(
    "T46",
    chapter=6,
    title="A cluster of order 1 is a triangle",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="49",
        section="5.4.2 El concepto Cluster C de orden n",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 46, PDF p. 49",
)
def order_one_triangle(ctx: FigureContext):
    """PDF p. 49, fig. 46: three nodes, three links, and the point of fig. 47.

    The classical coefficient counts the conglomerates where
    :math:`M_{i,j} = M_{i,k} = M_{j,k} = 1` -- that is, triangles -- and
    divides by :math:`k_i (k_i - 1)/2`.  The new definition,
    :math:`C_i(1) = 2\\,\\mathcal{E}_i(1)/(k_i (k_i - 1))`, counts the clusters
    of order 1 that the propagation finds instead.

    Drawing the triangle is the argument: a cluster of order 1 is a node
    reached at instant 2 from two neighbours of the seed, and two neighbours of
    the seed that are joined to each other *are* a triangle.  The two
    quantities are therefore the same object, which is what fig. 47 then checks
    node by node.
    """
    ctx.parameters(order=1, N=3, links=3, formula="C_i = 2 E_i / (k_i (k_i - 1))")

    pos = np.array([[-0.85, 0.75], [0.75, 0.0], [-0.35, -0.95]])

    # Three nodes drawn in three hues would say nothing.  Here the colours carry
    # the argument: i is the seed, its two links out are the reach of instant 1,
    # and the link between its neighbours -- purple, as everywhere else in this
    # section -- is the one that closes the cluster.
    # Three nodes and three links: a schematic, sized like one.
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    ax.add_collection(
        LineCollection([(pos[1], pos[0]), (pos[1], pos[2])], colors=PALE, linewidths=2.0, zorder=1)
    )
    ax.add_collection(LineCollection([(pos[0], pos[2])], colors=PURPLE, linewidths=3.0, zorder=2))
    ax.scatter(pos[[0, 2], 0], pos[[0, 2], 1], s=90, c=BASE, zorder=3, linewidths=0)
    ax.scatter(pos[1, 0], pos[1, 1], s=110, c=ACCENT, zorder=4, linewidths=0)

    for label, (x, y), (dx, dy) in zip(
        ("$k$", "$i$", "$j$"), pos, ((-16, 6), (12, 4), (-4, -18)), strict=True
    ):
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=11,
            style="italic",
            color=INK["primary"],
        )
    ax.annotate(
        "the seed",
        xy=pos[1],
        xytext=(12, -12),
        textcoords="offset points",
        fontsize=8,
        color=ACCENT,
    )
    ax.annotate(
        "the $j\\!-\\!k$ link\ncloses the cluster",
        xy=(0.5 * (pos[0, 0] + pos[2, 0]), 0.5 * (pos[0, 1] + pos[2, 1])),
        xytext=(-8, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=8,
        color=PURPLE,
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.margins(0.30)
    # No internal title: the printed caption carries it.
    return fig


@figure(
    "T47",
    chapter=6,
    title="The cluster coefficient node by node, two methods, four networks",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="50",
        section="5.4.2 El concepto Cluster C de orden n",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 47 (panels a-h), PDF p. 50",
)
def cluster_coefficient_by_node(ctx: FigureContext):
    """PDF p. 50, fig. 47: C per node by the method of section 2.3 and by propagation.

    A table of eight charts, as the thesis prints it: one row per network of
    100 nodes -- ER, an ordered ring where every node joins its four nearest,
    WS and scale free -- and one column per method, the classical coefficient
    of section 2.3 in red and the one found by propagation, from the clusters
    of order 1, in blue.  Each chart gives its mean; the largest difference
    between the two methods at any node is recorded.
    """
    n = 100
    rows = (
        ("ER", lambda: erdos_renyi(n, 0.085, rng=ctx.rng)),
        ("ordered", lambda: ring_lattice(n, 4)),
        ("WS", lambda: watts_strogatz(n, 4, 0.35, rng=ctx.rng)),
        ("scale free", lambda: barabasi_albert(n, links_per_new_node=2, rng=ctx.rng)),
    )
    fig, axes = plt.subplots(4, 2, figsize=(5.05, 5.6), sharex=True, layout="constrained")
    worst, means = 0.0, {}
    for k, ((left, right), (name, build)) in enumerate(zip(axes, rows, strict=True)):
        g = build()
        classical = local_clustering(g)
        propagated = local_cluster_coefficient_order1(g)
        worst = max(worst, float(np.max(np.abs(classical - propagated))))
        means[name] = round(float(classical.mean()), 4)
        for ax, values, colour, letter in (
            (left, classical, RED, "abcdefgh"[2 * k]),
            (right, propagated, BLUE, "abcdefgh"[2 * k + 1]),
        ):
            per_node(ax, values, color=colour)
            panel_letter(ax, letter)
            ax.annotate(
                f"mean {values.mean():.4f}", xy=(0.02, 0.97), xycoords="axes fraction",
                ha="left", va="top", fontsize=7.5, color=INK["secondary"],
                path_effects=[pe.withStroke(linewidth=2.4, foreground=INK["surface"])], zorder=5,
            )
        top = max(float(classical.max()), 0.05) * 1.25
        left.set_ylim(0, top)
        right.set_ylim(0, top)
        right.tick_params(labelleft=False)
        left.set_ylabel(f"{name}\n$C_i$", fontsize=9)
    axes[0, 0].set_title("method of Section 3.3", fontsize=9)
    axes[0, 1].set_title("by propagation", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("node $i$")
    ctx.parameters(N=n, networks=[r[0] for r in rows], means=means, largest_difference=worst)
    return fig


# ---------------------------------------------------------------------------
# 5.4.3 Infecciones: otra perspectiva -- figs. 48 to 53
# ---------------------------------------------------------------------------


def _thesis_city(rng):
    """The city of figs. 48-53, built to the two vectors printed on PDF p. 51.

    Ten types of place with counts ``[16 19 15 11 20 13 15 14 20 100]`` -- 243
    places, the last hundred being the houses, one per habitant -- and per-type
    contagion percentages ``[92 60 99 61 88 84 56 72 98 97]``.  Each of the 100
    habitants walks a closed route out of his house through three to five
    places and back.

    ``home_counts_toward_length=True`` is the literal reading of *"caminos
    aleatorios de cada habitante: minimo 3 lugares, maximo 5 lugares"*: three
    to five places in total, the house among them.  It matters.  Counting the
    house as extra gives routes of four to six places, the city saturates in
    three instants, and figs. 51-53 lose the two instants they are drawn over;
    the literal reading saturates in five, which is what the thesis prints.
    """
    return build_city(
        place_counts=THESIS_PLACE_COUNTS,
        n_habitants=100,
        min_places=3,
        max_places=5,
        contagion_percentages=THESIS_CONTAGION_PCT,
        home_counts_toward_length=True,
        rng=rng,
        name="thesis city, figs. 48-53",
    )


def _draw_city(ax, pos: np.ndarray, edges) -> None:
    """The routes, drawn as recessively as they can be and still be there.

    Four hundred chords over 243 places will win any contest with the places
    themselves unless they are held down, so they are a hairline of the pale
    link colour and the frame is switched off around the lot.
    """
    ax.add_collection(
        LineCollection(
            [(pos[u], pos[v]) for u, v in edges],
            colors=PALE,
            linewidths=0.4,
            alpha=0.85,
            zorder=1,
        )
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.margins(0.04)


def _city_positions(city, *, separate_homes: bool) -> np.ndarray:
    """Where the places are drawn.

    The thesis has no coordinates for its places -- the drawing is Maple's own
    scatter -- so the layout is chosen to make the two pictures it prints do
    what they are printed to do.  Figs. 48 and 49 are a dense square; fig. 50
    pulls the hundred houses away from the other 143 places so the bundle of
    routes between the two groups becomes visible.

    The generator is fixed rather than taken from the figure's context, so that
    every one of figs. 49-51 draws the same city in the same place.
    """
    rng = np.random.default_rng(THESIS_SEED + 1)
    is_home = np.zeros(city.n_places, dtype=bool)
    is_home[city.home_of] = True
    pos = rng.uniform(-1.0, 1.0, size=(city.n_places, 2))
    if separate_homes:
        pos[is_home] = pos[is_home] * 0.42 + np.array([0.78, 0.72])
        pos[~is_home] = pos[~is_home] * 0.60 + np.array([-0.70, -0.55])
    return pos


@figure(
    "T48",
    chapter=6,
    title="The city as a network of places",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="51-52",
        section="5.4.3 Infecciones: otra perspectiva",
        note="The place coordinates are not recoverable from the printed drawing; the layout is chosen, the structure is the thesis's.",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 48, PDF p. 52",
)
def place_network(ctx: FigureContext):
    """PDF p. 52, fig. 48: healthy places, blue routes, and a new model.

    Section 5.4.1 put people on the nodes and let the disease cross an
    acquaintance.  Section 5.4.3 throws that away: the nodes are *places* --
    schools, cafes, offices, transport, and one house per habitant -- and a
    link is drawn between two places when some habitant walks from one to the
    other.  Two people need never have met; the thesis's example is having used
    the same phone box.

    The vectors printed just above the figure are the model: place counts
    ``[16 19 15 11 20 13 15 14 20 100]`` and contagion percentages
    ``[92 60 99 61 88 84 56 72 98 97]``.  243 places, 100 habitants, routes of
    three to five places.
    """
    city = _thesis_city(ctx.rng)
    ctx.parameters(
        place_counts=list(THESIS_PLACE_COUNTS),
        contagion_pct=list(THESIS_CONTAGION_PCT),
        n_places=city.n_places,
        n_habitants=city.n_habitants,
        route_length="3 to 5 places, the house included",
    )

    pos = _city_positions(city, separate_homes=False)
    edges = city.place_edges()

    # A 243-node drawing of uniform dots is a hairball that says only "many".
    # The one thing the picture can say is which places are on many routes, so
    # the dot carries the degree and the busiest place is picked out and named.
    degree = np.zeros(city.n_places, dtype=int)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1
    busiest = int(np.argmax(degree))

    fig, ax = plt.subplots(figsize=(5.05, 5.45))
    _draw_city(ax, pos, edges)
    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        s=4.0 + 1.6 * degree,
        c=BASE,
        zorder=2,
        linewidths=0,
    )
    ax.scatter(
        pos[busiest, 0],
        pos[busiest, 1],
        s=60.0,
        facecolors="none",
        edgecolors=ACCENT,
        linewidths=1.4,
        zorder=4,
    )
    ax.annotate(
        f"dot size is the number of links at the place; ringed, the busiest place ({degree[busiest]} links)",
        xy=(0.0, -0.02),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=8,
        color=INK["muted"],
    )
    # No internal title: the printed caption carries it.
    return fig


@figure(
    "T49",
    chapter=6,
    title="The same city, with the houses picked out",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="52",
        section="5.4.3 Infecciones: otra perspectiva",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 49, PDF p. 52",
)
def place_network_by_type(ctx: FigureContext):
    """PDF p. 52, fig. 49: the same drawing, with the houses picked out.

    The thesis gives each of the ten types of place its own colour.  Nine of
    those hues carry nothing the reader can use -- the types are names, and
    which type a given dot belongs to is never referred to again -- while the
    tenth is the point of the figure: a hundred of the 243 places are somebody's
    home, every route starts and ends at one, and that is what fig. 50 then
    separates out.  So the houses take the light green the caption names, and
    the other nine types take hues kept clear of that green.
    """
    city = _thesis_city(ctx.rng)
    ctx.parameters(
        n_types=city.n_types,
        place_counts=list(THESIS_PLACE_COUNTS),
        houses=int(city.n_habitants),
    )

    pos = _city_positions(city, separate_homes=False)
    edges = city.place_edges()
    # This is the figure that gives each type of place its own colour, which is
    # what the sentence above it announces and what tells it from fig. 48.
    is_home = np.zeros(city.n_places, dtype=bool)
    is_home[city.home_of] = True
    home_type = int(city.place_type[city.home_of[0]])
    others = [t for t in range(city.n_types) if t != home_type]
    colour_of = {home_type: HOMES}
    colour_of.update(zip(others, PLACE_TYPES, strict=False))

    fig, ax = plt.subplots(figsize=(5.05, 5.2))
    _draw_city(ax, pos, edges)
    for t in others:
        of_type = city.place_type == t
        ax.scatter(
            pos[of_type, 0], pos[of_type, 1], s=13, c=colour_of[t], zorder=2, linewidths=0
        )
    # The houses last, and larger: they are 100 of the 243 places and the ends
    # of every route, and they are what the caption sends the reader to.
    ax.scatter(pos[is_home, 0], pos[is_home, 1], s=24, c=HOMES, zorder=3, linewidths=0)
    ax.annotate(
        f"the {int(is_home.sum())} houses",
        xy=(0.99, 0.01),
        xycoords="axes fraction",
        ha="right",
        fontsize=8.5,
        color=HOMES,
    )
    ax.annotate(
        f"{city.n_types} types of place",
        xy=(0.01, 0.01),
        xycoords="axes fraction",
        fontsize=8.5,
        color=INK["secondary"],
    )
    # No internal title: the printed caption carries it.
    return fig


@figure(
    "T50",
    chapter=6,
    title="The houses drawn apart from the rest of the city",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="52",
        section="5.4.3 Infecciones: otra perspectiva",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 50, PDF p. 52",
)
def homes_separated(ctx: FigureContext):
    """PDF p. 52, fig. 50: the same network, relaid so the houses sit apart.

    Nothing about the network changes -- this is a re-layout, not a re-model.
    What it shows is that the city is nearly bipartite in practice: a hundred
    houses on one side, the 143 shared places on the other, and a thick bundle
    of routes running between them.  Almost every link crosses the gap, because
    almost every step of a route goes from home out or from somewhere back
    home.

    It is also the layout figs. 51 uses, so that the infection can be watched
    arriving at the shared places and going home again.
    """
    city = _thesis_city(ctx.rng)
    pos = _city_positions(city, separate_homes=True)
    edges = city.place_edges()
    is_home = np.zeros(city.n_places, dtype=bool)
    is_home[city.home_of] = True
    crossing = sum(1 for u, v in edges if is_home[u] != is_home[v])
    ctx.parameters(
        houses=int(is_home.sum()),
        other_places=int((~is_home).sum()),
        links=len(edges),
        links_between_the_two_groups=crossing,
    )

    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    _draw_city(ax, pos, edges)
    ax.scatter(pos[~is_home, 0], pos[~is_home, 1], s=13, c=BASE, zorder=2, linewidths=0)
    ax.scatter(pos[is_home, 0], pos[is_home, 1], s=22, c=HOMES, zorder=3, linewidths=0)
    for name, colour, xy, ha in (
        (f"{int(is_home.sum())} houses", HOMES, (0.99, 0.99), "right"),
        (f"{int((~is_home).sum())} other places", BASE, (0.01, 0.03), "left"),
    ):
        ax.annotate(
            name,
            xy=xy,
            xycoords="axes fraction",
            ha=ha,
            va="top",
            fontsize=9,
            color=colour,
        )
    ax.annotate(
        f"{crossing} of the {len(edges)} links\ncross from one group to the other",
        xy=(0.02, 0.62),
        xycoords="axes fraction",
        fontsize=8,
        color=INK["secondary"],
    )
    # No internal title: the printed caption carries it.
    return fig


def _city_run(ctx: FigureContext):
    """The city, the habitant who is infected first, and the propagation.

    The seed habitant is drawn from the figure's own generator, so figs. 51, 52
    and 53 are three views of one run rather than three runs.
    """
    city = _thesis_city(ctx.rng)
    habitant = int(ctx.rng.integers(city.n_habitants))
    run = propagate_places(city, habitant)
    return city, habitant, run


@figure(
    "T51",
    chapter=6,
    title="The infection spreading through the city, t = 1 to 5",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="53",
        section="5.4.3 Infecciones: otra perspectiva",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 51 (panels a-e), PDF p. 53",
)
def city_infection_snapshots(ctx: FigureContext):
    """PDF p. 53, fig. 51: five cells, headed t = 1 ... 5, on the layout of fig. 50.

    One habitant picked at random is infected.  He leaves the infection at
    every place on his route at once -- not one link at a time, because the
    dynamics run on the incidence between habitants and places, not on the
    drawn place-graph.  Everybody who passes through one of those places is
    infected, and their own routes are contaminated in turn.

    A route is drawn red when the habitant who walks it is infected, blue while
    he is not; a place is red once it has been contaminated.  The red bundle
    grows from a single route to the whole city in five instants.

    The two mechanisms of section 5.4 look identical on the page and are not
    the same process: here two people who never met are infected at the same
    instant because they both use the same cafe.
    """
    city, habitant, run = _city_run(ctx)
    ctx.parameters(
        n_places=city.n_places,
        n_habitants=city.n_habitants,
        first_infected=habitant,
        t=[1, 2, 3, 4, 5],
        rule="an infected inhabitant contaminates every place on his route at once",
    )

    pos = _city_positions(city, separate_homes=True)
    walks = [
        (h, [(city.paths[h][i], city.paths[h][i + 1]) for i in range(len(city.paths[h]) - 1)])
        for h in range(city.n_habitants)
    ]

    fig = plt.figure(figsize=(5.05, 3.9))
    axes = panel_rows(fig, 5)
    for ax, t in zip(axes, range(1, 6), strict=True):
        healthy_segments, infected_segments = [], []
        for h, pairs in walks:
            target = infected_segments if run.infected_at[h] <= t else healthy_segments
            target.extend((pos[u], pos[v]) for u, v in pairs)
        for segments, colour, z in ((healthy_segments, BLUE, 1), (infected_segments, RED, 2)):
            if segments:
                ax.add_collection(
                    LineCollection(segments, colors=colour, linewidths=0.28, alpha=0.55, zorder=z)
                )
        contaminated = run.contaminated_at <= t
        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            s=3.2,
            c=np.where(contaminated, RED, HEALTHY),
            zorder=3,
            linewidths=0,
        )
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.margins(0.03)
        ax.set_title(f"$t={t}$", fontsize=9, loc="center", color=INK["primary"])
        panel_letter(ax, "abcde"[t - 1])

    # No internal title: the printed caption carries it.  The short titles
    # above identify the panels, which a single caption cannot.
    _state_key(fig)
    return fig


@figure(
    "T52",
    chapter=6,
    title="Healthy and infected inhabitants in the city",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="53",
        section="5.4.3 Infecciones: otra perspectiva",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 52, PDF p. 53",
)
def city_counts(ctx: FigureContext):
    """PDF p. 53, fig. 52: *Sano* in blue and *Infectado* in red, over t = 0 ... 5.

    The two curves saturate at exactly 100, which is how one can tell what the
    thesis was counting: there are 243 places and 100 habitants, so the axis is
    people.  The crossing falls between :math:`t = 2` and :math:`t = 3` in the
    printed figure and in this one.
    """
    city, habitant, run = _city_run(ctx)
    ctx.parameters(
        n_habitants=city.n_habitants,
        first_infected=habitant,
        t=f"0..{run.steps}",
        counted="inhabitants, not places",
    )

    t = np.arange(run.people_counts.size)
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    ax.plot(t, run.healthy_counts, color=HEALTHY, lw=1.8, marker="o", ms=4.5)
    ax.plot(t, run.people_counts, color=RED, lw=1.8, marker="o", ms=4.5)
    # Direct labels rather than a legend, each in open space beside its own
    # curve: "healthy" right of its fall, "infected" under its plateau.
    for name, colour, xy in (
        ("healthy", HEALTHY, (3.15, 24.0)),
        ("infected", RED, (3.9, 88.0)),
    ):
        ax.annotate(name, xy=xy, color=colour, fontsize=9, va="center")

    # The limits come before the reference line. reference_line reads them to
    # decide whether its label goes above the line, and above the line is where
    # the label stays clear of the healthy curve, which starts at 99.
    ax.set_ylim(-4, 114)
    reference_line(ax, float(city.n_habitants), f"all {city.n_habitants} inhabitants", side="left")
    ax.set_xlabel("$t$")
    ax.set_ylabel("number of inhabitants")
    ax.set_xticks(t)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.15, 5.15)
    return fig


@figure(
    "T53",
    chapter=6,
    title="The infected fraction in the city",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="53",
        section="5.4.3 Infecciones: otra perspectiva",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 53, PDF p. 53",
)
def city_fraction(ctx: FigureContext):
    """PDF p. 53, fig. 53: the same run as a fraction, t = 1 ... 5.

    The thesis prints this beside fig. 52 for the same reason it printed fig.
    34 beside fig. 33: the shape is what matters, and the shape here is not the
    same as in section 5.4.1.  The place model has no plateau of slow growth at
    the start -- the seed habitant contaminates his whole route in one move, so
    one move from the seed already carries a tenth of the city -- and it rises
    almost as a straight line to saturation instead of an exponential.  Two
    models of the same word, with different curves.
    """
    city, habitant, run = _city_run(ctx)
    ctx.parameters(
        n_habitants=city.n_habitants,
        first_infected=habitant,
        t=f"1..{run.steps}",
    )

    t = np.arange(1, run.people_counts.size)
    fraction = run.people_curve[1:]

    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    ax.plot(t, fraction, color=RED, lw=1.8, marker="o", ms=4.5)
    # The point of the figure is where the curve *starts*, not where it ends:
    # The axis starts at t = 1.  The seeding instant t = 0, where the seed
    # alone is infected, is carried by fig. 52; what this figure shows is the
    # jump his whole route makes in one move.
    reference_line(ax, 1.0, "every inhabitant", side="left")
    ax.set_xlabel("$t$")
    ax.set_ylabel("fraction of inhabitants infected")
    ax.set_xticks(t)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0.85, 5.2)
    return fig
