"""Section 5.6 of the thesis, figure by figure: networks made of communities.

PDF pp. 64-77.  Where :mod:`chapter_09` re-casts the interbank results as
analyses -- box plots, sweeps, comparisons -- this module reproduces the
*pages*: every numbered figure of section 5.6 gets a counterpart with the
thesis's own panel layout, its own parameters and its own quantities, named
after the thesis figure it replicates (``T69`` for Fig. 69, and so on).

The figures split across two book chapters, as the thesis splits across two
subjects: 5.6.1-5.6.3 (PDF pp. 64-67) is the community *network*, chapter 8;
5.6.4 onwards (PDF pp. 68-77) is the interbank debt system built on it,
chapter 9.  They live in one module because they share the drawing helpers --
a community layout and a degree-class layout -- that the section needs and
that no earlier chapter did.

    python -m netmotion.experiments.cli T69 T70 T71
"""

from __future__ import annotations

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator, PercentFormatter

from ..finance import InterbankNetwork, bank_capital, four_state_cascade
from ..finance.bank_growth import strategy_growth
from ..graph import Network
from ..metrics import mean_degree
from ..networks import barabasi_albert, erdos_renyi, newman_watts, watts_strogatz
from ..networks.communities import CommunityNetwork, community_scale_free
from ..propagation import propagate
from ..provenance import Origin, Source
from ..visualization import INK, SEQUENTIAL, SERIES, draw_network, spring_layout
from ..visualization.charts import panel_letter, panel_rows, per_node
from .chapter_06_extra import _early_alpha, _first_alpha, _thesis_city
from .registry import FigureContext, figure

THESIS_SEED = 20110930

#: The degree window the trendline of thesis fig. 74.c is drawn over.  That
#: panel plots nine markers, at k = 3, 4, 5, 6, 7, 8, 9, 10, 11; the k = 2 bin
#: of its own histogram is not among them, which is the practice printed on
#: p. 29 -- *"se han omitido algunos datos para realizar la regresion"*.
#: Refitting the nine plotted points by unweighted least squares on log-log
#: axes returns y = 4.9074 k^-2.3326 against the printed y = 4.8685 k^-2.33,
#: so this is the window the printed exponent was measured in, and it is the
#: window the reproduction measures its own exponent in.
FIG74_FIT_RANGE = (3, 11)


# ---------------------------------------------------------------------------
# Drawing helpers.  Section 5.6 introduces two layouts the earlier chapters
# never needed: nodes grouped onto one small circle per community, and nodes
# grouped onto one small circle per degree class.  Both are the point of the
# figures rather than decoration, so they are built here rather than borrowed.
# ---------------------------------------------------------------------------


def _circle(centre: np.ndarray, radius: float, count: int) -> np.ndarray:
    """*count* points evenly spaced on a circle, as the thesis places nodes."""
    if count == 1:
        return centre[None, :]
    angles = 2.0 * np.pi * np.arange(count) / count
    return centre[None, :] + radius * np.column_stack([np.cos(angles), np.sin(angles)])


def _contracted(network: Network, group: np.ndarray, n_groups: int) -> Network:
    """The graph on the groups: a link where any node of one joins another."""
    matrix = np.zeros((n_groups, n_groups))
    for i, j, _ in network.edges():
        a, b = int(group[i]), int(group[j])
        if a != b:
            matrix[a, b] = matrix[b, a] = 1.0
    return Network(matrix, name="contracted")


def _group_layout(
    network: Network,
    group: np.ndarray,
    *,
    rng,
    n_groups: int | None = None,
    tightness: float = 0.16,
) -> np.ndarray:
    """One small circle per group, the circles placed where the groups pull.

    The thesis's own words for figs. 69-72: *"se ha situado en la
    representacion los diferentes tipos de lugares sobre diferentes
    circunferencias situadas en diferentes posiciones"*.  Where the circles go
    is not stated; here the contracted graph -- one node per group -- is laid
    out by the same force-directed rule used for the grown networks of chapter
    4, so that groups which lend to each other sit near each other.
    """
    group = np.asarray(group, dtype=int)
    n_groups = int(group.max()) + 1 if n_groups is None else n_groups
    centres = spring_layout(_contracted(network, group, n_groups), rng=rng)
    radii = np.array([tightness * np.sqrt(max((group == c).sum(), 1)) for c in range(n_groups)])
    centres = _spread(centres, radii)

    pos = np.zeros((network.n, 2))
    for c in range(n_groups):
        members = np.flatnonzero(group == c)
        if members.size == 0:
            continue
        pos[members] = _circle(centres[c], radii[c], members.size)
    return pos


def _spread(
    centres: np.ndarray, radii: np.ndarray, pad: float = 1.2, rounds: int = 200
) -> np.ndarray:
    """Push the circle centres apart until no two circles overlap.

    A force-directed layout knows nothing of how many nodes a community holds,
    so a community of a hundred houses lands on top of one of eleven cafes.
    Scaling the whole arrangement would separate them but blow the drawing
    apart; relaxing only the pairs that actually collide keeps the placement
    compact, which is how the thesis draws these figures.
    """
    centres = np.array(centres, dtype=float)
    if centres.shape[0] < 2:
        return centres
    for _ in range(rounds):
        moved = False
        for a in range(centres.shape[0]):
            for b in range(a + 1, centres.shape[0]):
                delta = centres[b] - centres[a]
                gap = float(np.linalg.norm(delta))
                want = pad * (radii[a] + radii[b])
                if gap >= want:
                    continue
                if gap < 1e-9:
                    delta, gap = np.array([1.0, 0.0]), 1e-9
                push = 0.5 * (want - gap) * delta / gap
                centres[a] -= push
                centres[b] += push
                moved = True
        if not moved:
            break
    return centres


def _degree_class_layout(
    network: Network, *, rng
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group the nodes by their degree *k*, one circle per class.

    Thesis PDF p. 64: *"los nodos se separan segun su grado k y se situan
    sobre diferentes circunferencias situadas en diferentes posiciones"*.  A
    class holding one node draws as one node; a class holding twenty draws as a
    dense ring, and reading how many rings there are and how full they are is
    exactly the visual test the thesis proposes for how ordered a network is.

    Returns the node coordinates, the class index of every node, the centre of
    every class circle and the degree each class stands for.
    """
    k = network.degrees()
    classes = np.unique(k)
    label = np.searchsorted(classes, k)
    counts = np.array([(label == c).sum() for c in range(classes.size)])

    # The thesis does not say where it puts the sub-circles, only that they are
    # "situadas en diferentes posiciones". Ordering them by degree around a ring
    # keeps the reading it wants from the drawing -- which classes are populous,
    # which hold a single node -- legible however many classes there turn out
    # to be.
    centres = _circle(np.zeros(2), 1.0, classes.size)
    radii = np.array([0.085 * np.sqrt(max(int(count), 1)) for count in counts])
    centres = _spread(centres, radii, pad=1.35)

    pos = np.zeros((network.n, 2))
    for c in range(classes.size):
        members = np.flatnonzero(label == c)
        pos[members] = _circle(centres[c], radii[c], members.size)
    return pos, label, centres, classes


def _class_ramp(classes: np.ndarray) -> np.ndarray:
    """One colour per degree class, along the sequential ramp.

    Degree is an ordered quantity, so the classes get an ordered colour: pale
    for the sparsely connected circles, dark for the hubs.  A categorical hue
    per class would say the classes are unrelated names, and would run out of
    distinguishable colours by the eighth one besides.
    """
    if classes.size == 1:
        return np.array([SEQUENTIAL[-3]])
    span = np.linspace(0.15, 1.0, classes.size)
    return np.array([SEQUENTIAL[int(round(f * (len(SEQUENTIAL) - 1)))] for f in span])


def _label_classes(ax, centres: np.ndarray, classes: np.ndarray, counts: np.ndarray) -> None:
    """Write the degree each sub-circle stands for at its centre.

    The nodes of a populous class sit on its circumference, so the middle is
    empty and the label costs nothing.  A class of one or two nodes has no empty
    middle, so its label steps aside rather than sitting on top of the only node
    it is there to name.
    """
    for centre, k, count in zip(centres, classes, counts, strict=True):
        crowded = int(count) <= 4
        if crowded:
            # Small classes sit on the outside of the ring of circles, and their
            # centres bunch there, so a label always pushed straight down lands
            # on the next class's label.  Pushing it out along its own radius
            # instead fans them apart, because the centres already fan apart.
            radial = np.asarray(centre, dtype=float)
            norm = float(np.hypot(*radial)) or 1.0
            offset = tuple(22.0 * radial / norm)
            ha = "left" if radial[0] >= 0 else "right"
            va = "bottom" if radial[1] >= 0 else "top"
        else:
            offset = (0.0, 0.0)
            ha, va = "center", "center"
        ax.annotate(
            f"$k={int(k)}$",
            xy=(centre[0], centre[1]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.0,
            color=INK["secondary"],
            ha=ha,
            va=va,
            zorder=4,
            path_effects=[pe.withStroke(linewidth=2.4, foreground=INK["surface"])],
            arrowprops=(
                {"arrowstyle": "-", "lw": 0.6, "color": INK["muted"], "shrinkB": 5}
                if crowded
                else None
            ),
        )


def _draw_edges_by_group(
    ax,
    network: Network,
    pos: np.ndarray,
    group: np.ndarray,
    *,
    width: float = 0.5,
    alpha: float = 0.7,
) -> None:
    """Links inside a group faint, links that leave one picked out.

    PDF p. 65 gives every sub-circle its own colour of link -- *"Las ramas se
    han coloreado de forma distinta segun de que subcirculo a que subcirculo
    esten dirigidas"*.

    What the whole section is about is the *ratio* of interior links to
    exterior ones, so that is the distinction the colour draws here: the base
    hue inside a group, the accent for the few links that cross between.
    """
    inside, across = [], []
    for i, j, _ in network.edges():
        (inside if int(group[i]) == int(group[j]) else across).append((pos[i], pos[j]))
    if inside:
        ax.add_collection(
            LineCollection(inside, colors=SERIES[0], linewidths=width, alpha=alpha, zorder=1)
        )
    if across:
        ax.add_collection(
            LineCollection(
                across,
                colors=SERIES[1],
                linewidths=width * 1.5,
                alpha=min(1.0, alpha * 1.35),
                zorder=2,
            )
        )


def _draw_degree_classes(
    ax,
    network: Network,
    pos: np.ndarray,
    label: np.ndarray,
    classes: np.ndarray,
    centres: np.ndarray,
    *,
    node_size: float = 9.0,
) -> None:
    """The degree-class drawing: nodes on the ramp, links out of the way.

    Colouring the links by class as well turns the middle of the drawing into a
    plaid of a dozen hues through which nothing can be read.  The reading the
    thesis wants -- how many circles there are, how full each one is -- is
    carried entirely by the nodes, so the links drop back to a hairline and the
    colour is spent on the one thing that is ordered: the degree itself.
    """
    segments = [(pos[i], pos[j]) for i, j, _ in network.edges()]
    if segments:
        ax.add_collection(
            LineCollection(segments, colors=[to_rgba(SERIES[0], 0.16)], linewidths=0.3, zorder=1)
        )
    ramp = _class_ramp(classes)
    draw_network(
        network, ax, layout=pos, node_color=ramp[label], node_size=node_size, edge_alpha=0.0
    )
    counts = np.array([(label == c).sum() for c in range(classes.size)])
    _label_classes(ax, centres, classes, counts)


def _random_community_network(
    rng,
    *,
    n_nodes: int = 140,
    n_communities: int = 20,
    interior: int = 291,
    exterior: int = 40,
) -> CommunityNetwork:
    """A community network built directly, not grown: thesis fig. 72.

    PDF p. 65 describes it in full -- *"20 comunidades con un numero aleatorio
    de nodos en cada una"*, 291 links inside communities, 40 between them, 140
    nodes -- and those four numbers fix the network completely up to the draw.
    They also fix its mean degree: 2 x 331 / 140 = 4.728571429, which is the
    number printed above fig. 73.

    Community sizes are drawn with a floor of five nodes, because twenty
    communities sharing 140 nodes must still offer 291 distinct interior pairs.
    """
    floor = 5
    sizes = np.full(n_communities, floor, dtype=int)
    sizes += rng.multinomial(
        n_nodes - floor * n_communities, np.full(n_communities, 1.0 / n_communities)
    )
    community = np.repeat(np.arange(n_communities), sizes)

    inside = [
        (int(a), int(b))
        for c in range(n_communities)
        for idx, a in enumerate(np.flatnonzero(community == c))
        for b in np.flatnonzero(community == c)[idx + 1 :]
    ]
    if len(inside) < interior:
        raise RuntimeError(f"only {len(inside)} interior pairs available for {interior} links")
    chosen = rng.choice(len(inside), size=interior, replace=False)
    edges = [inside[i] for i in chosen]

    across: set[tuple[int, int]] = set()
    while len(across) < exterior:
        a, b = (int(x) for x in rng.integers(0, n_nodes, size=2))
        if a != b and community[a] != community[b]:
            across.add((min(a, b), max(a, b)))
    edges.extend(sorted(across))

    net = Network.from_edges(n_nodes, edges, name="random communities")
    return CommunityNetwork(network=net, community=community)


def _per_node_degree(ax, cn: CommunityNetwork, *, boundaries: bool = True) -> float:
    """The thesis's *grado k por nodo*: k against node index, one stem each.

    Not a degree distribution -- figs. 73 and 74.b give the degree of node 1,
    then node 2, then node 3, in index order, and the shape of that trace is
    what the thesis reads.  A polyline joining consecutive nodes draws a slope
    between two numbers that have nothing to do with each other; stems on a
    common baseline plot the same values and let the profile through.

    ``boundaries`` draws the blue verticals of fig. 73, which mark where one
    community ends and the next begins.  They are hairlines behind the stems,
    so twenty of them rule the axes without cutting through the data.  They
    only mean anything when the nodes are numbered
    community by community, which is true of the network built directly in
    fig. 72 and false of the network *grown* in fig. 74.a, where each arrival
    joins whichever community the kernel picked.  Fig. 74.b has none for
    exactly that reason.
    """
    k = cn.network.degrees()
    mean_k = float(mean_degree(cn.network))

    if boundaries:
        cuts = np.concatenate(([0], np.flatnonzero(np.diff(cn.community)) + 1, [k.size]))
        for cut in cuts:
            ax.axvline(cut + 0.5, color=SERIES[0], lw=0.7, alpha=0.85, zorder=0)

    # Where the verticals are drawn the stems step out of the blue, so that a
    # community boundary can never be read as a node of very high degree.
    per_node(ax, k, color=INK["secondary"] if boundaries else SERIES[0])
    ax.set_xlabel("node index")
    ax.set_ylabel("$k$")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(False)
    return mean_k


def _arrow(fig, x: float = 0.5, y: float = 0.5) -> None:
    """The thesis joins its before/after pairs with a large double arrow."""
    fig.text(x, y, r"$\Rightarrow$", fontsize=20, color=INK["secondary"], ha="center", va="center")


# ---------------------------------------------------------------------------
# 5.6.1 Definiciones, ejemplos y caracteristicas (PDF pp. 64-65)
# ---------------------------------------------------------------------------


@figure(
    "T69",
    chapter=8,
    title="The infection network redrawn as communities of place types",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="64",
        section="5.6.1 Definiciones, ejemplos y caracteristicas",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 69, PDF p. 64",
)
def places_as_communities(ctx: FigureContext):
    """PDF p. 64, fig. 69: the same network drawn twice.

    Left, the place network of section 5.4.3 as it comes out of the city --
    243 places joined by the steps its 100 habitants take between them, an
    undifferentiated blob. Right, the identical network with each *type of
    place* put on its own circle. Nothing has changed but the placement, and
    the community structure that was invisible on the left is the whole
    picture on the right.

    The thesis draws nine or ten circles; the city has ten types, the last of
    them the 100 houses, which is the large circle.
    """
    ctx.parameters(
        places=243,
        habitants=100,
        types=10,
        counts="thesis list, PDF p. 51",
        route_length="3 to 5 places, the house included",
    )

    # The place network of section 5.4.3, built exactly as it is there: the
    # house counts among the three to five places of a route, which is the
    # reading fig. 51 is drawn under.
    city = _thesis_city(ctx.rng)
    g = city.place_graph()
    kind = city.place_type

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.8))

    never = int((g.degrees() == 0).sum())
    # A force-directed layout has nothing to hold an isolated node with, so it
    # throws every one of them to the far corners of the frame and the places
    # that are actually joined collapse into a thumbnail in the middle -- the
    # panel then shows the exceptions at full size and the network not at all.
    # The isolated places are put on a ring just outside the connected cloud
    # instead: still visibly apart, still counted in the title, and no longer
    # setting the scale of the drawing.
    layout = spring_layout(g, rng=ctx.rng)
    lonely = g.degrees() == 0
    if lonely.any() and (~lonely).any():
        core = layout[~lonely]
        centre = np.median(core, axis=0)
        offset = core - centre
        dist = np.linalg.norm(offset, axis=1)
        reach = float(np.percentile(dist, 95))
        # A few places joined only among themselves are thrown far out by the
        # layout; drawn at the edge of the cloud they no longer set the scale.
        far = dist > 1.5 * reach
        if far.any():
            offset[far] *= 1.1 * reach / float(dist[far].max())
        layout[~lonely] = centre + offset
        radius = 1.25 * reach
        angles = np.linspace(0.0, 2.0 * np.pi, int(lonely.sum()), endpoint=False)
        layout[lonely] = centre + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    draw_network(
        g,
        left,
        layout=layout,
        node_color=np.where(lonely, INK["muted"], SERIES[0]),
        node_size=7.0,
        edge_width=0.25,
        edge_color=SERIES[0],
        edge_alpha=0.25,
    )
    left.set_title(
        f"the {g.n} places\n({never} in grey, on no route)",
        fontsize=9.5,
        fontweight="normal",
    )

    # One hue for the places, the accent spent on the houses alone -- the last
    # type, and the only circle the paragraph singles out.  Ten types in ten
    # colours would ask the reader to decode a key that carries nothing, and
    # nearly every link here crosses a type, so colouring the links by type
    # would just be the same hairball in ten colours instead of one.
    houses = city.n_types - 1
    palette = np.where(kind == houses, SERIES[1], SERIES[0])
    pos = _group_layout(g, kind, rng=ctx.rng, tightness=0.10)
    segments = [(pos[i], pos[j]) for i, j, _ in g.edges()]
    right.add_collection(
        LineCollection(segments, colors=[to_rgba(SERIES[0], 0.16)], linewidths=0.22, zorder=1)
    )
    draw_network(g, right, layout=pos, node_color=palette, node_size=7.0, edge_alpha=0.0)
    right.set_title(
        f"one circle per type of place\n(the {int((kind == houses).sum())} houses in orange)",
        fontsize=9.5,
        fontweight="normal",
    )

    _arrow(fig, 0.5, 0.52)
    return fig


@figure(
    "T70",
    chapter=8,
    title="A Newman-Watts network redrawn as communities of equal degree",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="64",
        section="5.6.1 Definiciones, ejemplos y caracteristicas",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 70, PDF p. 64",
)
def wn_by_degree_class(ctx: FigureContext):
    """PDF p. 64, fig. 70: the same WN network on the ring, then by degree.

    The thesis's second construction takes *any* network and builds an
    alternative drawing in which each node is placed according to its degree
    k, so that each sub-circle holds the nodes of one degree. It says what
    this is for: *"esta inspeccion visual permite observar el grado de
    ordenamiento existente en la red y visualizar la existencia de nodos con
    mayor importancia"*.

    On a Newman-Watts network the answer is: very ordered. Shortcuts are added
    without removing ring links, so almost every node keeps the ring degree
    and only the few that received a shortcut stand apart.

    The thesis prints no N, k or p for this figure; N = 100, k = 4, p = 0.1 is
    used, the parameters of its own Newman-Watts examples in section 4.3.
    """
    n, k, p = 100, 4, 0.1
    ctx.parameters(N=n, k=k, p=p, model="Newman-Watts", note="N, k, p not printed in the thesis")

    g = newman_watts(n, k, p, rng=ctx.rng)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.8))

    draw_network(
        g,
        left,
        node_color=SERIES[0],
        node_size=9.0,
        edge_width=0.3,
        edge_color=SERIES[0],
        edge_alpha=0.22,
    )
    # Split across two lines: at the narrow per-panel width the KDP trim gives
    # each half, a single line carrying "on the ring: N=..., k=..., p=..." ran
    # past its own axes, which is exactly what fit_to_print_width cannot widen
    # around.
    left.set_title(f"on the ring\n$N={n}$, $k={k}$, $p={p}$", fontsize=9.5, fontweight="normal")

    pos, label, centres, classes = _degree_class_layout(g, rng=ctx.rng)
    _draw_degree_classes(right, g, pos, label, classes, centres, node_size=9.0)
    counts = np.array([(label == c).sum() for c in range(classes.size)])
    right.set_title(
        f"by degree\n{classes.size} degree groups, max {counts.max()}/{n} nodes",
        fontsize=9.5,
        fontweight="normal",
    )

    _arrow(fig, 0.5, 0.52)
    return fig


@figure(
    "T71",
    chapter=8,
    title="An Erdos-Renyi network redrawn as communities of equal degree",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="65",
        section="5.6.1 Definiciones, ejemplos y caracteristicas",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 71, PDF p. 65",
)
def er_by_degree_class(ctx: FigureContext):
    """PDF p. 65, fig. 71: the same construction on a random network.

    The contrast with fig. 70 is the point of the pair. The Newman-Watts
    network collapses onto two or three degree classes; the Erdos-Renyi
    network of the same size spreads over eight or nine, because a binomial
    degree distribution has no preferred value beyond its mean. Reading the
    two drawings side by side is the thesis's visual test for order.

    As for fig. 70 the thesis prints no parameters; N = 100 and p = 0.1 give
    the dense disc it draws.
    """
    n, p = 100, 0.06
    ctx.parameters(N=n, p=p, model="Erdos-Renyi", note="N and p not printed in the thesis")

    g = erdos_renyi(n, p, rng=ctx.rng)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.8))

    draw_network(
        g,
        left,
        node_color=SERIES[0],
        node_size=9.0,
        edge_width=0.28,
        edge_color=SERIES[0],
        edge_alpha=0.18,
    )
    left.set_title(f"on the circle: $N={n}$, $p={p}$", fontsize=9.5, fontweight="normal")

    pos, label, centres, classes = _degree_class_layout(g, rng=ctx.rng)
    _draw_degree_classes(right, g, pos, label, classes, centres, node_size=9.0)
    right.set_title(
        f"by degree: {classes.size} circles, $k$ from {classes.min()} to {classes.max()}",
        fontsize=9.5,
        fontweight="normal",
    )

    _arrow(fig, 0.5, 0.52)
    return fig


@figure(
    "T72",
    chapter=8,
    title="A community network built directly, not grown",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="65",
        section="5.6.1 Definiciones, ejemplos y caracteristicas",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 72, PDF p. 65",
)
def random_community_network(ctx: FigureContext):
    """PDF p. 65, fig. 72: 20 communities, 140 nodes, 291 + 40 links.

    Before growing a community network the thesis simply builds one: pick a
    number of communities, give each a random number of nodes, then scatter a
    given number of links inside communities and a much smaller number between
    them. The figure is the definition of a community made visible -- *"la
    cantidad de uniones sera mas densa entre nodos de la misma comunidad que
    entre nodos de diferentes comunidades"* -- with the ratio 291 : 40.
    """
    ctx.parameters(N=140, communities=20, interior_links=291, exterior_links=40)

    cn = _random_community_network(ctx.rng)
    pos = _group_layout(cn.network, cn.community, rng=ctx.rng, tightness=0.09)

    fig, ax = plt.subplots(figsize=(5.05, 5.05))
    _draw_edges_by_group(ax, cn.network, pos, cn.community, width=0.4, alpha=0.55)
    draw_network(
        cn.network,
        ax,
        layout=pos,
        node_color=SERIES[0],
        node_size=11.0,
        edge_alpha=0.0,
    )
    return fig


@figure(
    "T73",
    chapter=8,
    title="Degree per node, with the communities marked",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="65",
        section="5.6.1 Definiciones, ejemplos y caracteristicas",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 73, PDF p. 65",
)
def degree_per_node(ctx: FigureContext):
    """PDF p. 65, fig. 73: the degree of node 1, then node 2, then node 3.

    Not a degree distribution: the thesis plots k against the node *index*,
    one point per node, and rules a blue vertical line wherever one community
    ends and the next begins. Between two blue lines the trace is the degree
    profile of a single community, and the figure's message is that the
    profile repeats -- every community looks like every other, because they
    were all built by the same random rule.

    The thesis prints the mean degree above the axes as 4.728571429, which is
    2 x 331 / 140 exactly: 291 + 40 = 331 links over 140 nodes. The number
    therefore follows from the link counts of fig. 72, and the reproduction
    returns the same value.
    """
    ctx.parameters(N=140, communities=20, interior_links=291, exterior_links=40)

    cn = _random_community_network(ctx.rng)
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    _per_node_degree(ax, cn)

    return fig


# ---------------------------------------------------------------------------
# 5.6.2 Red Scale Free de comunidades (PDF p. 66)
# ---------------------------------------------------------------------------


@figure(
    "T74",
    chapter=8,
    title="The scale-free network of communities, its degrees and its P(k)",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="66",
        section="5.6.2 Red Scale Free de comunidades",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis figs. 74.a, 74.b and 74.c, PDF p. 66",
)
def scale_free_communities(ctx: FigureContext):
    """PDF p. 66, fig. 74.a-c: Cn = 10, n0 = 3, r = 1, n = 2, m = 1, t = 100.

    All three panels of the thesis's figure, in its own order: the grown
    network, the degree of every node against its index, and P(k) on log-log
    axes with a power-law trendline fitted the way the original was, by least
    squares in the logarithms.

    The power law of panel c is fitted over k = 3 to 11, the points the
    thesis's panel plots -- see FIG74_FIT_RANGE.  The density d and the
    exponent gamma are recorded; the text gives them.
    """
    ctx.parameters(
        Cn=10,
        n0=3,
        r=1,
        n=2,
        m=1,
        t=100,
        fit_range=f"k = {FIG74_FIT_RANGE[0]} to {FIG74_FIT_RANGE[1]}, the points fig. 74.c plots",
    )

    cn = community_scale_free(rng=ctx.rng)
    k = cn.network.degrees()

    fig = plt.figure(figsize=(5.05, 5.6), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.5, 1.0))
    net_ax = fig.add_subplot(grid[0, :])
    deg_ax = fig.add_subplot(grid[1, 0])
    pk_ax = fig.add_subplot(grid[1, 1])
    for ax, letter in ((net_ax, "a"), (deg_ax, "b"), (pk_ax, "c")):
        panel_letter(ax, letter)

    pos = _cluster_layout(cn.network, cn.community, rng=ctx.rng)
    _draw_edges_by_group(net_ax, cn.network, pos, cn.community, width=0.3, alpha=0.4)
    draw_network(
        cn.network,
        net_ax,
        layout=pos,
        node_color=SERIES[0],
        node_size=3.0 + 1.6 * k,
        edge_alpha=0.0,
    )
    net_ax.set_title(
        f"$N={cn.network.n}$, {cn.n_communities} communities",
        fontsize=8.5,
        fontweight="normal",
    )

    _per_node_degree(deg_ax, cn, boundaries=False)
    deg_ax.set_title(
        "degree $k$ per node",
        fontsize=8.5,
        fontweight="normal",
    )

    values, counts = np.unique(k, return_counts=True)
    pk = counts / k.size
    lo, hi = FIG74_FIT_RANGE
    window = (values >= lo) & (values <= hi)
    slope, intercept = np.polyfit(np.log(values[window]), np.log(pk[window]), 1)
    prefactor = float(np.exp(intercept))
    ctx.parameters(density=round(cn.density, 2), gamma=round(-float(slope), 2))
    fit = prefactor * values[window].astype(float) ** slope
    pk_ax.loglog(values[window], fit, color=INK["muted"], lw=1.0, zorder=2)
    # Every bin is plotted; the bins the trendline is read from carry the full
    # hue and the ones outside the window are drawn open, so the aperture the
    # exponent was measured through is visible rather than merely stated.
    pk_ax.loglog(
        values[window], pk[window], ls="none", marker="o", ms=4.5, color=SERIES[0], zorder=3
    )
    if (~window).any():
        pk_ax.loglog(
            values[~window],
            pk[~window],
            ls="none",
            marker="o",
            ms=4.5,
            mfc="none",
            mec=SERIES[0],
            mew=0.9,
            alpha=0.55,
            zorder=3,
        )
    pk_ax.set_xlabel("$k$")
    pk_ax.set_ylabel("$P(k)$")
    pk_ax.set_title(
        f"a power law, fitted over $k = {lo}$ to ${hi}$",
        fontsize=8.5,
        fontweight="normal",
    )
    pk_ax.grid(True, which="major", axis="both")
    # A log axis spanning k = 2 to 18 gets exactly one decade tick, so the
    # panel came out with a single "10^1" under it and no way to read a value
    # off it.  The degrees the data actually reaches are ticked instead.
    pk_ticks = [t for t in (2, 3, 5, 10, 20, 50) if t <= 1.4 * float(np.max(k))]
    pk_ax.set_xticks(pk_ticks)
    pk_ax.set_xticklabels([str(t) for t in pk_ticks])
    pk_ax.minorticks_off()


    return fig


# ---------------------------------------------------------------------------
# 5.6.3 Propagacion de infeccion en red de comunidades (PDF p. 67)
# ---------------------------------------------------------------------------


def _curves_from_every_node(network: Network, *, length: int) -> np.ndarray:
    """p(t) from every node in turn, padded to a common length by holding.

    The thesis's own procedure -- *"cuando esta se inicia en cada nodo de la
    red"* -- gives one curve per node; a curve that saturates early is drawn
    flat to the right edge, which is how its figures show them.
    """
    curves = np.ones((network.n, length))
    for seed in range(network.n):
        curve = propagate(network, seed).curve
        take = min(curve.size, length)
        curves[seed, :take] = curve[:take]
        curves[seed, take:] = curve[-1]
    return curves


@figure(
    "T75",
    chapter=8,
    title="Infection from every node of the community network",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="67",
        section="5.6.3 Propagacion de infeccion en red de comunidades",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 75, PDF p. 67",
)
def propagation_on_communities(ctx: FigureContext):
    """PDF p. 67, fig. 75: one curve per starting node, on the network of 74.a.

    The thesis reads the *dispersion* of the bundle, not its height: every
    curve ends at 1, but they take between six and eleven steps to get there,
    and that spread is what a community structure does to an infection. A node
    inside a well-connected community infects it in three steps and then waits
    at the border.

    The thesis reports <alpha> = 1.10 for this bundle, alpha being the
    exponential rate of section 5.4.1.
    """
    ctx.parameters(Cn=10, n0=3, r=1, n=2, m=1, t=100, seeds="every node")

    cn = community_scale_free(rng=ctx.rng)
    length = 12
    curves = _curves_from_every_node(cn.network, length=length)
    ctx.parameters(mean_alpha=round(_early_alpha(cn.network), 2))

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    t = np.arange(1, length + 1)
    ax.plot(t, curves.T, color=SERIES[0], lw=0.7, alpha=0.28, zorder=2)
    ax.plot(t, curves.mean(axis=0), color=SERIES[1], lw=1.8, zorder=3)

    # The dispersion is what the thesis reads, so it is what gets named: the
    # first and the last starting node to finish the network.
    done = np.array([float(np.argmax(c >= 1.0)) + 1.0 for c in curves])
    ax.annotate(
        "mean of all\nstarting nodes",
        xy=(t[-1], curves.mean(axis=0)[-1]),
        xytext=(-6, -10),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8,
        color=SERIES[1],
    )
    ctx.parameters(full_after=f"{done.min():.0f} to {done.max():.0f} steps")
    ax.set_xlabel("$t$")
    ax.set_xlim(1, length)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("infected")

    return fig


@figure(
    "T76",
    chapter=8,
    title="Whether propagation tells a network of communities from one without",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="67",
        section="5.6.3 Propagacion de infeccion en red de comunidades",
    ),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis figs. 76.a and 76.b, PDF p. 67",
)
def communities_versus_none(ctx: FigureContext):
    """PDF p. 67, fig. 76.a-b: N = 430 either way, with and without communities.

    This is the thesis turning its own propagation model into a measuring
    instrument. Two scale-free networks are matched on everything it can match
    them on -- N = 430, gamma near 2.3, about two links per node -- and differ
    only in whether the nodes are grouped into communities. Its figure shows
    the left bundle visibly more dispersed, and concludes that the dispersion
    of propagation curves detects community structure.

    Left, the scale-free network of communities grown to t = 400; right, a
    scale-free network of 430 nodes without communities.  The density and the
    exponent gamma (fitted over k = 3 to 20) of each are recorded, with the
    spread of the instant at which each starting node has infected half the
    network: the width of the bundle the text reads.
    """
    length = 12
    ctx.parameters(N=430, Cn=10, n0=3, t=400, m_ba=2, seeds="every node")

    cn = community_scale_free(rng=ctx.rng, steps=400)
    plain = barabasi_albert(430, links_per_new_node=2, rng=ctx.rng)

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True, sharex=True)
    panel_letter(left, "a")
    panel_letter(right, "b")
    t = np.arange(1, length + 1)

    for ax, net, label in (
        (left, cn.network, "grown in communities"),
        (right, plain, "grown without them"),
    ):
        curves = _curves_from_every_node(net, length=length)
        ax.plot(t, curves.T, color=SERIES[0], lw=0.6, alpha=0.2, zorder=2)
        ax.plot(t, curves.mean(axis=0), color=SERIES[1], lw=1.8, zorder=3)
        if ax is right:
            ax.annotate("mean of all\nstarting nodes", xy=(t[-1], curves.mean(axis=0)[-1]),
                        xytext=(-6, -10), textcoords="offset points", ha="right", va="top",
                        fontsize=8, color=SERIES[1])

        # The quantity the thesis reads off the bundle is its width. Measured
        # here as the spread in the instant at which each starting node has
        # infected half the network -- the part of the curve where a community
        # boundary either has or has not been crossed yet.
        half = np.array([float(np.argmax(c >= 0.5)) + 1.0 for c in curves])
        ax.set_xlim(1, length)
        ax.set_ylim(0, 1.02)
        k = net.degrees()
        values, counts = np.unique(k, return_counts=True)
        fit = (values >= 3) & (values <= 20)
        gamma = -float(np.polyfit(np.log(values[fit]), np.log(counts[fit] / k.size), 1)[0])
        # The panel is named; the measurements go in the caption.  Three lines
        # of parameters set wider than the panel they belong to ran into the
        # title of the panel beside them.
        ax.set_title(
            label,
            fontsize=8.5,
        )
        ctx.parameters(
            **{
                f"{label}_density": round(net.n_edges / net.n, 2),
                f"{label}_half_infected_t": f"{half.min():.0f}-{half.max():.0f}",
                f"{label}_half_infected_sd": round(float(half.std()), 2),
                f"{label}_gamma_k3_to_20": round(gamma, 2),
                f"{label}_half_infected_sd_all": round(float(half.std()), 3),
            }
        )

    left.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    left.xaxis.set_major_locator(MaxNLocator(integer=True))  # t counts instants; the panels share x
    left.set_ylabel("infected")
    fig.supxlabel("$t$", fontsize=9.5, color=INK["secondary"])

    return fig


# ---------------------------------------------------------------------------
# 5.6.4 Red de deuda del sistema bancario (PDF pp. 68-77).  A community is a
# country, a node is a bank, and the weight on a link is the debt along it.
# Every figure below is a run of the package's own models: the debt network
# of netmotion.finance.interbank, the growth by strategies of
# netmotion.finance.bank_growth, and the financial virus of
# netmotion.finance.contagion, started as the text starts it.
# ---------------------------------------------------------------------------

#: The three balances of a bank, in the colours the caption of fig. 77 gives
#: them: brown for one that owes more than it lends, green for one that lends
#: more than it owes, purple for one that lends as much as it owes.
DEBT_BALANCE = {
    "owes": ("#8c5a2b", "owes more than it lends"),
    "lends": ("#1baf7a", "lends more than it owes"),
    "balanced": ("#8e44ad", "lends as much as it owes"),
}

#: The colours the text of fig. 82 gives the links -- green between two banks
#: still standing, blue between a standing bank and a fallen one, red between
#: two fallen -- and the red of a fallen bank.
LINK_STANDING, LINK_EXPOSED, LINK_FALLEN = SERIES[5], SERIES[0], SERIES[7]
BANK_FALLEN = SERIES[7]


#: The reserves and the reaction threshold of the financial virus: a bank in
#: debt keeps eta per cent of what it owes as reserves, and a bank reacts --
#: calls in what its debtors owe it, weakest link first -- once it has lost r
#: per cent of its money.  The values of the 2012 article.
RESERVES_ETA, REACTION_R = 20.0, 30.0


def _virus(network: Network, seed: int, capital: np.ndarray):
    """The financial virus from *seed*, with reserves and reactions."""
    return four_state_cascade(
        network, seed, capital, reserves=RESERVES_ETA, reaction_threshold=REACTION_R
    )


def _balance(capital: np.ndarray) -> np.ndarray:
    """``"owes"``, ``"lends"`` or ``"balanced"`` for each bank, by the sign of D."""
    return np.where(capital < 0, "owes", np.where(capital > 0, "lends", "balanced"))


def _bank_sizes(capital: np.ndarray, *, largest: float = 320.0) -> np.ndarray:
    """Marker areas proportional to the size of each bank's debt |D|."""
    size = np.abs(capital)
    return 5.0 + largest * size / max(float(size.max()), 1.0)


def _cluster_layout(
    network: Network, group: np.ndarray, *, rng, spacing: float = 1.3, spread: float = 0.2
) -> np.ndarray:
    """Each country a compact cluster, laid out by its own links, the clusters apart.

    The countries are placed where their links pull them -- a force-directed
    layout of the graph with one node per country -- and pushed apart until no
    two overlap; inside each, the banks are laid out by the links between
    them, so that the large banks sit at the centre of their country and the
    small ones around them, rather than on a ring.
    """
    group = np.asarray(group, dtype=int)
    n_groups = int(group.max()) + 1
    centres = spring_layout(_contracted(network, group, n_groups), rng=rng)
    sizes = np.bincount(group, minlength=n_groups)
    radii = spread * np.sqrt(np.maximum(sizes, 1))
    centres = _spread(centres, radii, pad=spacing)
    pos = np.zeros((network.n, 2))
    for c in range(n_groups):
        members = np.flatnonzero(group == c)
        if members.size == 0:
            continue
        if members.size == 1:
            pos[members] = centres[c]
            continue
        local = spring_layout(network.subgraph(members), rng=rng, iterations=120)
        local = local - local.mean(axis=0)
        reach = float(np.abs(local).max()) or 1.0
        pos[members] = centres[c] + radii[c] * local / reach
    return pos


def _debt_network(rng, *, steps: int = 100) -> tuple[InterbankNetwork, np.ndarray]:
    """The network of fig. 74.a with the debts on its links, and its layout.

    Built from the figure's seed in the order fig. 74 builds it, so it is the
    very network fig. 74.a draws.
    """
    cn = community_scale_free(rng=rng, steps=steps)
    pos = _cluster_layout(cn.network, cn.community, rng=rng)
    return InterbankNetwork.from_community_network(cn), pos


def _strategy_network(rng, *, steps: int = 100, restricted: bool = False, spread: float = 0.2):
    """A network grown by the seven strategies, with its debts and its layout."""
    cn, _ = strategy_growth(
        rng=rng, steps=steps, restricted=restricted, restriction_sides="both"
    )
    pos = _cluster_layout(cn.network, cn.community, rng=rng, spread=spread)
    return InterbankNetwork.from_community_network(cn), pos


def _draw_banks(
    ax,
    network: Network,
    pos: np.ndarray,
    capital: np.ndarray,
    *,
    fallen: np.ndarray | None = None,
    link_width: float = 0.3,
    largest: float = 320.0,
    ringed: np.ndarray | None = None,
    link_alpha: float = 0.28,
    sizes: np.ndarray | None = None,
    giants_on_top: bool = False,
) -> None:
    """A debt network: each bank coloured by its balance, sized by its debt.

    Without *fallen*, the links are drawn pale.  With it -- a moment of a
    cascade -- each link takes the colour of fig. 82 for the state of its two
    banks, the blue of the front drawn last and heaviest, and each fallen bank
    is drawn red.
    """
    if fallen is None:
        segments = [(pos[i], pos[j]) for i, j, _ in network.edges()]
        ax.add_collection(
            LineCollection(
                segments, colors=[to_rgba(SERIES[0], link_alpha)], linewidths=link_width, zorder=1
            )
        )
    else:
        groups: dict[int, list] = {0: [], 1: [], 2: []}
        for i, j, _ in network.edges():
            groups[int(fallen[i]) + int(fallen[j])].append((pos[i], pos[j]))
        for hit, colour, alpha, scale in (
            (0, LINK_STANDING, 0.45, 1.0),
            (2, LINK_FALLEN, 0.55, 1.0),
            (1, LINK_EXPOSED, 0.95, 2.0),
        ):
            if groups[hit]:
                ax.add_collection(
                    LineCollection(
                        groups[hit],
                        colors=[to_rgba(colour, alpha)],
                        linewidths=link_width * scale,
                        zorder=1 + (hit == 1),
                    )
                )
    colours = np.array([DEBT_BALANCE[b][0] for b in _balance(capital)], dtype=object)
    if fallen is not None:
        colours[fallen] = BANK_FALLEN
    # Largest drawn first, so that a small bank on a large one stays visible.
    area = _bank_sizes(capital, largest=largest) if sizes is None else np.asarray(sizes)
    order = np.argsort(area if giants_on_top else -area)
    if fallen is not None:
        # The fallen banks last, so that a bank's own country cannot hide it.
        order = order[np.argsort(fallen[order], kind="stable")]
    ax.scatter(
        pos[order, 0],
        pos[order, 1],
        s=area[order],
        c=list(colours[order]),
        zorder=3,
        linewidths=0.35,
        edgecolors="white",
    )
    if ringed is not None and len(ringed):
        ax.scatter(
            pos[ringed, 0],
            pos[ringed, 1],
            s=(np.sqrt(area[ringed]) + 5.0) ** 2,
            facecolors="none",
            edgecolors=INK["primary"],
            linewidths=1.1,
            zorder=4,
        )
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.margins(0.04)


def _debt_key(
    fig, capital: np.ndarray, *, cascade: bool = False, ncol: int | None = None, ringed: bool = False
) -> None:
    """The colours of the drawing, named as the captions name them, above it."""
    present = [b for b in DEBT_BALANCE if b in set(_balance(capital).tolist())]
    handles = [
        plt.Line2D([], [], ls="none", marker="o", ms=5.5, color=DEBT_BALANCE[b][0],
                   label=DEBT_BALANCE[b][1])
        for b in present
    ]
    if ringed:
        handles.append(
            plt.Line2D([], [], ls="none", marker="o", ms=8, mfc="none", mec=INK["primary"],
                       mew=1.1, label="too big to fail")
        )
    if cascade:
        handles += [
            plt.Line2D([], [], ls="none", marker="o", ms=5.5, color=BANK_FALLEN, label="fallen"),
            plt.Line2D([], [], color=LINK_STANDING, lw=1.6, label="both standing"),
            plt.Line2D([], [], color=LINK_EXPOSED, lw=1.6, label="one fallen"),
            plt.Line2D([], [], color=LINK_FALLEN, lw=1.6, label="both fallen"),
        ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=ncol or 2,
        frameon=False,
        fontsize=7.5,
        handletextpad=0.3,
        columnspacing=1.0,
    )


def _fallen_at(run, t: int) -> np.ndarray:
    """The banks fallen at instant *t*, counted as the thesis counts.

    At t = 0 no bank has yet been hit; at t = 1 the first one falls; from then
    on each instant is one step of the cascade, and the last state holds.
    """
    if t <= 0:
        return np.zeros_like(run.crashed_history[0])
    return run.crashed_history[min(t - 1, len(run.crashed_history) - 1)]


def _cascade_curves(network: Network, capital: np.ndarray, *, length: int) -> np.ndarray:
    """The fraction of banks fallen against t, from every bank in turn."""
    curves = np.zeros((network.n, length))
    for seed in range(network.n):
        history = _virus(network, seed, capital.copy()).history
        take = min(history.size, length)
        curves[seed, :take] = history[:take]
        curves[seed, take:] = history[-1]
    return curves


def _plain_curves(network: Network, *, length: int) -> np.ndarray:
    """The infection of section 5.4.1, without weights, from every node in turn."""
    curves = np.ones((network.n, length))
    for seed in range(network.n):
        curve = propagate(network, seed).curve
        take = min(curve.size, length)
        curves[seed, :take] = curve[:take]
        curves[seed, take:] = curve[-1]
    return curves


def _cascade_alpha(curves: np.ndarray) -> float:
    """The mean factor alpha of a bundle of cascade curves.

    As in section 5.4.1, alpha is the rate of an exponential fitted by least
    squares to ln p against t over the first FIT_FIRST instants of each curve,
    and averaged over every starting bank, as for fig. 5.54.  A bank whose
    failure spreads to no one has a flat curve, and its alpha of 0 counts in
    the mean.
    """
    return float(np.nanmean([_first_alpha(curve) for curve in curves]))


def _curve_axes(ax, length: int) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(1, length)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("$t$")
    ax.set_ylabel("nodes infected")


@figure(
    "T77",
    chapter=8,
    title="The debt network, each bank coloured by its balance",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="68",
        section="5.6.4 Red de deuda del sistema bancario",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 77, PDF p. 68",
)
def debt_network(ctx: FigureContext):
    """PDF p. 68, fig. 77: the network of fig. 74.a, now carrying debts.

    Nothing about the network has changed; what has been added is a debt on
    every link, W_ij from the degrees of its two banks, and the balance D_i of
    every bank, which sets its colour -- brown if it owes more than it lends,
    green if it lends more than it owes, purple if the two are equal -- and
    the size it is drawn at.
    """
    ib, pos = _debt_network(ctx.rng)
    d = ib.capital
    ctx.parameters(
        Cn=10, n0=3, r=1, n=2, m=1, t=100,
        banks=ib.n_banks,
        owe=int((d < 0).sum()), lend=int((d > 0).sum()), balanced=int((d == 0).sum()),
    )
    fig, ax = plt.subplots(figsize=(5.05, 4.2), layout="constrained")
    _draw_banks(ax, ib.network, pos, d)
    _debt_key(fig, d)
    return fig


@figure(
    "T78",
    chapter=8,
    title="The virus from every bank of the debt network",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="70",
        section="5.6.4 Red de deuda del sistema bancario",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 78, PDF p. 70",
)
def financial_virus_curves(ctx: FigureContext):
    """PDF p. 70, fig. 78: one infection curve per bank the virus starts from.

    Every bank of fig. 77 is attacked in turn and the fraction of banks fallen
    is drawn against t.  Each curve stops where the network absorbs the
    contagion, and where that is depends on the bank it starts at.
    """
    length = 41
    ib, _ = _debt_network(ctx.rng)
    curves = _cascade_curves(ib.network, ib.capital, length=length)
    finals = curves[:, -1]
    ctx.parameters(
        seeds="every bank",
        alpha=round(_cascade_alpha(curves), 2),
        final_min=round(float(finals.min()), 3),
        final_max=round(float(finals.max()), 3),
        below_20pct=round(float(np.mean(finals < 0.2)), 3),
    )
    fig, ax = plt.subplots(figsize=(5.0, 3.2), layout="constrained")
    t = np.arange(1, length + 1)
    ax.plot(t, curves.T, color=SERIES[7], lw=0.8, alpha=0.35)
    _curve_axes(ax, length)
    return fig


def _four_types(rng, n: int):
    """The four basic networks of section 5.4.1 at d = 2 links per node."""
    return (
        ("Erdős–Rényi", lambda: erdos_renyi(n, 2 * 2.0 / (n - 1), rng=rng)),
        ("Watts–Strogatz", lambda: watts_strogatz(n, 4, 0.1, rng=rng)),
        ("Newman–Watts", lambda: newman_watts(n, 4, 0.1, rng=rng)),
        ("scale free", lambda: barabasi_albert(n, links_per_new_node=2, rng=rng)),
    )


@figure(
    "T79",
    chapter=8,
    title="The virus with and without weights, on four types of network",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="71",
        section="5.6.4 Red de deuda del sistema bancario",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 79 (panels a-d), PDF p. 71",
)
def weighted_table(ctx: FigureContext):
    """PDF p. 71, fig. 79: the infection from every node, with and without weights.

    One chart per type of network -- ER, SW, NW and scale free, 100 nodes and
    about two links per node -- each holding, for ten networks of the type,
    one curve per seed node twice over: in blue the infection of section 5.4.1
    over the network without weights, in red the financial virus over the
    same network with the debts on its links.  Each chart gives the mean
    percentage of nodes that survive the virus, the table the text reads.
    """
    n, realisations, length = 100, 10, 41
    ctx.parameters(N=n, d="~2 links/node", realisations=realisations, seeds="every node")
    fig, axes = plt.subplots(2, 2, figsize=(5.05, 4.6), sharex=True, sharey=True,
                             layout="constrained")
    t = np.arange(1, length + 1)
    survival = {}
    for k, (ax, (name, build)) in enumerate(zip(axes.ravel(), _four_types(ctx.rng, n), strict=True)):
        means = []
        for _ in range(realisations):
            g = build()
            ax.plot(t, _plain_curves(g, length=length).T, color=SERIES[0], lw=0.35, alpha=0.08)
            weighted = _cascade_curves(g, bank_capital(g), length=length)
            ax.plot(t, weighted.T, color=SERIES[7], lw=0.35, alpha=0.10)
            means.append(1.0 - float(weighted[:, -1].mean()))
        survival[name] = 100.0 * float(np.mean(means))
        ax.set_title(name, fontsize=9)
        ax.set_title(f"survive: {survival[name]:.1f}%", loc="right", fontsize=8,
                     fontweight="normal", color=INK["primary"])
        panel_letter(ax, "abcd"[k])
        ax.set_xlim(1, length)
        ax.set_ylim(0, 1.02)
    ctx.parameters(**{f"survive_{name}": round(v, 1) for name, v in survival.items()})
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    for ax in axes[:, 0]:
        ax.set_ylabel("nodes infected")
    for ax in axes[-1]:
        ax.set_xlabel("$t$")
    fig.legend(
        handles=[
            plt.Line2D([], [], color=SERIES[0], lw=1.6, label="without weights"),
            plt.Line2D([], [], color=SERIES[7], lw=1.6, label="with weights"),
        ],
        loc="outside upper center", ncol=2, frameon=False, fontsize=8,
    )
    return fig


def _abc_configuration(with_b: bool) -> tuple[Network, np.ndarray, dict[int, str]]:
    """The two configurations of fig. 80: A and C joined through B, or directly.

    A has eight smaller banks around it and C three; B, when present, sits
    between them with only those two links.  The numbers are the smallest for
    which the text's account holds in the model: A's fall takes B, which is in
    debt, and C absorbs the loss of B; without B, the loss of A is larger than
    C's balance and C falls, and with it the banks around it.
    """
    a_leaves, c_leaves = 8, 3
    edges, positions, names = [], {0: (0.0, 0.0), 1: (2.4, 0.0)}, {0: "A", 1: "C"}
    nxt = 2
    if with_b:
        positions[2], names[2] = (1.2, 0.0), "B"
        edges += [(0, 2), (2, 1)]
        nxt = 3
    else:
        edges.append((0, 1))
    for i in range(a_leaves):
        angle = np.pi * (0.62 + 0.76 * i / (a_leaves - 1))
        positions[nxt] = (0.9 * np.cos(angle), 0.9 * np.sin(angle))
        edges.append((0, nxt))
        nxt += 1
    for i in range(c_leaves):
        angle = np.pi * (-0.34 + 0.68 * i / (c_leaves - 1))
        positions[nxt] = (2.4 + 0.8 * np.cos(angle), 0.8 * np.sin(angle))
        edges.append((1, nxt))
        nxt += 1
    g = Network.from_edges(nxt, edges)
    return g, np.array([positions[i] for i in range(nxt)]), names


@figure(
    "T80",
    chapter=8,
    title="What the bank in between does",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="72",
        section="5.6.4 Red de deuda del sistema bancario",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis figs. 80.a and 80.b, PDF p. 72",
)
def absorption_configurations(ctx: FigureContext):
    """PDF p. 72, fig. 80: A falls, with the bank B between A and C and without it.

    Both panels show the end of the cascade the fall of A starts, run by the
    model of the text.  In a, A takes B with it and C cushions the fall of B,
    so the contagion stops; in b, B is gone, A is joined to C directly, C
    cannot cushion the fall of A and falls, and so do the banks joined to it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(5.05, 2.3), layout="constrained")
    outcome = {}
    for ax, with_b, letter in zip(axes, (True, False), "ab", strict=True):
        g, pos, names = _abc_configuration(with_b)
        d = bank_capital(g)
        run = _virus(g, 0, d.copy())
        fallen = run.crashed_history[-1]
        outcome[letter] = {names[i]: bool(fallen[i]) for i in names}
        _draw_banks(ax, g, pos, d, fallen=fallen, link_width=1.2, largest=320.0)
        for node, name in names.items():
            ax.annotate(name, xy=pos[node], xytext=(0, -13), textcoords="offset points",
                        ha="center", va="top", fontsize=9, color=INK["primary"],
                        weight="semibold")
        ax.set_ylim(-1.15, 1.05)
        panel_letter(ax, letter)
    # The account the text gives has to be the account the model gives.
    assert outcome["a"] == {"A": True, "B": True, "C": False}, outcome
    assert outcome["b"] == {"A": True, "C": True}, outcome
    ctx.parameters(A_leaves=8, C_leaves=3, outcome=str(outcome))
    return fig


@figure(
    "T81",
    chapter=8,
    title="The network grown by strategies, and the virus on it",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="74",
        section="5.6.4 Red de deuda del sistema bancario",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis figs. 81.a-c, PDF p. 74",
)
def strategy_network(ctx: FigureContext):
    """PDF p. 74, fig. 81: the network grown by strategies, the virus on it, P(k).

    Cn = 10, n0 = 3, r = 1, n = 2, m = 1, t = 100, each arriving bank given one
    of the seven strategies at random.  a is the network, its banks coloured
    and sized as in fig. 77; b is the infection from every bank, without
    weights in blue and as a financial virus in red; c is P(k).
    """
    length = 24
    ib, pos = _strategy_network(ctx.rng)
    g, d = ib.network, ib.capital
    k = g.degrees()
    plain = _plain_curves(g, length=length)
    weighted = _cascade_curves(g, d, length=length)
    largest = _virus(g, ib.largest_bank(), d.copy())
    reference = community_scale_free(rng=np.random.default_rng(THESIS_SEED))
    ctx.parameters(
        Cn=10, n0=3, r=1, n=2, m=1, t=100, strategies="all seven, at random",
        banks=g.n, density=round(g.n_edges / g.n, 2),
        scale_free_density=round(reference.network.n_edges / reference.network.n, 2),
        below_20pct=round(float(np.mean(weighted[:, -1] < 0.2)), 3),
        largest_bank_fallen=round(largest.crashed_fraction, 3),
    )

    fig = plt.figure(figsize=(5.05, 5.0), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.25, 1.0))
    net = fig.add_subplot(grid[0, :])
    curves = fig.add_subplot(grid[1, 0])
    pk = fig.add_subplot(grid[1, 1])

    _draw_banks(net, g, pos, d, largest=110.0)
    t = np.arange(1, length + 1)
    curves.plot(t, plain.T, color=SERIES[0], lw=0.5, alpha=0.25)
    curves.plot(t, weighted.T, color=SERIES[7], lw=0.5, alpha=0.3)
    _curve_axes(curves, length)

    values, counts = np.unique(k, return_counts=True)
    pk.bar(values, counts / k.size, width=0.8, color=SERIES[0])
    pk.set_xlabel("$k$")
    pk.set_ylabel("$P(k)$")
    for ax, letter in zip((net, curves, pk), "abc", strict=True):
        panel_letter(ax, letter)
    # One key above the figure, in two rows: the bundles of b, then the
    # balances of a.  The legend fills column by column, hence the order, and
    # an empty entry closes the first row.
    bank = {
        b: plt.Line2D([], [], ls="none", marker="o", ms=5.5, color=DEBT_BALANCE[b][0],
                      label=DEBT_BALANCE[b][1])
        for b in DEBT_BALANCE
    }
    fig.legend(
        handles=[
            plt.Line2D([], [], color=SERIES[0], lw=1.6, label="without weights"),
            bank["owes"],
            plt.Line2D([], [], color=SERIES[7], lw=1.6, label="with weights"),
            bank["lends"],
            plt.Line2D([], [], alpha=0.0, label=" "),
            bank["balanced"],
        ],
        loc="outside upper center", ncol=3, frameon=False, fontsize=7.5,
        handletextpad=0.3, columnspacing=1.0,
    )
    return fig


def _cascade_panels(ctx, ib, pos, instants, *, largest: float, height: float = 5.6,
                    ratio: float = 1.0) -> plt.Figure:
    """Five moments of the cascade from the bank with the largest D."""
    g, d = ib.network, ib.capital
    run = _virus(g, ib.largest_bank(), d.copy())
    fig = plt.figure(figsize=(5.05, height), layout="constrained")
    axes = panel_rows(fig, len(instants), top=0.55, ratio=ratio)
    for ax, t, letter in zip(axes, instants, "abcde", strict=True):
        fallen = _fallen_at(run, t)
        _draw_banks(ax, g, pos, d, fallen=fallen, link_width=0.25, largest=largest)
        ax.set_title(f"$t={t}$", fontsize=9)
        panel_letter(ax, letter)
    _debt_key(fig, d, cascade=True, ncol=4)
    ctx.parameters(
        instants=list(instants), steps=run.steps,
        fallen_at_end=round(run.crashed_fraction, 3),
    )
    return fig


def _instants(run, first: int = 0) -> list[int]:
    """Five instants from *first* to the end of the cascade, as the thesis picks them."""
    end = run.steps + 1
    if first == 0:
        return [0, 1, 2, max(3, round((2 + end) / 2)), end]
    return [1, max(2, round(end / 4)), max(3, round(end / 2)), max(4, round(3 * end / 4)), end]


@figure(
    "T82",
    chapter=8,
    title="The cascade through the debt network, step by step",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="74",
        section="5.6.4 Red de deuda del sistema bancario",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 82 (panels a-e), PDF p. 74",
)
def cascade_sequence(ctx: FigureContext):
    """PDF p. 74, fig. 82: the virus from the bank with the largest D, step by step.

    The network of fig. 81.a.  At t = 0 nothing has fallen; at t = 1 the bank
    with the largest D falls; from then on each panel is a later instant, up
    to the last, when the cascade has stopped.
    """
    ib, pos = _strategy_network(ctx.rng)
    run = _virus(ib.network, ib.largest_bank(), ib.capital.copy())
    return _cascade_panels(ctx, ib, pos, _instants(run), largest=60.0, ratio=0.68)


@figure(
    "T83",
    chapter=8,
    title="What is left when the cascade stops",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="75",
        section="5.6.4 Red de deuda del sistema bancario",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 83, PDF p. 75",
)
def survivors(ctx: FigureContext):
    """PDF p. 75, fig. 83: the banks of fig. 82 still standing once it has stopped.

    Only the survivors are drawn, where fig. 81.a places them and coloured by
    their balance, with the green links still joining them; the banks that
    fell are left as faint marks, so that it can be seen where the survivors
    sat.
    """
    ib, pos = _strategy_network(ctx.rng)
    g, d, c = ib.network, ib.capital, ib.country
    run = _virus(g, ib.largest_bank(), d.copy())
    fallen = run.crashed_history[-1]
    alive = np.flatnonzero(~fallen)
    a = g.adjacency
    # A surviving bank in debt, and whether it lends or borrows abroad through
    # a bank that is still standing: what the text reads off the figure.
    debtors = [i for i in alive if d[i] < 0]
    abroad = [i for i in debtors if np.any((a[i] > 0) & (c != c[i]) & ~fallen)]

    fig, ax = plt.subplots(figsize=(5.05, 3.6), layout="constrained")
    # The network that fell, as a faint ground, so the countries can be seen.
    gone = np.flatnonzero(fallen)
    ax.add_collection(LineCollection(
        [(pos[i], pos[j]) for i, j, _ in g.edges() if fallen[i] or fallen[j]],
        colors=[to_rgba(INK["muted"], 0.12)], linewidths=0.3, zorder=1,
    ))
    ax.scatter(pos[gone, 0], pos[gone, 1], s=10, c=INK["grid"], linewidths=0, zorder=1)
    # What still stands: the banks, and the links between them.
    segments = [(pos[i], pos[j]) for i, j, _ in g.edges() if not fallen[i] and not fallen[j]]
    if segments:
        ax.add_collection(LineCollection(segments, colors=LINK_STANDING, linewidths=1.1, zorder=2))
    sizes = _bank_sizes(d, largest=260.0)
    colours = [DEBT_BALANCE[b][0] for b in _balance(d[alive])]
    ax.scatter(pos[alive, 0], pos[alive, 1], s=np.maximum(sizes[alive], 40.0), c=colours,
               zorder=3, linewidths=0.5, edgecolors="white")
    # The banks in debt that stood: what the text reads off the figure.
    if debtors:
        ax.scatter(pos[debtors, 0], pos[debtors, 1],
                   s=(np.sqrt(np.maximum(sizes[debtors], 40.0)) + 6.0) ** 2,
                   facecolors="none", edgecolors=INK["primary"], linewidths=1.1, zorder=4)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.margins(0.04)
    handles = [
        plt.Line2D([], [], ls="none", marker="o", ms=6, color=DEBT_BALANCE[b][0], label=DEBT_BALANCE[b][1])
        for b in DEBT_BALANCE if b in set(_balance(d[alive]).tolist())
    ] + ([
        plt.Line2D([], [], ls="none", marker="o", ms=8, mfc="none", mec=INK["primary"], mew=1.1,
                   label="in debt, still standing"),
    ] if debtors else []) + [
        plt.Line2D([], [], ls="none", marker="o", ms=5, color=INK["grid"], label="fallen"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False, fontsize=7.5,
               handletextpad=0.3, columnspacing=1.0)
    ctx.parameters(
        survivors=int(alive.size), surviving_debtors=len(debtors),
        debtors_joined_abroad_to_standing_banks=len(abroad),
    )
    return fig


@figure(
    "T84",
    chapter=8,
    title="The network grown by strategies, at t = 200",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="75",
        section="5.6.4 Red de deuda del sistema bancario",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 84, PDF p. 75",
)
def long_growth(ctx: FigureContext):
    """PDF p. 75, fig. 84: the growth by strategies carried to t = 200.

    The largest banks -- those whose D lies more than two standard deviations
    above the mean, the banks too big to fail -- are the largest discs; the
    parameters record how many countries hold at least one of them.
    """
    ib, pos = _strategy_network(ctx.rng, steps=200, spread=0.3)
    giants = ib.too_big_to_fail(sigma=2.0)
    ctx.parameters(
        Cn=10, t=200, banks=ib.n_banks,
        too_big_to_fail=int(giants.size),
        countries_with_one=int(np.unique(ib.country[giants]).size),
    )
    fig, ax = plt.subplots(figsize=(5.05, 5.0), layout="constrained")
    k = ib.network.degrees()
    _draw_banks(ax, ib.network, pos, ib.capital, link_width=0.18, ringed=giants, link_alpha=0.16,
                sizes=4.0 + 380.0 * (k / k.max()) ** 2, giants_on_top=True)
    _debt_key(fig, ib.capital, ringed=True)
    return fig


@figure(
    "T85",
    chapter=8,
    title="The debt network grown under the restriction",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="76",
        section="5.6.4 Red de deuda del sistema bancario",
    ),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 85, PDF p. 76",
)
def restricted_network(ctx: FigureContext):
    """PDF p. 76, fig. 85: grown to t = 1000 when a link abroad needs D > 0.

    Cn = 10, n0 = 3, r = 1, n = 2, m = 1, t = 1000: a bank may join a bank of
    another country only if that bank's balance D is positive.
    """
    ib, pos = _strategy_network(ctx.rng, steps=1000, restricted=True)
    g = ib.network
    exterior = sum(1 for i, j, _ in g.edges() if ib.country[i] != ib.country[j])
    ctx.parameters(Cn=10, t=1000, restriction="exterior link only to a bank with D > 0",
                   banks=g.n, links=g.n_edges, exterior_links=exterior)
    fig, ax = plt.subplots(figsize=(5.05, 5.0), layout="constrained")
    # Sized as fig. 84 is: by the number of loans of each bank, so the two
    # drawings can be read side by side.
    k = g.degrees()
    _draw_banks(ax, g, pos, ib.capital, link_width=0.12, link_alpha=0.16,
                sizes=3.0 + 200.0 * (k / k.max()) ** 2, giants_on_top=True)
    _debt_key(fig, ib.capital)
    return fig


@figure(
    "T86",
    chapter=8,
    title="The cascade under the restriction, step by step",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="76",
        section="5.6.4 Red de deuda del sistema bancario",
    ),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 86 (panels a-e), PDF p. 76",
)
def restricted_cascade(ctx: FigureContext):
    """PDF p. 76, fig. 86: the virus from the largest bank of fig. 85, step by step."""
    ib, pos = _strategy_network(ctx.rng, steps=1000, restricted=True)
    run = _virus(ib.network, ib.largest_bank(), ib.capital.copy())
    fig = _cascade_panels(ctx, ib, pos, _instants(run, first=1), largest=40.0, height=4.0)
    fallen = run.crashed_history[-1]
    ctx.parameters(countries_reached=int(np.unique(ib.country[fallen]).size))
    return fig


@figure(
    "T87",
    chapter=8,
    title="How far the contagion gets once lending is restricted",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="77",
        section="5.6.4 Red de deuda del sistema bancario",
    ),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 87, PDF p. 77",
)
def restricted_curve(ctx: FigureContext):
    """PDF p. 77, fig. 87: the fraction of banks fallen, from the largest bank of fig. 85."""
    ib, _ = _strategy_network(ctx.rng, steps=1000, restricted=True)
    run = _virus(ib.network, ib.largest_bank(), ib.capital.copy())
    length = max(run.history.size, 10)
    curve = np.full(length, run.history[-1])
    curve[: run.history.size] = run.history
    ctx.parameters(seed_bank="the largest D", fallen=round(run.crashed_fraction, 3))
    fig, ax = plt.subplots(figsize=(4.8, 2.0), layout="constrained")
    t = np.arange(1, length + 1)
    ax.plot(t, curve, color=SERIES[7], lw=1.8, marker="o", ms=3.5)
    _curve_axes(ax, length)
    return fig
