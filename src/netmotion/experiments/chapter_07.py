"""Figures for chapter 7: vulnerability and immunisation.

Thesis section 5.5 (PDF pp. 54-63).  If propagation speed measures a network,
then it also measures a node -- by how fast an infection starting there covers
everything -- and the nodes that score highest are the ones worth immunising.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..metrics.higher_order_degree import k_order_profile
from ..metrics.propagation_centrality import estrada_hatano_network
from ..networks import barabasi_albert, erdos_renyi, ring_lattice, watts_strogatz
from ..propagation import propagate
from ..propagation.immunity import (
    THESIS_SCALE_FREE_100,
    compare_immunisation_criteria,
    compare_orders,
)
from ..provenance import Origin, Source
from ..visualization import INK, SERIES, draw_network
from ..visualization.charts import reference_line
from .registry import FigureContext, figure

THESIS_SEED = 20110930


@figure(
    "F07-01",
    chapter=7,
    title="The degree k of order n, on the worked example",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="54-56",
        section="5.5.1.3 El metodo del grado k de orden n del nodo m",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 54 and the values on PDF p. 56",
)
def k_order_example(ctx: FigureContext):
    """PDF pp. 54-56, fig. 54: the nine-node network the thesis works through.

    The degree of order n counts the *links* reachable within n-1 hops, not the
    nodes. The distinction is the whole point, and the thesis's own numbers
    turn on it: K_1(2) = 4 while the ball around node 1 holds five nodes.

    The right panel is why the measure exists. At order 1 node 5 looks
    unimportant -- it has two links against node 4's four. By order 2 it has
    overtaken everything: it is the node from which the whole network is
    closest.

    Node 5 carries the accent because it is the node the section is about; the
    two nodes it is measured against share the one hue, and the remaining six
    profiles are drawn as context rather than as six more series.
    """
    g = estrada_hatano_network()
    ctx.parameters(N=g.n, E=g.n_edges, source="thesis fig. 54")

    orders = range(1, 6)
    profiles = {m: k_order_profile(g, m, max(orders)) for m in range(g.n)}

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # Node 5 is the subject; nodes 1 and 4 are what it is compared against.
    named = {0: SERIES[0], 3: SERIES[0], 4: SERIES[1]}
    colours = np.array([named.get(i, INK["muted"]) for i in range(g.n)])
    draw_network(g, left, node_color=colours, scale_by_degree=True, edge_width=0.9)
    # Push each label radially outward from the centre of the ring layout, so
    # that it clears its own marker whatever the degree-scaled marker size is.
    # The offset is derived per node from the same size formula draw_network
    # uses for scale_by_degree, rather than one constant for every node, so a
    # degree-4 hub's larger marker gets a larger offset instead of relying on
    # one fixed number to clear the biggest marker in the network.
    degrees = g.degrees()
    base_size = max(6.0, 260.0 / max(g.n, 1) ** 0.55)
    span = max(int(degrees.max()), 1)
    marker_radii = np.sqrt(base_size * (0.55 + 5.5 * (degrees / span)) / np.pi)
    for i, (x, y) in enumerate(_layout(g.n)):
        offset = marker_radii[i] + 5.0
        left.annotate(
            str(i + 1),
            xy=(x, y),
            xytext=(offset * x, offset * y),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=7.5,
            color=named.get(i, INK["secondary"]),
        )
    # The labels are pushed outward from the ring, so the topmost one sits above
    # everything drawn and was reaching the title. The margin buys it room.
    left.margins(0.16)
    left.set_title("Node 5 bridges\nthe two stars", fontsize=9.5, pad=10.0)

    for m in range(g.n):
        if m not in named:
            right.plot(list(orders), profiles[m], color=INK["grid"], lw=1.0, zorder=1)
    for m, colour in named.items():
        right.plot(list(orders), profiles[m], color=colour, lw=1.8, marker="o", ms=4.5, zorder=3)
        # Named in the shared flat region at the right edge, where every
        # profile has already reached g.n_edges, not at each curve's own
        # turn point: a label planted at its own turn sits on top of the
        # neighbours that are still rising through that same order, which is
        # exactly the collision node 5's label used to land in on node 4's
        # climb. Past the last node to saturate, all three lie on the same
        # flat line, so the labels can only stack on each other, not on data.
        right.annotate(
            f"node {m + 1}",
            xy=(list(orders)[-1], g.n_edges),
            xytext=(3, -9 - 10 * list(named).index(m)),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=8,
            color=colour,
        )

    reference_line(right, g.n_edges, f"all {g.n_edges} links", side="left")
    right.set_xlabel("order $n$")
    right.set_ylabel("$K_m(n)$")
    right.set_xticks(list(orders))
    right.set_xlim(0.05, 5.2)
    right.set_title("By order 2, node 5\nreaches every link", fontsize=9.5)

    return fig


def _layout(n: int):
    angles = 2.0 * np.pi * np.arange(1, n + 1) / n
    return list(zip(np.cos(angles), np.sin(angles), strict=True))


@figure(
    "F07-02",
    chapter=7,
    title="Immunisation: which nodes to protect",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="56-63",
        section="5.5.2 Inmunizacion frente a infeccion",
    ),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis figs. 58-63, PDF pp. 58-63",
)
def immunisation_criteria(ctx: FigureContext):
    """PDF pp. 58-63: immunise the top-ranked nodes, and watch the avalanche.

    An immune node is frozen at S = +1, so it is never infected and therefore
    never transmits. Immunising the right ones cuts the network's ability to
    carry an infection at all.

    The two criteria the thesis compares are degree and the degree of order n;
    the other three here are 2026 additions, included so that the two can be
    compared against alternatives rather than read in isolation. That
    is what the drawing says as well: the two criteria under discussion carry
    the colour, and the three controls are grey and told apart by their dash.

    The higher-order criterion is drawn at order 3, because at order 1 it *is*
    the degree -- K_m(1) = k_m by definition -- and the two curves would lie
    exactly on top of each other. That coincidence is the reason the thesis can
    treat ordinary degree as the n = 1 case of its own measure.
    """
    n = 100
    # The sweep is quadratic in N -- every immunisation count crossed with every
    # seed -- so the figure subsamples both. The thesis averages over all seeds;
    # a quarter of them is enough to draw a curve, and the parameters are
    # recorded in the sidecar so the reduction is visible.
    seeds = np.arange(0, n, 4)
    counts = np.arange(0, 61, 4)
    order = 3
    ctx.parameters(
        N=n,
        model="scale-free",
        m=2,
        k_order=order,
        seeds=f"every 4th node ({seeds.size} of {n})",
        immune_counts=counts.tolist(),
    )
    g = barabasi_albert(n, links_per_new_node=2, rng=ctx.rng)
    sweeps = compare_immunisation_criteria(g, start_nodes=seeds, counts=counts, order=order, rng=ctx.rng)
    ctx.realisations(seeds.size)

    labels = {
        "degree": "degree  (= order 1)",
        "k_order_n": f"degree of order {order}",
        "random": "random",
        "min_propagation_time": "min. propagation time",
        "node_removal": "node removal",
    }
    # Colour is spent on the two criteria the section is about.  The three 2026
    # controls share the muted ink and are separated by dash pattern, which is
    # what says "these are the baseline" without inventing three more hues.
    styles = {
        "degree": (SERIES[1], "-", 2.0, 3),
        "k_order_n": (SERIES[0], "-", 2.0, 3),
        "random": (INK["muted"], (0, (5, 2)), 1.2, 2),
        "min_propagation_time": (INK["muted"], (0, (1, 2)), 1.2, 2),
        "node_removal": (INK["muted"], (0, (4, 2, 1, 2)), 1.2, 2),
    }

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.4))

    for name, sweep in sweeps.items():
        colour, dash, width, order_z = styles[name]
        label = labels.get(name, name.replace("_", " "))
        left.plot(
            sweep.counts,
            sweep.infected,
            color=colour,
            ls=dash,
            lw=width,
            zorder=order_z,
            label=label,
        )
        usable = np.isfinite(sweep.speed)
        right.plot(
            np.asarray(sweep.counts)[usable],
            np.asarray(sweep.speed)[usable],
            color=colour,
            ls=dash,
            lw=width,
            zorder=order_z,
            label=label,
        )

    left.set_xlabel("immune nodes")
    left.set_ylabel("nodes eventually infected")
    left.set_ylim(bottom=0)
    left.set_title("Degree extinguishes\nit fastest", fontsize=9.5)

    right.set_xlabel("immune nodes")
    right.set_ylabel(r"$\langle \alpha \rangle$")
    right.set_title("All five slow it\nby about as much", fontsize=9.5)

    # The two criteria the chapter is about are named on their own curves, at
    # the sweep point where they are furthest apart, so the reader's eye never
    # leaves the data.  Only the three grey alternatives -- which are told apart
    # by dash pattern and nothing else -- still need naming off the marks, and
    # they go in the left panel's empty upper corner without a box.
    named = ("degree", "k_order_n")
    if all(k in sweeps for k in named):
        a = np.asarray(sweeps[named[0]].infected, dtype=float)
        b = np.asarray(sweeps[named[1]].infected, dtype=float)
        counts = np.asarray(sweeps[named[0]].counts, dtype=float)
        split = int(np.argmax(np.abs(a - b)))
        for name, values, dy in ((named[0], a, -12), (named[1], b, 12)):
            left.annotate(
                labels.get(name, name.replace("_", " ")),
                xy=(counts[split], values[split]),
                xytext=(6, dy),
                textcoords="offset points",
                fontsize=8.5,
                color=styles[name][0],
                ha="left",
                va="center",
            )

    grey_handles = [
        plt.Line2D([], [], color=styles[n][0], ls=styles[n][1], lw=styles[n][2])
        for n in ("random", "min_propagation_time", "node_removal")
        if n in sweeps
    ]
    grey_names = [
        labels.get(n, n.replace("_", " "))
        for n in ("random", "min_propagation_time", "node_removal")
        if n in sweeps
    ]
    if grey_handles:
        # The key goes in the right panel's upper corner, which every curve
        # actually reaches: at q = 0 every series sits near \langle\alpha\rangle
        # \approx 0.95, and this box currently hides that shared value (plus two
        # more points of the random curve). Padding the y-axis above that shared
        # peak opens real headroom for the box instead -- same fix already used
        # for charts.reference_line's off-axes label, kept inside the axes so
        # fit_to_print_width's width guard still measures it.
        right.set_ylim(top=1.3)
        right.legend(
            grey_handles,
            grey_names,
            fontsize=8,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            borderaxespad=0.4,
            title="the three alternatives",
            title_fontsize=8,
            frameon=True,
            facecolor=INK["surface"],
            edgecolor="none",
            framealpha=1.0,
        )

    return fig


@figure(
    "F07-03",
    chapter=7,
    title="Which order of the degree immunises best",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="58-63",
        section="5.5.2 Inmunizacion frente a infeccion",
    ),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis figs. 59, 62 and 63, PDF pp. 59-63",
)
def immunisation_orders(ctx: FigureContext):
    """PDF pp. 59-63: ranking by the degree of order n, for n = 1 to 8.

    The thesis reports how many nodes must be immunised before the infection
    dies out, for each order, and finds order 1 -- ordinary degree -- best for
    the scale-free network, with higher orders needing more nodes.

    The reported numbers are read off its figures rather than tabulated, so
    what is reproduced is the shape: a minimum at low order, rising and then
    flattening as the higher orders saturate and stop discriminating between
    nodes.
    """
    n = 100
    orders = list(range(1, 9))
    # Counts must reach far enough to see extinction -- the thesis reports it
    # between 53 and 62 immune nodes -- while the seeds are subsampled, since
    # the sweep costs one propagation per (count, seed) pair.
    seeds = np.arange(0, n, 4)
    counts = np.arange(0, n + 1, 2)
    ctx.parameters(
        N=n,
        model="scale-free",
        m=2,
        orders=orders,
        seeds=f"every 4th node ({seeds.size} of {n})",
        immune_counts="0..100 in steps of 2",
    )
    ctx.realisations(seeds.size)

    g = barabasi_albert(n, links_per_new_node=2, rng=ctx.rng)
    sweeps = compare_orders(g, orders=orders, start_nodes=seeds, counts=counts, rng=ctx.rng)

    measured = np.array([sweeps[o].mean_nodes_to_extinction for o in orders])
    reported = np.array(THESIS_SCALE_FREE_100.nodes_to_zero)

    betas = []
    for o in orders:
        try:
            betas.append(sweeps[o].beta)
        except ValueError:
            betas.append(np.nan)
    betas = np.array(betas)

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    for ax, here, quoted, best_is in (
        (left, measured, reported, np.argmin),
        (right, betas, np.array(THESIS_SCALE_FREE_100.beta), np.nanargmin),
    ):
        ax.plot(
            orders, here, color=SERIES[0], lw=1.8, marker="o", ms=5, zorder=3, label="measured here"
        )
        ax.plot(
            orders,
            quoted,
            color=SERIES[1],
            lw=1.5,
            ls=(0, (5, 2)),
            marker="D",
            ms=4,
            zorder=2,
            label="the values of section 5.5.1",
        )
        best = orders[int(best_is(here))]
        ax.plot(best, here[best - 1], "o", ms=9, mfc="none", mec=SERIES[0], mew=1.4, zorder=4)
        ax.set_xlabel("order $n$ of the degree")
        ax.set_xticks(orders)
        ax.set_xlim(0.6, 8.6)
        # Two curves named once here rather than at each line's own end: an
        # end label that fits at the book's old 6in width no longer fits
        # beside a panel this narrow, and the two panels share the same two
        # series anyway, so one small key per panel replaces two annotations
        # that fit_to_print_width's width guard could not see past the axes'
        # own edge (annotation_clip=False) -- see charts.reference_line for
        # the same fix applied to the shared helper.
        ax.legend(fontsize=8, loc="best", borderaxespad=0.3, handlelength=1.4)

    left.set_ylabel("immune nodes needed")
    left.set_title("Low orders extinguish it\nwith fewer nodes", fontsize=9.5)
    right.set_ylabel(r"decay factor $\beta$")
    right.set_title("The same question\nasked of the speed", fontsize=9.5)

    return fig


@figure(
    "F07-04",
    chapter=7,
    title="The ordered network is the exception",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="58-63",
        section="5.5.2 Inmunizacion frente a infeccion",
    ),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis table of best orders, PDF pp. 62-63",
)
def best_order_by_network(ctx: FigureContext):
    """PDF pp. 62-63: the same sweep, across network types.

    The thesis reports that ordinary degree is the best immunisation criterion
    for every network type it tries -- except the ordered ring, where order 3
    wins.

    That exception is the interesting part, and it makes sense: in a ring every
    node has the same degree, so degree cannot rank anything at all. Only a
    higher-order measure, which sees further than one hop, can tell one node
    from another.

    Four families are four names, not four magnitudes, so they are four panels
    on one shared vertical scale rather than four hues in one frame. The ring
    is the accent because it is the exception the section is about, and on the
    shared scale its flatness -- no order beats any other -- is the point.
    """
    n = 80
    orders = list(range(1, 6))
    seeds = np.arange(0, n, 4)
    counts = np.arange(0, n + 1, 2)
    ctx.parameters(
        N=n,
        orders=orders,
        seeds=f"every 4th node ({seeds.size} of {n})",
        immune_counts=f"0..{n} in steps of 2",
    )
    ctx.realisations(seeds.size)

    builders = {
        "ordered ring": lambda: ring_lattice(n, 4),
        "Erdos-Renyi": lambda: erdos_renyi(n, 4 / (n - 1), rng=ctx.rng),
        "Watts-Strogatz": lambda: watts_strogatz(n, 4, 0.1, rng=ctx.rng),
        "scale-free": lambda: barabasi_albert(n, links_per_new_node=2, rng=ctx.rng),
    }

    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.9), sharex=True, sharey=True)

    for ax, (label, build) in zip(axes, builders.items(), strict=True):
        g = build()
        sweeps = compare_orders(g, orders=orders, start_nodes=seeds, counts=counts, rng=ctx.rng)
        needed = np.array([sweeps[o].mean_nodes_to_extinction for o in orders])
        colour = SERIES[1] if label == "ordered ring" else SERIES[0]
        ax.plot(orders, needed / n, color=colour, lw=1.8, marker="o", ms=4.5, zorder=3)
        best = orders[int(np.argmin(needed))]
        ax.plot(
            best,
            needed[best - 1] / n,
            "o",
            ms=9,
            mfc="none",
            mec=colour,
            mew=1.4,
            zorder=4,
        )
        # "best n = 1" is true of all four panels, so printing it four times
        # says nothing and the accent on the ring would be a hue spent on a
        # distinction the data does not make.  What the ring actually does is
        # not vary at all -- every order costs the same there -- and that is
        # what its panel is made to say.
        spread = float(needed.max() - needed.min()) / n
        if spread < 0.005:
            ax.set_title(f"{label}\ncosts the same", fontsize=7, color=colour)
        else:
            ax.set_title(
                f"{label}\n$n = {best}$, spread {spread:.02f}",
                fontsize=7,
                color=colour,
            )
        ax.set_xticks(orders)

    axes[0].set_ylabel("fraction immunised\nto extinguish")
    fig.supxlabel("order $n$ of the degree", fontsize=9.5, color=INK["secondary"])
    return fig


@figure(
    "F07-05",
    chapter=7,
    title="What immunisation does to the avalanche",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="56-58",
        section="5.5.2 Inmunizacion frente a infeccion",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis figs. 56-58, PDF pp. 57-58",
)
def avalanche_with_immunity(ctx: FigureContext):
    """PDF pp. 57-58: the same network, before and after immunising its hubs.

    On the left the infection reaches everything. On the right, ten immune
    nodes chosen by degree leave most of the network untouched -- and the
    curves that remain are not merely lower, they are flatter, which is the
    point: immunisation slows propagation as well as containing it.

    Both panels share the time axis, because the flattening is only visible
    against a common ruler: on its own scale the right-hand family would look
    just as steep as the left-hand one.
    """
    n, n_immune = 100, 10
    ctx.parameters(N=n, model="scale-free", m=2, immune=n_immune, criterion="degree")

    g = barabasi_albert(n, links_per_new_node=2, rng=ctx.rng)
    hubs = np.argsort(g.degrees())[::-1][:n_immune]

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True, sharex=True)

    longest = 1
    for ax, immune, title, colour in (
        (left, None, "Nobody immune", SERIES[0]),
        (right, hubs, f"{n_immune} hubs immune", SERIES[1]),
    ):
        finals = []
        for seed in range(n):
            if immune is not None and seed in set(hubs.tolist()):
                continue
            run = propagate(g, seed, immune=immune)
            longest = max(longest, run.curve.size)
            ax.plot(
                np.arange(1, run.curve.size + 1),
                run.curve,
                color=colour,
                alpha=0.16,
                lw=1.1,
            )
            finals.append(run.final_fraction)
        ax.set_title(f"{title}\n{np.mean(finals):.0%} reached", fontsize=9.5)
        ax.set_ylim(0, 1.02)

    left.set_ylabel("$p(t)$")
    left.set_xticks(np.arange(2, longest + 1, 2))
    left.set_xlim(1, longest)
    fig.supxlabel("$t$", fontsize=9.5, color=INK["secondary"])
    return fig
