"""Figures for chapter 9: the interbank network and financial contagion.

Thesis section 5.6.4 and the 2012 IJMPC article.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from ..finance import (
    BankState,
    InterbankNetwork,
    bank_capital,
    two_state_cascade,
)
from ..networks import barabasi_albert, erdos_renyi, newman_watts, watts_strogatz
from ..networks.communities import community_scale_free
from ..propagation import propagate
from ..provenance import Origin, Source
from ..visualization import INK, SERIES, draw_network, spring_layout
from ..visualization.charts import per_node, ranked_bars, reference_line
from .registry import FigureContext, figure

THESIS_SEED = 20110930


def _key(ax, entries, *, x=0.02, y=0.98, dy=0.075, fontsize=8.0):
    """Two or three coloured words in dead space, in place of a legend box.

    Each entry is ``(text, colour)``.  Written in axes coordinates so it sits
    in whatever corner of the panel is empty.
    """
    from matplotlib import patheffects

    halo = [patheffects.withStroke(linewidth=2.6, foreground=INK["surface"])]
    for i, (text, colour) in enumerate(entries):
        ax.text(
            x,
            y - i * dy,
            text,
            transform=ax.transAxes,
            color=colour,
            fontsize=fontsize,
            ha="left",
            va="top",
            weight="semibold",
            path_effects=halo,
            zorder=6,
        )


@figure(
    "F08-01",
    chapter=8,
    title="A scale-free network of communities",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="66",
        section="5.6.2 Red Scale Free de comunidades",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 74.a-c, PDF p. 66",
)
def community_network(ctx: FigureContext):
    """PDF p. 66, fig. 74: Cn = 10, n0 = 3, r = 1, n = 2, m = 1, t = 100.

    Ten communities grown together, each by preferential attachment inside
    itself while reaching across to the others. Nodes are sized by degree and
    the links that cross a community boundary are picked out, because those
    links are the only thing holding the ten communities together.

    The thesis reports a density of 2.06 links per node and an exponent of 2.33
    for this network.
    """
    ctx.parameters(Cn=10, n0=3, r=1, n=2, m=1, t=100)
    cn = community_scale_free(rng=ctx.rng)
    k = cn.network.degrees()

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.4))

    pos = spring_layout(cn.network, rng=ctx.rng)
    draw_network(
        cn.network,
        left,
        layout=pos,
        node_color=SERIES[0],
        scale_by_degree=True,
        edge_color=INK["grid"],
        edge_width=0.6,
    )
    crossing = [
        (pos[i], pos[j]) for i, j, _ in cn.network.edges() if cn.community[i] != cn.community[j]
    ]
    left.add_collection(
        LineCollection(crossing, colors=SERIES[1], linewidths=0.7, alpha=0.9, zorder=1.5)
    )
    left.set_title(f"{cn.network.n} nodes, 10 communities", fontsize=8.5)
    left.annotate(
        f"{cn.exterior_edges()} cross-community links",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0, -2),
        textcoords="offset points",
        color=SERIES[1],
        fontsize=8.0,
        va="top",
    )

    # Node index is the order of arrival, not the community, so the series is
    # regrouped before it is drawn: the bands below are then the communities.
    order = np.argsort(cn.community, kind="stable")
    grouped = k[order]
    per_node(
        right,
        grouped,
        mean_label=f"mean $k$ = {k.mean():.1f}",
        highlight=np.flatnonzero(grouped == grouped.max()),
    )

    sizes = cn.sizes()
    edges = np.concatenate([[0], np.cumsum(sizes)])
    for c in range(0, cn.n_communities, 2):
        right.axvspan(edges[c] + 0.5, edges[c + 1] + 0.5, color=INK["grid"], alpha=0.45, zorder=0)
    right.set_xticks([])
    right.set_xlabel("node, grouped by community")
    right.set_ylabel("$k$")
    right.set_title(
        f"A few hubs at $k = {int(k.max())}$\nmost nodes at two or three",
        fontsize=8.5,
    )

    return fig


@figure(
    "F09-01",
    chapter=9,
    title="The debt network: who owes whom",
    source=Source(
        Origin.PAPER_2012,
        pdf_pages="4",
        section="2.2 The weighted network",
        document="both",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 77, PDF p. 68",
)
def debt_network(ctx: FigureContext):
    """Thesis fig. 77 (PDF p. 68): the same network, now carrying debts.

    Node size is the size of the bank's net position and colour is its sign.
    The whole system nets to zero by construction, so the two colours are
    exactly in balance: the profile on the right has as much area below the
    line as above it.
    """
    ctx.parameters(Cn=10, n0=3, t=100, convention="paper")
    cn = community_scale_free(rng=ctx.rng)
    ib = InterbankNetwork.from_community_network(cn)
    d = ib.capital

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.4))

    colours = np.where(d > 0, SERIES[1], np.where(d < 0, SERIES[0], INK["muted"]))
    scale = 200.0 * np.abs(d) / max(np.abs(d).max(), 1.0) + 4.0
    pos = spring_layout(ib.network, rng=ctx.rng)
    draw_network(
        ib.network,
        left,
        layout=pos,
        node_color=colours,
        node_size=scale,
        edge_color=INK["grid"],
        edge_width=0.6,
    )
    left.set_title("Size is $|D_i|$, colour its sign", fontsize=9.5)
    _key(left, [("owed money", SERIES[1]), ("in debt", SERIES[0])], y=0.13, dy=0.08)

    ordered = np.sort(d)
    x = np.arange(1, d.size + 1)
    right.fill_between(x, ordered, 0, where=ordered <= 0, color=SERIES[0], lw=0, interpolate=True)
    right.fill_between(x, ordered, 0, where=ordered >= 0, color=SERIES[1], lw=0, interpolate=True)
    right.plot(x, ordered, color=INK["primary"], lw=0.9, zorder=3)
    right.axhline(0, color=INK["axis"], lw=0.8, zorder=2)
    right.set_xlim(1, d.size)
    right.set_xlabel("bank, ordered by net position")
    right.set_ylabel("$D_i$")
    right.set_title("The two sides balance exactly", fontsize=9.5)

    debtors, creditors = ib.debtors().size, ib.creditors().size
    right.annotate(
        f"{debtors} banks in debt",
        xy=(debtors * 0.45, ordered.max() * 0.06),
        color=SERIES[0],
        fontsize=8.0,
        ha="center",
        va="bottom",
    )
    right.annotate(
        f"{creditors} banks owed",
        xy=(d.size - creditors * 0.6, ordered.max() * 0.42),
        color=SERIES[1],
        fontsize=8.0,
        ha="right",
    )

    return fig


@figure(
    "F09-02",
    chapter=9,
    title="Crashing the largest bank",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="69-71",
        section="5.6.4 Analisis de la estructura de comunidades",
        document="both",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis figs. 81.b and 82, PDF p. 74",
)
def cascade_snapshots(ctx: FigureContext):
    """Thesis figs. 81.b and 82: the contagion, step by step.

    The largest bank is attacked and fails. Each neighbour loses an amount set
    by the difference in their connectivity; a bank whose capital cannot absorb
    that loss fails in turn.

    The panels are the snapshots the thesis draws, over the strip that says
    where in the cascade each one sits. What they show is that the failure does
    not spread uniformly: it runs along the chains of banks whose capital is
    thin relative to their neighbours', and stops where it meets a bank large
    enough to take the loss.
    """
    ctx.parameters(Cn=10, n0=3, t=100, seed_bank="largest by D", drain="every_step")
    cn = community_scale_free(rng=ctx.rng)
    ib = InterbankNetwork.from_community_network(cn)
    capital = np.abs(ib.capital)
    seed = int(np.argmax(capital))

    run = two_state_cascade(ib.network, seed, capital, drain="every_step")
    pos = spring_layout(ib.network, rng=ctx.rng)

    snapshots = np.unique(np.linspace(0, len(run.capital_history) - 1, 4).astype(int))
    rows = 2 if snapshots.size > 2 else 1
    cols = int(np.ceil(snapshots.size / rows))

    fig = plt.figure(figsize=(2.9 * cols, 2.35 * rows + 0.95))
    grid = fig.add_gridspec(rows + 1, cols, height_ratios=[1.0] * rows + [0.42])
    axes = [fig.add_subplot(grid[r, c]) for r in range(rows) for c in range(cols)]

    for ax, t in zip(axes, snapshots, strict=True):
        d_t = run.capital_history[t]
        failed = d_t < 0
        # The attacked bank starts crashed and its capital is never reduced,
        # so the test above would draw it solvent; it is down from t = 0.
        failed[seed] = True
        colours = np.where(failed, SERIES[7], SERIES[0])
        draw_network(
            ib.network,
            ax,
            layout=pos,
            node_color=colours,
            node_size=17.0,
            edge_color=INK["grid"],
            edge_width=0.5,
        )
        ax.scatter(
            pos[seed, 0],
            pos[seed, 1],
            s=90,
            facecolors="none",
            edgecolors=INK["primary"],
            linewidths=1.0,
            zorder=5,
        )
        ax.set_title(f"$t = {t}$   {failed.mean():.0%} failed", fontsize=10)

    strip = fig.add_subplot(grid[rows, :])
    steps = np.arange(run.history.size)
    strip.plot(steps, run.history * 100, color=SERIES[7], lw=1.8)
    strip.plot(snapshots, run.history[snapshots] * 100, "o", ms=5.0, color=INK["primary"], zorder=3)
    for t in snapshots:
        strip.annotate(
            f"$t={t}$",
            xy=(t, run.history[t] * 100),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=INK["secondary"],
        )
    strip.set_xlim(-0.5, steps[-1] + 0.5)
    strip.set_ylim(0, 118)
    strip.set_yticks([0, 50, 100])
    strip.set_xlabel("step")
    strip.set_ylabel("% failed")

    # A spring-laid cloud fills its axes edge to edge, so there is no empty
    # corner in a network panel for a key: the three words were being written
    # across the nodes of the t = 0 panel.  They go above the curve strip
    # instead, clear of the strip's own snapshot-time annotations.
    strip.legend(
        handles=[
            Line2D([], [], ls="none", marker="o", ms=5, color=SERIES[7], label="failed"),
            Line2D([], [], ls="none", marker="o", ms=5, color=SERIES[0], label="still solvent"),
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                ms=6,
                mfc="none",
                mec=INK["primary"],
                mew=1.0,
                label="the bank attacked",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=3,
        frameon=False,
        fontsize=8.0,
        handletextpad=0.3,
        columnspacing=1.6,
        borderaxespad=0.3,
    )

    return fig


@figure(
    "F09-04",
    chapter=9,
    title="Whether any network type absorbs contagion better than another",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="70-71",
        section="5.6.4 Analisis de la estructura de comunidades",
    ),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 79 and its table, PDF p. 71",
)
def weighted_versus_unweighted(ctx: FigureContext):
    """PDF p. 71, fig. 79: the conclusion the thesis draws from the whole model.

    Run two processes on the same networks. The unweighted infection of section
    5.4 always reaches every node -- topology cannot stop it. The weighted
    cascade, where capital absorbs losses, saturates anywhere between a tenth
    and nine tenths of the system depending only on where it started.

    The thesis reports mean survival of 56.5%, 58.1%, 55.4% and 56.6% for
    Erdos-Renyi, Watts-Strogatz, Newman-Watts and scale-free networks: four
    numbers within two points of each other. Its conclusion is that no topology
    is safer than another, and that what decides survival is the local
    configuration of a bank and its neighbours.

    The four networks are rebuilt here at about two links per node, since the
    ER p, the WS/NW k and the SF m are not printed. Both sets of means are
    drawn on the left panel: the 2011 diamonds sit inside the spread this
    reproduction measures at every one of the four types, so the comparison is
    read at the level of the relation -- a narrow spread across types, and a
    wide spread across seeds -- rather than at the level of the individual
    percentages, which no stated parameter set pins down.
    """
    n, realisations = 100, 10
    ctx.parameters(N=n, density="~2 links/node", seeds="every 5th bank")
    ctx.realisations(realisations)

    builders = {
        "Erdos-Renyi": lambda: erdos_renyi(n, 2 * 2.0 / (n - 1), rng=ctx.rng),
        "Watts-Strogatz": lambda: watts_strogatz(n, 4, 0.1, rng=ctx.rng),
        "Newman-Watts": lambda: newman_watts(n, 4, 0.1, rng=ctx.rng),
        "Scale-free": lambda: barabasi_albert(n, links_per_new_node=2, rng=ctx.rng),
    }
    reported = {
        "Erdos-Renyi": 56.5,
        "Watts-Strogatz": 58.1,
        "Newman-Watts": 55.4,
        "Scale-free": 56.6,
    }

    survival, unweighted = {}, {}
    for label, build in builders.items():
        weighted_rates, plain_rates = [], []
        for _ in range(realisations):
            g = build()
            capital = np.abs(bank_capital(g))
            for seed in range(0, n, 5):
                weighted_rates.append(two_state_cascade(g, seed, capital.copy()).survived_fraction)
                plain_rates.append(1.0 - propagate(g, seed).final_fraction)
        survival[label] = np.array(weighted_rates)
        unweighted[label] = float(np.mean(plain_rates))

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.2))

    labels = list(builders)
    short = [k.replace("-", "-\n") for k in labels]
    jitter = ctx.rng.uniform(-0.14, 0.14, size=survival[labels[0]].size)

    for i, label in enumerate(labels):
        values = survival[label] * 100
        left.scatter(
            i + jitter[: values.size],
            values,
            s=9,
            color=SERIES[0],
            alpha=0.3,
            zorder=2,
        )
        left.plot(
            [i - 0.26, i + 0.26],
            [values.mean()] * 2,
            color=INK["primary"],
            lw=1.6,
            solid_capstyle="butt",
            zorder=4,
        )
    left.scatter(
        range(len(labels)),
        [reported[k] for k in labels],
        color=SERIES[1],
        zorder=5,
        s=26,
        marker="D",
    )
    left.annotate(
        "the runs of section 5.6.4",
        # Anchored under the middle of the diamond row rather than under the
        # first one, where it ran into the y-axis and its bottom tick label.
        xy=((len(labels) - 1) / 2.0, min(reported.values())),
        xytext=(0, -8),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=8.0,
        color=SERIES[1],
    )
    left.set_xticks(range(len(labels)))
    left.set_xticklabels(short, fontsize=7.5)
    left.set_xlim(-0.6, len(labels) - 0.4)
    left.set_ylim(48, 104)
    left.set_ylabel("banks surviving (%)")
    left.set_title("Survival barely moves\nwith the type", fontsize=9.5, pad=18)

    for i, label in enumerate(labels):
        w, u = survival[label].mean() * 100, unweighted[label] * 100
        right.plot([u, w], [i, i], color=INK["grid"], lw=2.4, zorder=1, solid_capstyle="round")
        right.plot(w, i, "o", ms=7, color=SERIES[0], zorder=3)
        right.plot(u, i, "o", ms=7, color=SERIES[1], zorder=3)
        right.annotate(
            f"{w:.0f}%",
            xy=(w, i),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color=INK["secondary"],
        )
        right.annotate(
            f"{u:.0f}%",
            xy=(u, i),
            xytext=(-8, 0),
            textcoords="offset points",
            va="center",
            ha="right",
            fontsize=7.5,
            color=INK["secondary"],
        )
    top = len(labels) - 1
    right.annotate(
        "weighted\ncascade",
        xy=(survival[labels[top]].mean() * 100, top),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        fontsize=8.0,
        color=SERIES[0],
    )
    right.annotate(
        "unweighted\ninfection",
        xy=(unweighted[labels[top]] * 100, top),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        fontsize=8.0,
        color=SERIES[1],
    )
    right.set_yticks(range(len(labels)))
    right.set_yticklabels(short, fontsize=7.5)
    right.tick_params(axis="y", length=0)
    right.set_ylim(-0.6, top + 1.05)
    right.set_xlim(-28, 118)
    right.xaxis.set_visible(False)
    right.grid(False)
    for side in ("bottom", "left"):
        right.spines[side].set_visible(False)
    right.set_title("Capital stops it,\ntopology does not", fontsize=9.5, pad=18)

    # No suptitle: the caption carries the title, and the spread it quoted has
    # been moved into the caption in the manuscript. The two panel titles stay,
    # because the caption cannot say which panel is which.
    return fig


@figure(
    "F09-06",
    chapter=9,
    title="Growth strategies, and the banks they grow",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="72-75",
        section="5.6.4 Analisis de la estructura de comunidades",
    ),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis figs. 81 and 84, PDF pp. 74-75",
)
def growth_strategies(ctx: FigureContext):
    """PDF pp. 72-75: seven strategies, and the heterogeneity they produce.

    Every arriving bank is given one of seven rules at random -- attach to the
    biggest, imitate the average, be two links more cautious, two links more
    reckless, average globally, attach at random, or never lend abroad. The
    strategies exist to make banks differ from one another, and it is that
    difference which grows banks too big to fail.

    The left panel gives each strategy on its own, so the spread is visible;
    the right panel is the long run of fig. 84, where the thesis observes that
    at t = 200 there is roughly one dominant bank per country -- *"en cada pais
    existe por lo menos un banco de los denominados too big too fail"*; this run
    gives nine or ten of the ten communities holding one.
    """
    from ..finance.bank_growth import Strategy, strategy_growth

    ctx.parameters(Cn=10, n0=3, r=1, t="100 and 200", strategies="all seven, uniformly")
    ctx.realisations(4)

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.4))

    names, densities = [], []
    for strategy in Strategy:
        runs = [
            strategy_growth(rng=ctx.rng, strategies=(int(strategy),))[0].density for _ in range(4)
        ]
        names.append(strategy.name.lower())
        densities.append(float(np.mean(runs)))

    mixed = float(np.mean([strategy_growth(rng=ctx.rng)[0].density for _ in range(4)]))

    ranked_bars(left, names, densities, highlight=names.index("risky"), value_format="{:.1f}")
    left.tick_params(axis="y", length=0)
    reference_line(left, mixed, f"all seven mixed: {mixed:.2f}", axis="x")
    left.set_xlabel("links per node")
    left.set_title("The reckless rule\nbuilds twice the network", fontsize=9.5, pad=18)

    long_run, _ = strategy_growth(rng=ctx.rng, steps=200)
    ib = InterbankNetwork.from_community_network(long_run)
    tbtf = set(ib.too_big_to_fail(sigma=2.0).tolist())
    capital = ib.capital

    # The claim of fig. 84 is about *countries*, so show it per country: the
    # biggest bank in each, against the too-big-to-fail cut-off.
    biggest, holds_one = [], []
    for c in range(long_run.n_communities):
        members = long_run.members(c)
        best = members[int(np.argmax(capital[members]))]
        biggest.append(float(capital[best]))
        holds_one.append(best in tbtf)

    countries = int(np.sum(holds_one))
    missing = [c for c, held in enumerate(holds_one) if not held]
    ranked_bars(
        right,
        [f"country {c + 1}" for c in range(long_run.n_communities)],
        biggest,
        highlight=missing[0] if len(missing) == 1 else None,
        value_format="{:.0f}",
    )
    right.tick_params(axis="y", labelsize=8, length=0)
    cut = capital.mean() + 2.0 * capital.std()
    reference_line(right, cut, "too-big-to-fail cut-off", axis="x")
    right.set_xlabel("$D$ of the largest bank")
    right.set_title(
        f"{countries} of {long_run.n_communities} countries hold\none too big to fail",
        fontsize=9.5,
        pad=18,
    )

    return fig


@figure(
    "F09-07",
    chapter=9,
    title="Restricting foreign lending to solvent banks",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="76-77",
        section="5.6.4 Analisis de la estructura de comunidades",
        document="both",
    ),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis figs. 85-87, PDF pp. 76-77; article fig. 8, PDF p. 16",
)
def restricted_growth(ctx: FigureContext):
    """PDF pp. 76-77: one condition, and the policy conclusion drawn from it.

    A bank may lend across a border only to a bank whose net position is
    currently positive. The thesis reports that this single restriction takes
    the cascade from about 80% of the network to about 10%, and confines it to
    one community.

    The two networks are grown for different lengths of time, and printed p. 73
    says why: *"la restriccion impuesta para las conexiones exteriores dificulta
    el crecimiento de las ramas entre comunidades"* -- with the restriction in
    place the branches between communities are slow to appear, so the restricted
    network is given t = 1000 iterations where the unrestricted one is given
    t = 100. Both are grown here at the lengths the thesis states.

    The cascade is the four-state one of the 2012 article, which is the version
    of the model carrying the two parameters this comparison needs: the reserves
    eta held by the net borrowers and the reaction threshold r. The left panel
    sweeps eta with r = 0 and draws both networks at each value, against the two
    percentages the thesis reports. The first of them is close at the low end of
    the sweep, where the unrestricted network loses about seven banks in ten and
    the thesis's 80% is at the top of the realised spread.

    The right panel is the article's own Fig. 8, which states the same
    restriction for its three-strategy network, on the growth parameters that
    figure's caption prints. Its caption does not give r, so both ends of the
    range the article works in are drawn.

    The restriction lowers the cascade on both networks, and by much more on the
    thesis's. What this reproduction does not recover is the second thesis
    percentage and the confinement to a single community: the restricted
    cascade lands well above a tenth of the network and reaches all ten
    communities at every cell measured, with the share of the losses falling
    inside the seed's own community close to the one tenth a uniform spread
    would give. Three things the sources do not fix would each move it. Figs. 86-87 pre-date the
    article's reserves rule, so the eta and r they were run at are not stated
    anywhere. Neither source says what an arriving bank does when the
    restriction refuses it a foreign partner -- whether it looks for another, as
    it does here, or goes without the link. And this cascade's response to r is
    shallower than the article's Fig. 10 prints, so a cell chosen by r alone
    does not sit where the article's own curve would put it.
    """
    from ..finance import four_state_cascade
    from ..finance.bank_growth import confidence_strategy_growth, strategy_growth

    # Per-realisation spread reaches 11 points on the unrestricted network, so
    # the five realisations this figure used to average put the standard error
    # wider than the differences the panels are drawn to show.
    realisations = 25
    etas = (20.0, 30.0, 50.0)
    threshold = 0.0
    ctx.parameters(
        Cn=10,
        n0=3,
        exterior_seed_links=1,
        t="100 unrestricted, 1000 restricted",
        cascade="2012 four-state",
        eta=etas,
        reaction_threshold=threshold,
        drain="once",
        seed_bank="largest by D",
    )
    ctx.realisations(realisations)

    def crash(cn, eta: float, reaction: float = threshold) -> tuple[float, int]:
        """Fraction of banks lost, and how many communities the loss reached."""
        ib = InterbankNetwork.from_community_network(cn)
        seed = int(np.argmax(ib.capital))
        run = four_state_cascade(
            ib.network,
            seed,
            ib.capital,
            reserves=eta,
            reaction_threshold=reaction,
            drain="once",
        )
        return run.crashed_fraction, int(np.unique(cn.community[run.crashed]).size)

    failed: dict[str, dict[float, list[float]]] = {
        label: {eta: [] for eta in etas} for label in ("unrestricted", "restricted")
    }
    reached: dict[float, list[int]] = {eta: [] for eta in etas}

    for _ in range(realisations):
        for label, steps, restricted in (
            ("unrestricted", 100, False),
            ("restricted", 1000, True),
        ):
            cn, _ = strategy_growth(rng=ctx.rng, steps=steps, restricted=restricted)
            for eta in etas:
                fraction, communities = crash(cn, eta)
                failed[label][eta].append(fraction)
                if restricted:
                    reached[eta].append(communities)

    # The article's Fig. 8, on the parameters printed in its own caption:
    # C = 10, e = 1, n0 = 4, c_min = 1, C_min = 0, d_in = 2, d_ex = 1,
    # alpha = 2, t = 200, contagion begun in the greatest node.  The caption
    # does not give r, so both ends of the article's own working range are
    # drawn: r = 0, where its fig. 10 prints 68%, and r = 30, which is where
    # that same curve passes the 59% of fig. 8.
    paper_eta = 20.0
    paper_rs = (0.0, 30.0)
    paper: dict[str, dict[float, list[float]]] = {
        label: {r: [] for r in paper_rs} for label in ("unrestricted", "restricted")
    }
    for label, restricted in (("unrestricted", False), ("restricted", True)):
        for _ in range(realisations):
            cn, _ = confidence_strategy_growth(
                rng=ctx.rng,
                founding_banks=4,
                steps=200,
                interior_max=2,
                exterior_max=1,
                confidence=2.0,
                restricted=restricted,
            )
            for r in paper_rs:
                paper[label][r].append(crash(cn, paper_eta, r)[0])

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.3), width_ratios=[1.55, 1.0])

    # --- the thesis: both networks, across the reserves the article works in --
    x = np.arange(len(etas))
    width = 0.34
    for i, label in enumerate(("unrestricted", "restricted")):
        means = np.array([np.mean(failed[label][eta]) for eta in etas]) * 100
        spread = np.array([np.std(failed[label][eta]) for eta in etas]) * 100
        positions = x + (i - 0.5) * width
        left.bar(positions, means, width, color=SERIES[i], zorder=2)
        left.errorbar(
            positions,
            means,
            yerr=spread,
            fmt="none",
            ecolor=INK["primary"],
            elinewidth=0.9,
            capsize=2.5,
            zorder=4,
        )
        # Written clear of the error bar rather than on the bar top, where the
        # upper cap ran through the digits.
        for pos, value, bar in zip(positions, means, spread, strict=True):
            left.annotate(
                f"{value:.0f}%",
                xy=(pos, value + bar),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=INK["secondary"],
            )

    # Both cut-offs used to sit in the same top-right corner (reference_line
    # anchors there for side="right"), so the two labels landed on top of one
    # another. They are stacked in that corner instead, since the top-left is
    # where the key goes -- "unrestricted"/"restricted" is said once already, by
    # the key and by each line's own colour, so a label only needs its number.
    reference_line(left, 80, "80%", axis="y", side="right", color=SERIES[0])
    reference_line(left, 10, "10%", axis="y", side="right", color=SERIES[1])
    _key(left, [("unrestricted, $t = 100$", SERIES[0]), ("restricted, $t = 1000$", SERIES[1])])
    communities = float(np.mean([np.mean(reached[eta]) for eta in etas]))
    left.annotate(
        f"the loss reaches {communities:.0f} of the 10 communities",
        xy=(0.02, 0.83),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=8.0,
        color=INK["secondary"],
    )
    left.set_xticks(x)
    left.set_xticklabels([f"{eta:.0f}" for eta in etas])
    left.set_xlabel(r"reserves $\eta$ held by the net borrowers (%)")
    left.set_ylim(0, 125)
    left.set_ylabel("banks that fail (%)")
    left.set_title("Fewer banks fail under the restriction", fontsize=9.0)

    # --- the article's Fig. 8, the same restriction on its own network --------
    xr = np.arange(len(paper_rs))
    for i, label in enumerate(("unrestricted", "restricted")):
        means = np.array([np.mean(paper[label][r]) for r in paper_rs]) * 100
        spread = np.array([np.std(paper[label][r]) for r in paper_rs]) * 100
        positions = xr + (i - 0.5) * width
        right.bar(positions, means, width, color=SERIES[i], zorder=2)
        right.errorbar(
            positions,
            means,
            yerr=spread,
            fmt="none",
            ecolor=INK["primary"],
            elinewidth=0.9,
            capsize=2.5,
            zorder=4,
        )
        for pos, value, bar in zip(positions, means, spread, strict=True):
            right.annotate(
                f"{value:.0f}%",
                xy=(pos, value + bar),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=INK["secondary"],
            )
    reference_line(right, 59, "2012: 59%", axis="y", side="right", color=SERIES[0])
    reference_line(right, 48, "2012: 48%", axis="y", side="right", color=SERIES[1])
    right.set_xticks(xr)
    right.set_xticklabels([f"{r:.0f}" for r in paper_rs])
    right.set_xlabel(f"reaction threshold $r$ (%), $\\eta = {paper_eta:.0f}$")
    right.set_ylim(0, 125)
    right.set_ylabel("banks that fail (%)")
    right.set_title("The article's own fig. 8", fontsize=9.0)

    return fig


def _unused_marker():
    """Keep BankState importable for downstream figure work."""
    return BankState.CRASHED
