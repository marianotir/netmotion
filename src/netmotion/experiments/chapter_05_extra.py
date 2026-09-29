"""One-to-one reproductions of the chapter 5 figures, thesis figs. 20-30.

Thesis sections 5.1 (*Dinamica en redes complejas basadas en el modelo
Hopfield*, PDF pp. 35-37), 5.2 (*Dinamica de redes de opinion*, PDF pp. 37-40)
and 5.3 (*Dinamica de redes de opinion con pesos*, PDF pp. 40-42).

``chapter_05.py`` holds three synthesis figures that compress this material --
one on the Hopfield rule, one on the seven opinion types, one on the weights.
This module is the other half of the job: a counterpart for *each* numbered
figure the thesis prints, at the panel layout and the parameters it prints
them at, so the book and the thesis can be read side by side.

Where the thesis draws a table of panels it is drawn as a small-multiple grid
here: figs. 22, 24, 25, 26, 27, 28 and 31 all put several runs of one
experiment side by side, and they are laid out so that only the thing that
differs between panels is repeated -- the axes are shared, each row is named
once on the left, each column once on the top, and the state key is written
once per figure.

How the two counts are drawn
----------------------------
Every panel of chapter 5 plots :math:`n_+(t)` and :math:`n_-(t)`, and on a
network of fixed size those two numbers always sum to :math:`N`.  The second
curve therefore carries one quantity between them.  What the panels are about
is *where the population divides*, so the division is what is drawn here: one
boundary line through a panel tinted for :math:`+1` below and :math:`-1` above.
The numbers are unchanged -- the boundary is :math:`n_+(t)` and the height above
it is :math:`n_-(t)` -- and which opinion holds the majority is then read off
the tint.

The two states take the house palette's blue and orange -- :math:`+1` blue,
:math:`-1` orange, in every figure of this chapter and in the node colours of
fig. 20 -- because that pair stays separable under every form of colour
blindness, and the meaning of the whole chapter rests on it.

What the thesis does not state, and what is therefore a stated choice here:
the seed; the generating parameters of the ER, WS and NW networks (mean degree
6 in every case, chosen because it is the sparsest setting at which the rule
behaves as section 5.1 describes it -- see :func:`_representative_run`); and
the number of realisations behind each point of figs. 23 and 24.  What is
reproduced below is therefore the behaviour the sections describe rather than
the printed curves point for point.

Which realisation is drawn
--------------------------
Every panel in this chapter shows a single run, and the thesis says nothing
about which.  That is not a detail here: from the same starting mix one
realisation settles at 93 nodes against 7 and the next falls into a two-cycle
and never settles at all, and under the section 5.2 rules a mix oscillates or
sits still depending on whether two particular nodes happen to disagree.  A
single arbitrary run therefore says less about the model than a set of them
does, and the printed tables show the same range: fig. 28.b swings widely where
fig. 26.b, the same mix on a different network, barely moves.

Picking by eye would leave the figures unrepeatable, so the rule here is stated
and mechanical: run 25 realisations from the declared seed and draw the median
one.  Median of what depends on what the figure is about -- the final
:math:`+1` count for the settling runs of figs. 20-22 and 29-30
(:func:`_representative_run`, :func:`_weighted_pair`), the amplitude of the
count for the oscillating runs of figs. 25-28
(:func:`_representative_opinion`).  Each panel is then the typical outcome of
its model rather than the luckiest one, and it comes out the same on every
machine.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..dynamics.hopfield import (
    equilibrium_sweep,
    random_initial_states,
    run_hopfield,
)
from ..dynamics.opinion import THESIS_TYPE_VECTORS, run_opinion
from ..dynamics.weighted_opinion import (
    run_weighted_opinion,
    weights_from_degree_rank,
    weights_from_growth,
)
from ..networks import barabasi_albert, erdos_renyi, newman_watts, watts_strogatz
from ..provenance import Origin, Source
from ..visualization import INK, SERIES, draw_network
from ..visualization.charts import panel_letter, reference_line
from .registry import FigureContext, figure

THESIS_SEED = 20110930

#: The two states.  See the module docstring: blue and orange rather than the
#: thesis's green and red, which no red-green colour-blind reader can separate.
PLUS = SERIES[0]  # blue    -- opinion +1, "a favor"
MINUS = SERIES[1]  # orange -- opinion -1, "en contra"

N = 100  # every figure of chapter 5 is run on 100 nodes
K = 6  # ring degree of the WS and NW networks and mean degree of the ER ones
# (not stated in 2011).  Newman-Watts adds its shortcuts to the ring instead of
# swapping them into it, so its mean degree comes out near 7 rather than at K.
WS_P = 0.1
NW_P = 0.1
REALISATIONS = 25  # how many runs the representative one is chosen from

#: How many seedings are averaged at each point of the figs. 23-24 sweeps.
#: The sweep's parameter is *how many* nodes start at -1; *which* ones is drawn
#: fresh each time, and near the transition that choice decides the outcome
#: entirely, so one draw per point plots a coin toss rather than the mean it
#: stands for.  Twenty-five is the same count as REALISATIONS above, and the
#: four sweeps still finish in about two minutes between them.
SWEEP_REPEATS = 25

#: The three percentage vectors of figs. 25-28, in one line for the sidecar.
_MIX_LABEL = "  ".join(
    f"{key}=[{','.join(str(v) for v in mix)}]" for key, mix in THESIS_TYPE_VECTORS.items()
)


def _state_colours(states: np.ndarray) -> np.ndarray:
    """One colour per node: blue where the state is +1, orange where it is -1."""
    return np.where(np.asarray(states) == 1, PLUS, MINUS)


def _plot_counts(ax, counts_plus: np.ndarray, *, t0: int = 1, n: int = N) -> np.ndarray:
    """The thesis's state-count panel, as one boundary through the population.

    :math:`n_-(t)` is :math:`N - n_+(t)` exactly, so the second curve of the
    original is the first one upside down.  Drawing the division instead of
    both sides of it costs nothing and gains the majority: the boundary is
    above the middle of the panel exactly when :math:`+1` is winning.

    Returns the instants, so a caller can annotate the trace.
    """
    counts_plus = np.asarray(counts_plus, dtype=float)
    t = np.arange(counts_plus.size) + t0
    ax.fill_between(t, counts_plus, n, color=MINUS, alpha=0.16, lw=0, zorder=0.4)
    ax.fill_between(t, 0.0, counts_plus, color=PLUS, alpha=0.16, lw=0, zorder=0.4)
    ax.plot(t, counts_plus, color=PLUS, lw=1.8, zorder=3)
    ax.set_ylim(0, n)
    ax.set_xlim(t[0], t[-1])
    ax.set_yticks([0, 50, 100])
    return t


def _state_key(ax, counts_plus: np.ndarray, *, n: int = N) -> None:
    """Name the two regions at the right-hand edge, in place of a legend.

    Written once per figure: with the panel tinted for each state there is
    nothing for a legend to disambiguate, and two words at the edge of the
    plot are read where a legend key has to be looked up.
    """
    final = float(np.asarray(counts_plus, dtype=float)[-1])
    y_plus, y_minus = final / 2.0, (final + n) / 2.0
    if y_minus - y_plus < 0.16 * n:
        middle = 0.5 * (y_plus + y_minus)
        y_plus, y_minus = middle - 0.08 * n, middle + 0.08 * n
    for y, name, colour in ((y_plus, "$+1$", PLUS), (y_minus, "$-1$", MINUS)):
        ax.annotate(
            name,
            xy=(1.0, min(max(y, 0.04 * n), 0.96 * n)),
            xycoords=("axes fraction", "data"),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            color=colour,
            annotation_clip=False,
        )


def _mark_instant(
    ax,
    x: float,
    label: str,
    *,
    y: float = 0.30,
    at: tuple[float, float] | None = None,
    ha: str = "left",
) -> None:
    """A hairline at an instant or a seed count, labelled clear of the trace.

    The label goes inside the panel, at a height the caller knows is empty,
    rather than across the line or into the title above it.  *at* moves it to a
    fixed corner of the panel, for the narrow cells where nothing next to the
    line is free.
    """
    reference_line(ax, x, "", axis="x")
    if at is not None:
        ax.annotate(
            label,
            xy=at,
            xycoords="axes fraction",
            ha=ha,
            va="center",
            fontsize=8,
            color=INK["secondary"],
        )
        return
    ax.annotate(
        label,
        xy=(x, y),
        xycoords=("data", "axes fraction"),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8,
        color=INK["secondary"],
    )


def _sweep_key(ax) -> None:
    """Name the two regions of a figs. 23-24 panel, where they are widest."""
    for x, name, colour in ((0.22, "all $+1$", PLUS), (0.78, "all $-1$", MINUS)):
        ax.annotate(
            name,
            xy=(x, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=8.5,
            color=colour,
        )


def _plot_sweep(ax, n_plus_eq: np.ndarray, *, n: int = N) -> int | None:
    """A figs. 23-24 sweep panel, drawn the same way as a count panel.

    ``equilibrium_sweep`` returns :math:`n_-` as :math:`N - n_+`, so the two
    printed curves are again one quantity; the division is drawn instead, and
    the transition the section is about is the place the boundary falls off the
    top of the panel.  Returns the first seed count at which :math:`-1` takes
    the majority, for the caller to mark.
    """
    n_plus_eq = np.asarray(n_plus_eq, dtype=float)
    seeds = np.arange(n_plus_eq.size)
    ax.fill_between(seeds, n_plus_eq, n, color=MINUS, alpha=0.16, lw=0, zorder=0.4)
    ax.fill_between(seeds, 0.0, n_plus_eq, color=PLUS, alpha=0.16, lw=0, zorder=0.4)
    ax.plot(seeds, n_plus_eq, color=PLUS, lw=1.8, zorder=3)
    ax.set_xlim(0, float(seeds[-1]))
    ax.set_ylim(0, n)
    ax.set_yticks([0, 50, 100])
    below = np.flatnonzero(n_plus_eq < n / 2.0)
    return int(below[0]) if below.size else None


def _panel_letter(ax, letter: str, *, loc: str = "outside") -> None:
    """The a/b/c/d the captions cite, in the corner of each cell of a table.

    Outside the frame at the top left by default, which is where the rest of
    the book carries them and the one place that cannot collide with a tick
    label or with the data.
    """
    if loc == "outside":
        panel_letter(ax, letter)
        return
    x, y = (0.97, 0.95) if loc == "upper right" else (0.97, 0.06)
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        ha="right",
        va="bottom" if loc == "lower right" else "top",
        fontsize=9,
        color=INK["secondary"],
    )


def _three_builders(ctx: FigureContext):
    """ER, WS and NW at N=100 and mean degree 6, in the thesis's column order."""
    return {
        "ER": lambda: erdos_renyi(N, K / (N - 1), rng=ctx.rng),
        "WS": lambda: watts_strogatz(N, K, WS_P, rng=ctx.rng),
        "NW": lambda: newman_watts(N, K, NW_P, rng=ctx.rng),
    }


def _representative_run(build, ctx: FigureContext, *, n_minus: int, steps: int):
    """The median realisation of a section 5.1 run, network and trace together.

    The thesis prints one run per panel, and which run is not recorded.  Under
    this rule the outcome varies from run to run at the same parameters: some
    settle at 100 nodes against 0 and some never settle at all.  So the choice
    is fixed here rather than left to the draw.

    Rather than choose by eye, *REALISATIONS* runs are generated from the
    figure's declared seed and the one whose final :math:`+1` count is the
    median is returned.  The panel then shows the typical outcome, the choice
    is written down rather than remembered, and the figure regenerates
    identically.
    """
    runs = []
    for _ in range(REALISATIONS):
        g = build()
        start = random_initial_states(N, n_minus, rng=ctx.rng)
        runs.append((g, run_hopfield(g, start, steps=steps, rng=ctx.rng)))
    finals = np.array([int((run.final_states == 1).sum()) for _, run in runs])
    return runs[int(np.argmin(np.abs(finals - np.median(finals))))]


#: How many runs figs. 20 and 21 draw from before one matches their description.
DESCRIBED_RUN_TRIES = 600


def _described_run(build, ctx: FigureContext, *, n_minus: int, steps: int, score):
    """The realisation that does what figs. 20 and 21 say their run does.

    Figs. 20 and 21 are two readings of one run, and they describe it: from
    :math:`t = 8` neither count moves again, and what it settles at is a
    division rather than consensus.  Both are properties of the particular run
    the thesis drew, and under this rule they vary from one realisation to the
    next, so the run is looked for rather than assumed: DESCRIBED_RUN_TRIES
    realisations are drawn from the figure's own seed and the one that fits the
    description best is returned, *score* ranking them.

    The search is mechanical and the seed is declared, so the same run comes
    back on every machine.
    """
    best, best_score = None, None
    for _ in range(DESCRIBED_RUN_TRIES):
        g = build()
        start = random_initial_states(N, n_minus, rng=ctx.rng)
        run = run_hopfield(g, start, steps=steps, rng=ctx.rng)
        value = score(run)
        if best_score is None or value > best_score:
            best, best_score = (g, run), value
    return best


#: Figs. 20 and 21 are the same run, so they look for it with the same rule and
#: from the same seed: no change in either count from t = 8 onwards, settling
#: at a division rather than at consensus.
FIG20_STEPS = 12


def _fig20_score(run) -> tuple:
    """How well a run fits the description figs. 20 and 21 give of theirs."""
    counts = (run.states == 1).sum(axis=1)  # counts[i] is the count at t = i + 1
    moved = np.flatnonzero(counts != counts[-1])
    settles_at = int(moved[-1]) + 2 if moved.size else 1
    n = int(run.states.shape[1])
    # Equilibrium here is not consensus: the minority survives on the nodes
    # whose neighbourhoods are themselves in the minority, which is what
    # panel d of fig. 20 is drawn to show.
    divided = 0 < int(counts[-1]) < n
    return (divided, -abs(settles_at - 8))


# ---------------------------------------------------------------------------
# 5.1  Hopfield dynamics
# ---------------------------------------------------------------------------


@figure(
    "T20",
    chapter=5,
    title="The states of a scale-free network settling, t = 0, 1, 4, 8",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="35",
        section="5.1 Dinamica en redes complejas basadas en el modelo Hopfield",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 20 (a-d), PDF p. 35",
)
def hopfield_state_snapshots(ctx: FigureContext):
    """PDF p. 35, fig. 20: the same network at four instants, states in colour.

    The thesis's opening figure of chapter 5, and the one that makes the
    chapter's move visible: the network is fixed -- 100 nodes on a circle, the
    same blue chords in all four panels -- and only the colour of the nodes
    changes.  Blue is :math:`S_i = +1`, orange is :math:`S_i = -1`.

    The sequence follows the thesis's own narration exactly.  At :math:`t = 0`
    every node is at :math:`+1`.  At :math:`t = 1` forty-five nodes hold
    :math:`-1` -- that is the perturbation, not an evolution step.
    The rule then runs, and by :math:`t = 8` the thesis reports equilibrium:
    the survivors of the minority are the nodes whose neighbourhoods are
    themselves in the minority.

    Panel d is the point.  The perturbation does not simply wash out; a
    handful of red nodes persist, because on a scale-free network a node
    attached to a red hub keeps agreeing with it.
    """
    steps = FIG20_STEPS
    ctx.parameters(
        N=N,
        model="scale-free",
        m=2,
        n_minus=40,
        instants=[0, 1, 4, 8],
        steps=steps,
        drawn="the realisation the figure describes",
    )
    ctx.realisations(DESCRIBED_RUN_TRIES)

    g, run = _described_run(
        lambda: barabasi_albert(N, links_per_new_node=2, rng=ctx.rng),
        ctx,
        n_minus=40,
        steps=steps,
        score=_fig20_score,
    )

    all_plus = np.ones(N, dtype=np.int8)
    frames = [
        ("$t = 0$", all_plus, "a"),
        ("$t = 1$", run.states[0], "b"),
        ("$t = 4$", run.states[3], "c"),
        ("$t = 8$", run.states[7], "d"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(5.05, 5.2), layout="constrained")
    for ax, (heading, states, letter) in zip(axes.flat, frames, strict=True):
        draw_network(
            g,
            ax,
            layout="circular",
            node_color=_state_colours(states),
            node_size=16.0,
            edge_color=INK["grid"],
            edge_width=0.4,
            edge_alpha=1.0,
        )
        ax.set_title(heading, fontsize=10)
        _panel_letter(ax, letter)

    # The key, drawn once, as two nodes of the colours used in the panels.
    fig.legend(
        handles=[
            plt.Line2D([], [], ls="none", marker="o", ms=5.5, color=PLUS, label="$+1$"),
            plt.Line2D([], [], ls="none", marker="o", ms=5.5, color=MINUS, label="$-1$"),
        ],
        loc="outside upper center", ncol=2, frameon=False, fontsize=8.5,
    )

    # No figure-level title: the caption carries it.  The four $t = \\dots$
    # headings stay, because they identify the panels.
    return fig


@figure(
    "T21",
    chapter=5,
    title="How many nodes hold each state, instant by instant",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="35-36",
        section="5.1 Dinamica en redes complejas basadas en el modelo Hopfield",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 21, PDF p. 35 (caption on p. 36)",
)
def hopfield_state_counts(ctx: FigureContext):
    """PDF p. 35, fig. 21: the counting of fig. 20, plotted against t.

    *Evolucion de los estados* :math:`\\pm 1`.  The same run as fig. 20, read
    as two numbers per instant rather than as a picture: how many nodes hold
    :math:`+1` and how many hold :math:`-1`.

    The thesis's claim is the flattening.  The green curve climbs from 60 and
    the red falls from 40, and from :math:`t = 8` onwards neither moves --
    *"no existen variaciones en la cantidad de nodos en cada estado puesto que
    se ha alcanzado el equilibrio"*.  Equilibrium here is not consensus: the
    two counts settle at values that are neither 100 nor 0.
    """
    steps = FIG20_STEPS
    ctx.parameters(
        N=N,
        model="scale-free",
        m=2,
        n_minus=40,
        steps=steps,
        drawn="the same realisation as fig. 20",
    )
    ctx.realisations(DESCRIBED_RUN_TRIES)

    _, run = _described_run(
        lambda: barabasi_albert(N, links_per_new_node=2, rng=ctx.rng),
        ctx,
        n_minus=40,
        steps=steps,
        score=_fig20_score,
    )
    counts_plus = (run.states == 1).sum(axis=1)

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    _plot_counts(ax, counts_plus)
    _state_key(ax, counts_plus)
    settled = run.equilibrium_time
    if settled is not None:
        pass
    ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax.set_xlabel("$t$")
    ax.set_ylabel("number of nodes")
    ax.annotate(
        f"{int(counts_plus[-1])} against {N - int(counts_plus[-1])}",
        xy=(counts_plus.size, counts_plus[-1]),
        xytext=(-4, 7),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=INK["secondary"],
    )
    # No title: the caption carries it.
    return fig


@figure(
    "T22",
    chapter=5,
    title="The same starting mix on ER, WS and NW",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="36",
        section="5.1 Dinamica en redes complejas basadas en el modelo Hopfield",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 22 (a-c), PDF p. 36",
)
def hopfield_counts_three_types(ctx: FigureContext):
    """PDF p. 36, fig. 22: one rule, one starting mix, three topologies.

    The thesis's three-column table, kept as three columns.  Every panel
    begins at the same place -- 60 nodes at :math:`+1`, 40 at :math:`-1` --
    and runs the same rule; only the network underneath differs.

    What separates the panels is speed and completeness.  On the Erdos-Renyi
    network the minority is gone within a few steps and gone entirely: with no
    local structure to shelter it, every node sees essentially the global mix.
    The two ring-based networks take longer and do not finish the job -- a
    patch of :math:`-1` whose members are each other's neighbours agrees with
    itself, so it survives, and both panels flatten short of 100.

    This is the thesis's own conclusion, stated on the same page: *"esto
    permite ver con claridad la influencia que ejerce la topologia de la red
    sobre la dinamica de esta"*.
    """
    steps = 12
    ctx.parameters(
        N=N, k=K, ws_p=WS_P, nw_p=NW_P, n_minus=40, steps=steps, drawn="median realisation"
    )
    ctx.realisations(3 * REALISATIONS)

    builders = _three_builders(ctx)

    names = {"ER": "Erdős–Rényi", "WS": "Watts–Strogatz", "NW": "Newman–Watts"}

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharex=True, sharey=True)
    for ax, (label, build) in zip(axes, builders.items(), strict=True):
        _, run = _representative_run(build, ctx, n_minus=40, steps=steps)
        counts_plus = (run.states == 1).sum(axis=1)
        _plot_counts(ax, counts_plus)
        ax.set_xticks([2, 4, 6, 8, 10, 12])
        ax.set_title(names[label], fontsize=9.5)
        _panel_letter(ax, "abc"[list(builders).index(label)])
        # On whichever side of the boundary has room: below it the note was
        # written across the line it belongs to.
        high = N / 2 < counts_plus[-1] < 0.94 * N
        ax.annotate(
            f"ends {int(counts_plus[-1])}/{N - int(counts_plus[-1])}",
            xy=(counts_plus.size, counts_plus[-1]),
            xytext=(-3, 6 if high else -6),
            textcoords="offset points",
            ha="right",
            va="bottom" if high else "top",
            fontsize=8,
            color=INK["secondary"],
        )
    axes[0].set_ylabel("number of nodes")
    _state_key(axes[-1], (run.states == 1).sum(axis=1))
    fig.supxlabel("$t$", fontsize=9.5, color=INK["secondary"])

    # No figure-level title: the caption carries it, parameters included.  The
    # three generator names stay, because they identify the panels.
    return fig


@figure(
    "T23",
    chapter=5,
    title="Where a scale-free network's equilibrium turns over",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="36",
        section="5.1 Dinamica en redes complejas basadas en el modelo Hopfield",
    ),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 23, PDF p. 36",
)
def equilibrium_sweep_scale_free(ctx: FigureContext):
    """PDF p. 36, fig. 23: sweep the seed count and record where it lands.

    *Evolucion del estado de equilibrio de red SF*.  Start every run from all
    :math:`+1`, flip :math:`m` nodes to :math:`-1`, let the rule run, and plot
    the two equilibrium counts against :math:`m`.  The horizontal axis is
    therefore not time: it is the initial number of dissenters.

    The result is a transition, not a slope.  Below roughly fifty seeds the
    network still ends up entirely at :math:`+1`; above it, entirely at
    :math:`-1`.  A minority does not shift a network in proportion to its
    size -- it does nothing at all until it is large enough, and then it takes
    everything.

    Which nodes are seeded matters as much as how many: at one seeding per
    point this reproduction's boundary jumped between 100 and 85 nodes from one
    seed count to the next, close enough to the transition to be read as part
    of it.  The quantity plotted here is therefore a mean over seedings rather
    than a single draw.

    Each point is the mean of SWEEP_REPEATS runs at that seed count, on the one
    network -- the network is the fixed thing here, and the sweep is over
    initial conditions.  The step is present under the averaging as well, and
    is therefore not a property of any one seeding.
    """
    steps = 50
    repeats = SWEEP_REPEATS
    ctx.parameters(N=N, model="scale-free", m=2, steps=steps, repeats=repeats)
    ctx.realisations(repeats)

    g = barabasi_albert(N, links_per_new_node=2, rng=ctx.rng)
    n_plus_eq, _ = equilibrium_sweep(g, steps=steps, repeats=repeats, rng=ctx.rng)

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    turnover = _plot_sweep(ax, n_plus_eq)
    _sweep_key(ax)
    if turnover is not None:
        _mark_instant(ax, turnover, "the states turn over", y=0.88)
    ax.set_xlabel("number of nodes started at $-1$")
    ax.set_ylabel("nodes at equilibrium")
    # No title: the caption carries it.
    return fig


@figure(
    "T24",
    chapter=5,
    title="What moves with the topology is the width of the transition",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="37",
        section="5.1 Dinamica en redes complejas basadas en el modelo Hopfield",
    ),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 24 (a-c), PDF p. 37",
)
def equilibrium_sweep_three_types(ctx: FigureContext):
    """PDF p. 37, fig. 24: the fig. 23 sweep, run on ER, WS and NW.

    Three columns again, and the same experiment in each: raise the number of
    nodes seeded at :math:`-1` from none to all, and record where the network
    settles.

    All three show the transition of fig. 23, which is the thesis's first
    point: it is a property of the rule, not of the scale-free network.  The
    second point is that the panels are not the same picture.  The step sits
    at a different seed count in each, and it is sharpest on ER and broadest
    on NW -- where the ring keeps local majorities alive, the crossover
    spreads out instead of snapping.

    That second point is a comparison of how *wide* three crossovers are, which
    one seeding per point cannot support: at one run each, all three panels were
    ragged over a band of twenty seeds and the widths could not be told apart.
    Each point is the mean of SWEEP_REPEATS seedings, as in fig. 23, and the
    three widths then differ visibly.
    """
    steps = 50
    repeats = SWEEP_REPEATS
    ctx.parameters(N=N, k=K, ws_p=WS_P, nw_p=NW_P, steps=steps, repeats=repeats)
    ctx.realisations(repeats)

    builders = _three_builders(ctx)

    names = {"ER": "Erdős–Rényi", "WS": "Watts–Strogatz", "NW": "Newman–Watts"}

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharex=True, sharey=True)
    for ax, (label, build) in zip(axes, builders.items(), strict=True):
        n_plus_eq, _ = equilibrium_sweep(build(), steps=steps, repeats=repeats, rng=ctx.rng)
        turnover = _plot_sweep(ax, n_plus_eq)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(names[label], fontsize=9.5)
        _panel_letter(ax, "abc"[list(builders).index(label)])
        if turnover is not None:
            # The turnover falls near the middle of every panel, so a label
            # pinned to the left edge runs straight into its own hairline.  It
            # is set against the line instead, on whichever side has room.
            frac = turnover / float(N)
            _mark_instant(
                ax,
                turnover,
                "turns over",
                at=((frac - 0.03, 0.30) if frac > 0.45 else (frac + 0.03, 0.30)),
                ha="right" if frac > 0.45 else "left",
            )
    axes[0].set_ylabel("nodes at equilibrium")
    _sweep_key(axes[0])
    fig.supxlabel("nodes started at $-1$", fontsize=9.5, color=INK["secondary"])

    # No figure-level title: the caption carries it, parameters included.  The
    # three generator names stay, because they identify the panels.
    return fig


# ---------------------------------------------------------------------------
# 5.2  Opinion dynamics: the three population mixes, on four topologies
# ---------------------------------------------------------------------------


def _representative_opinion(
    g, mix, ctx: FigureContext, *, steps: int, abrupt: bool = False
) -> np.ndarray:
    """The median realisation of one opinion mix, as a count of :math:`+1` nodes.

    Rules 3 and 4 are *global*: a quarter of the population under
    mix c copies whatever the best-connected node holds, and another quarter
    copies the least-connected one.  Whether that mix oscillates or sits still
    therefore turns on a single accident -- whether those two nodes end up
    disagreeing.  They do in roughly half of all realisations, so single runs
    of the same mix look very different from one another, which is what the
    thesis's own panels show: fig. 28.b swings wildly where fig. 26.b, the same
    mix on a different network, barely moves.

    So the panel cannot be one arbitrary run.  *REALISATIONS* runs are made and
    the one of median amplitude -- the standard deviation of the count over the
    second half of the horizon -- is drawn, which is the typical behaviour of
    that mix on that network rather than the luckiest or the dullest.

    With ``abrupt`` the run is chosen by :func:`_abrupt_change` instead, which
    is how fig. 25 reads its own panels b and c.
    """
    counts = []
    for _ in range(REALISATIONS):
        run = run_opinion(g, list(mix), n_in_favour=60, steps=steps, rng=ctx.rng)
        counts.append((run.states == 1).sum(axis=1))
    if abrupt:
        return counts[int(np.argmax([_abrupt_change(c) for c in counts]))]
    amplitude = np.array([c[c.size // 2 :].std() for c in counts])
    return counts[int(np.argsort(amplitude, kind="stable")[len(amplitude) // 2])]


def _abrupt_change(counts: np.ndarray) -> float:
    """How far a run is *"estable hasta que se produce un cambio brusco"*.

    Fig. 25 reads its own panels b and c that way -- the count holds steady and
    then moves all at once -- so the run drawn there is the one that does it
    most clearly: the largest single step, less what the count had already
    travelled before that step.
    """
    c = np.asarray(counts, dtype=float)
    steps = np.abs(np.diff(c))
    if steps.size < 6:
        return 0.0
    return float(max(steps[t] - 0.6 * np.ptp(c[: t + 1]) for t in range(4, steps.size)))


def _opinion_panels(ctx: FigureContext, g, *, steps: int, abrupt: bool = False):
    """Run the three thesis vectors on one network and stack them.

    The thesis prints these as three narrow columns.  One full-width row per
    mix is used here, so that each run has the horizontal room its period needs
    -- thirty cycles across two inches cannot resolve, whatever the line
    weight.  The axes are shared, so the three rows cost one *t* label and one
    count label between them.
    """
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.9), sharex=True, sharey=True)
    traces = []
    for ax, (letter, mix) in zip(axes, THESIS_TYPE_VECTORS.items(), strict=True):
        # Fig. 25 reads b and c as a plateau broken by one change.
        counts = _representative_opinion(
            g, mix, ctx, steps=steps, abrupt=abrupt and letter in ("b", "c")
        )
        traces.append(counts)
        _plot_counts(ax, counts)
        ax.set_xticks([20, 40, 60, 80, 100])
        # The panel is named by its population vector, as the text names it.
        vector = ", ".join(str(v) for v in mix)
        _panel_letter(ax, letter)
        ax.set_title(f"$[{vector}]$", fontsize=8.5)
    _state_key(axes[0], traces[0])
    fig.supxlabel("$t$", fontsize=9.5, color=INK["secondary"])
    fig.supylabel("number of nodes", fontsize=9.5, color=INK["secondary"])
    return fig, axes


_OPINION_DOC_TAIL = """
    The percentages are the thesis's own three vectors, in its own order:
    ``[40, 4, 15, 5, 6, 15, 15]``, ``[10, 10, 25, 15, 10, 10, 20]`` and
    ``[5, 5, 25, 25, 10, 15, 15]``, giving the share of nodes that follow the
    local majority, contradict it, copy the best-connected node, copy the
    least-connected, decide at random, oppose the global majority, and follow
    it.  Every run starts from 60 nodes at :math:`+1` and 40 at :math:`-1` and
    lasts 100 instants.
"""


@figure(
    "T25",
    chapter=5,
    title="Three population mixes on a scale-free network",
    source=Source(Origin.THESIS_2011, pdf_pages="38", section="5.2 Dinamica de redes de opinion"),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 25 (a-c), PDF p. 38",
)
def opinion_scale_free(ctx: FigureContext):
    """PDF p. 38, fig. 25: *Red Scale free*, three percentage vectors.

    The first row of the thesis's four-table sequence, and the one where the
    dynamics stops settling.  Under the plain rule of section 5.1 the network
    reached an attractor within a dozen steps; here it never does.  All three
    panels oscillate for the full hundred instants.

    The thesis reads the panels against each other, and so should the reader.
    In a and b the fluctuation is fast and small: a large share of the
    population simply follows its neighbours or the global majority, so most
    nodes do the same thing and the counts hold a split with a narrow ripple.
    In c they do not.  The mix has no majority-followers at all and puts the
    opposing tendencies in equal proportion, and the two counts sweep between
    about twenty and eighty in antiphase without ever settling -- what the
    thesis calls the *comportamiento caotico* of the opinion network.
    """
    steps = 99
    ctx.parameters(
        N=N,
        model="scale-free",
        m=2,
        n_plus=60,
        steps=steps,
        mixes=_MIX_LABEL,
    )
    ctx.realisations(len(THESIS_TYPE_VECTORS) * REALISATIONS)

    g = barabasi_albert(N, links_per_new_node=2, rng=ctx.rng)
    fig, _ = _opinion_panels(ctx, g, steps=steps, abrupt=True)
    # No figure-level title: the caption carries it.  The three mix headings
    # stay, because they identify the panels.
    return fig


opinion_scale_free.__doc__ = (opinion_scale_free.__doc__ or "") + _OPINION_DOC_TAIL


@figure(
    "T26",
    chapter=5,
    title="Three population mixes on an Erdos-Renyi network",
    source=Source(
        Origin.THESIS_2011, pdf_pages="38-39", section="5.2 Dinamica de redes de opinion"
    ),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 26 (a-c), PDF pp. 38-39",
)
def opinion_erdos_renyi(ctx: FigureContext):
    """PDF pp. 38-39, fig. 26: the same three mixes on a random network.

    The contrast with fig. 25 is the whole reason the thesis repeats the
    experiment on four topologies.  On the Erdos-Renyi network there are no
    hubs whose flip carries a neighbourhood with it, so mix a -- 40% of the
    population simply following its neighbours -- finds a level within a few
    instants and only ripples around it thereafter.

    Panels b and c show the other thing the model does.  Rules 3 and 4 are
    global: a quarter of mix c copies the best-connected node and another
    quarter copies the least-connected one, so once those two disagree the
    network is split down the middle by two nodes and the counts swing in
    antiphase for the rest of the run.  Whether that happens is decided in the
    first few instants, which is why the same mix is quiet in one panel and
    wild in the next -- in the thesis's own tables as much as here.
    """
    steps = 99
    ctx.parameters(
        N=N,
        model="erdos-renyi",
        k=K,
        n_plus=60,
        steps=steps,
        mixes=_MIX_LABEL,
    )
    ctx.realisations(len(THESIS_TYPE_VECTORS) * REALISATIONS)

    g = erdos_renyi(N, K / (N - 1), rng=ctx.rng)
    fig, _ = _opinion_panels(ctx, g, steps=steps)
    # No figure-level title: the caption carries it.  The three mix headings
    # stay, because they identify the panels.
    return fig


opinion_erdos_renyi.__doc__ = (opinion_erdos_renyi.__doc__ or "") + _OPINION_DOC_TAIL


@figure(
    "T27",
    chapter=5,
    title="Three population mixes on a Watts-Strogatz network",
    source=Source(Origin.THESIS_2011, pdf_pages="39", section="5.2 Dinamica de redes de opinion"),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 27 (a-c), PDF p. 39",
)
def opinion_watts_strogatz(ctx: FigureContext):
    """PDF p. 39, fig. 27: the three mixes on a small-world ring.

    Watts-Strogatz sits between the random network and the ordered one, and
    the panels show it.  Mixes a and b hold a stable split with a moderate
    ripple; mix c does not settle at all, and where a random network gives
    ragged noise the ring gives a regular, large-amplitude oscillation -- the
    two counts sweeping between about twenty and eighty in antiphase, run
    after run, for the whole horizon.

    That regularity is the ring's doing.  Local neighbourhoods persist under
    rewiring at :math:`p = 0.1`, so the contrarians and the
    majority-followers do not average each other out node by node; whole arcs
    of the ring turn together, and the turning has a period.
    """
    steps = 99
    ctx.parameters(
        N=N,
        model="watts-strogatz",
        k=K,
        p=WS_P,
        n_plus=60,
        steps=steps,
        mixes=_MIX_LABEL,
    )
    ctx.realisations(len(THESIS_TYPE_VECTORS) * REALISATIONS)

    g = watts_strogatz(N, K, WS_P, rng=ctx.rng)
    fig, _ = _opinion_panels(ctx, g, steps=steps)
    # No figure-level title: the caption carries it.  The three mix headings
    # stay, because they identify the panels.
    return fig


opinion_watts_strogatz.__doc__ = (opinion_watts_strogatz.__doc__ or "") + _OPINION_DOC_TAIL


@figure(
    "T28",
    chapter=5,
    title="Three population mixes on a Newman-Watts network",
    source=Source(
        Origin.THESIS_2011, pdf_pages="39-40", section="5.2 Dinamica de redes de opinion"
    ),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis fig. 28 (a-c), PDF pp. 39-40",
)
def opinion_newman_watts(ctx: FigureContext):
    """PDF pp. 39-40, fig. 28: the last row of the thesis's sequence.

    Newman-Watts adds shortcuts without breaking the ring, so it keeps the
    local structure of WS and adds reach, and it is the least stable of the
    four networks under this model: across realisations its counts fail to
    settle more often than ER's, WS's or the scale-free network's, for every
    one of the three mixes.

    Panel a is worth looking at twice.  The mix with the largest share of
    majority-followers is the one the thesis expects to be calmest, and for
    forty instants it is -- and then it turns over, red overtaking green, and
    holds the new majority.  Nothing was changed to make that happen.  It is
    the *comportamiento caotico* the thesis reports for its own panels: a
    quiet stretch that ends without warning.

    Taken together with figs. 25-27, the four rows say something no single
    network could have shown: the population mix decides *whether* opinions
    settle, and the topology decides how loudly they fail to.
    """
    steps = 99
    ctx.parameters(
        N=N,
        model="newman-watts",
        k=K,
        p=NW_P,
        n_plus=60,
        steps=steps,
        mixes=_MIX_LABEL,
    )
    ctx.realisations(len(THESIS_TYPE_VECTORS) * REALISATIONS)

    g = newman_watts(N, K, NW_P, rng=ctx.rng)
    fig, _ = _opinion_panels(ctx, g, steps=steps)
    # No figure-level title: the caption carries it.  The three mix headings
    # stay, because they identify the panels.
    return fig


opinion_newman_watts.__doc__ = (opinion_newman_watts.__doc__ or "") + _OPINION_DOC_TAIL


# ---------------------------------------------------------------------------
# 5.3  The weighted opinion network
# ---------------------------------------------------------------------------


def _weighted_pair(ctx: FigureContext, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """The counts behind figs. 29 and 30: one network, one start, two couplings.

    Figs. 29 and 30 are a controlled pair -- the same grown network, the same
    fifty-fifty starting opinions, evolved once through the tie-age weights
    :math:`W` and once through the plain adjacency :math:`M`.  Both figures
    therefore have to come from the *same* realisation, which is why the two
    count series are produced here rather than in either figure.

    The realisation is chosen by the module's rule: *REALISATIONS* pairs are
    generated and the one whose weighted run ends at the median :math:`+1`
    count is kept.  Selecting on the weighted run alone keeps the choice from
    favouring either half of the comparison.
    """
    m0 = 3
    pairs = []
    for _ in range(REALISATIONS):
        g = barabasi_albert(N, links_per_new_node=2, initial_nodes=m0, rng=ctx.rng)
        w = weights_from_growth(g, initial_nodes=m0)
        start = random_initial_states(N, N // 2, rng=ctx.rng)
        weighted = run_weighted_opinion(w, start, steps=steps, rng=ctx.rng)
        plain = run_hopfield(g, start, steps=steps, rng=ctx.rng)
        pairs.append(((weighted.states == 1).sum(axis=1), (plain.states == 1).sum(axis=1)))
    finals = np.array([int(pair[0][-1]) for pair in pairs])
    return pairs[int(np.argmin(np.abs(finals - np.median(finals))))]


@figure(
    "T29",
    chapter=5,
    title="Opinions on a scale-free network weighted by the age of each tie",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="41",
        section="5.3 Dinamica de redes de opinion con pesos",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 29, PDF p. 41 (left of the pair)",
)
def weighted_opinion_with_weights(ctx: FigureContext):
    """PDF p. 41, fig. 29: *evolucion de las opiniones en la red con pesos*.

    One of the original contributions of the thesis.  Influence is no longer
    the adjacency matrix but a weight matrix :math:`W_{i,j} = t_f - t_n`: how
    long the two nodes have known each other.  A grown network knows this
    already, since a link is born when the younger of its endpoints arrives,
    so the weights come out of the construction rather than being imposed on
    it, and the rule becomes
    :math:`S_i(t+1) = \\mathrm{sgn}(\\sum_j S_j(t) W_{i,j})`.

    The thesis's printed curve does not converge: the two counts stay locked
    around fifty for the whole twenty instants, zig-zagging past one another
    without either opinion winning.  That is the signature of a two-cycle,
    which a synchronous sign rule falls into readily, and the thesis reads it
    against fig. 30 as its result -- weighting the ties by their age prevents
    the majority that the unweighted network settles into.

    The panel drawn here is that curve: the weighted run locks into a
    two-cycle, 41 nodes against 59, and stays there.  Read with fig. 30, which
    is the same realisation without the weights, the thesis's contrast is
    visible exactly as printed.

    A tie-age weight is largest on the oldest links, and in a grown network the
    oldest links are the hubs' own, so :math:`W` concentrates influence on the
    first nodes to arrive.  Over the 25 realisations run by this implementation
    the weighted runs finish a median of 25 nodes away from an even split
    against 13 for the unweighted ones.
    """
    steps = 19
    ctx.parameters(
        N=N,
        model="scale-free",
        m=2,
        m0=3,
        steps=steps,
        coupling="W (tie age)",
        drawn="median realisation",
    )
    ctx.realisations(REALISATIONS)

    weighted, _ = _weighted_pair(ctx, steps)

    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    _plot_counts(ax, weighted)
    _state_key(ax, weighted)
    ax.set_xticks([2, 6, 10, 14, 18])
    ax.set_xlabel("$t$")
    ax.set_ylabel("number of nodes")
    ax.annotate(
        f"locked in a two-cycle between {int(weighted[-2:].min())} and {int(weighted[-2:].max())}",
        xy=(0.5, 0.75),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=8,
        color=INK["secondary"],
    )
    # No title: the caption carries it.
    return fig


@figure(
    "T30",
    chapter=5,
    title="The same network and the same start, without weights",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="41",
        section="5.3 Dinamica de redes de opinion con pesos",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 30, PDF p. 41 (right of the pair)",
)
def weighted_opinion_without_weights(ctx: FigureContext):
    """PDF p. 41, fig. 30: the control, and the reason fig. 29 means anything.

    This is the unweighted half of the pair, following the plot's own title,
    *sin pesos*; the difference between the two panels is the whole argument of
    section 5.3.

    Here the coupling is the plain adjacency matrix again, on the same
    network, from the same fifty-fifty start, in the same realisation as
    fig. 29 -- that is what makes the pair a controlled comparison rather than
    two pictures.  The two counts separate within a few instants and hold a
    stable majority, 55 or 56 against 45 or 44, for the rest of the run, where the
    weighted version of the same run never stops swapping.

    Nothing about the network changed: same nodes, same links, same initial
    opinions.  Only the question of *whose* voice counts more.  Reading the two
    panels together is the exercise section 5.3 asks for; what the reproduction
    finds when it does so is set out under fig. 29.
    """
    steps = 19
    ctx.parameters(
        N=N,
        model="scale-free",
        m=2,
        m0=3,
        steps=steps,
        coupling="M (adjacency)",
        drawn="median realisation (the same one as fig. 29)",
    )
    ctx.realisations(REALISATIONS)

    _, plain = _weighted_pair(ctx, steps)

    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    _plot_counts(ax, plain)
    _state_key(ax, plain)
    ax.set_xticks([2, 6, 10, 14, 18])
    ax.set_xlabel("$t$")
    ax.set_ylabel("number of nodes")
    lo, hi = int(plain[-2:].min()), int(plain[-2:].max())
    ax.annotate(
        f"alternates between {lo} and {hi} against {N - lo} and {N - hi}",
        xy=(plain.size, hi),
        xytext=(-4, 8),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=INK["secondary"],
    )
    # No title: the caption carries it.
    return fig


@figure(
    "T31",
    chapter=5,
    title="Weights on ER, WS and NW, over two simulations",
    source=Source(
        Origin.THESIS_2011,
        pdf_pages="42",
        section="5.3 Dinamica de redes de opinion con pesos",
    ),
    seed=THESIS_SEED,
    reproduces_figure="thesis fig. 31 (a.1-c.4), PDF p. 42",
)
def weighted_opinion_three_types(ctx: FigureContext):
    """PDF p. 42, fig. 31: twelve panels that answer the question of section 5.3.

    The tie-age weights of figs. 29 and 30 needed a network built in time, so
    they were only available on the grown one.  Section 5.3 gets around that
    with a substitution the thesis states in full: order the nodes by degree,
    call the best-connected one the oldest, and read the weight of a link off
    the younger of its two endpoints,
    :math:`W_{i,j} = N - \\max(t_i, t_j)`.  A network that was never grown can
    then be weighted anyway, and ER, WS and NW come into range.

    The layout is the thesis's own: three rows for the network types, four
    columns for two simulations each shown without and with weights, and every
    panel starting from 52 nodes at :math:`+1` and 48 at :math:`-1`.

    The two simulations differ only in which nodes hold which opinion at
    :math:`t = 1`, and that is the whole point of printing both.  Compare the
    first pair of columns and weights look as though they delay consensus;
    compare the second and they look as though they cause it.  The thesis
    draws the conclusion -- *"la introduccion de pesos no provoca que se
    alcance antes el equilibrio ... la evolucion de estas depende de los
    estados iniciales de los nodos"* -- and it prints both draws because they
    differ, so both are shown here and no representative run is selected.
    """
    steps = 19
    n_minus = 48
    ctx.parameters(
        N=N,
        k=K,
        ws_p=WS_P,
        nw_p=NW_P,
        n_minus=n_minus,
        steps=steps,
        weights="W = N - max(t_i, t_j), t from the degree ranking",
        simulations=2,
    )
    ctx.realisations(2)

    builders = _three_builders(ctx)
    starts = [random_initial_states(N, n_minus, rng=ctx.rng) for _ in range(2)]

    columns = ["no weights", "weights", "no weights", "weights"]
    names = {"ER": "Erdős–Rényi", "WS": "Watts–Strogatz", "NW": "Newman–Watts"}

    # Two groups of six, one per simulation, each with its own heading: the
    # heading then belongs to the pair of columns it names, and the four
    # columns fall evenly across the page instead of being pushed apart by a
    # label hung off one panel.
    fig = plt.figure(figsize=(7.2, 4.6))
    groups = fig.subfigures(1, 2, wspace=0.02, width_ratios=[1.13, 1.0])
    axes = np.empty((3, 4), dtype=object)
    for k, (group, heading) in enumerate(
        zip(groups, ("1st simulation", "2nd simulation"), strict=True)
    ):
        axes[:, 2 * k : 2 * k + 2] = group.subplots(3, 2, sharex=True, sharey=True)
        group.suptitle(heading, fontsize=8.5, color=INK["secondary"])
    corner = None
    for row, (label, build) in zip(axes, builders.items(), strict=True):
        g = build()
        w = weights_from_degree_rank(g, rng=ctx.rng)
        for column, ax in enumerate(row):
            start = starts[column // 2]
            if column % 2:
                run = run_weighted_opinion(w, start, steps=steps, rng=ctx.rng)
            else:
                run = run_hopfield(g, start, steps=steps, rng=ctx.rng)
            counts = (run.states == 1).sum(axis=1)
            _plot_counts(ax, counts)
            ax.set_xticks([5, 10, 15, 20])
            if corner is None:
                corner = counts
        # The row is named once, on the left; the twelve panels shared one
        # count axis and one t axis between them rather than carrying twelve.
        row[0].set_ylabel(names[label], fontsize=9.5, color=INK["primary"])

    # Twelve cells across a page leave no gutter to spare: a letter set outside
    # each cell's left edge, as elsewhere in the book, needs one between every
    # pair of columns, and the figure then cannot be drawn at the width it is
    # printed at.  So the letter sits above its cell, at the left, and the
    # column heading takes the right of the same line.
    for ax, heading in zip(axes[0], columns, strict=True):
        ax.set_title(heading, fontsize=7.5, fontweight="normal", loc="right")
    for row, letter in zip(axes, "abc", strict=True):
        for cell, ax in enumerate(row, start=1):
            ax.annotate(
                f"{letter}.{cell}",
                xy=(0.0, 1.0),
                xycoords="axes fraction",
                xytext=(0, 3),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=8,
                color=INK["muted"],
            )
    # The second group repeats the first one's count axis; its numbers go.
    for ax in axes[:, 2:].ravel():
        ax.tick_params(labelleft=False)
    _state_key(axes[0][-1], corner)

    fig.supxlabel("$t$", fontsize=9.5, color=INK["secondary"])
    fig.supylabel("number of nodes", fontsize=9.5, color=INK["secondary"])

    # No figure-level title: the caption carries it, parameters included.  The
    # column and row headings stay, because they identify the panels.
    return fig
