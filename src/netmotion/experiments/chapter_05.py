"""Figures for chapter 5: dynamics on networks.

Thesis sections 5.1-5.3 (PDF pp. 35-42).  The network stops being the object of
study here and becomes the stage: states live on nodes, and links decide who
moves whom.

None of these sections states a seed, a number of realisations, or the
generating parameters of the networks it runs on, so the figures below
reproduce the *behaviour* the thesis describes rather than its exact curves.
Each caption says which.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..dynamics.opinion import THESIS_TYPE_VECTORS, run_opinion
from ..networks import watts_strogatz
from ..provenance import Origin, Source
from ..visualization import INK, SERIES
from ..visualization.charts import reference_line
from .registry import FigureContext, figure

THESIS_SEED = 20110930

#: What each of the thesis's three percentage vectors actually says.
MIX_MEANING = {
    "a": "40% simply follow their neighbours",
    "b": "40% copy the hub or the loneliest node",
    "c": "half copy the hub or the loneliest node",
}


def _name_curve(ax, x, y, label: str, colour: str, *, dy: float = 6.0) -> None:
    """Write a curve's name at its own end, in its own colour.

    Three curves do not need a legend: the eye should not have to travel from a
    key to a line and back to find out which is which.
    """
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(-2, dy),
        textcoords="offset points",
        ha="right",
        va="bottom" if dy > 0 else "top",
        fontsize=8.5,
        color=colour,
    )


@figure(
    "F05-02",
    chapter=5,
    title="Opinion dynamics: seven kinds of person",
    source=Source(
        Origin.THESIS_2011, pdf_pages="37-40", section="5.2 Dinamica de redes de opinion"
    ),
    seed=THESIS_SEED,
    slow=True,
    reproduces_figure="thesis figs. 25-28, PDF pp. 38-40",
)
def opinion_dynamics(ctx: FigureContext):
    """PDF pp. 37-40: nodes that form their opinion by different criteria.

    This is where the thesis departs from the plain Hopfield rule. Each node is
    one of seven types -- follow the local majority, contradict it, copy the
    best-connected node, copy the least-connected one, decide at random, follow
    the global majority, oppose it -- and the mix of types decides whether the
    network reaches consensus at all.

    The three mixes are the ones the thesis uses. What they show is that a
    population's outcome is not set by its topology alone: the same network
    reaches consensus under one mix and oscillates under another.
    """
    n, steps = 100, 99
    ctx.parameters(
        N=n,
        model="watts-strogatz",
        k=4,
        p=0.1,
        n_plus=60,
        steps=steps,
        mixes=list(THESIS_TYPE_VECTORS),
        drawn="one run per mix",
    )
    ctx.realisations(len(THESIS_TYPE_VECTORS))

    g = watts_strogatz(n, 4, 0.1, rng=ctx.rng)

    # Stacked, not side by side: these runs oscillate with a period of about
    # three instants over a hundred of them, and a third of the page's width
    # cannot resolve that -- the trace turns into a solid block.  Full width
    # each, with the axes shared, and the oscillation can be counted.
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.4), sharex=True, sharey=True)
    for ax, (label, mix) in zip(axes, THESIS_TYPE_VECTORS.items(), strict=True):
        run = run_opinion(g, list(mix), n_in_favour=60, steps=steps, rng=ctx.rng)
        fraction = (run.states == 1).sum(axis=1) / n
        ax.plot(np.arange(fraction.size), fraction, color=SERIES[0], lw=1.8)
        # The title names what the seven numbers of the population vector
        # count, so the bracket can be read without a key.
        vector = ", ".join(str(v) for v in mix)
        ax.set_title(
            f"{label}   {MIX_MEANING[label]}\n[{vector}] people, one count per rule",
            fontsize=7.5,
        )
        ax.set_xlim(0, fraction.size - 1)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_yticks([0.0, 0.5, 1.0])
    reference_line(axes[0], 0.5, "even split")
    for ax in axes[1:]:
        reference_line(ax, 0.5, "")

    fig.supxlabel("$t$", fontsize=9.5, color=INK["secondary"])
    fig.supylabel("fraction holding $+1$", fontsize=9.5, color=INK["secondary"])
    # No figure-level title: the caption carries it.  The three panel titles
    # stay, because they name the mixes and the caption cannot.
    return fig


