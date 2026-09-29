"""Node dynamics: what happens to a network once time is switched on.

Chapter 5.1-5.3 of the thesis.  All three sections share one equation, the
Hopfield/Ising update :math:`S_i(t+1) = \\mathrm{sgn}(\\sum_j S_j(t) C_{ij})`
over spins in :math:`\\{+1, -1\\}`, and differ only in what the coupling
:math:`C` is and in who is allowed to use the equation at all:

``hopfield``
    Section 5.1.  :math:`C = M`, the plain adjacency matrix, and every node
    obeys the same rule.  The thesis uses it to show that the same initial
    condition evolves differently on a scale-free, an Erdos-Renyi, a
    Watts-Strogatz and a Newman-Watts network, which is its way of making the
    influence of topology on dynamics visible.
``opinion``
    Section 5.2.  :math:`+1` and :math:`-1` become opinions for and against,
    and the single rule is replaced by seven kinds of individual, assigned once
    at random in proportions given by a percentage vector.  The thesis reports
    that the opinions oscillate without reaching equilibrium.
``weighted_opinion``
    Section 5.3, the thesis's own contribution.  :math:`C = W`, a matrix whose
    entries measure the *age* of each relationship, so that a long-standing
    acquaintance influences an opinion more than a recent one.

The three build on one another in that order, and the shared machinery -- the
synchronous step, the trace object, the treatment of a vanishing local field --
lives in :mod:`netmotion.dynamics.hopfield` so that all three inherit exactly
the same conventions.
"""

from __future__ import annotations

from .hopfield import (
    HopfieldRun,
    ZeroRule,
    cycle_period,
    equilibrium_sweep,
    equilibrium_time,
    hopfield_step,
    random_initial_states,
    run_hopfield,
    sgn_states,
)
from .opinion import (
    OPINION_RULES,
    THESIS_TYPE_VECTORS,
    OpinionRun,
    OpinionScope,
    TieRule,
    assign_opinion_types,
    degree_extremes,
    opinion_step,
    run_opinion,
)
from .weighted_opinion import (
    WeightComparison,
    compare_weighted_unweighted,
    run_weighted_opinion,
    weights_from_degree_rank,
    weights_from_edge_ages,
    weights_from_growth,
)

__all__ = [
    # 5.1
    "ZeroRule",
    "HopfieldRun",
    "sgn_states",
    "hopfield_step",
    "run_hopfield",
    "random_initial_states",
    "equilibrium_time",
    "cycle_period",
    "equilibrium_sweep",
    # 5.2
    "OpinionScope",
    "TieRule",
    "OPINION_RULES",
    "THESIS_TYPE_VECTORS",
    "OpinionRun",
    "assign_opinion_types",
    "degree_extremes",
    "opinion_step",
    "run_opinion",
    # 5.3
    "WeightComparison",
    "weights_from_edge_ages",
    "weights_from_growth",
    "weights_from_degree_rank",
    "run_weighted_opinion",
    "compare_weighted_unweighted",
]
