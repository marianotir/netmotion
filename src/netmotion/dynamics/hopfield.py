"""Hopfield-type node dynamics on a complex network.

Thesis section 5.1, *Dinamica en redes complejas basadas en el modelo Hopfield*
(PDF pp. 35-37).

Chapter 5 is where the thesis introduces time.  Up to chapter 4 a network is a
static object measured by distance, degree and clustering; from section 5.1
onwards every node carries a state that changes, and the question becomes what
the topology does to that change.  The simplest dynamics the thesis can write
down is taken from the Hopfield model, itself an Ising model: node *i* holds a
spin :math:`S_i(t) \\in \\{+1, -1\\}` and adopts, at the next instant, the sign
of the sum of the spins it is linked to (PDF p. 35):

.. math:: S_i(t+1) = \\mathrm{sgn}\\Big( \\sum_{j}^{N} S_j(t)\\, M_{i,j} \\Big)

The coupling matrix is the adjacency matrix of the network, so the only free
ingredient of the dynamics is the topology: the same initial condition run on a
scale-free, an Erdos-Renyi, a Watts-Strogatz and a Newman-Watts network gives
different evolutions (figs. 21 and 22) -- *"esto permite ver con claridad la
influencia que ejerce la topologia de la red sobre la dinamica de esta"*.

Equilibrium
-----------
The thesis calls the endpoint an *atractor*: equilibrium is reached *"cuando no
existen mas cambios en los estados de los nodos"*, which may be all :math:`+1`,
all :math:`-1`, or a mixture.  For the scale-free network of fig. 21 it is
reached at :math:`t = 8`.  :func:`run_hopfield` runs a fixed horizon, as the
figures do, and reports whether the trace ended in a fixed point
(:attr:`HopfieldRun.equilibrium_time`) or in a cycle (:attr:`HopfieldRun.period`).

Readings taken
--------------
*The update order.*  Every rule of chapter 5 is written
:math:`S_i(t+1) = f(S(t))`, so all nodes are updated simultaneously.

*A zero field.*  A node whose neighbours are evenly split, or which has none,
keeps its state (``zero_rule="hold"``), the convention the thesis writes for
opinion types 6 and 7 -- *"si* :math:`\\sum_j S_j(t) = 0`
:math:`\\Rightarrow S_i(t+1) = S_i(t)`".  ``"positive"`` and ``"random"`` are
offered as options.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..graph import Network
from ..provenance import Origin, sourced
from ..rng import SeedLike, as_rng

__all__ = [
    "ZeroRule",
    "HopfieldRun",
    "sgn_states",
    "hopfield_step",
    "run_hopfield",
    "random_initial_states",
    "equilibrium_time",
    "cycle_period",
    "equilibrium_sweep",
]

#: How a zero local field is resolved.  See the module docstring.
ZeroRule = Literal["hold", "positive", "random"]

StateLike = np.ndarray | Sequence[int]

_ZERO_RULES = ("hold", "positive", "random")


def _as_states(states: StateLike, n: int) -> np.ndarray:
    """Validate a spin vector and return it as an ``int8`` copy."""
    s = np.asarray(states, dtype=np.int8).copy()
    if s.shape != (n,):
        raise ValueError(f"state vector must have shape ({n},), got {s.shape}")
    if not np.isin(s, (-1, 1)).all():
        raise ValueError("states must be +1 or -1")
    return s


@sourced(
    Origin.REPRODUCTION,
    pdf_pages="35-36",
    section="5.1 Dinamica en redes complejas basadas en el modelo Hopfield",
    reproduces="netmotion.dynamics.hopfield.run_hopfield",
    note=(
        "The trajectory the thesis plots in figs. 21-22 and 25-31: the full "
        "state history plus the n_+(t) and n_-(t) counts that are the y-axis "
        "of every one of those figures. The thesis stores no such object -- it "
        "plots the counts directly -- so this is a container for a reproduction, "
        "not a thesis concept."
    ),
    tags=("chapter-5", "container"),
)
@dataclass
class HopfieldRun:
    """A whole trajectory of the section 5.1 rule, plus what it settled into.

    Attributes
    ----------
    states:
        ``(steps + 1, N)`` array of spins.  Row ``k`` is the state after ``k``
        updates, so row 0 is the initial condition the caller supplied.
    """

    states: np.ndarray

    @property
    def n_plus(self) -> np.ndarray:
        """:math:`n_+(t)`, the count of nodes at :math:`+1`; the green curve."""
        return (self.states == 1).sum(axis=1)

    @property
    def n_minus(self) -> np.ndarray:
        """:math:`n_-(t)`, the count of nodes at :math:`-1`; the red curve."""
        return (self.states == -1).sum(axis=1)

    @property
    def final_states(self) -> np.ndarray:
        """The last state of the trace."""
        return self.states[-1]

    @property
    def steps(self) -> int:
        """Number of updates performed."""
        return int(self.states.shape[0] - 1)

    @property
    def equilibrium_time(self) -> int | None:
        """The thesis's :math:`t^*`, or ``None`` if the trace never settled."""
        return equilibrium_time(self.states)

    @property
    def period(self) -> int | None:
        """Period of the orbit the trace ended in; 1 means a fixed point."""
        return cycle_period(self.states)


@sourced(
    Origin.REPRODUCTION,
    pdf_pages="35",
    section="5.1 Dinamica en redes complejas basadas en el modelo Hopfield",
    reproduces="sgn in S_i(t+1) = sgn(sum_j S_j(t) M_ij)",
    note=(
        "Element-wise sign returning values in {+1, -1}. Factored out because "
        "every dynamics in chapter 5 -- the base rule of 5.1, opinion types 1 "
        "and 2 of 5.2, and the weighted rule of 5.3 -- goes through it."
    ),
    reading=(
        "A zero field keeps the state (zero_rule='hold', the default), as for "
        "opinion types 6 and 7 (PDF p. 38); 'positive' and 'random' are options."
    ),
    tags=("convention", "chapter-5"),
)
def sgn_states(
    field: np.ndarray,
    previous: np.ndarray,
    *,
    zero_rule: ZeroRule = "hold",
    rng: SeedLike = None,
) -> np.ndarray:
    """:math:`\\mathrm{sgn}` of *field*, with zeros resolved by *zero_rule*.

    Parameters
    ----------
    field:
        The local fields :math:`h_i = \\sum_j S_j C_{ij}`.
    previous:
        The states at instant *t*, used when ``zero_rule="hold"``.
    zero_rule:
        ``"hold"`` keeps :math:`S_i(t)`, ``"positive"`` returns :math:`+1`,
        ``"random"`` draws a fair coin.  See the module docstring.

    Examples
    --------
    >>> import numpy as np
    >>> sgn_states(np.array([2.0, -3.0, 0.0]), np.array([1, 1, -1])).tolist()
    [1, -1, -1]

    The zero entry is the only one the convention touches:

    >>> sgn_states(np.array([2.0, -3.0, 0.0]), np.array([1, 1, -1]),
    ...            zero_rule="positive").tolist()
    [1, -1, 1]
    """
    if zero_rule not in _ZERO_RULES:
        raise ValueError(f"zero_rule must be one of {_ZERO_RULES}, got {zero_rule!r}")

    if zero_rule == "hold":
        fill = np.asarray(previous, dtype=np.int8)
    elif zero_rule == "positive":
        fill = np.ones(field.shape, dtype=np.int8)
    else:
        fill = (2 * as_rng(rng).integers(0, 2, size=field.shape) - 1).astype(np.int8)

    return np.where(field > 0, np.int8(1), np.where(field < 0, np.int8(-1), fill)).astype(np.int8)


@sourced(
    Origin.THESIS_2011,
    pdf_pages="35",
    section="5.1 Dinamica en redes complejas basadas en el modelo Hopfield",
    note=(
        "One update of S_i(t+1) = sgn(sum_j S_j(t) M_ij), the base rule of "
        "chapter 5. The coupling is network.matrix, which is the unweighted "
        "adjacency M in section 5.1 and the tie-age weight matrix W in section "
        "5.3 -- the thesis reuses the identical equation with W in place of M."
    ),
    reading=(
        "The rule is written S_i(t+1) = f(S(t)), so the N nodes update "
        "simultaneously."
    ),
    tags=("chapter-5", "synchronous-update"),
)
def hopfield_step(
    coupling: Network,
    states: StateLike,
    *,
    zero_rule: ZeroRule = "hold",
    rng: SeedLike = None,
) -> np.ndarray:
    """Advance every node one instant, simultaneously.

    Parameters
    ----------
    coupling:
        The network whose ``matrix`` couples the nodes: :math:`M` for section
        5.1, :math:`W` for section 5.3.
    states:
        The spins at instant *t*, any sequence of :math:`\\pm 1`.

    Examples
    --------
    On :math:`K_4` every field is an odd integer, so no zero field occurs:

    >>> from netmotion import Network
    >>> k4 = Network.from_edges(4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])
    >>> hopfield_step(k4, [1, 1, -1, -1]).tolist()
    [-1, -1, 1, 1]

    The update is simultaneous, not a sweep: on a single edge the two nodes
    exchange states rather than both falling to the first one's value.

    >>> edge = Network.from_edges(2, [(0, 1)])
    >>> hopfield_step(edge, [1, -1]).tolist()
    [-1, 1]
    """
    s = _as_states(states, coupling.n)
    field = coupling.matrix @ s.astype(float)
    return sgn_states(field, s, zero_rule=zero_rule, rng=rng)


@sourced(
    Origin.THESIS_2011,
    pdf_pages="35-36",
    section="5.1 Dinamica en redes complejas basadas en el modelo Hopfield",
    note=(
        "The trajectory plotted in figs. 20-22: 100 nodes, 60 at +1 and 40 at "
        "-1 initially, iterated over a fixed horizon, as the figures plot it."
    ),
    reading=(
        "Equilibrium, 'cuando no existen mas cambios en los estados', is "
        "reported by HopfieldRun.equilibrium_time, and a cycle by .period."
    ),
    tags=("chapter-5",),
)
def run_hopfield(
    coupling: Network,
    states: StateLike,
    *,
    steps: int = 12,
    zero_rule: ZeroRule = "hold",
    rng: SeedLike = None,
) -> HopfieldRun:
    """Iterate :func:`hopfield_step` and keep the whole trace.

    Parameters
    ----------
    steps:
        Number of updates.  The trace has ``steps + 1`` rows.  The default of
        12 gives the 13 instants plotted in figs. 21 and 22.

    Examples
    --------
    A triangle with a pendant node reaches consensus in one step and then holds
    it, an attractor:

    >>> from netmotion import Network
    >>> tri = Network.from_edges(4, [(0,1),(0,2),(1,2),(2,3)])
    >>> run = run_hopfield(tri, [1, 1, 1, -1], steps=4)
    >>> run.n_plus.tolist()
    [3, 4, 4, 4, 4]
    >>> run.equilibrium_time
    1

    On :math:`K_4` from this state the run alternates with period 2:

    >>> k4 = Network.from_edges(4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])
    >>> cycling = run_hopfield(k4, [1, 1, -1, -1], steps=4)
    >>> cycling.equilibrium_time is None
    True
    >>> cycling.period
    2
    >>> cycling.n_plus.tolist()
    [2, 2, 2, 2, 2]
    """
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")

    generator = as_rng(rng)
    s = _as_states(states, coupling.n)
    trace = np.empty((steps + 1, coupling.n), dtype=np.int8)
    trace[0] = s
    for k in range(steps):
        s = hopfield_step(coupling, s, zero_rule=zero_rule, rng=generator)
        trace[k + 1] = s
    return HopfieldRun(states=trace)


@sourced(
    Origin.REPRODUCTION,
    pdf_pages="35",
    section="5.1 Dinamica en redes complejas basadas en el modelo Hopfield",
    reproduces="netmotion.dynamics.hopfield.run_hopfield",
    note=(
        "The initial condition of figs. 20-22 and of section 5.2: all nodes at "
        "+1 with n_minus of them chosen uniformly at random and flipped to -1. "
        "The thesis's own runs use N = 100 with n_minus = 40 (sections 5.1 and "
        "5.2) and n_minus = 48 (fig. 31)."
    ),
    tags=("initial-condition", "chapter-5"),
)
def random_initial_states(n_nodes: int, n_against: int, *, rng: SeedLike = None) -> np.ndarray:
    """All nodes at :math:`+1` except *n_against* random ones, set to :math:`-1`.

    Examples
    --------
    The thesis's standard initial condition for sections 5.1 and 5.2:

    >>> s = random_initial_states(100, 40, rng=0)
    >>> int((s == 1).sum()), int((s == -1).sum())
    (60, 40)
    """
    n, n_minus = n_nodes, n_against
    if not 0 <= n_minus <= n:
        raise ValueError(f"n_minus must lie in 0..{n}, got {n_minus}")
    s = np.ones(n, dtype=np.int8)
    flipped = as_rng(rng).choice(n, size=n_minus, replace=False)
    s[flipped] = -1
    return s


@sourced(
    Origin.THESIS_2011,
    pdf_pages="35",
    section="5.1 Dinamica en redes complejas basadas en el modelo Hopfield",
    note=(
        "t*, the instant from which 'no existen mas cambios en los estados de "
        "los nodos'; t* = 8 in fig. 21 for the scale-free network. Returns None "
        "when the trace does not settle."
    ),
)
def equilibrium_time(states: np.ndarray) -> int | None:
    """First instant at which the state stops changing, or ``None``.

    Examples
    --------
    >>> import numpy as np
    >>> equilibrium_time(np.array([[1, -1], [1, 1], [1, 1], [1, 1]]))
    1
    >>> equilibrium_time(np.array([[1, -1], [-1, 1], [1, -1]])) is None
    True
    """
    states = np.asarray(states)
    for t in range(states.shape[0] - 1):
        if np.array_equal(states[t], states[t + 1]):
            return t
    return None


@sourced(
    Origin.EXTENSION,
    pdf_pages="35",
    section="5.1 Dinamica en redes complejas basadas en el modelo Hopfield",
    reproduces="netmotion.dynamics.hopfield.run_hopfield",
    note=(
        "2026 addition: the period of the orbit the trace ended in, 1 for a "
        "fixed point."
    ),
    tags=("chapter-5", "diagnostic"),
)
def cycle_period(states: np.ndarray) -> int | None:
    """Period of the orbit the trace ends in, or ``None`` if none is visible.

    A return value of 1 means a fixed point, an attractor; 2 means the state
    alternates.

    Examples
    --------
    >>> import numpy as np
    >>> cycle_period(np.array([[1, 1], [1, 1]]))
    1
    >>> cycle_period(np.array([[1, -1], [-1, 1], [1, -1]]))
    2
    >>> cycle_period(np.array([[1, -1], [-1, 1]])) is None
    True
    """
    states = np.asarray(states)
    last = states[-1]
    for period in range(1, states.shape[0]):
        if np.array_equal(states[-1 - period], last):
            return period
    return None


@sourced(
    Origin.THESIS_2011,
    pdf_pages="36-37",
    section="5.1 Dinamica en redes complejas basadas en el modelo Hopfield",
    note=(
        "Figs. 23 and 24: start from all +1, raise the number of nodes seeded "
        "at -1 from 0 to N one at a time, and record the equilibrium counts. "
        "The finding is a transition -- past a critical seed count the "
        "equilibrium number of -1 nodes overtakes the +1 nodes -- whose "
        "location depends on the topology."
    ),
    reading=(
        "One realisation per point by default (repeats=1). The final state of a "
        "fixed-length run is recorded."
    ),
    tags=("chapter-5", "figure-23", "figure-24"),
)
def equilibrium_sweep(
    coupling: Network,
    *,
    steps: int = 50,
    zero_rule: ZeroRule = "hold",
    repeats: int = 1,
    rng: SeedLike = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Equilibrium :math:`n_+` and :math:`n_-` against the number of seeds.

    Returns
    -------
    tuple of ndarray
        Two arrays of length :math:`N + 1`; entry *m* is the mean final count
        over *repeats* runs started with *m* nodes at :math:`-1`.

    Examples
    --------
    The two endpoints are the only points of figs. 23 and 24 that are exactly
    determined and independent of the topology: all :math:`+1` and all
    :math:`-1` are both fixed points of the rule.

    >>> from netmotion import Network
    >>> tri = Network.from_edges(4, [(0,1),(0,2),(1,2),(2,3)])
    >>> n_plus_eq, n_minus_eq = equilibrium_sweep(tri, steps=10, rng=0)
    >>> float(n_plus_eq[0]), float(n_minus_eq[0])
    (4.0, 0.0)
    >>> float(n_plus_eq[-1]), float(n_minus_eq[-1])
    (0.0, 4.0)
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")

    generator = as_rng(rng)
    n = coupling.n
    n_plus_eq = np.zeros(n + 1)
    for m in range(n + 1):
        totals = 0.0
        for _ in range(repeats):
            start = random_initial_states(n, m, rng=generator)
            run = run_hopfield(coupling, start, steps=steps, zero_rule=zero_rule, rng=generator)
            totals += float((run.final_states == 1).sum())
        n_plus_eq[m] = totals / repeats
    return n_plus_eq, float(n) - n_plus_eq
