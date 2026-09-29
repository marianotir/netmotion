"""Opinion dynamics with heterogeneous individuals.

Thesis section 5.2, *Dinamica de redes de opinion* (PDF pp. 37-40).

Section 5.1 gives every node the same rule, and the thesis explains why
section 5.2 extends it: *"en la definicion anterior vemos que todos los nodos j
a los cuales este conectado el nodo i influyen de la misma forma sobre la
opinion del nodo i. Sin embargo en la sociedad la opinion de un
individuo es influenciable de diferente forma segun el tipo de individuo"*.
So section 5.2 keeps the state space and the synchronous update of 5.1 --
:math:`S_i(t) = +1` now means *opinion a favor* and :math:`-1` *opinion en
contra* -- and replaces the single rule by seven, one per kind of person
(PDF pp. 37-38):

1. the majority of my neighbours, :math:`\\mathrm{sgn}\\,h_i(t)`;
2. the opposite of my neighbours, :math:`-\\mathrm{sgn}\\,h_i(t)`;
3. whatever the highest-degree node in the network says;
4. whatever the lowest-degree node in the network says;
5. at random, :math:`(-1)^b` with *b* a random integer;
6. the opposite of the global majority, holding when the network is tied;
7. with the global majority, holding when the network is tied;

where :math:`h_i(t) = \\sum_j S_j(t) M_{i,j}` is the local field of 5.1 and the
global rules read :math:`H(t) = \\sum_j S_j(t)` instead.  Each node is assigned
one of the seven **once**, at random, in proportions given by a percentage
vector; the assignment never changes afterwards, because it is a personality,
not a state.  The thesis runs three vectors, ``[40, 4, 15, 5, 6, 15, 15]``,
``[10, 10, 25, 15, 10, 10, 20]`` and ``[5, 5, 25, 25, 10, 15, 15]``, across all
four topologies (figs. 25-28), all at :math:`N = 100` with 60 nodes at
:math:`+1` and 40 at :math:`-1` at :math:`t = 1`.

The states *"oscilan con lo que no se termina de alcanzar el equilibrio"*, with
abrupt changes for the second and third vectors, *"el comportamiento caotico de
la red de opiniones asi generada"*.  :func:`run_opinion` runs the full horizon,
the 100 instants of the figures.

The thesis reads the three vectors on PDF p. 40: in the first, 40% follow their
neighbours, so fluctuations are small; the second has opposed tendencies of
unequal size (25% against 15%, 20% against 10%), and the topology sets the level
of fluctuation; the third has the opposed tendencies in equal proportion, and
the behaviour is unstable.

Readings taken
--------------
*Rules 3 and 4* are written *"con* :math:`k_j > k_d\\ \\forall d \\in R` *(d nodo
perteneciente a la red R)"*: every type-3 node copies the highest-degree node
of the network (``scope="global"``, the default); ``scope="neighbourhood"``
looks among a node's neighbours instead.

*Ties in the degree extremes*: ``tie="first"`` takes the lowest index,
``tie="random"`` draws.

*Rule 5*, *"b es un numero entero aleatorio par o impar"*: a fair coin per node
per instant.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..graph import Network
from ..provenance import Origin, sourced
from ..rng import SeedLike, as_rng
from .hopfield import (
    HopfieldRun,
    StateLike,
    ZeroRule,
    _as_states,
    random_initial_states,
    sgn_states,
)

__all__ = [
    "OpinionScope",
    "TieRule",
    "OPINION_RULES",
    "THESIS_TYPE_VECTORS",
    "OpinionRun",
    "assign_opinion_types",
    "degree_extremes",
    "opinion_step",
    "run_opinion",
]

#: Whether rules 3 and 4 look at the whole network or only at the neighbours.
OpinionScope = Literal["global", "neighbourhood"]

#: How a tie in the degree extremes is broken.
TieRule = Literal["first", "random"]

#: The seven ways of forming an opinion, in the order of the percentage vector.
#: The order is the numbered list of PDF pp. 37-38; the colour coding on PDF
#: p. 40 pairs the opposed rules (1, 2), (3, 4) and (6, 7) and leaves 5 alone.
OPINION_RULES: tuple[str, ...] = (
    "majority of my neighbours",
    "opposite of my neighbours",
    "copy the highest-degree node",
    "copy the lowest-degree node",
    "at random",
    "opposite of the global majority",
    "with the global majority",
)

#: The three percentage vectors of figs. 25-28, keyed by the panel letter.
THESIS_TYPE_VECTORS: dict[str, tuple[int, ...]] = {
    "a": (40, 4, 15, 5, 6, 15, 15),
    "b": (10, 10, 25, 15, 10, 10, 20),
    "c": (5, 5, 25, 25, 10, 15, 15),
}

_SCOPES = ("global", "neighbourhood")
_TIE_RULES = ("first", "random")


@sourced(
    Origin.REPRODUCTION,
    pdf_pages="38",
    section="5.2 Dinamica de redes de opinion",
    reproduces="netmotion.dynamics.opinion.run_opinion",
    note=(
        "A section 5.1 trajectory carrying, in addition, the per-node opinion "
        "type, so a run can be checked against its percentage vector."
    ),
    tags=("chapter-5", "container"),
)
@dataclass
class OpinionRun(HopfieldRun):
    """A section 5.2 trajectory, together with the personalities behind it.

    Attributes
    ----------
    types:
        ``(N,)`` array of values 1..7, assigned once before the run and never
        touched again -- the thesis's *tipos de individuos*.
    """

    types: np.ndarray = None  # type: ignore[assignment]


@sourced(
    Origin.THESIS_2011,
    pdf_pages="38",
    section="5.2 Dinamica de redes de opinion",
    note=(
        "Turns the percentage vector [p1..p7] into a fixed per-node type. The "
        "thesis: 'la asociacion de estos tipos de opiniones con los nodos de la "
        "red se realizara de forma aleatoria'. Exact counts are built and then "
        "shuffled, so a vector of [40, 4, 15, 5, 6, 15, 15] on N = 100 gives "
        "exactly 40 nodes of type 1, as the thesis's own runs do."
    ),
    reading=(
        "The assignment is a shuffle of exact counts; for N other than 100 the "
        "counts are rounded by the largest-remainder method."
    ),
    tags=("chapter-5",),
)
def assign_opinion_types(
    n_nodes: int,
    percentages: Sequence[float],
    *,
    rng: SeedLike = None,
) -> np.ndarray:
    """Assign each of *n_nodes* nodes one of the seven rules, at random.

    Parameters
    ----------
    percentages:
        Seven non-negative percentages summing to 100, in the order of
        :data:`OPINION_RULES`.

    Examples
    --------
    The first of the thesis's three vectors, at its own network size:

    >>> types = assign_opinion_types(100, [40, 4, 15, 5, 6, 15, 15], rng=0)
    >>> [int((types == r).sum()) for r in range(1, 8)]
    [40, 4, 15, 5, 6, 15, 15]
    """
    n = n_nodes
    pct = np.asarray(percentages, dtype=float)
    if pct.shape != (7,):
        raise ValueError(f"the type vector has one entry per rule, so 7, got {pct.shape}")
    if (pct < 0).any():
        raise ValueError("percentages must be non-negative")
    if not np.isclose(pct.sum(), 100.0):
        raise ValueError(f"percentages must sum to 100, got {pct.sum():g}")

    exact = pct * n / 100.0
    counts = np.floor(exact).astype(int)
    # Largest-remainder allocation of whatever the flooring left over; at
    # N = 100 every count is exact.
    shortfall = n - int(counts.sum())
    if shortfall:
        order = np.argsort(-(exact - counts), kind="stable")
        counts[order[:shortfall]] += 1

    types = np.repeat(np.arange(1, 8, dtype=np.int8), counts)
    as_rng(rng).shuffle(types)
    return types


@sourced(
    Origin.THESIS_2011,
    pdf_pages="38",
    section="5.2 Dinamica de redes de opinion",
    note=(
        "The reference nodes of opinion rules 3 and 4: the node of largest and "
        "of smallest degree. The topology is static throughout chapter 5, so "
        "these are computed once and reused at every instant."
    ),
    reading=(
        "The formula quantifies over 'd perteneciente a la red R', the whole "
        "network: scope='global', the default. scope='neighbourhood' takes the "
        "extreme among a node's neighbours. tie='first' takes the lowest index "
        "and tie='random' draws."
    ),
    tags=("chapter-5",),
)
def degree_extremes(
    network: Network,
    *,
    scope: OpinionScope = "global",
    tie: TieRule = "first",
    rng: SeedLike = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The node each node watches under rules 3 and 4.

    Returns
    -------
    tuple of ndarray
        Two ``(N,)`` integer arrays, ``(hub_of, sink_of)``.  Under
        ``scope="global"`` both are constant.  Under
        ``scope="neighbourhood"`` a node with no neighbours is mapped to
        itself, so the rule leaves it alone.

    Examples
    --------
    A triangle with a pendant node has degrees ``[2, 2, 3, 1]``, so every node
    watches node 2 for rule 3 and node 3 for rule 4:

    >>> from netmotion import Network
    >>> tri = Network.from_edges(4, [(0,1),(0,2),(1,2),(2,3)])
    >>> hub_of, sink_of = degree_extremes(tri)
    >>> hub_of.tolist(), sink_of.tolist()
    ([2, 2, 2, 2], [3, 3, 3, 3])

    Read locally instead, the pendant node watches its only neighbour both
    ways:

    >>> hub_of, sink_of = degree_extremes(tri, scope="neighbourhood")
    >>> hub_of.tolist(), sink_of.tolist()
    ([2, 2, 0, 2], [1, 0, 3, 2])
    """
    if scope not in _SCOPES:
        raise ValueError(f"scope must be one of {_SCOPES}, got {scope!r}")
    if tie not in _TIE_RULES:
        raise ValueError(f"tie must be one of {_TIE_RULES}, got {tie!r}")

    n = network.n
    k = network.degrees().astype(float)
    # A random tie-break is expressed as a tiny random key added to the degree
    # so that argmin/argmax stay single vectorised calls; the perturbation is
    # far smaller than the unit spacing of integer degrees, so it can only
    # reorder nodes that were already equal.
    key = k if tie == "first" else k + as_rng(rng).uniform(0.0, 0.5, size=n)

    if scope == "global":
        hub = int(np.argmax(key))
        sink = int(np.argmin(key))
        return np.full(n, hub, dtype=int), np.full(n, sink, dtype=int)

    adjacency = network.adjacency
    hub_of = np.arange(n)
    sink_of = np.arange(n)
    for i in range(n):
        neighbours = np.flatnonzero(adjacency[i])
        if neighbours.size == 0:
            continue
        hub_of[i] = int(neighbours[int(np.argmax(key[neighbours]))])
        sink_of[i] = int(neighbours[int(np.argmin(key[neighbours]))])
    return hub_of, sink_of


@sourced(
    Origin.THESIS_2011,
    pdf_pages="37-38",
    section="5.2 Dinamica de redes de opinion",
    note=(
        "One synchronous step of the seven-rule opinion model. Every rule reads "
        "the state at instant t and none reads the partially built next state. "
        "Rules 6 and 7: when the global sum vanishes the node keeps its opinion."
    ),
    reading=(
        "Rule 5, S_i(t+1) = (-1)^b with 'b un numero entero aleatorio par o "
        "impar', is a fair coin per node per instant. For rules 1 and 2 a zero "
        "local field follows zero_rule, so under 'hold' the node keeps its "
        "opinion."
    ),
    tags=("chapter-5", "synchronous-update"),
)
def opinion_step(
    network: Network,
    states: StateLike,
    types: np.ndarray,
    *,
    hub_of: np.ndarray | None = None,
    sink_of: np.ndarray | None = None,
    zero_rule: ZeroRule = "hold",
    scope: OpinionScope = "global",
    tie: TieRule = "first",
    rng: SeedLike = None,
) -> np.ndarray:
    """Advance every node one instant under its own rule.

    Parameters
    ----------
    types:
        ``(N,)`` array of values 1..7, from :func:`assign_opinion_types`.
    hub_of, sink_of:
        Precomputed reference nodes for rules 3 and 4.  Left to
        :func:`degree_extremes` when omitted; passing them keeps a long run
        from recomputing a quantity that cannot change.

    Examples
    --------
    Every node of type 3 copies the highest-degree node, whatever its own
    neighbourhood says:

    >>> import numpy as np
    >>> from netmotion import Network
    >>> tri = Network.from_edges(4, [(0,1),(0,2),(1,2),(2,3)])
    >>> opinion_step(tri, [1, -1, 1, -1], np.full(4, 3)).tolist()
    [1, 1, 1, 1]

    Rules 6 and 7 ignore the topology entirely and read only the global
    magnetisation, here :math:`H = +2`:

    >>> opinion_step(tri, [1, 1, 1, -1], np.full(4, 6)).tolist()
    [-1, -1, -1, -1]
    >>> opinion_step(tri, [1, 1, 1, -1], np.full(4, 7)).tolist()
    [1, 1, 1, 1]

    When the sum is zero they hold, as rules 6 and 7 state:

    >>> opinion_step(tri, [1, 1, -1, -1], np.full(4, 6)).tolist()
    [1, 1, -1, -1]
    """
    n = network.n
    s = _as_states(states, n)
    types = np.asarray(types, dtype=np.int8)
    if types.shape != (n,):
        raise ValueError(f"types must have shape ({n},), got {types.shape}")
    if not np.isin(types, np.arange(1, 8)).all():
        raise ValueError("opinion types must lie in 1..7")

    generator = as_rng(rng)
    if hub_of is None or sink_of is None:
        hub_of, sink_of = degree_extremes(network, scope=scope, tie=tie, rng=generator)

    field = network.adjacency @ s.astype(float)
    magnetisation = int(s.sum())

    follower = sgn_states(field, s, zero_rule=zero_rule, rng=generator)
    # Where the field is non-zero the contrarian is the exact negation of the
    # follower; where it vanishes both fall back to the same zero convention,
    # so that "hold" holds the opinion rather than inverting a held one.
    contrarian = np.where(field != 0, -follower, follower).astype(np.int8)
    coin = (2 * generator.integers(0, 2, size=n) - 1).astype(np.int8)
    global_sign = np.int8(np.sign(magnetisation))

    new = np.empty(n, dtype=np.int8)
    new[types == 1] = follower[types == 1]
    new[types == 2] = contrarian[types == 2]
    new[types == 3] = s[hub_of][types == 3]
    new[types == 4] = s[sink_of][types == 4]
    new[types == 5] = coin[types == 5]
    if magnetisation == 0:
        new[types == 6] = s[types == 6]
        new[types == 7] = s[types == 7]
    else:
        new[types == 6] = -global_sign
        new[types == 7] = global_sign
    return new


@sourced(
    Origin.THESIS_2011,
    pdf_pages="38-40",
    section="5.2 Dinamica de redes de opinion",
    note=(
        "The runs behind figs. 25-28: N = 100, 60 opinions in favour and 40 "
        "against at t = 1, types drawn once from the percentage vector, and "
        "100 instants plotted -- 'los estados oscilan con lo que no se termina "
        "de alcanzar el equilibrio'."
    ),
    reading="Each run is one realisation; runs are compared by their bands and excursions.",
    tags=("chapter-5", "figure-25", "figure-26", "figure-27", "figure-28"),
)
def run_opinion(
    network: Network,
    percentages: Sequence[float],
    *,
    states: StateLike | None = None,
    n_in_favour: int = 60,
    steps: int = 99,
    zero_rule: ZeroRule = "hold",
    scope: OpinionScope = "global",
    tie: TieRule = "first",
    rng: SeedLike = None,
) -> OpinionRun:
    """Run the seven-rule opinion model over a fixed horizon.

    Parameters
    ----------
    states:
        The opinions at :math:`t = 1`.  Drawn from *n_in_favour* when omitted.
    n_in_favour:
        Number of nodes starting at :math:`+1`; the thesis uses 60 of 100.
    steps:
        Number of updates.  The default of 99 gives the 100 instants
        :math:`t = 1 \\ldots 100` plotted in figs. 25-28.

    Examples
    --------
    A network of pure type-1 individuals *is* the base rule of section 5.1, so
    the two must agree step for step:

    >>> import numpy as np
    >>> from netmotion import Network
    >>> from netmotion.dynamics import run_hopfield
    >>> tri = Network.from_edges(4, [(0,1),(0,2),(1,2),(2,3)])
    >>> start = np.array([1, 1, 1, -1])
    >>> opinions = run_opinion(tri, [100, 0, 0, 0, 0, 0, 0], states=start, steps=4)
    >>> bool(np.array_equal(opinions.states, run_hopfield(tri, start, steps=4).states))
    True

    The personalities are drawn once and are the same array at the end of the
    run as at the start:

    >>> run = run_opinion(tri, [0, 0, 25, 25, 0, 25, 25], states=start, steps=4, rng=0)
    >>> run.types.shape, sorted(set(run.types.tolist()))
    ((4,), [3, 4, 6, 7])
    >>> run.states.shape
    (5, 4)
    """
    n_plus = n_in_favour
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")

    generator = as_rng(rng)
    types = assign_opinion_types(network.n, percentages, rng=generator)
    hub_of, sink_of = degree_extremes(network, scope=scope, tie=tie, rng=generator)

    if states is None:
        s = random_initial_states(network.n, network.n - n_plus, rng=generator)
    else:
        s = _as_states(states, network.n)

    trace = np.empty((steps + 1, network.n), dtype=np.int8)
    trace[0] = s
    for k in range(steps):
        s = opinion_step(
            network,
            s,
            types,
            hub_of=hub_of,
            sink_of=sink_of,
            zero_rule=zero_rule,
            rng=generator,
        )
        trace[k + 1] = s
    return OpinionRun(states=trace, types=types)
