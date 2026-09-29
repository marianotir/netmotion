"""Infection propagation, and propagation speed as a network characteristic.

Thesis section 5.4.1 (PDF pp. 43-48).

Each node carries a state :math:`S_i(t) \\in \\{+1, -1\\}`, and :math:`-1` means
infected.  The rule (PDF p. 43) is

.. math::

    S_i(t+1) = \\begin{cases}
        -1 & \\text{si } \\exists\\, j \\ /\\ M_{ij} \\neq 0 \\text{ y } S_j(t) = -1 \\\\
        +1 & \\text{si } \\nexists\\, j \\ /\\ M_{ij} \\neq 0 \\text{ y } S_j(t) = -1
    \\end{cases}

with a single seed node infected at :math:`t = 1`.  The model has no recovery
-- *no se desarrolla el concepto de sanacion* -- so an infected node stays
infected and the infected set only grows, as in the thesis's figures.
``heals=True`` gives a variant with recovery, in which a node with no infected
neighbour returns to :math:`+1`.

The infected set at instant *t* is

.. math:: I_t = B_{t-1}(s) = \\{v : d(s, v) \\le t - 1\\}

so the number of steps needed to reach the whole network is the eccentricity of
the seed.

Propagation speed
-----------------
The thesis fits the first instants of the infected-fraction curve, where the
infection grows exponentially, to an exponential :math:`p(t) = A_0 e^{\\alpha t}` and treats :math:`\\alpha` as
*"una forma de evaluar la rapidez con la cual la infeccion se propaga a traves
de la red"*.  Averaging over every possible seed gives
:math:`\\langle \\alpha \\rangle`, which the thesis calls the parameter
characterising propagation on a network of that type and size.

The maximum of :math:`\\langle \\alpha \\rangle` over link density grows
logarithmically with network size, independently of the network type:

.. math:: \\alpha_{\\max}(N) = 0.4974 \\ln N + 0.012

:func:`propagation_speed` fits by default every instant up to and including the
first with :math:`p = 1`; ``window="unsaturated"`` fits only the instants with
:math:`p < 1`, and ``window="first"`` the first :data:`FIRST_INSTANTS` of them.
``tests/reproduction/test_propagation_speed.py`` checks the law.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..graph import Network
from ..provenance import Origin, sourced

__all__ = [
    "PropagationRun",
    "propagate",
    "infected_fraction_curve",
    "propagation_speed",
    "mean_propagation_speed",
    "propagation_time",
    "alpha_max_size_law",
    "FIRST_INSTANTS",
]

#: How many instants ``window="first"`` fits: the instants before the infection
#: has reached the whole network, at most the first four of them.  Fig. 35's
#: curve has exactly four such instants, as the printed regression does.
FIRST_INSTANTS: int = 4


@sourced(
    Origin.THESIS_2011,
    pdf_pages="43-45",
    section="5.4.1 Definicion y analisis",
    note=(
        "The state of one propagation: the instant each node reached S = -1, "
        "and the infected-fraction curve p(t) the thesis plots in figs. 32-35. "
        "Indexing follows the thesis, whose seed is infected at t = 1."
    ),
)
@dataclass
class PropagationRun:
    """The outcome of one propagation, from one seed.

    Attributes
    ----------
    first_infected:
        The node the infection started at.
    infected_at:
        ``infected_at[v]`` is the instant *t* at which *v* became infected, or
        ``np.inf`` if it never did.  The seed has ``t = 1``, matching the
        thesis's indexing.
    curve:
        ``curve[k]`` is :math:`p(k+1)`, the infected fraction at instant
        :math:`k+1`.  ``curve[0] = 1/N``.
    complete:
        Whether the infection eventually reached every node.
    """

    first_infected: int
    infected_at: np.ndarray
    curve: np.ndarray
    complete: bool
    immune: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    @property
    def final_fraction(self) -> float:
        """The fraction of the network the infection reached."""
        return float(self.curve[-1])

    @property
    def steps(self) -> int:
        """Iterations needed to reach the final state; the thesis's :math:`t_n`."""
        return int(self.curve.size - 1)

    def infected_set(self, t: int) -> np.ndarray:
        """Node indices infected at instant *t*."""
        return np.flatnonzero(self.infected_at <= t)


@sourced(
    Origin.THESIS_2011,
    pdf_pages="43-45",
    section="5.4.1 Definicion y analisis",
    note=(
        "State rule S_i(t+1) = -1 if any neighbour is infected, seeded with one "
        "infected node at t = 1. Immune nodes are frozen at S = +1 and, being "
        "never infected, never transmit -- which is the thesis's own definition "
        "in section 5.5 (PDF p. 54)."
    ),
    reading=(
        "There is no recovery, so an infected node stays infected. heals=True "
        "gives a variant with recovery."
    ),
    tags=("original-behaviour", "chapter-5"),
)
def propagate(
    network: Network,
    first_infected: int,
    *,
    immune: np.ndarray | list[int] | None = None,
    heals: bool = False,
    max_steps: int | None = None,
) -> PropagationRun:
    """Run the infection from the node *first_infected* until it stops spreading.

    Parameters
    ----------
    immune:
        Nodes frozen at :math:`S = +1`.  Following the thesis's definition in
        section 5.5 (PDF p. 54), an immune node is never infected and therefore never
        transmits, so immunising a node is operationally the same as cutting it
        out of the propagation.
    heals:
        The variant with recovery.  It need not settle -- on a path the state
        oscillates with period two -- so a healing run stops when it revisits a
        state, or after ``max_steps`` (default ``4N``).

    Examples
    --------
    On a path the infection advances one node per step in each direction:

    >>> from netmotion import Network
    >>> path = Network.from_edges(5, [(0,1),(1,2),(2,3),(3,4)])
    >>> run = propagate(path, 0)
    >>> run.infected_at.tolist()
    [1.0, 2.0, 3.0, 4.0, 5.0]
    >>> run.steps                      # the eccentricity of node 0
    4

    Starting in the middle halves the time:

    >>> propagate(path, 2).steps
    2
    """
    seed = first_infected
    n = network.n
    if not 0 <= seed < n:
        raise IndexError(f"seed {seed} outside 0..{n - 1}")

    immune_mask = np.zeros(n, dtype=bool)
    if immune is not None:
        immune_mask[np.asarray(immune, dtype=int)] = True
    if immune_mask[seed]:
        raise ValueError(f"seed {seed} cannot also be immune")

    adjacency = network.adjacency
    infected = np.zeros(n, dtype=bool)
    infected[seed] = True
    infected_at = np.full(n, np.inf)
    infected_at[seed] = 1.0

    curve = [infected.sum() / n]
    t = 1
    # The variant with recovery can oscillate -- on a path with period two -- so
    # healing runs need a cycle guard as well as a step cap. Without recovery
    # the infected set only grows and cannot cycle.
    seen: set[bytes] = {infected.tobytes()} if heals else set()
    limit = max_steps if max_steps is not None else (4 * n if heals else None)

    while True:
        if limit is not None and t > limit:
            break
        exposed = (adjacency @ infected) > 0
        nxt = exposed if heals else (exposed | infected)
        nxt = nxt & ~immune_mask
        if np.array_equal(nxt, infected):
            break
        newly = nxt & ~infected
        infected = nxt
        t += 1
        infected_at[newly] = np.minimum(infected_at[newly], float(t))
        curve.append(infected.sum() / n)
        if heals:
            key = infected.tobytes()
            if key in seen:
                break
            seen.add(key)
        elif infected.sum() == n - immune_mask.sum():
            break

    return PropagationRun(
        first_infected=seed,
        infected_at=infected_at,
        curve=np.asarray(curve, dtype=float),
        complete=bool(infected.sum() == n - immune_mask.sum()),
        immune=np.flatnonzero(immune_mask),
    )


@sourced(
    Origin.THESIS_2011,
    pdf_pages="43-46",
    section="5.4.1 Definicion y analisis",
    note="p(t), the infected fraction against the instant t, as plotted in figs. 32-35.",
)
def infected_fraction_curve(network: Network, first_infected: int, **kwargs) -> np.ndarray:
    """The curve :math:`p(t)` the thesis plots, indexed from :math:`t = 1`."""
    seed = first_infected
    return propagate(network, seed, **kwargs).curve


@sourced(
    Origin.THESIS_2011,
    pdf_pages="44",
    section="5.4.1 Definicion y analisis",
    note=(
        "alpha from a least-squares fit of ln p(t) against t, over every "
        "instant up to and including the first at which the infection has "
        "covered the network."
    ),
    reading=(
        "The fit runs over every instant up to and including the first "
        "saturated one (window='to_saturation'), which gives the law "
        "alpha_max = 0.4974 ln N + 0.012. window='unsaturated' fits only the "
        "instants with p < 1, and window='first' the first FIRST_INSTANTS of "
        "them, 'los primeros datos' of fig. 35."
    ),
    tags=("chapter-5", "fit-window"),
)
def propagation_speed(curve: np.ndarray, *, window: str = "to_saturation") -> float:
    """The exponential rate :math:`\\alpha` in :math:`p(t) = A_0 e^{\\alpha t}`.

    Fitted by ordinary least squares of :math:`\\ln p` on :math:`t`.

    Parameters
    ----------
    window:
        ``"to_saturation"``
            Every instant up to and including the first at which
            :math:`p = 1`.  The default.
        ``"unsaturated"``
            Only the instants with :math:`p(t) < 1`.
        ``"first"``
            The first :data:`FIRST_INSTANTS` instants with :math:`p(t) < 1`,
            the region where the infection grows exponentially (fig. 35).

    Raises
    ------
    ValueError
        If fewer than two instants are usable, in which case the rate is not
        determined.

    Examples
    --------
    >>> import numpy as np
    >>> curve = np.array([0.01, 0.9, 1.0])
    >>> round(propagation_speed(curve), 3)
    2.303
    >>> round(propagation_speed(curve, window="unsaturated"), 3)
    4.5
    >>> slow = np.array([0.01, 0.02, 0.04, 0.08, 0.1, 0.1])
    >>> round(propagation_speed(slow, window="first"), 3)
    0.693
    """
    if window not in ("to_saturation", "unsaturated", "first"):
        raise ValueError(
            f"window must be 'to_saturation', 'unsaturated' or 'first', got {window!r}"
        )

    curve = np.asarray(curve, dtype=float)
    t = np.arange(1, curve.size + 1, dtype=float)

    usable = curve > 0.0
    if window in ("unsaturated", "first"):
        usable &= curve < 1.0
        if window == "first":
            usable[np.flatnonzero(usable)[FIRST_INSTANTS:]] = False
    else:
        saturated = np.flatnonzero(curve >= 1.0)
        cutoff = int(saturated[0]) if saturated.size else curve.size - 1
        keep = np.zeros(curve.size, dtype=bool)
        keep[: cutoff + 1] = True
        usable &= keep

    if usable.sum() < 2:
        raise ValueError(
            "propagation speed is undefined: fewer than two usable instants "
            "(the infection covered the network too quickly to fit)"
        )
    return float(np.polyfit(t[usable], np.log(curve[usable]), 1)[0])


@sourced(
    Origin.THESIS_2011,
    pdf_pages="44-47",
    section="5.4.1 Definicion y analisis",
    note=(
        "<alpha>, the mean of the propagation speed over every possible seed. "
        "The thesis calls this the parameter that characterises propagation on "
        "a network of a given type and size."
    ),
)
def mean_propagation_speed(network: Network, **kwargs) -> float:
    """:math:`\\langle \\alpha \\rangle`: propagate from every node and average.

    Seeds whose curve saturates too fast to fit are skipped rather than counted
    as zero, since the thesis's quantity is a mean of fitted rates.
    """
    rates = []
    for seed in range(network.n):
        try:
            rates.append(propagation_speed(propagate(network, seed, **kwargs).curve))
        except ValueError:
            continue
    if not rates:
        raise ValueError("no seed produced a fittable propagation curve")
    return float(np.mean(rates))


@sourced(
    Origin.THESIS_2011,
    pdf_pages="55",
    section="5.5.1.2 El metodo de propagacion en tiempo minimo",
    note=(
        "t_n: the number of iterations an infection starting at n needs to "
        "reach the whole network. This is the eccentricity of n, so the node "
        "minimising it is the graph centre and the minimum is the graph radius."
    ),
)
def propagation_time(network: Network, first_infected: int, **kwargs) -> float:
    """:math:`t_n`, or ``inf`` when the infection cannot reach every node."""
    seed = first_infected
    run = propagate(network, seed, **kwargs)
    return float(run.steps) if run.complete else float("inf")


@sourced(
    Origin.THESIS_2011,
    pdf_pages="47",
    section="5.4.1 Definicion y analisis",
    note=(
        "The size law the thesis reports for the maximum mean propagation "
        "speed over link density: alpha_max(N) = 0.4974 ln N + 0.012, found to "
        "hold for all four network types."
    ),
)
def alpha_max_size_law(n_nodes: int | np.ndarray) -> np.ndarray:
    """The thesis's fitted law :math:`\\alpha_{\\max} = 0.4974 \\ln N + 0.012`.

    >>> round(float(alpha_max_size_law(100)), 3)
    2.303
    """
    n = n_nodes
    return 0.4974 * np.log(np.asarray(n, dtype=float)) + 0.012
