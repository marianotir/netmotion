"""Immunisation against infection: which nodes to protect, and how much it buys.

Thesis section 5.5.2, *Inmunizacion frente a infeccion* (PDF pp. 56-63).

Sections 5.4 and 5.5.1 built two things: a deterministic infection that crosses
a network in a countable number of steps, and a set of scores saying which node
is the most "vulnerable" -- the one an infection starting there reaches the rest
of the network from fastest.  Section 5.5.2 puts them together and asks the
practical question.  If a limited number of nodes can be made immune, which ones
should they be, and what does immunising them actually do to the epidemic?

The thesis's protocol is printed as a numbered recipe on PDF p. 56:

1. start an infection at node 1 with nobody immune;
2. make the single highest-ranked node immune and start the infection at node 1
   again;
3. increase the number of immune nodes by one, always taking the next node down
   the ranking, and start the infection at node 1 again;

repeating 2-3 *"hasta que toda la red ha sido inmunizada"*.  The ranking itself
is the criterion under study: the thesis takes the selective-immunisation idea
of reference [38] -- protect the best-connected nodes first -- and generalises
it by ranking on its own higher-order degree :math:`K_m(n)` instead of on plain
degree.  Since :math:`K_m(1)` *is* the degree, sweeping :math:`n = 1 \\ldots 8`
puts the standard criterion at one end of a family and asks whether looking
further into the network than one link does better.

Two quantities are measured, and both are averaged over every possible seed
node rather than over node 1 alone, *"para obtener una estadistica adecuada"*:

**Size.**  The number of immune nodes at which the infection stops spreading
(Figs. 58-59, PDF p. 57).  On a 100-node scale-free network the thesis reports
roughly 53 nodes when :math:`K(1)` is the criterion, rising to about 62 for
:math:`K(3)` and :math:`K(4)`.

**Speed.**  The propagation speed :math:`\\alpha` of section 5.4 falls as more
nodes are immunised.  Averaging over seeds gives
:math:`\\langle \\alpha \\rangle_{cn}` for :math:`cn = 1 \\ldots N-1` immune
nodes (Fig. 61), a decaying curve that the thesis fits to

.. math:: \\langle \\alpha \\rangle_{cn} = B e^{\\beta\\, cn}

and reads :math:`\\beta = -0.023` off for the scale-free network with the
:math:`K(1)` criterion (PDF p. 59).  A more negative :math:`\\beta` means each
additional immune node slows the epidemic more, so :math:`\\beta` is the
thesis's single-number score for a criterion.

Its conclusion is that plain degree wins on both measures: for the
Erdos-Renyi, Watts-Strogatz, Newman-Watts and scale-free networks at
:math:`N = 50, 75, 100`, :math:`K(1)` both extinguishes the infection with the
fewest immune nodes and gives the largest :math:`|\\beta|` (PDF p. 62).  The one
counter-example is the family of *ordered* networks with a defined structure
drawn in Fig. 68, where :math:`K(3)` wins instead, from
which the thesis concludes that the best criterion depends on how ordered the
network is (PDF p. 63).

What "immune" means
-------------------
The thesis defines immunity on PDF p. 54 as a node frozen at :math:`S_n(t)=+1`
whatever its neighbours do.  Under the monotone infection rule that is exactly
:func:`~netmotion.propagation.infection.propagate`'s ``immune=`` argument: an
immune node is never infected and therefore never passes the infection on, so
immunising it is operationally the same as cutting it out of the network.  This
module is only the strategy layer on top of that -- it chooses *which* nodes to
freeze and measures the consequence.

Readings taken
--------------
Each is recorded in the ``reading`` field of the object that depends on it.

*How :math:`\\alpha` is read.*  *"Se repetiran los calculos antes expuestos"*
(PDF p. 58): each curve's :math:`\\alpha` is the one of section 5.4.1, fitted
over the first instants of the curve before the infection has reached every
node (``window="first"``).  Read that way the sweep lands on the printed values
of Figs. 61-63 -- :math:`\\langle\\alpha\\rangle` near 1.3 with nobody immune,
about 0.9 with ten nodes immune, :math:`\\beta` near -0.023.  ``horizon=``
instead reads every curve over a common span of instants, held at its final
value once the infection has stopped (:data:`FIG_60_HORIZON`, the
:math:`t = 1 \\ldots 8` axis of Fig. 60).

*The tie order.*  Nodes with equal scores are ordered by a drawn permutation;
``ties="index"`` orders them by node index.

*"Podemos considerar de modo aproximativo que la infeccion se ha frenado con
exito cuando el numero de nodos infectados se reduce a cero"* (PDF p. 57).  The
seed is not immunised, and the infection counts as stopped when it does not
leave the seed (``threshold=1``).

``tests/reproduction/test_immunity.py`` asserts the findings of the section -- that :math:`K(1)` needs the fewest immune nodes
and gives the largest :math:`|\\beta|`, and that :math:`\\langle\\alpha\\rangle`
decays with the number of immune nodes.  The reported values are recorded in
:data:`THESIS_SCALE_FREE_100`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..graph import Network
from ..metrics.degree import degrees as _degrees
from ..provenance import Origin, sourced
from ..rng import SeedLike, as_rng
from .infection import propagate, propagation_speed

__all__ = [
    "IMMUNISATION_CRITERIA",
    "CRITERION_ORIGIN",
    "FIG_60_HORIZON",
    "DEFAULT_TIE_SEED",
    "immunisation_ranking",
    "ImmunisationSweep",
    "immunisation_sweep",
    "nodes_to_extinction",
    "beta_factor",
    "beta_fit_range",
    "BETA_FIT_FLOOR",
    "compare_orders",
    "compare_immunisation_criteria",
    "ThesisImmunisationResult",
    "THESIS_SCALE_FREE_100",
    "THESIS_BEST_ORDER_BY_NETWORK",
]


#: The ranking criteria :func:`immunisation_ranking` understands.
IMMUNISATION_CRITERIA: tuple[str, ...] = (
    "degree",
    "k_order_n",
    "random",
    "min_propagation_time",
    "node_removal",
)

#: Which of them the thesis actually used, and which are 2026 additions.
#:
#: Section 5.5.2 ranks on the higher-order degree :math:`K_m(n)`, of which plain
#: degree is the :math:`n = 1` case; the other three are 2026 additions offered
#: through the same interface.
CRITERION_ORIGIN: dict[str, Origin] = {
    "degree": Origin.THESIS_2011,
    "k_order_n": Origin.THESIS_2011,
    "random": Origin.EXTENSION,
    "min_propagation_time": Origin.EXTENSION,
    "node_removal": Origin.EXTENSION,
}

#: The time axis of Fig. 60 (PDF p. 58), :math:`t = 1 \ldots 8`.
#:
#: Fig. 60 draws the whole family of :math:`p(t)` curves -- one per number of
#: immune nodes, all with the :math:`K(1)` criterion and the infection started
#: at node 1 -- on a single axis running :math:`t = 1 \ldots 8`.  Passed as
#: ``horizon=`` to :func:`immunisation_sweep`, every curve is read over that
#: common span instead of over its own first instants.  Read that way
#: :math:`\langle\alpha\rangle` starts near half its section 5.4.1 value and
#: :math:`\beta` comes out about twice the printed -0.023, so it is not the
#: default.
FIG_60_HORIZON: int = 8

#: The tie order used by :func:`immunisation_ranking` when no ``rng`` is given.
#:
#: Fixed -- the thesis's submission date, as elsewhere in the package -- so that
#: a sweep run without a random state is still reproducible; pass ``rng=`` to
#: draw a different tie order, or several, and average over them.
DEFAULT_TIE_SEED: int = 20110930


# ---------------------------------------------------------------------------
# Scores borrowed from netmotion.metrics
#
# The higher-order degree K_m(n) of section 5.5.1.3, the minimum-propagation-
# time score t_m of section 5.5.1.2 and the node-elimination peeling of section
# 5.5.1.1 are network metrics, not immunisation strategies. They live in
# netmotion.metrics and nothing about them is recomputed here.
#
# The imports sit inside the functions because
# netmotion.metrics.propagation_centrality imports back from
# netmotion.propagation.infection, so importing it at module scope would close
# an import cycle through this package's __init__. That is the only reason they
# are not at the top of the file.
# ---------------------------------------------------------------------------


def _k_order_n(network: Network, order: int) -> np.ndarray:
    """:math:`K_m(n)` for every node, from :mod:`netmotion.metrics`."""
    from ..metrics.higher_order_degree import k_order_all

    return k_order_all(network, order)[:, order - 1].astype(float)


def _peeling_depth(network: Network) -> np.ndarray:
    """Section 5.5.1.1 peeling depth, from :mod:`netmotion.metrics`.

    A node stripped in the first round scores zero and the survivor scores
    highest, so a large depth is a node the peeling considered central.
    """
    from ..metrics.propagation_centrality import node_removal_scores

    return np.asarray(node_removal_scores(network), dtype=float)


def _min_propagation_time(network: Network, **kwargs) -> np.ndarray:
    """:math:`t_m` for every node, from :mod:`netmotion.metrics`."""
    from ..metrics.propagation_centrality import min_time_scores

    return np.asarray(min_time_scores(network, **kwargs), dtype=float)


# ---------------------------------------------------------------------------
# the ranking
# ---------------------------------------------------------------------------


@sourced(
    Origin.THESIS_2011,
    pdf_pages="56",
    section="5.5.2 Inmunizacion frente a infeccion",
    note=(
        "The immunisation order of the printed recipe: score every node, sort "
        "from most to least important, immunise down the list one node at a "
        "time. criterion='degree' is the selective-immunisation criterion the "
        "thesis borrows from [38]; criterion='k_order_n' with order=1..8 is the "
        "generalisation the section is about, and order=1 reproduces 'degree'. "
        "The remaining three criteria are 2026 additions -- see "
        "CRITERION_ORIGIN."
    ),
    reading=(
        "Tied nodes are ordered by a drawn permutation (ties='shuffle', seeded "
        "from rng=, or from DEFAULT_TIE_SEED when none is given); ties='index' "
        "orders them by node index. The ranking is computed once, on the intact "
        "network: 'la cantidad de nodos inmunes "
        "correspondientes a los nodos con mayor grado K(1)'."
    ),
    tags=("chapter-5", "convention"),
)
def immunisation_ranking(
    network: Network,
    criterion: str = "degree",
    *,
    order: int = 1,
    ties: str = "shuffle",
    rng: SeedLike = None,
) -> np.ndarray:
    """Node indices in the order the criterion would immunise them.

    Parameters
    ----------
    criterion:
        ``"degree"``
            Highest degree first: the thesis's selective immunisation.
        ``"k_order_n"``
            Highest :math:`K_m(n)` first, with *n* given by ``order``.  This is
            the family section 5.5.2 sweeps over; ``order=1`` is ``"degree"``.
        ``"random"``
            A uniform random permutation, the baseline selective immunisation
            is normally judged against.  2026 addition.
        ``"min_propagation_time"``
            Smallest :math:`t_m` first, i.e. the fastest spreaders of section
            5.5.1.2 first.  2026 addition.
        ``"node_removal"``
            Deepest survivor of the section 5.5.1.1 peeling first.  2026
            addition.
    order:
        The :math:`n` of :math:`K_m(n)`; used by ``"k_order_n"`` only.
    ties:
        How nodes with the same score are ordered.  ``"shuffle"`` (the default)
        draws a tie order from ``rng``; ``"index"`` puts the lowest node index
        first.
    rng:
        Random state: the permutation for ``"random"``, and the tie order for
        every criterion when ``ties="shuffle"``.  ``None`` means a fresh draw
        for ``"random"`` and :data:`DEFAULT_TIE_SEED` for the tie order, so a
        ranking is reproducible without a seed being passed.

    Returns
    -------
    ndarray
        A permutation of ``0 .. N-1``, most-important-first.

    Examples
    --------
    On the nine-node tree of Fig. 54 (PDF p. 54, here indexed from zero) the
    degree criterion picks the two hubs first, which is Barabasi's answer.  They
    have the same degree, so which of the two comes first is the tie order's
    business, not the criterion's:

    >>> from netmotion import Network
    >>> tree = Network.from_edges(
    ...     9, [(0,3),(1,3),(2,3),(3,4),(4,5),(5,6),(5,7),(5,8)]
    ... )
    >>> sorted(immunisation_ranking(tree, "degree")[:2].tolist())
    [3, 5]

    The three criteria that look further than one link all put node 4 -- the
    thesis's "nodo 5", its central node -- first instead:

    >>> for name in ("k_order_n", "min_propagation_time", "node_removal"):
    ...     int(immunisation_ranking(tree, name, order=2)[0])
    4
    4
    4
    """
    if criterion not in IMMUNISATION_CRITERIA:
        raise ValueError(f"criterion must be one of {IMMUNISATION_CRITERIA}, got {criterion!r}")
    if ties not in ("shuffle", "index"):
        raise ValueError(f"ties must be 'shuffle' or 'index', got {ties!r}")

    if criterion == "random":
        return as_rng(rng).permutation(network.n)

    if criterion == "degree":
        score = -_degrees(network).astype(float)
    elif criterion == "k_order_n":
        score = -_k_order_n(network, order)
    elif criterion == "min_propagation_time":
        score = _min_propagation_time(network)
    else:
        score = -_peeling_depth(network).astype(float)

    if ties == "index":
        # A stable sort is what turns "break ties by node index" into a fact
        # rather than an accident of the sorting algorithm.
        return np.argsort(score, kind="stable")

    # The score decides the ranking wherever it discriminates; where it does not,
    # a drawn order decides, so that nothing of the node numbering -- which on a
    # ring lattice is the ring itself -- leaks into which nodes get protected.
    tie_order = as_rng(DEFAULT_TIE_SEED if rng is None else rng).permutation(network.n)
    return np.lexsort((tie_order, score))


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------


def _over_horizon(curve: np.ndarray, horizon: int | None) -> np.ndarray:
    """*curve* as Fig. 60 draws it: the same *t* axis whatever the immune count.

    A run stops as soon as the infection stops moving, so its curve is as long
    as that infection lasted.  Fig. 60 puts all of them on one axis running to
    :math:`t = 8`; a short curve is held at its final value, since nothing
    changes after the last instant.
    """
    if horizon is None or curve.size == horizon:
        return curve
    if curve.size > horizon:
        return curve[:horizon]
    return np.concatenate([curve, np.full(horizon - curve.size, curve[-1])])


@sourced(
    Origin.THESIS_2011,
    pdf_pages="57-59",
    section="5.5.2 Inmunizacion frente a infeccion",
    note=(
        "The record of one immunisation sweep: the two curves section 5.5.2 "
        "plots -- infected nodes against immune nodes (Fig. 58) and <alpha> "
        "against immune nodes (Fig. 61) -- together with the two numbers it "
        "reduces them to, the mean immune nodes needed to stop the infection "
        "(Fig. 59) and the decay factor beta (Fig. 63)."
    ),
)
@dataclass
class ImmunisationSweep:
    """One criterion swept across every number of immune nodes.

    Attributes
    ----------
    criterion, order:
        The ranking that produced it.
    ranking:
        The immunisation order, as returned by :func:`immunisation_ranking`.
    counts:
        The numbers of immune nodes examined, ascending.
    infected:
        ``infected[i]`` is the final number of infected nodes averaged over the
        seeds, with ``counts[i]`` nodes immune.  This is the quantity Fig. 58
        plots.
    speed:
        ``speed[i]`` is :math:`\\langle\\alpha\\rangle_{cn}`, the propagation
        speed averaged over the seeds whose curve can be fitted, or ``nan``
        when none can.  This is the quantity Fig. 61 plots.
    extinction:
        ``extinction[j]`` is the smallest number of immune nodes at which the
        infection started at ``seeds[j]`` stopped spreading, or ``inf`` if it
        never did within ``counts``.
    start_nodes:
        The nodes the infection was started from, averaged over.
    threshold:
        The final infected count still counted as extinguished.
    horizon:
        The number of instants each :math:`p(t)` curve was cut or held to
        before its fit, common to every immune count, or ``None`` if each
        curve was read as it is.
    window:
        The fitting window of :func:`~netmotion.propagation.propagation_speed`
        each curve was read with.
    """

    criterion: str
    order: int
    ranking: np.ndarray
    counts: np.ndarray
    infected: np.ndarray
    speed: np.ndarray
    extinction: np.ndarray
    start_nodes: np.ndarray
    threshold: int = 1
    horizon: int | None = None
    window: str = "first"
    immune_sets: list[np.ndarray] = field(default_factory=list)

    @property
    def mean_nodes_to_extinction(self) -> float:
        """The Fig. 59 quantity: immune nodes needed, averaged over seeds."""
        return float(np.mean(self.extinction))

    @property
    def beta(self) -> float:
        """The Fig. 61/63 quantity: the decay rate of :math:`\\langle\\alpha\\rangle`."""
        return beta_factor(self.counts, self.speed)


@sourced(
    Origin.THESIS_2011,
    pdf_pages="56-57",
    section="5.5.2 Inmunizacion frente a infeccion",
    note=(
        "The printed three-step recipe: infect, add the next node of the "
        "ranking to the immune set, infect again, until the network is "
        "immunised. Repeated from every seed node rather than from node 1 "
        "alone, which is what the thesis does 'para obtener una estadistica "
        "adecuada' before averaging."
    ),
    reading=(
        "(1) 'Se repetiran los calculos antes expuestos': alpha is read as in "
        "section 5.4.1, over the first instants of each curve before the "
        "infection has reached every node (window='first'); "
        "horizon=FIG_60_HORIZON reads every curve over the t = 1..8 axis of "
        "Fig. 60 instead. (2) The seed is not immunised: it is skipped when the ranking "
        "reaches it. (3) The infection counts as reduced to zero, 'de modo "
        "aproximativo', when it does not leave the seed (threshold=1). "
        "(4) 'tendremos N factores alpha ... uno por cada nodo n de la red en la "
        "cual se ha iniciado la infeccion': a seed whose infection does not "
        "spread contributes alpha = 0 (stopped='zero', the default); "
        "stopped='skip' averages over the seeds that spread."
    ),
    tags=("chapter-5", "original-behaviour"),
)
def immunisation_sweep(
    network: Network,
    criterion: str = "degree",
    *,
    order: int = 1,
    start_nodes: np.ndarray | list[int] | None = None,
    counts: np.ndarray | list[int] | None = None,
    threshold: int = 1,
    stopped: str = "zero",
    horizon: int | None = None,
    window: str = "first",
    ties: str = "shuffle",
    rng: SeedLike = None,
    keep_immune_sets: bool = False,
    **propagate_kwargs,
) -> ImmunisationSweep:
    """Immunise down the ranking one node at a time and watch the avalanche.

    Parameters
    ----------
    start_nodes:
        Seed nodes to average over.  Defaults to every node, which is the
        thesis's statistic; pass ``[0]`` for the single-seed curves of Figs. 58
        and 60, where the infection always starts at node 1.
    counts:
        Numbers of immune nodes to examine, ascending.  Defaults to
        ``0 .. N-1``, the whole sweep.
    threshold:
        Largest final infected count still counted as extinguished.  ``1``, the
        default, means the infection never left the seed; see the module
        docstring.
    stopped:
        What a seed whose infection never spreads contributes to
        :math:`\\langle\\alpha\\rangle`.  ``"zero"`` (the default) contributes
        :math:`\\alpha = 0`, one factor per seed, *"tendremos N factores
        alpha"*.  ``"skip"`` averages only over seeds that did spread.
    horizon:
        ``None`` (the default) reads each curve as it is.  An integer cuts or
        holds every curve to that many instants first, each held at its final
        value past the instant the infection stops; :data:`FIG_60_HORIZON` is
        the :math:`t = 1 \\ldots 8` axis of Fig. 60.
    window:
        The fitting window passed to
        :func:`~netmotion.propagation.propagation_speed`.  The default,
        ``"first"``, is the :math:`\\alpha` of section 5.4.1.
    ties:
        Tie-break for the ranking; see :func:`immunisation_ranking`.
    keep_immune_sets:
        Store the immune set used at each count, for drawing.

    Examples
    --------
    On the nine-node tree of Fig. 54 (PDF p. 54, indexed from zero) the whole
    network is reached from every seed while nobody is immune.  Immunising the
    first hub of the degree ranking cuts the mean reach to under four nodes, and
    immunising the second cuts it to under two:

    >>> from netmotion import Network
    >>> tree = Network.from_edges(
    ...     9, [(0,3),(1,3),(2,3),(3,4),(4,5),(5,6),(5,7),(5,8)]
    ... )
    >>> sweep = immunisation_sweep(tree, "degree", counts=[0, 1, 2])
    >>> sweep.infected.round(3).tolist()
    [9.0, 3.667, 1.667]

    Extinction is very uneven across seeds, which is why the thesis averages
    over all of them.  The two hubs have the same degree, so the tie order
    decides which is protected first; here it is index 5, and an infection
    started at one of its own leaves is stopped by that single immune node,
    while one started at the far hub survives until eight of the nine nodes are
    immune, because the ranking protects the other side of the tree first:

    >>> full = immunisation_sweep(tree, "degree")
    >>> full.extinction.tolist()
    [2.0, 2.0, 2.0, 6.0, 2.0, 8.0, 1.0, 1.0, 1.0]
    >>> round(full.mean_nodes_to_extinction, 3)
    2.778
    """
    seeds = start_nodes
    if stopped not in ("zero", "skip"):
        raise ValueError(f"stopped must be 'zero' or 'skip', got {stopped!r}")
    if horizon is not None and horizon < 2:
        raise ValueError(f"horizon must be at least two instants, got {horizon!r}")
    n = network.n
    seed_nodes = np.arange(n) if seeds is None else np.asarray(seeds, dtype=int)
    count_grid = np.arange(n) if counts is None else np.asarray(counts, dtype=int)
    ranking = immunisation_ranking(network, criterion, order=order, ties=ties, rng=rng)

    infected = np.full(count_grid.size, np.nan)
    speed = np.full(count_grid.size, np.nan)
    extinction = np.full(seed_nodes.size, np.inf)
    immune_sets: list[np.ndarray] = []

    for i, count in enumerate(count_grid):
        totals = []
        rates = []
        for j, seed in enumerate(seed_nodes):
            immune = ranking[ranking != seed][:count]
            run = propagate(network, int(seed), immune=immune, **propagate_kwargs)
            reached = int(np.isfinite(run.infected_at).sum())
            totals.append(reached)
            if reached <= threshold and not np.isfinite(extinction[j]):
                extinction[j] = float(count)
            if run.curve.size == 1:
                # A curve of one instant is an infection that never moved; one
                # alpha is counted per seed, and for that seed the rate is zero.
                if stopped == "zero":
                    rates.append(0.0)
                continue
            try:
                rates.append(propagation_speed(_over_horizon(run.curve, horizon), window=window))
            except ValueError:
                pass
        infected[i] = float(np.mean(totals))
        if rates:
            speed[i] = float(np.mean(rates))
        if keep_immune_sets:
            immune_sets.append(ranking[:count].copy())

    return ImmunisationSweep(
        criterion=criterion,
        order=order,
        ranking=ranking,
        counts=count_grid,
        infected=infected,
        speed=speed,
        extinction=extinction,
        start_nodes=seed_nodes,
        threshold=threshold,
        horizon=horizon,
        window=window,
        immune_sets=immune_sets,
    )


@sourced(
    Origin.THESIS_2011,
    pdf_pages="57",
    section="5.5.2 Inmunizacion frente a infeccion",
    note=(
        "The Fig. 59 quantity: 'los valores medios de los momentos en los "
        "cuales la infeccion se ha hecho cero cuando esta comienza en cada nodo "
        "de la red'. The thesis plots it against the order n of the K(n) used "
        "as the criterion and picks the n that minimises it."
    ),
)
def nodes_to_extinction(
    network: Network,
    criterion: str = "degree",
    *,
    order: int = 1,
    threshold: int = 1,
    rng: SeedLike = None,
    **kwargs,
) -> float:
    """Immune nodes needed to stop the infection, averaged over seeds.

    Examples
    --------
    A star is the case the whole selective-immunisation idea rests on.
    Immunising the hub isolates every leaf from every other, so one immune node
    stops an infection starting at any of the four leaves; only an infection
    starting at the hub itself is expensive, needing all four leaves immune,
    since the seed is never immunised.  The mean is therefore
    :math:`(4 \\times 1 + 4)/5`:

    >>> from netmotion import Network
    >>> star = Network.from_edges(5, [(0, 1), (0, 2), (0, 3), (0, 4)])
    >>> nodes_to_extinction(star, "degree")
    1.6

    A random ranking has to reach the hub by luck, and on average pays for it:

    >>> round(nodes_to_extinction(star, "random", rng=0), 2)
    3.4
    """
    sweep = immunisation_sweep(
        network,
        criterion,
        order=order,
        threshold=threshold,
        rng=rng,
        **kwargs,
    )
    return sweep.mean_nodes_to_extinction


#: The smallest :math:`\langle\alpha\rangle` the fit of :math:`\beta` uses.
#:
#: In the last few immune counts nearly every infection is stopped at its seed,
#: :math:`\langle\alpha\rangle` falls towards zero, and its logarithm falls
#: without bound; left in, those few counts would decide the whole fit and the
#: fitted curve would miss the points it is drawn over.
BETA_FIT_FLOOR: float = 0.1


def beta_fit_range(
    counts: np.ndarray, speeds: np.ndarray, *, floor: float | None = None
) -> np.ndarray:
    """The immune counts :func:`beta_factor` fits: at least one immune node, and
    :math:`\\langle\\alpha\\rangle` of at least *floor*.

    Examples
    --------
    >>> import numpy as np
    >>> beta_fit_range(np.arange(5), np.array([1.3, 0.9, 0.4, 0.05, 0.0])).tolist()
    [False, True, True, False, False]
    """
    floor = BETA_FIT_FLOOR if floor is None else floor
    counts = np.asarray(counts, dtype=float)
    speeds = np.asarray(speeds, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.isfinite(speeds) & (speeds > 0.0) & (speeds >= floor) & (counts >= 1.0)


@sourced(
    Origin.THESIS_2011,
    pdf_pages="58-59",
    section="5.5.2 Inmunizacion frente a infeccion",
    note=(
        "beta from the fit <alpha>_cn = B exp(beta cn) of Fig. 61, where cn is "
        "the number of immune nodes. The thesis reports beta = -0.023 for the "
        "100-node scale-free network with K(1) as the criterion and treats the "
        "criterion with the largest |beta| as the best one."
    ),
    reading=(
        "Every immune count with <alpha> of at least BETA_FIT_FLOOR = 0.1 is "
        "used in the fit. Past it, in the last few counts, nearly every "
        "infection is stopped at its seed and ln<alpha> falls without bound, "
        "which would decide the whole fit."
    ),
    tags=("chapter-5", "fit-window"),
)
def beta_factor(
    counts: np.ndarray, speeds: np.ndarray, *, floor: float | None = None
) -> float:
    """The exponential decay rate :math:`\\beta` of :math:`\\langle\\alpha\\rangle`.

    Fitted by ordinary least squares of :math:`\\ln\\langle\\alpha\\rangle` on
    the number of immune nodes, over the counts where :math:`\\langle\\alpha\\rangle`
    is at least *floor* (:data:`BETA_FIT_FLOOR` when ``None``).

    Raises
    ------
    ValueError
        If fewer than two counts carry a usable speed.

    Examples
    --------
    Exact exponential data returns its own exponent:

    >>> import numpy as np
    >>> cn = np.arange(1, 21)
    >>> round(beta_factor(cn, 1.5 * np.exp(-0.03 * cn)), 6)
    -0.03
    """
    counts = np.asarray(counts, dtype=float)
    speeds = np.asarray(speeds, dtype=float)
    usable = beta_fit_range(counts, speeds, floor=floor)
    if usable.sum() < 2:
        raise ValueError(
            "beta is undefined: fewer than two immune-node counts produced a "
            "measurable propagation speed"
        )
    return float(np.polyfit(counts[usable], np.log(speeds[usable]), 1)[0])


@sourced(
    Origin.THESIS_2011,
    pdf_pages="57-59",
    section="5.5.2 Inmunizacion frente a infeccion",
    note=(
        "The experiment of Figs. 59 and 63: run the whole sweep once per order "
        "n of K(n), n = 1..8, and compare the orders on the two measures. The "
        "thesis's conclusion is that n = 1 wins on both for ER, WS, NW and "
        "scale-free networks. Every order is swept on the same terms -- the "
        "same seeds, the same immune counts, and in particular the one reading "
        "of alpha -- since the orders are only comparable on the "
        "measure that is read the same way for each of them."
    ),
)
def compare_orders(
    network: Network,
    orders: range | list[int] = range(1, 9),
    **kwargs,
) -> dict[int, ImmunisationSweep]:
    """Sweep :math:`K(n)` as the criterion for each *n*, as Figs. 59 and 63 do.

    Examples
    --------
    A star is the smallest case of the saturation the thesis's Fig. 59 shows at
    the high orders.  :math:`K(1)` separates the hub from the leaves and
    immunises it first; from order 2 on, every node already sees all four
    branches, so every node scores the same, the criterion has nothing left to
    rank on, and it pays for that:

    >>> from netmotion import Network
    >>> star = Network.from_edges(5, [(0, 1), (0, 2), (0, 3), (0, 4)])
    >>> result = compare_orders(star, [1, 2, 3])
    >>> [r.mean_nodes_to_extinction for r in result.values()]
    [1.6, 2.2, 2.2]
    """
    return {
        int(n): immunisation_sweep(network, "k_order_n", order=int(n), **kwargs) for n in orders
    }


@sourced(
    Origin.EXTENSION,
    pdf_pages="54-59",
    section="5.5.2 Inmunizacion frente a infeccion",
    reproduces="netmotion.propagation.immunity.immunisation_sweep",
    note=(
        "New in 2026. Adds a random baseline and the two centrality scores of "
        "section 5.5.1 -- the minimum-propagation-time score and the "
        "node-elimination peeling -- as immunisation criteria, alongside the "
        "orders of K(n) that section 5.5.2 compares."
    ),
)
def compare_immunisation_criteria(
    network: Network,
    criteria: tuple[str, ...] | list[str] = IMMUNISATION_CRITERIA,
    **kwargs,
) -> dict[str, ImmunisationSweep]:
    """Sweep several criteria on one network and return them side by side.

    Examples
    --------
    On the nine-node tree of Fig. 54 every structural criterion needs fewer
    immune nodes than chance does:

    >>> from netmotion import Network
    >>> tree = Network.from_edges(
    ...     9, [(0,3),(1,3),(2,3),(3,4),(4,5),(5,6),(5,7),(5,8)]
    ... )
    >>> result = compare_immunisation_criteria(tree, rng=0)
    >>> best = min(result, key=lambda c: result[c].mean_nodes_to_extinction)
    >>> best
    'degree'
    >>> result["degree"].mean_nodes_to_extinction < result["random"].mean_nodes_to_extinction
    True
    """
    return {str(name): immunisation_sweep(network, str(name), **kwargs) for name in criteria}


# ---------------------------------------------------------------------------
# what the thesis reported
# ---------------------------------------------------------------------------


@sourced(
    Origin.THESIS_2011,
    pdf_pages="57-59",
    section="5.5.2 Inmunizacion frente a infeccion",
    note=(
        "The numbers section 5.5.2 reports for its worked case, a scale-free "
        "network of 100 nodes. beta = -0.023 for K(1) and alpha = 0.9 at ten "
        "immune nodes are printed in the text (PDF pp. 58-59); the remaining "
        "per-order values are digitised from Figs. 59 and 63."
    ),
    reading="The per-order entries are digitised from Figs. 59 and 63.",
    tags=("chapter-5", "reported-values", "digitised"),
)
@dataclass(frozen=True)
class ThesisImmunisationResult:
    """Reported immunisation results for one network of the thesis.

    Attributes
    ----------
    network, n:
        Which network the numbers belong to.
    nodes_to_zero:
        Fig. 59: immune nodes needed to reduce the infection to zero, per order
        :math:`n = 1 \\ldots 8` of the :math:`K(n)` used as the criterion.
    beta:
        Fig. 63: the fitted :math:`\\beta` per order, over the same range.
    alpha_at_ten_immune:
        PDF p. 58: *"si inmunizamos la red con los 10 nodos con mayor grado
        K(1) la infeccion se propagara en media de forma exponencial con un
        factor alpha aproximadamente 0.9"*.
    """

    network: str
    n: int
    nodes_to_zero: tuple[float, ...]
    beta: tuple[float, ...]
    alpha_at_ten_immune: float

    @property
    def best_order_by_size(self) -> int:
        """The order needing the fewest immune nodes; the thesis reads 1."""
        return int(np.argmin(self.nodes_to_zero)) + 1

    @property
    def best_order_by_speed(self) -> int:
        """The order with the largest :math:`|\\beta|`; the thesis reads 1."""
        return int(np.argmin(self.beta)) + 1


#: Section 5.5.2's worked case: a scale-free network of 100 nodes.
THESIS_SCALE_FREE_100 = ThesisImmunisationResult(
    network="scale free",
    n=100,
    nodes_to_zero=(52.8, 60.1, 62.4, 62.5, 61.0, 59.3, 59.3, 59.5),
    beta=(-0.0230, -0.0223, -0.0190, -0.0188, -0.0191, -0.0220, -0.0223, -0.0223),
    alpha_at_ten_immune=0.9,
)

#: The order of :math:`K(n)` the thesis concludes is the best criterion, by
#: network family.  PDF p. 62 for the four random families, PDF p. 63 for the
#: ordered networks of Fig. 68, where the answer changes -- which is the
#: section's closing point: *"el criterio mas adecuado para inmunizar una red
#: dependera del grado de ordenamiento de esta"*.
THESIS_BEST_ORDER_BY_NETWORK: dict[str, int] = {
    "erdos_renyi": 1,
    "watts_strogatz": 1,
    "newman_watts": 1,
    "scale_free": 1,
    "ordered": 3,
}
