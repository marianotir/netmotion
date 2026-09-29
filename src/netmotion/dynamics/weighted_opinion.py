"""Opinion dynamics weighted by the age of each relationship.

Thesis section 5.3, *Dinamica de redes de opinion con pesos* (PDF pp. 40-43).

Sections 5.1 and 5.2 treat every link alike; in section 5.3 the influence of a
link depends on its age, because *"un individuo que conozca a otro desde mas tiempo ejercera una
mayor influencia en su opinion que otro que lo conozca de menos tiempo"*.  The
influence one node has on another should therefore depend not only on the
existence of the tie but on its **age** -- how long the two have been linked.
After naming the weighted-network models it could have used instead
(Yook-Jeong-Barabasi-Tu, Zheng-Trimper-Zheng-Hui, Antal-Krapivsky), the thesis
builds a weight matrix that measures exactly that antiquity and nothing else,
and feeds it into the section 5.1 equation in place of the adjacency matrix
(PDF p. 41):

.. math:: S_i(t+1) = \\mathrm{sgn}\\Big( \\sum_{j}^{N} S_j(t)\\, W_{i,j} \\Big)

Section 5.3 writes this single rule for every node, so this module implements
the weighted version of the base rule of 5.1.

The two weight constructions
----------------------------
Tie age is only directly available on a network that was built in time, and of
the four topologies of chapter 4 only the scale-free network has a growth
history.  The thesis gives two constructions accordingly.

**Grown networks.**  If the entry :math:`M_{i,j}` becomes 1 at instant
:math:`t_n` and is still 1 at the final instant :math:`t_f`, then
:math:`W_{i,j} = t_f - t_n`, and :math:`W_{i,j} = 0` where :math:`M_{i,j} = 0`.
The weight is the lifetime of the link.  This is
:func:`weights_from_edge_ages`, with :func:`weights_from_growth` deriving the
birth instants from the arrival order of a network grown by
:func:`~netmotion.networks.barabasi_albert`.

**ER, Watts-Strogatz and Newman-Watts.**  These are not built in time, so the
thesis gives them a time from the degree sequence: *"un nodo con
un mayor grado k es mas antiguo"*, since an older node has had time to generate
more connections.  Nodes are sorted by decreasing
degree and given birth instants :math:`t_i = 1, \\ldots, N` by rank, the whole
network being assumed to have been built in a time :math:`t = N`.  A tie cannot
exist before both its endpoints do, so it is dated by the *younger* of the two
-- *"la union entre los nodos i y j se produce cuando aparece el nodo mas joven
en la red"* -- giving

.. math::

    W_{i,j} = \\begin{cases}
        N - t_i & t_i > t_j \\\\
        N - t_j & t_i < t_j
    \\end{cases}
    \\qquad = \\; N - \\max(t_i, t_j)

which is :func:`weights_from_degree_rank`.

State space, stopping criterion and results
-------------------------------------------
The state space and the synchronous update are those of section 5.1, and no
stopping criterion is applied: figs. 29-31 plot a fixed horizon of 20 instants,
and fig. 31 starts from 52 nodes at :math:`+1` and 48 at :math:`-1`.  The experiment is a
comparison, not a single run -- the same initial opinions are evolved twice,
once through :math:`W` and once through :math:`M`, for two independent initial
conditions and for each of ER, SW and NW (:func:`compare_weighted_unweighted`).

*"De esto se concluye que la introduccion de pesos no provoca que se alcance
antes el equilibrio entre las opiniones con lo que la evolucion de estas depende
de los estados iniciales de los nodos."*  In the first simulation the unweighted
run settles first, except on NW, where neither settles; in the second the two
settle at a similar time on ER, only the weighted run settles on WS and only
the unweighted one on NW.

Under both constructions a link created at the last instant has weight zero:
:math:`t_n = t_f` for a grown network, and :math:`t = N` for the lowest-degree
node under the degree ranking.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..graph import Network
from ..provenance import Origin, sourced
from ..rng import SeedLike, as_rng
from .hopfield import (
    HopfieldRun,
    StateLike,
    ZeroRule,
    run_hopfield,
)
from .opinion import _TIE_RULES, TieRule

__all__ = [
    "WeightComparison",
    "weights_from_edge_ages",
    "weights_from_growth",
    "weights_from_degree_rank",
    "run_weighted_opinion",
    "compare_weighted_unweighted",
]


@sourced(
    Origin.REPRODUCTION,
    pdf_pages="42",
    section="5.3 Dinamica de redes de opinion con pesos",
    reproduces="netmotion.dynamics.weighted_opinion.compare_weighted_unweighted",
    note=(
        "The pair of runs that make up one row of fig. 31: the same initial "
        "opinions evolved once through W and once through M."
    ),
    tags=("chapter-5", "container", "figure-31"),
)
@dataclass
class WeightComparison:
    """The two halves of the fig. 31 experiment, from one initial condition.

    Attributes
    ----------
    with_weights:
        The run coupled through the tie-age matrix :math:`W`.
    without_weights:
        The run coupled through the plain adjacency :math:`M`, from the same
        initial opinions.
    """

    with_weights: HopfieldRun
    without_weights: HopfieldRun

    @property
    def settles_sooner_with_weights(self) -> bool | None:
        """Whether :math:`W` reached a fixed point before :math:`M` did.

        ``None`` when at least one of the two runs did not settle within the
        horizon.
        """
        weighted = self.with_weights.equilibrium_time
        plain = self.without_weights.equilibrium_time
        if weighted is None or plain is None:
            return None
        return weighted < plain


@sourced(
    Origin.THESIS_2011,
    pdf_pages="41",
    section="5.3 Dinamica de redes de opinion con pesos",
    note=(
        "The thesis's own weighting rule, and the original contribution of "
        "section 5.3: W_ij = t_f - t_n for a link that appeared at t_n and "
        "survived to the final instant t_f, and W_ij = 0 where M_ij = 0. The "
        "weight is the antiquity of the relationship, so an old acquaintance "
        "influences an opinion more than a recent one."
    ),
    reading=(
        "t is counted in growth steps from the seed network; a link born at "
        "t_f has weight 0."
    ),
    tags=("chapter-5", "original-contribution", "weights"),
)
def weights_from_edge_ages(
    n_nodes: int,
    edge_birth: Mapping[tuple[int, int], float],
    *,
    final_time: float,
    name: str = "",
) -> Network:
    """The tie-age weight matrix :math:`W_{i,j} = t_f - t_n`.

    Parameters
    ----------
    n_nodes:
        Number of nodes.
    edge_birth:
        ``{(i, j): t_n}`` for every link that exists at :math:`t_f`.  Either
        orientation of a pair may be given; the matrix is symmetric.
    final_time:
        The instant :math:`t_f` at which the network is read.

    Examples
    --------
    Four nodes and four links, the last two of which were created at the final
    instant and therefore carry no weight at all:

    >>> w = weights_from_edge_ages(
    ...     4, {(0, 1): 2, (0, 2): 3, (1, 3): 4, (0, 3): 4}, final_time=4)
    >>> w.matrix.tolist()
    [[0.0, 2.0, 1.0, 0.0], [2.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    """
    n, t_final = n_nodes, final_time
    matrix = np.zeros((n, n))
    for (i, j), born in edge_birth.items():
        if i == j:
            raise ValueError("a node is not linked to itself")
        if born > t_final:
            raise ValueError(f"link ({i}, {j}) is born at {born}, after final_time={t_final}")
        matrix[i, j] = matrix[j, i] = float(t_final) - float(born)
    return Network(matrix, name=name or f"W tie-age N={n}")


@sourced(
    Origin.REPRODUCTION,
    pdf_pages="41",
    section="5.3 Dinamica de redes de opinion con pesos",
    reproduces="netmotion.dynamics.weighted_opinion.weights_from_edge_ages",
    note=(
        "Applies the tie-age rule to a network grown by "
        "netmotion.networks.barabasi_albert, where node indices are the "
        "arrival order: the m0 seed nodes exist at step 0 and node v arrives "
        "at step v - m0 + 1. A link is born when the later of its two "
        "endpoints arrives, so W_ij = t_f - max(step_i, step_j) and the whole "
        "matrix follows from the topology."
    ),
    reading=(
        "A link is born when its younger endpoint arrives, as the degree "
        "ranking of section 5.3 dates a link by its younger node."
    ),
    tags=("chapter-5", "weights"),
)
def weights_from_growth(
    network: Network,
    *,
    initial_nodes: int,
    final_time: float | None = None,
    name: str = "",
) -> Network:
    """Tie-age weights for a network grown by preferential attachment.

    Parameters
    ----------
    initial_nodes:
        Size of the seed network the growth started from; every one of those
        nodes is taken to exist at step 0.
    final_time:
        The instant the weights are read at.  Defaults to the step at which
        the last node arrived, ``network.n - m0``.

    Examples
    --------
    >>> from netmotion.networks import barabasi_albert
    >>> grown = barabasi_albert(20, links_per_new_node=2, initial_nodes=3, rng=0)
    >>> w = weights_from_growth(grown, initial_nodes=3)

    The oldest links are the ones inside the seed network, which have survived
    the whole growth:

    >>> float(w.matrix.max())
    17.0

    The links of the last node to arrive have weight zero:

    >>> float(w.matrix[19].max())
    0.0
    """
    m0, t_final = initial_nodes, final_time
    n = network.n
    if not 2 <= m0 <= n:
        raise ValueError(f"m0 must lie in 2..{n}, got {m0}")

    arrival = np.maximum(np.arange(n) - m0 + 1, 0).astype(float)
    if t_final is None:
        t_final = float(arrival.max())

    born = np.maximum.outer(arrival, arrival)
    matrix = np.where(network.adjacency != 0, float(t_final) - born, 0.0)
    np.fill_diagonal(matrix, 0.0)
    return Network(matrix, name=name or f"W growth-age N={n}")


@sourced(
    Origin.THESIS_2011,
    pdf_pages="41",
    section="5.3 Dinamica de redes de opinion con pesos",
    note=(
        "The weights the thesis uses for ER, SW and NW networks. Nodes are ranked by decreasing degree and given "
        "birth instants t = 1..N -- 'un nodo con un mayor grado k es mas "
        "antiguo' -- and a link is dated by its younger endpoint, so "
        "W_ij = N - max(t_i, t_j) on the links of the network."
    ),
    reading=(
        "Nodes of equal degree are ranked by index (tie='first') or at random "
        "(tie='random'). The weights sit on the links of the network, W_ij = 0 "
        "where M_ij = 0, as for the scale-free case."
    ),
    tags=("chapter-5", "weights", "degree-rank"),
)
def weights_from_degree_rank(
    network: Network,
    *,
    tie: TieRule = "first",
    rng: SeedLike = None,
    name: str = "",
) -> Network:
    """Tie-age weights inferred from the degree ranking.

    Examples
    --------
    On a triangle with a pendant node the degrees are ``[2, 2, 3, 1]``, so the
    birth instants by rank are ``[2, 3, 1, 4]`` and
    :math:`W_{i,j} = 4 - \\max(t_i, t_j)`:

    >>> from netmotion import Network
    >>> tri = Network.from_edges(4, [(0,1),(0,2),(1,2),(2,3)])
    >>> weights_from_degree_rank(tri).matrix.tolist()
    [[0.0, 1.0, 2.0, 0.0], [1.0, 0.0, 1.0, 0.0], [2.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]

    Node 3 has the lowest degree, so it is the youngest node and its only link
    carries weight zero:

    >>> float(weights_from_degree_rank(tri).matrix[3].max())
    0.0
    """
    if tie not in _TIE_RULES:
        raise ValueError(f"tie must be one of {_TIE_RULES}, got {tie!r}")

    n = network.n
    k = network.degrees().astype(float)
    key = k if tie == "first" else k + as_rng(rng).uniform(0.0, 0.5, size=n)

    order = np.argsort(-key, kind="stable")
    birth = np.empty(n, dtype=float)
    birth[order] = np.arange(1, n + 1, dtype=float)

    matrix = np.where(network.adjacency != 0, float(n) - np.maximum.outer(birth, birth), 0.0)
    np.fill_diagonal(matrix, 0.0)
    return Network(matrix, name=name or f"W degree-rank N={n}")


@sourced(
    Origin.THESIS_2011,
    pdf_pages="41-43",
    section="5.3 Dinamica de redes de opinion con pesos",
    note=(
        "The section 5.3 driver: the base rule of 5.1 with the weight matrix W "
        "in place of the adjacency M. Figs. 29-31 use N = 100 over 20 instants; "
        "fig. 31 starts with 52 opinions in favour and 48 against."
    ),
    reading="52 in favour and 48 against, as stated for fig. 31.",
    tags=("chapter-5", "weights", "figure-29", "figure-30", "figure-31"),
)
def run_weighted_opinion(
    weights: Network,
    states: StateLike,
    *,
    steps: int = 19,
    zero_rule: ZeroRule = "hold",
    rng: SeedLike = None,
) -> HopfieldRun:
    """Evolve opinions through the tie-age weights.

    Parameters
    ----------
    weights:
        The weight matrix :math:`W`, from :func:`weights_from_growth` or
        :func:`weights_from_degree_rank`.
    steps:
        Number of updates; the default of 19 gives the 20 instants of
        figs. 29-31.

    Examples
    --------
    A worked example, one step from
    ``[+1, -1, -1, +1]``.  The fields are :math:`[-3, 0, 1, 0]`, so node 0
    follows its heavily weighted neighbours while nodes 1 and 3, whose weighted
    fields cancel or vanish, keep their opinions:

    >>> import numpy as np
    >>> from netmotion import Network
    >>> w = Network(np.array([[0., 1., 2., 0.],
    ...                       [1., 0., 1., 0.],
    ...                       [2., 1., 0., 0.],
    ...                       [0., 0., 0., 0.]]))
    >>> run = run_weighted_opinion(w, [1, -1, -1, 1], steps=1)
    >>> run.states[1].tolist()
    [-1, -1, 1, 1]
    """
    return run_hopfield(weights, states, steps=steps, zero_rule=zero_rule, rng=rng)


@sourced(
    Origin.THESIS_2011,
    pdf_pages="42",
    section="5.3 Dinamica de redes de opinion con pesos",
    note=(
        "The fig. 31 experiment: the same initial opinions evolved through W "
        "and through M, so that the effect of the weights is isolated from the "
        "effect of the initial condition. The thesis runs it for two "
        "independent initial conditions on each of ER, SW and NW, and concludes "
        "that the weights do not bring equilibrium forward -- which of the two "
        "settles first depends on the initial states."
    ),
    reading=(
        "The tests assert the conclusion of the section: which run settles "
        "first depends on the initial states."
    ),
    tags=("chapter-5", "weights", "figure-31"),
)
def compare_weighted_unweighted(
    network: Network,
    weights: Network,
    states: StateLike,
    *,
    steps: int = 19,
    zero_rule: ZeroRule = "hold",
    rng: SeedLike = None,
) -> WeightComparison:
    """Run the same initial opinions through :math:`W` and through :math:`M`.

    Examples
    --------
    On a triangle with a pendant node the unweighted rule pulls the network to
    consensus at once, while under the tie-age weights the pendant node's link
    has weight zero and the network stays at three opinions in favour:

    >>> from netmotion import Network
    >>> tri = Network.from_edges(4, [(0,1),(0,2),(1,2),(2,3)])
    >>> both = compare_weighted_unweighted(
    ...     tri, weights_from_degree_rank(tri), [1, 1, 1, -1], steps=4)
    >>> both.without_weights.n_plus.tolist()
    [3, 4, 4, 4, 4]
    >>> both.with_weights.n_plus.tolist()
    [3, 3, 3, 3, 3]
    >>> both.settles_sooner_with_weights
    True
    """
    unweighted = Network(network.adjacency, directed=network.directed, name=network.name)
    generator = as_rng(rng)
    return WeightComparison(
        with_weights=run_hopfield(weights, states, steps=steps, zero_rule=zero_rule, rng=generator),
        without_weights=run_hopfield(
            unweighted, states, steps=steps, zero_rule=zero_rule, rng=generator
        ),
    )
