"""Vulnerability scores: node removal and minimum propagation time.

Thesis section 5.5.1, *Analisis del nodo de maxima vulnerabilidad* (PDF pp.
54-56).

Section 5.5 asks which node is the *most vulnerable* -- the **nodo central**.
According to Barabasi the central node is the one of largest degree; the
*thermal bath* method of Estrada and Hatano names another node on their
nine-node example (:func:`estrada_hatano_network`, fig. 54): the largest
degrees belong to nodes 4 and 6, and thermal bath names node 5.  The thesis
describes two further methods whose answer coincides with thermal bath.

**Node removal** (section 5.5.1.1, PDF pp. 54-55).  Compute every degree; find
the minimum degree and delete every node that has it; repeat on what is left
until a single node remains.  On fig. 54 the first round strips the six leaves
and leaves ``{4, 5, 6}``, the second strips 4 and 6, and node 5 remains.  The
thesis relates the method to k-shell decomposition [35].

**Minimum propagation time** (section 5.5.1.2, PDF p. 55).  The infection of
section 5.4.1 is started at each node *m* in turn and run until the whole
network is infected; the number of iterations it needs, :math:`t_m`, is the
node's score, and the central node is the one that infects the network
fastest.  On fig. 54 nodes 1, 2, 3, 7, 8, 9 need four iterations, nodes 4 and 6
need three, and node 5 needs two.  The dynamics are those of
:func:`netmotion.propagation.infection.propagation_time`.

A node at distance *d* from the seed is infected at the *d*-th iteration, so
:math:`t_m` equals the eccentricity of *m*; the tests use
:mod:`netmotion.metrics.distance` to check the scores on other networks.

:func:`rank_nodes` gives the criteria of section 5.5.1 -- node removal, minimum
time, and the degree :math:`K_m(n)` of order *n* of section 5.5.1.3 -- a common
shape, :class:`NodeRanking`.

Nodes are indexed from 0 here, so node *k* of the thesis is index *k-1*.  On a
network the infection cannot cover, :func:`min_time_scores` reports ``inf``.
When a round of node removal would delete every remaining node at once,
:func:`peel_by_minimum_degree` stops and reports ``terminated="emptied"``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..graph import Network
from ..propagation.infection import propagation_time
from ..provenance import Origin, sourced
from .higher_order_degree import k_order

__all__ = [
    "estrada_hatano_network",
    "PeelingResult",
    "peel_by_minimum_degree",
    "node_removal_scores",
    "central_node_by_removal",
    "min_time_scores",
    "central_nodes_by_min_time",
    "NodeRanking",
    "Criterion",
    "CENTRALITY_CRITERIA",
    "degree_order_n_criterion",
    "rank_nodes",
    "compare_centrality_criteria",
]


@sourced(
    Origin.THESIS_2011,
    pdf_pages="54",
    section="5.5.1 Analisis del nodo de maxima vulnerabilidad",
    note=(
        "The nine-node network of fig. 54, the example of Estrada and Hatano "
        "[34]. A tree: hub 4 carries leaves 1, 2, 3, hub 6 carries leaves 7, 8, "
        "9, and node 5 joins them. Thesis node k is index k-1 here."
    ),
    reading=(
        "The eight links are read off fig. 54: 'existen un total de 8 ramas' "
        "(PDF p. 56), and nodes 6 and 4 have the largest degree (PDF p. 54)."
    ),
    tags=("chapter-5", "fixture"),
)
def estrada_hatano_network() -> Network:
    """The nine-node example network of fig. 54 (PDF p. 54).

    Examples
    --------
    Nine nodes and the eight branches the thesis counts on PDF p. 56:

    >>> g = estrada_hatano_network()
    >>> g.n, g.n_edges
    (9, 8)

    The degrees, in thesis numbering, are 1, 1, 1, 4, 2, 4, 1, 1, 1 -- which is
    what makes nodes 4 and 6 the answer by Barabasi's criterion:

    >>> g.degrees().tolist()
    [1, 1, 1, 4, 2, 4, 1, 1, 1]
    """
    edges = [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (5, 8)]
    return Network.from_edges(9, edges, name="Estrada-Hatano (fig. 54)")


@sourced(
    Origin.THESIS_2011,
    pdf_pages="54-55",
    section="5.5.1.1 Metodo por eliminacion de nodos",
    note=(
        "The 'descomposicion de la red en capas' the method produces: layers[r] "
        "holds the batch removed in round r, outermost first, and survivors is "
        "what the peeling stopped at."
    ),
)
@dataclass(frozen=True, eq=False)
class PeelingResult:
    """The layer decomposition produced by :func:`peel_by_minimum_degree`.

    Attributes
    ----------
    layers:
        One array of node indices per round, outermost round first.  Each is
        the complete set of nodes attaining the minimum degree of the subgraph
        that survived the previous rounds.
    survivors:
        What the peeling stopped at: one node, several when the next round
        would have removed every remaining node, none when the network was
        empty to begin with.
    terminated:
        ``"single_node"``
            ``survivors`` has length 1 and that node is the central node.
        ``"emptied"``
            The next round would have deleted every remaining node at once; the
            peeling stopped before it and ``survivors`` holds those nodes.
        ``"empty"``
            The network had no nodes.
    """

    layers: tuple[np.ndarray, ...]
    survivors: np.ndarray
    terminated: str

    @property
    def n_rounds(self) -> int:
        """Number of removal rounds performed."""
        return len(self.layers)

    def as_lists(self) -> list[list[int]]:
        """The layers as plain sorted lists, for printing and for tests."""
        return [sorted(int(v) for v in layer) for layer in self.layers]


@sourced(
    Origin.THESIS_2011,
    pdf_pages="54-55",
    section="5.5.1.1 Metodo por eliminacion de nodos",
    note=(
        "The three printed steps, in order: (1) compute the degree of every "
        "node of the current subgraph, (2) find the minimum degree k and delete "
        "every node having it, in one simultaneous batch, (3) repeat on the "
        "induced subgraph until a single node remains. Degrees are recomputed "
        "between batches."
    ),
    reading=(
        "The three steps as described are implemented. A round that would "
        "remove every remaining node is not performed: the peeling stops and "
        "reports terminated='emptied'."
    ),
    tags=("chapter-5", "printed-steps"),
)
def peel_by_minimum_degree(network: Network) -> PeelingResult:
    """Strip the minimum-degree nodes repeatedly, as section 5.5.1.1 prescribes.

    Examples
    --------
    The worked example of PDF pp. 54-55, in the thesis's own numbering.  The
    first round removes the six leaves and leaves *"la red restante formada por
    los nodos 4, 5, 6"*; the second removes 4 and 6; node 5 survives.

    >>> result = peel_by_minimum_degree(estrada_hatano_network())
    >>> [[v + 1 for v in layer] for layer in result.as_lists()]
    [[1, 2, 3, 7, 8, 9], [4, 6]]
    >>> (result.survivors + 1).tolist(), result.terminated
    ([5], 'single_node')

    On a triangle the first round would delete all three nodes at once:

    >>> from netmotion import Network
    >>> triangle = Network.from_edges(3, [(0, 1), (1, 2), (2, 0)])
    >>> peel_by_minimum_degree(triangle).terminated
    'emptied'
    """
    adjacency = network.adjacency
    remaining = np.arange(network.n)
    layers: list[np.ndarray] = []

    while remaining.size > 1:
        sub = adjacency[np.ix_(remaining, remaining)]
        degrees = sub.sum(axis=1)
        removed = degrees == degrees.min()
        if removed.all():
            return PeelingResult(tuple(layers), remaining, "emptied")
        layers.append(remaining[removed])
        remaining = remaining[~removed]

    terminated = "single_node" if remaining.size == 1 else "empty"
    return PeelingResult(tuple(layers), remaining, terminated)


@sourced(
    Origin.REPRODUCTION,
    pdf_pages="54-55",
    section="5.5.1.1 Metodo por eliminacion de nodos",
    reproduces="netmotion.metrics.propagation_centrality.peel_by_minimum_degree",
    note=(
        "Per-node form of the layer decomposition, so that the node-removal "
        "method can be used as a ranking criterion alongside the other two. A "
        "node removed in round r scores r, and the survivors score n_rounds, so "
        "a higher score means a more central node, following the "
        "'descomposicion de la red en capas'."
    ),
    tags=("chapter-5", "ranking"),
)
def node_removal_scores(network: Network) -> np.ndarray:
    """Peeling depth of every node; higher means removed later, so more central.

    Examples
    --------
    On the fig. 54 network the six leaves are stripped in round 0, the two hubs
    in round 1, and node 5 (index 4) survives both:

    >>> node_removal_scores(estrada_hatano_network()).tolist()
    [0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0]
    """
    result = peel_by_minimum_degree(network)
    scores = np.full(network.n, float(result.n_rounds))
    for round_index, layer in enumerate(result.layers):
        scores[layer] = float(round_index)
    return scores


@sourced(
    Origin.THESIS_2011,
    pdf_pages="54-55",
    section="5.5.1.1 Metodo por eliminacion de nodos",
    note=(
        "The answer the method is built to give: 'nos queda por ultimo el nodo "
        "5 que coincide con el nodo mas vulnerable o nodo central obtenido por "
        "thermal bath'."
    ),
    reading=(
        "Defined when the peeling reaches a single node; otherwise this raises, "
        "and peel_by_minimum_degree exposes the remaining nodes."
    ),
)
def central_node_by_removal(network: Network) -> int:
    """The node that survives the peeling: the central node of section 5.5.1.1.

    Raises
    ------
    ValueError
        If the peeling never reaches a single node, which happens whenever the
        surviving subgraph becomes regular.  See :class:`PeelingResult`.

    Examples
    --------
    Node 5 of fig. 54, index 4 here:

    >>> central_node_by_removal(estrada_hatano_network())
    4
    """
    result = peel_by_minimum_degree(network)
    if result.terminated != "single_node":
        survivors = sorted(int(v) for v in result.survivors)
        raise ValueError(
            f"the peeling terminated as {result.terminated!r} with survivors "
            f"{survivors}: the minimum-degree round would have removed every "
            "remaining node at once. Use peel_by_minimum_degree to inspect the "
            "layers."
        )
    return int(result.survivors[0])


@sourced(
    Origin.THESIS_2011,
    pdf_pages="55",
    section="5.5.1.2 El metodo de propagacion en tiempo minimo",
    reproduces="netmotion.propagation.infection.propagation_time",
    note=(
        "t_n for every node: the infection of section 5.4.1 is started at each "
        "node in turn and the iterations needed to cover the network are "
        "counted. The dynamics are not re-implemented here; this drives "
        "netmotion.propagation.infection.propagation_time. t_n is the "
        "eccentricity of n, so the minimum of these scores is the graph radius."
    ),
    reading=(
        "t_n counts iterations, as in the values 4/3/2 of fig. 54. A network "
        "the infection cannot cover scores inf."
    ),
    tags=("chapter-5", "counting-convention"),
)
def min_time_scores(network: Network, **kwargs) -> np.ndarray:
    """:math:`t_m` for every node; lower means faster, so more central.

    Parameters
    ----------
    **kwargs:
        Passed to :func:`~netmotion.propagation.infection.propagate`.  Immune
        nodes cannot seed an infection, so they score ``inf``.

    Returns
    -------
    numpy.ndarray
        Float array of length *N*.  A node whose infection never covers the
        network scores ``inf``, which is also its eccentricity.

    Examples
    --------
    PDF p. 55: *"los nodos 1, 2, 3, 7, 8, 9 necesitan 4 iteraciones para
    propagarse por toda la red, los nodos 4 y 6 necesitan 3 iteraciones y el
    nodo 5 necesita de 2 iteraciones"*.

    >>> min_time_scores(estrada_hatano_network()).tolist()
    [4.0, 4.0, 4.0, 3.0, 2.0, 3.0, 4.0, 4.0, 4.0]

    The scores are eccentricities, so the minimum is the graph radius:

    >>> from netmotion import Network
    >>> path = Network.from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    >>> min_time_scores(path).tolist()
    [4.0, 3.0, 2.0, 3.0, 4.0]
    """
    immune_mask = np.zeros(network.n, dtype=bool)
    immune = kwargs.get("immune")
    if immune is not None:
        immune_mask[np.asarray(immune, dtype=int)] = True

    scores = np.full(network.n, np.inf)
    for seed in range(network.n):
        if immune_mask[seed]:
            continue
        scores[seed] = propagation_time(network, seed, **kwargs)
    return scores


@sourced(
    Origin.THESIS_2011,
    pdf_pages="55",
    section="5.5.1.2 El metodo de propagacion en tiempo minimo",
    note=(
        "'el tiempo minimo de estos se obtiene cuando la propagacion se origina "
        "en el nodo central'. Equivalently, by the identity t_n = ecc(n), the "
        "graph centre."
    ),
    reading="When several nodes attain the minimum, all of them are returned.",
    tags=("chapter-5", "ties"),
)
def central_nodes_by_min_time(network: Network, **kwargs) -> np.ndarray:
    """Every node attaining the minimum :math:`t_m`: the graph centre.

    Raises
    ------
    ValueError
        If no node can cover the network, so that no minimum exists.

    Examples
    --------
    The thesis's answer for fig. 54 is node 5, which is index 4:

    >>> (central_nodes_by_min_time(estrada_hatano_network()) + 1).tolist()
    [5]

    A path of four nodes has two centres, and both are returned:

    >>> from netmotion import Network
    >>> path = Network.from_edges(4, [(0, 1), (1, 2), (2, 3)])
    >>> central_nodes_by_min_time(path).tolist()
    [1, 2]
    """
    scores = min_time_scores(network, **kwargs)
    if not np.isfinite(scores).any():
        raise ValueError(
            "no node's infection covers the whole network, so section 5.5.1.2 "
            "defines no central node; the network is disconnected or every "
            "node is immune"
        )
    return np.flatnonzero(scores == scores.min())


@sourced(
    Origin.REPRODUCTION,
    pdf_pages="54-56",
    section="5.5.1 Analisis del nodo de maxima vulnerabilidad",
    note=(
        "Common shape for the vulnerability criteria of section 5.5.1 so they "
        "can be compared, and so section 5.5.2 can immunise in the order one of "
        "them dictates."
    ),
    tags=("chapter-5", "ranking"),
)
@dataclass(frozen=True, eq=False)
class NodeRanking:
    """Nodes scored and ordered by one criterion of section 5.5.1.

    Attributes
    ----------
    criterion:
        Name of the criterion that produced the scores.
    scores:
        The raw score of each node, in node order.  Their scale is the
        criterion's own and is not comparable across criteria; only the
        ordering is.
    higher_is_more_central:
        Which end of the score range is the central node.  ``False`` for
        minimum propagation time, ``True`` for degree and for peeling depth.
    """

    criterion: str
    scores: np.ndarray
    higher_is_more_central: bool

    @property
    def sort_key(self) -> np.ndarray:
        """Scores flipped, if need be, so that smaller always means more central."""
        return -self.scores if self.higher_is_more_central else self.scores

    @property
    def order(self) -> np.ndarray:
        """Node indices, most central first; ties broken by node index."""
        key = self.sort_key
        return np.lexsort((np.arange(key.size), key))

    @property
    def most_central(self) -> np.ndarray:
        """Every node tying for most central -- the criterion's answer."""
        key = self.sort_key
        if key.size == 0:
            return np.array([], dtype=int)
        return np.flatnonzero(key == key.min())

    def top(self, k: int) -> np.ndarray:
        """The *k* most central nodes: the immunisation order of section 5.5.2."""
        return self.order[:k]


@sourced(
    Origin.REPRODUCTION,
    pdf_pages="54-56",
    section="5.5.1 Analisis del nodo de maxima vulnerabilidad",
    note=(
        "One of the three criteria of section 5.5.1, packaged so that they are "
        "interchangeable: a scoring function plus the direction that means "
        "'more central'. The registered criteria are in CENTRALITY_CRITERIA."
    ),
    tags=("chapter-5", "ranking"),
)
@dataclass(frozen=True)
class Criterion:
    """A named way of scoring how central, hence how vulnerable, a node is.

    Attributes
    ----------
    name:
        Key under which the criterion appears in :data:`CENTRALITY_CRITERIA`.
    score:
        Callable mapping a :class:`~netmotion.graph.Network` to one score per
        node.
    higher_is_more_central:
        See :class:`NodeRanking`.
    section:
        The thesis section the criterion comes from.
    """

    name: str
    score: Callable[[Network], np.ndarray]
    higher_is_more_central: bool
    section: str
    description: str = ""

    def rank(self, network: Network) -> NodeRanking:
        """Score every node of *network* and wrap the result."""
        scores = np.asarray(self.score(network), dtype=float)
        if scores.shape != (network.n,):
            raise ValueError(
                f"criterion {self.name!r} returned {scores.shape} scores for a "
                f"network of {network.n} nodes"
            )
        return NodeRanking(self.name, scores, self.higher_is_more_central)


def _degree_scores(network: Network) -> np.ndarray:
    """Barabasi's criterion (PDF p. 54): the degree, K_m(1) in the thesis's terms."""
    return network.degrees().astype(float)


#: The criteria of section 5.5.1, keyed by name.  ``"degree_order_n"`` is not
#: registered here because it needs the order *n* as a parameter; build it with
#: :func:`degree_order_n_criterion`.
CENTRALITY_CRITERIA: dict[str, Criterion] = {
    "node_removal": Criterion(
        name="node_removal",
        score=node_removal_scores,
        higher_is_more_central=True,
        section="5.5.1.1 Metodo por eliminacion de nodos",
        description="Round at which the minimum-degree peeling removed the node.",
    ),
    "min_time": Criterion(
        name="min_time",
        score=min_time_scores,
        higher_is_more_central=False,
        section="5.5.1.2 El metodo de propagacion en tiempo minimo",
        description="Iterations an infection seeded at the node needs to cover the network.",
    ),
    "degree": Criterion(
        name="degree",
        score=_degree_scores,
        higher_is_more_central=True,
        section="5.5.1 Analisis del nodo de maxima vulnerabilidad",
        description="Barabasi's criterion: the degree k of the node.",
    ),
}


@sourced(
    Origin.REPRODUCTION,
    pdf_pages="55-56",
    section="5.5.1.3 El metodo del grado k de orden n del nodo m",
    reproduces="netmotion.metrics.higher_order_degree.k_order",
    note=(
        "The third criterion of section 5.5.1, the one section 5.5.2 immunises "
        "by, exposed through the same interface as the other two. K_m(n) itself belongs to section 5.5.1.3 and is computed by "
        "netmotion.metrics.higher_order_degree.k_order; this only wraps it, and "
        "reproduces that module's k_order_ranking node for node."
    ),
)
def degree_order_n_criterion(order: int, *, rule: str = "incident") -> Criterion:
    """The degree :math:`K_m(n)` of order :math:`n` (*order*, section 5.5.1.3) as a criterion.

    At :math:`n = 1` this is the ordinary degree, which is Barabasi's criterion;
    larger *order* widens the horizon the score sees.  ``rule`` is passed to
    :func:`~netmotion.metrics.higher_order_degree.k_order`, where the reading it
    selects is documented.

    Examples
    --------
    The ranking changes with the order: at order 1 the fig. 54 network puts the
    two hubs first, at order 2 node 5 comes first, as with the other two methods
    (PDF p. 56).

    >>> g = estrada_hatano_network()
    >>> (rank_nodes(g, degree_order_n_criterion(1)).most_central + 1).tolist()
    [4, 6]
    >>> (rank_nodes(g, degree_order_n_criterion(2)).most_central + 1).tolist()
    [5]
    """
    n = order
    if n < 1:
        raise ValueError(f"the order n of K_m(n) starts at 1, got {n}")

    def score(network: Network) -> np.ndarray:
        return np.array([float(k_order(network, m, n, rule=rule)) for m in range(network.n)])

    return Criterion(
        name=f"degree_order_{n}",
        score=score,
        higher_is_more_central=True,
        section="5.5.1.3 El metodo del grado k de orden n del nodo m",
        description=f"K_m({n}), the degree k of order {n} of the node.",
    )


@sourced(
    Origin.REPRODUCTION,
    pdf_pages="54-56",
    section="5.5.1 Analisis del nodo de maxima vulnerabilidad",
    note=(
        "Entry point for the comparison section 5.5.1 sets up and section 5.5.2 "
        "needs: score every node by one named criterion and order them from "
        "most to least central."
    ),
)
def rank_nodes(network: Network, criterion: str | Criterion = "min_time") -> NodeRanking:
    """Rank the nodes of *network* by one criterion of section 5.5.1.

    Parameters
    ----------
    criterion:
        A key of :data:`CENTRALITY_CRITERIA` -- ``"node_removal"``, ``"min_time"`` or
        ``"degree"`` -- or a :class:`Criterion` built elsewhere, such as
        :func:`degree_order_n_criterion`.

    Examples
    --------
    On the network of fig. 54 the two methods of section 5.5.1 name node 5, and
    the degree names 4 and 6:

    >>> g = estrada_hatano_network()
    >>> (rank_nodes(g, "min_time").most_central + 1).tolist()
    [5]
    >>> (rank_nodes(g, "node_removal").most_central + 1).tolist()
    [5]
    >>> (rank_nodes(g, "degree").most_central + 1).tolist()
    [4, 6]

    The full order is what section 5.5.2 immunises along:

    >>> (rank_nodes(g, "min_time").top(3) + 1).tolist()
    [5, 4, 6]
    """
    if isinstance(criterion, str):
        try:
            criterion = CENTRALITY_CRITERIA[criterion]
        except KeyError:
            raise KeyError(
                f"unknown criterion {criterion!r}; the criteria of section "
                f"5.5.1 are {sorted(CENTRALITY_CRITERIA)}, or pass a Criterion (see "
                "degree_order_n_criterion for the order-n degree)"
            ) from None
    return criterion.rank(network)


@sourced(
    Origin.REPRODUCTION,
    pdf_pages="54-56",
    section="5.5.1 Analisis del nodo de maxima vulnerabilidad",
    reproduces="netmotion.metrics.propagation_centrality.rank_nodes",
    note=(
        "Runs several criteria over the same network, which is what section "
        "5.5.1 does by hand on fig. 54 when it sets Barabasi's answer against "
        "thermal bath's and against its own two methods."
    ),
)
def compare_centrality_criteria(
    network: Network,
    criteria: Sequence[str | Criterion] | None = None,
) -> dict[str, NodeRanking]:
    """Rank the same network by several criteria at once.

    Examples
    --------
    PDF p. 54, in one call: the two methods name node 5, as thermal bath does,
    and the degree names 4 and 6.

    >>> rankings = compare_centrality_criteria(estrada_hatano_network())
    >>> {name: (r.most_central + 1).tolist() for name, r in rankings.items()}
    {'node_removal': [5], 'min_time': [5], 'degree': [4, 6]}
    """
    chosen = list(CENTRALITY_CRITERIA) if criteria is None else list(criteria)
    rankings = {}
    for item in chosen:
        ranking = rank_nodes(network, item)
        rankings[ranking.criterion] = ranking
    return rankings
