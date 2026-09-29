"""The degree *k* of order *n*, and the centrality the thesis builds from it.

Thesis section 5.5.1.3, *El metodo del grado k de orden n del nodo m*
(PDF pp. 55-56, printed 52-53), applied in section 5.5.2 (PDF pp. 56-59).

Section 5.5 asks which node of a network is the most important one, and tries
three answers in turn: removing nodes and watching the network fall apart
(5.5.1.1), timing an infection started at each node (5.5.1.2), and finally the
measure this module implements.  The complaint that motivates it is stated
plainly on PDF p. 55: the classical degree of Barabasi *"no coincide para la red
anterior con el nodo central obtenido por el metodo thermal bath"*, so the
thesis looks for a degree that *"encerrase las caracteristicas de conectividad
dentro de la red mas alla de los nodos a los que este unidos"*.

The construction is to widen the horizon one hop at a time.  Writing
:math:`B_r(m)` for the set of nodes within *r* hops of *m*,

.. math:: K_m(n) = \\bigl|\\{\\, \\{i,j\\} \\in E : \\min(d(m,i), d(m,j)) \\le n-1 \\,\\}\\bigr|

is the number of **distinct links** with at least one end inside
:math:`B_{n-1}(m)`.  At :math:`n = 1` the ball is the node itself and
:math:`K_m(1) = k_m`, the ordinary degree -- the thesis says so explicitly.  At
:math:`n = 2` a node also counts the links of its neighbours, so it knows not
only who its friends are but roughly what kind of company they keep; the thesis
makes exactly that analogy with friends of friends.  As *n* grows the quantity
saturates at the number of links in the node's connected component, and the
node that gets there first is declared the central, most vulnerable node.

Links are counted once
----------------------
The worked example (Fig. 54, the nine-node tree
``1-4, 2-4, 3-4, 4-5, 5-6, 6-7, 6-8, 6-9``) gives

.. math:: K_1(2) = 4, \\qquad K_4(2) = 5, \\qquad K_5(2) = 8

and Fig. 57 draws the link joining the red source to each blue neighbour once,
with the remaining links of those neighbours in green.  Each link is counted
once, however many of its ends lie in the ball; this module counts links.

How it is computed
------------------
PDF p. 56 describes the procedure: give every node the state
:math:`S_i(t) = +1`, set :math:`S_m(1) = -1`, and run the infection rule of
section 5.4.1 until :math:`t = n`, counting what has been reached.  That dynamic
reaches, at instant *t*, the nodes of :math:`B_{t-1}(m)`, so a single
breadth-first search from *m*, a minimum over each link's two endpoint
distances, and a cumulative histogram give every order at once.  It returns
the same integers as the state iteration, and the test file checks that against
:func:`netmotion.propagation.propagate` on random networks.

What the measure was used for
-----------------------------
Section 5.5.2 immunises networks by :func:`k_order_ranking`: sort the nodes
by :math:`K_m(n)` for a fixed *n*, freeze the top *q* at :math:`S = +1`, and see
how far an infection still spreads.  Sweeping *n* from 1 to 8 the thesis
concludes (PDF pp. 57-62) that :math:`K(1)` -- the plain degree -- is the best
criterion for the disordered topologies ER, WS, NW and SF at
:math:`N = 50, 75, 100`, while for the ordered networks with a defined structure drawn
in Fig. 68 the best criterion is :math:`K(3)` (PDF p. 63).
The general conclusion of PDF p. 78 is that the right immunisation criterion
*"dependera de lo ordenada que sea la estructura de la red"*.  This module is
the measure.

On the network of Fig. 54 node 5 has a smaller degree than node 4 and a larger
degree of order 2, the point section 5.5.1.3 makes.
"""

from __future__ import annotations

import numpy as np

from ..graph import Network
from ..provenance import Origin, sourced
from .distance import shortest_path_lengths

__all__ = [
    "k_order",
    "k_order_profile",
    "k_order_all",
    "saturation_order",
    "central_node",
    "k_order_ranking",
]

_RULES = ("incident", "induced")


def _edge_orders(network: Network, m: int, rule: str = "incident") -> np.ndarray:
    """The smallest order at which each link enters :math:`K_m(n)`.

    One entry per undirected link, in ``i < j`` order, so a link is represented
    once -- that single-visit enumeration *is* the de-duplication the definition
    calls for.  Unreachable links get ``np.inf``.
    """
    if network.directed:
        raise ValueError(
            "the degree of order n is defined for undirected networks"
        )
    if rule not in _RULES:
        raise ValueError(f"rule must be one of {_RULES}, got {rule!r}")
    if not 0 <= m < network.n:
        raise IndexError(f"node {m} outside 0..{network.n - 1}")

    dist = shortest_path_lengths(network, m)
    rows, cols = np.nonzero(np.triu(network.adjacency, 1))
    if rule == "incident":
        # A link counts from the order after the nearer of its two ends is
        # reached, since the ball at order n has radius n - 1.
        return np.minimum(dist[rows], dist[cols]) + 1.0
    return np.maximum(np.maximum(dist[rows], dist[cols]), 1.0)


@sourced(
    Origin.THESIS_2011,
    pdf_pages="55-56",
    section="5.5.1.3 El metodo del grado k de orden n del nodo m",
    note=(
        "K_m(n): the number of distinct links with at least one end within "
        "n-1 hops of m. K_m(1) is the classical degree. Verified against the "
        "three values printed on PDF p. 56 for the Fig. 54 network: "
        "K_1(2) = 4, K_4(2) = 5, K_5(2) = 8."
    ),
    reading=(
        "(1) Each link is counted once, as in the printed values and fig. 57. "
        "(2) A link counts when at least one of its ends is within n-1 hops "
        "(rule='incident'), following 'las ramas de los nodos a los cuales el "
        "nodo esta unido'. rule='induced' counts the links of the subgraph "
        "induced on B_n(m) instead."
    ),
    tags=("original-behaviour", "chapter-5", "convention"),
)
def k_order(network: Network, node: int, order: int, *, rule: str = "incident") -> int:
    """:math:`K_m(n)`, the degree of order :math:`n` (*order*) of node :math:`m` (*node*).

    Parameters
    ----------
    network:
        An undirected network.  Directed networks are rejected: the thesis's
        adjacency test ``exists j / M_ij != 0`` has no meaning for them.
    node:
        The node whose horizon is being measured.
    order:
        The order, an integer :math:`\\ge 1`.  Order 1 is the classical degree.
    rule:
        ``"incident"``
            Every link with at least one end within :math:`n-1` hops of *node*
            (the default).
        ``"induced"``
            Every link of the subgraph induced on :math:`B_n(m)`, which also
            counts links joining two nodes both exactly *order* hops away.

    Returns
    -------
    int
        The number of distinct links, never a sum of degrees.

    Examples
    --------
    The network of Fig. 54 (PDF p. 54), the double star the thesis works
    through.  Nodes are 0-indexed here and 1-indexed in the thesis, so its node
    *i* is index ``i - 1``:

    >>> from netmotion import Network
    >>> fig54 = Network.from_edges(
    ...     9, [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (5, 8)]
    ... )
    >>> k_order(fig54, 0, 2), k_order(fig54, 3, 2), k_order(fig54, 4, 2)
    (4, 5, 8)

    Order 1 is the ordinary degree, so node 5 of the thesis starts behind node
    4 and overtakes it at order 2 -- the inversion section 5.5.1.3 exists to
    show:

    >>> k_order(fig54, 4, 1), k_order(fig54, 3, 1)
    (2, 4)
    """
    m, n = node, order
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool):
        raise TypeError(f"the order n must be an integer, got {type(n).__name__}")
    if n < 1:
        raise ValueError(f"the order n starts at 1 (K_m(1) is the classical degree), got {n}")
    orders = _edge_orders(network, m, rule)
    return int((orders <= n).sum())


@sourced(
    Origin.THESIS_2011,
    pdf_pages="56",
    section="5.5.1.3 El metodo del grado k de orden n del nodo m",
    note=(
        "The vector [K_m(1), ..., K_m(n_max)]. PDF p. 56 describes obtaining "
        "'los grados k de ordenes 1, 2 ..., n-1, n del nodo m' from a single "
        "run of the propagation, which is what this computes: one breadth-first "
        "sweep and a cumulative histogram over the links' activation radii."
    ),
)
def k_order_profile(network: Network, node: int, max_order: int, *, rule: str = "incident") -> list[int]:
    """:math:`[K_m(1), \\ldots, K_m(n_{\\max})]`, from one sweep of the network.

    The profile is non-decreasing and flat once it reaches the number of links
    in *node*'s connected component.

    Examples
    --------
    The values for the Fig. 54 network (PDF p. 54), derived from the definition.  Node 1 of the thesis
    learns of the eight links only at order 4:

    >>> from netmotion import Network
    >>> fig54 = Network.from_edges(
    ...     9, [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (5, 8)]
    ... )
    >>> k_order_profile(fig54, 0, 5)
    [1, 4, 5, 8, 8]

    while node 5, sitting on the bridge, knows all of them at order 2:

    >>> k_order_profile(fig54, 4, 5)
    [2, 8, 8, 8, 8]

    An isolated node never learns anything:

    >>> lonely = Network.from_edges(4, [(1, 2), (2, 3)])
    >>> k_order_profile(lonely, 0, 4)
    [0, 0, 0, 0]
    """
    m, n_max = node, max_order
    if not isinstance(n_max, (int, np.integer)) or isinstance(n_max, bool):
        raise TypeError(f"n_max must be an integer, got {type(n_max).__name__}")
    if n_max < 1:
        raise ValueError(f"the order n starts at 1, so n_max must be at least 1, got {n_max}")

    orders = _edge_orders(network, m, rule)
    # Links beyond n_max are dropped, not clipped into the last bucket: the
    # profile is a prefix of the full one, so extending n_max only appends.
    reached = orders[orders <= n_max].astype(int)
    counts = np.bincount(reached, minlength=n_max + 1)
    return np.cumsum(counts[1:]).tolist()


@sourced(
    Origin.THESIS_2011,
    pdf_pages="56",
    section="5.5.1.3 El metodo del grado k de orden n del nodo m",
    note=(
        "K_m(n) for every node and every order up to n_max. The thesis computes "
        "exactly this table before ranking the nodes: 'se calcularan los grados "
        "k de orden desde 1 hasta n para cada nodo de la red' (PDF p. 56)."
    ),
)
def k_order_all(network: Network, max_order: int, *, rule: str = "incident") -> np.ndarray:
    """The ``(N, n_max)`` table with ``table[m, n - 1]`` equal to :math:`K_m(n)`.

    Examples
    --------
    The first two columns for the Fig. 54 network, derived from the definition;
    the thesis prints K_1(2), K_4(2) and K_5(2) (PDF p. 56):

    >>> from netmotion import Network
    >>> fig54 = Network.from_edges(
    ...     9, [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (5, 8)]
    ... )
    >>> k_order_all(fig54, 2).tolist()
    [[1, 4], [1, 4], [1, 4], [4, 5], [2, 8], [4, 5], [1, 4], [1, 4], [1, 4]]

    Column 1 is the degree, as the definition requires:

    >>> bool((k_order_all(fig54, 2)[:, 0] == fig54.degrees()).all())
    True
    """
    n_max = max_order
    rows = [k_order_profile(network, m, n_max, rule=rule) for m in range(network.n)]
    if not rows:
        return np.zeros((0, n_max), dtype=int)
    return np.asarray(rows, dtype=int)


@sourced(
    Origin.THESIS_2011,
    pdf_pages="56",
    section="5.5.1.3 El metodo del grado k de orden n del nodo m",
    note=(
        "The smallest n at which m 'adquiere el conocimiento de la existencia "
        "de las 8 ramas', i.e. K_m(n) = |E|. On the Fig. 54 network this is 2 "
        "for node 5, 3 for nodes 4 and 6 and 4 for the six leaves, reproducing "
        "the propagation times the thesis reports on PDF p. 55."
    ),
    reading=(
        "On a disconnected network K_m(n) saturates at the links of m's own "
        "component and never reaches |E|; np.inf is returned."
    ),
    tags=("chapter-5", "disconnected"),
)
def saturation_order(network: Network, node: int, *, rule: str = "incident") -> float:
    """The smallest *n* with :math:`K_m(n) = |E|`, or ``inf`` if there is none.

    This is the thesis's measure of how quickly a node acquires complete
    knowledge of the network.  It is closely related to, but not identical
    with, the eccentricity of *node*: a link whose two ends are both at the
    maximum distance is only learned one order later, so on a graph with an odd
    cycle the saturation order can exceed the eccentricity by one.  On a tree,
    such as the network of Fig. 54, the two coincide.

    Examples
    --------
    Node 5 of Fig. 54 reaches all eight links at order 2, before any other node
    (PDF p. 56):

    >>> from netmotion import Network
    >>> fig54 = Network.from_edges(
    ...     9, [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (5, 8)]
    ... )
    >>> [saturation_order(fig54, m) for m in (4, 3, 5, 0)]
    [2.0, 3.0, 3.0, 4.0]

    A node in a network it cannot cross never gets there:

    >>> split = Network.from_edges(4, [(0, 1), (2, 3)])
    >>> saturation_order(split, 0)
    inf
    """
    m = node
    orders = _edge_orders(network, m, rule)
    if int(np.isfinite(orders).sum()) < network.n_edges:
        return float("inf")
    return float(orders.max()) if orders.size else 1.0


@sourced(
    Origin.THESIS_2011,
    pdf_pages="56",
    section="5.5.1.3 El metodo del grado k de orden n del nodo m",
    note=(
        "'el nodo que adquiera el conocimiento de la red para un K(n) con n "
        "menor sera el nodo que transmitira informacion al resto de la red de "
        "forma mas rapida'. On the Fig. 54 network this returns node 5 "
        "(index 4), agreeing with the thermal-bath result of Estrada and "
        "Hatano and with sections 5.5.1.1 and 5.5.1.2."
    ),
    reading="Of two nodes that saturate at the same order, the lower index is returned.",
    tags=("chapter-5", "tie-break"),
)
def central_node(network: Network, *, rule: str = "incident") -> int:
    """The node that first reaches :math:`K_m(n) = |E|`.

    Raises
    ------
    ValueError
        If the network is empty, or disconnected so that no node can ever see
        every link.  See :func:`saturation_order`.

    Examples
    --------
    >>> from netmotion import Network
    >>> fig54 = Network.from_edges(
    ...     9, [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (5, 8)]
    ... )
    >>> central_node(fig54)          # thesis node 5
    4

    On a path the centre wins, as it does for every distance-based centrality:

    >>> central_node(Network.from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)]))
    2
    """
    if network.n == 0:
        raise ValueError("an empty network has no central node")
    orders = np.array([saturation_order(network, m, rule=rule) for m in range(network.n)])
    if not np.isfinite(orders).any():
        raise ValueError(
            "no node ever reaches K_m(n) = |E|: the network is disconnected"
        )
    return int(np.argmin(orders))


@sourced(
    Origin.THESIS_2011,
    pdf_pages="56-57",
    section="5.5.2 Inmunizacion frente a infeccion",
    note=(
        "'ordenar los nodos de mayor a menor grado k de orden n'. The first q "
        "nodes of this ordering are the q nodes the thesis immunises before "
        "re-running the infection, for each criterion K(1) ... K(8)."
    ),
    reading="Nodes with equal K_m(n) are ordered by index.",
    tags=("chapter-5", "tie-break"),
)
def k_order_ranking(network: Network, order: int, *, rule: str = "incident") -> list[int]:
    """Node indices ordered by decreasing :math:`K_m(n)`, ties by index.

    Examples
    --------
    The ranking depends on the order.  Under :math:`K(1)` the Fig. 54 network puts the two hubs
    first; under :math:`K(2)` the bridge node 5 takes the lead:

    >>> from netmotion import Network
    >>> fig54 = Network.from_edges(
    ...     9, [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (5, 8)]
    ... )
    >>> k_order_ranking(fig54, 1)[:3]
    [3, 5, 4]
    >>> k_order_ranking(fig54, 2)[:3]
    [4, 3, 5]
    """
    n = order
    scores = np.array([k_order(network, m, n, rule=rule) for m in range(network.n)])
    return np.lexsort((np.arange(network.n), -scores)).tolist()
