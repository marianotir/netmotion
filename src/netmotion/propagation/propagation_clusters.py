"""Clusters of order *n*, found by watching an infection close on itself.

Thesis section 5.4.2, *Aplicaciones del concepto de propagacion de una
infeccion a traves de la red. El concepto Cluster C de orden n*
(PDF pp. 48-51, printed pp. 45-48).

The clustering coefficient of section 3.3 counts one shape: node *m* joined to
two neighbours that are themselves joined.  A network also contains
*conglomerados de mayor tamano* -- *m* joined to :math:`n_1, n_2`, which are
joined to :math:`n_3, n_4`, which are joined to each other -- and finding them
in the adjacency matrix *"seria muy costoso"*.  The thesis finds them with the
propagation of an infection: start an infection at *m*, let it spread, and
watch where the wavefront closes on itself.

The newly infected nodes at step *t* are the nodes at distance *t* from *m*.  A
cycle through *m* is announced by one of two events, which the thesis marks in
purple in figures 44 and 45 -- *"contabilizar la cantidad de ramas y nodos
morados"*:

*purple nodes*
    a node infected at :math:`t+1` by **more than one** node infected at *t*.
    Two shortest paths of length *n* out of *m* have met head-on, closing a
    cycle of length :math:`2n`.

*purple edges*
    two nodes infected at the same instant :math:`t+1` that are joined to each
    other.  Two shortest paths of length *n* are bridged by one link, closing a
    cycle of length :math:`2n + 1`.

Their sum is :math:`\\mathcal{E}_m(n)`, the number of *"clusters de orden n
del nodo m"*.  The thesis then defines the order-1
*coefficient* (PDF p. 49)

.. math:: C_i(1) = \\frac{2\\,\\mathcal{E}_i(1)}{k_i (k_i - 1)}

and checks, for four kinds of 100-node network, that it coincides with the
classical coefficient node by node (fig. 47).  At :math:`n = 1` every newly
infected node has the single parent *m*, so :math:`\\mathcal{E}_m(1)` is the
number of purple edges among the neighbours of *m* -- each of them a triangle
on *m*, the cluster of fig. 46 -- and equals the classical :math:`E_m`.  The
test suite asserts that identity.

Readings taken
--------------
*Infected nodes stay infected*, as figures 44 and 45 draw them, and as in
section 5.4.1.

*"a partir de nodos distintos".*  The purple-edge rule reads *"tambien se forman
clusters cuando dos nodos n1, n2 son infectados en el instante t + 1 a partir de
nodos distintos y a su vez existe una rama que conecta a los nodos n1 y n2"*.
``distinct_clause="ends"`` (the default) takes *nodos distintos* to be the two
different nodes :math:`n_1, n_2` of the same wavefront, which gives the order-1
cluster of fig. 46 and the agreement of fig. 47; ``distinct_clause="parents"``
asks instead that the two ends have different parents.

*A purple node counts once, however many parents it has*, as in fig. 44 at
:math:`t = 3`, where a node reached from three directions is *"un cluster de
orden 3"*.

Nothing here is stochastic: the propagation is a breadth-first sweep and the
counts are deterministic functions of the adjacency matrix, so these functions
take no ``rng``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..graph import Network
from ..provenance import Origin, sourced
from .infection import propagate

__all__ = [
    "ClusterOrders",
    "infection_levels",
    "cluster_orders",
    "cluster_coefficient_order1",
    "local_cluster_coefficient_order1",
    "cluster_order_matrix",
    "nodes_in_largest_clusters",
]


@sourced(
    Origin.THESIS_2011,
    pdf_pages="48-49",
    section="5.4.2 Aplicaciones del concepto de propagacion de una infeccion a traves de la red. El concepto Cluster C de orden n",
    note=(
        "The result of the order-n cluster count for one seed. The two "
        "mechanisms the thesis marks in purple are kept apart: purple_nodes "
        "counts closures at a node (cycles of length 2n through the seed), "
        "purple_edges counts closures across a link (cycles of length 2n + 1). "
        "The number of clusters is their sum."
    ),
    reading="The two counts are kept separate, and each closure is counted as it occurs.",
    tags=("chapter-5", "original-method"),
)
@dataclass
class ClusterOrders:
    """Clusters of every order found from one seed.

    Attributes
    ----------
    first_infected:
        The node the infection started at, the thesis's *m*.
    purple_nodes:
        ``purple_nodes[n - 1]`` is :math:`A_m(n)`, the number of nodes infected
        at instant *n* by two or more nodes infected at instant :math:`n - 1`.
    purple_edges:
        ``purple_edges[n - 1]`` is :math:`B_m(n)`, the number of links joining
        two nodes that were both infected at instant *n*.

    Both arrays run from order 1 to the last order the infection reached, so an
    order that found nothing is present as a zero rather than missing.
    """

    first_infected: int
    purple_nodes: np.ndarray
    purple_edges: np.ndarray

    @property
    def counts(self) -> np.ndarray:
        """:math:`\\mathcal{E}_m(n) = A_m(n) + B_m(n)` for each order."""
        return self.purple_nodes + self.purple_edges

    @property
    def max_order(self) -> int:
        """The highest order computed; the number of infection advance cycles."""
        return int(self.counts.size)

    def __getitem__(self, order: int) -> int:
        """:math:`\\mathcal{E}_m(n)`, with orders numbered from 1 as in the thesis."""
        if not 1 <= order <= self.max_order:
            raise IndexError(f"order {order} outside 1..{self.max_order}")
        return int(self.counts[order - 1])

    def as_dict(self) -> dict[int, int]:
        """``{n: E_m(n)}``, the form the worked examples are quoted in."""
        return {n: int(c) for n, c in enumerate(self.counts, start=1)}

    def cycle_lengths(self) -> dict[int, int]:
        """Lengths of the cycles through the seed the counts correspond to.

        Read off the outlines drawn in figs. 44 and 45: a purple node at order
        *n* closes a cycle of length :math:`2n`, a purple edge one of length
        :math:`2n + 1`.  Returns ``{length: how many}``.
        """
        found: dict[int, int] = {}
        pairs = zip(self.purple_nodes, self.purple_edges, strict=True)
        for n, (a, b) in enumerate(pairs, start=1):
            if a:
                found[2 * n] = found.get(2 * n, 0) + int(a)
            if b:
                found[2 * n + 1] = found.get(2 * n + 1, 0) + int(b)
        return found


@sourced(
    Origin.REPRODUCTION,
    pdf_pages="48",
    section="5.4.2 Aplicaciones del concepto de propagacion de una infeccion a traves de la red. El concepto Cluster C de orden n",
    reproduces="netmotion.propagation.infection.propagate",
    note=(
        "The wavefront F(t) of the section-5.4.1 propagation, seeded at a "
        "single node: F(0) = {seed} and F(t) is the set newly infected at "
        "instant t. Derived from propagate rather than reimplemented, so the "
        "cluster method and the epidemic results share one dynamics."
    ),
    reading=(
        "The propagation is run until it stops spreading, which for a connected "
        "network means until the whole network is infected; max_order stops it "
        "earlier."
    ),
    tags=("chapter-5",),
)
def infection_levels(
    network: Network,
    first_infected: int,
    *,
    max_order: int | None = None,
) -> list[np.ndarray]:
    """The infection wavefront at each instant, as node-index arrays.

    Element *t* holds the nodes newly infected at instant *t*, the nodes at
    distance *t* from *first_infected*.  Element 0 is always ``[first_infected]``.

    Parameters
    ----------
    max_order:
        Stop after this many advance cycles, giving ``max_order + 1`` levels.
        ``None`` runs until the infection stops spreading.

    Examples
    --------
    On the hypercube of fig. 44 the infection leaves the corner it started
    from, reaches the three nodes joined to it, then the three opposite those,
    and finally the far corner:

    >>> from netmotion import Network
    >>> q3 = Network.from_edges(8, [(0, 1), (1, 2), (2, 3), (3, 0),
    ...                             (4, 5), (5, 6), (6, 7), (7, 4),
    ...                             (0, 4), (1, 5), (2, 6), (3, 7)])
    >>> [level.tolist() for level in infection_levels(q3, 3)]
    [[3], [0, 2, 7], [1, 4, 6], [5]]

    A node with no links infects nobody, so there is a single level:

    >>> lonely = Network.empty(3)
    >>> [level.tolist() for level in infection_levels(lonely, 1)]
    [[1]]
    """
    seed = first_infected
    if max_order is not None and max_order < 1:
        raise ValueError(f"max_order must be at least 1, got {max_order}")

    run = propagate(network, seed, max_steps=max_order)
    depth = run.infected_at

    levels: list[np.ndarray] = []
    # The thesis indexes the seed's instant as t = 1 (see PropagationRun), so
    # order n reads level n, whose nodes carry infected_at == n + 1.
    t = 1.0
    while True:
        nodes = np.flatnonzero(depth == t)
        if nodes.size == 0:
            break
        levels.append(nodes)
        t += 1.0
    return levels


@sourced(
    Origin.THESIS_2011,
    pdf_pages="48-49",
    section="5.4.2 Aplicaciones del concepto de propagacion de una infeccion a traves de la red. El concepto Cluster C de orden n",
    note=(
        "The counting rule stated in words on PDF pp. 48-49: count the nodes "
        "infected at t + 1 by more than one node infected at t ('nodos "
        "morados'), plus the links joining two nodes infected at the same "
        "instant ('ramas moradas'). Their sum is the number of clusters of "
        "order n of node m. Verified against both worked examples: the "
        "hypercube of fig. 44 gives 0, 3, 1 for orders 1, 2, 3."
    ),
    reading=(
        "(1) A node reached from several directions counts once, as in fig. 44 "
        "at t = 3, 'un cluster de orden 3'. (2) 'a partir de nodos distintos' "
        "is read as the two different ends n1, n2 of one wavefront "
        "(distinct_clause='ends', the default), which gives the triangle of "
        "fig. 46 at order 1 and the agreement of fig. 47; 'parents' asks that "
        "the two ends have different parents."
    ),
    tags=("chapter-5", "original-method"),
)
def cluster_orders(
    network: Network,
    first_infected: int,
    *,
    max_order: int | None = None,
    distinct_clause: str = "ends",
) -> ClusterOrders:
    """Count the clusters of every order that contain *first_infected*.

    Parameters
    ----------
    max_order:
        Highest order to compute.  ``None`` runs the infection to exhaustion.
    distinct_clause:
        Which way the *"a partir de nodos distintos"* clause of the purple-edge
        rule is attached.  ``"ends"``, the default, attaches it to the pair
        :math:`n_1, n_2`: two different nodes of the same wavefront, so the
        branch closes two separate paths out of the seed, as in figs. 46 and
        47.  ``"parents"`` asks instead that the nodes that infected them
        differ.  See the module docstring.

    Notes
    -----
    :math:`A_m(1)` is always zero -- at the first instant the only infected
    node is the seed, so every newly infected node has exactly one parent.
    Order 1 is therefore carried entirely by the purple-edge mechanism, and
    under the ``"ends"`` reading of the *nodos distintos* clause it reduces to
    the number of links among the neighbours of *m*, which is the classical
    :math:`E_m`; :func:`cluster_coefficient_order1` is then the Watts-Strogatz
    coefficient, exactly as fig. 47 reports.  Each such link is *m* plus two
    adjacent neighbours, the triangle fig. 46 draws.

    Examples
    --------
    The worked example of fig. 44 (PDF p. 48): the 8-node hypercube, infection
    started at an outer corner.  The thesis reports *"no tenemos ningun cluster
    de orden 1, 3 clusteres de orden 2 y un cluster de orden 3"*:

    >>> from netmotion import Network
    >>> q3 = Network.from_edges(8, [(0, 1), (1, 2), (2, 3), (3, 0),
    ...                             (4, 5), (5, 6), (6, 7), (7, 4),
    ...                             (0, 4), (1, 5), (2, 6), (3, 7)])
    >>> cluster_orders(q3, 3).as_dict()
    {1: 0, 2: 3, 3: 1}

    All four of those closures happen at a node, never across a link, so they
    are cycles of even length -- three squares and the hexagon the thesis
    traces in bold in panel d:

    >>> cluster_orders(q3, 3).cycle_lengths()
    {4: 3, 6: 1}

    A triangle is the smallest order-1 cluster, and the classical shape of
    fig. 46:

    >>> triangle = Network.from_edges(3, [(0, 1), (1, 2), (2, 0)])
    >>> cluster_orders(triangle, 0).as_dict()
    {1: 1}

    With ``distinct_clause="parents"`` order 1 is empty, since both ends are
    reached from the seed:

    >>> cluster_orders(triangle, 0, distinct_clause="parents").as_dict()
    {1: 0}

    The five-cycle closes across a link instead, at order 2:

    >>> c5 = Network.from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    >>> run = cluster_orders(c5, 0)
    >>> run.as_dict(), int(run.purple_nodes[1]), int(run.purple_edges[1])
    ({1: 0, 2: 1}, 0, 1)
    """
    seed = first_infected
    if distinct_clause not in ("ends", "parents"):
        raise ValueError(f"distinct_clause must be 'ends' or 'parents', got {distinct_clause!r}")

    adjacency = network.adjacency
    levels = infection_levels(network, seed, max_order=max_order)

    purple_nodes = np.zeros(max(len(levels) - 1, 0), dtype=int)
    purple_edges = np.zeros_like(purple_nodes)

    for order in range(1, len(levels)):
        previous, front = levels[order - 1], levels[order]

        # Only the previous wavefront can be a parent: an older infected node
        # adjacent to a front node would have infected it a step sooner.
        parents = adjacency[np.ix_(front, previous)] > 0
        parent_count = parents.sum(axis=1)
        purple_nodes[order - 1] = int((parent_count >= 2).sum())

        internal = adjacency[np.ix_(front, front)] > 0
        if distinct_clause == "parents":
            shared = parents.astype(int) @ parents.astype(int).T
            distinct = parent_count[:, None] + parent_count[None, :] - shared >= 2
            internal = internal & distinct
        # Under the "ends" reading the clause asks that n1 and n2 be different
        # nodes and that neither was reached along the branch joining them.
        # Both hold for any link internal to one wavefront -- the ends are two
        # entries of `front`, and a node's parents lie in the previous level --
        # so the clause is applied by counting each such link exactly once,
        # which is the strict upper triangle.
        purple_edges[order - 1] = int(np.triu(internal, k=1).sum())

    return ClusterOrders(first_infected=seed, purple_nodes=purple_nodes, purple_edges=purple_edges)


@sourced(
    Origin.THESIS_2011,
    pdf_pages="49",
    section="5.4.2 Aplicaciones del concepto de propagacion de una infeccion a traves de la red. El concepto Cluster C de orden n",
    note=(
        "C_i(1) = 2 E_i(1) / [k_i (k_i - 1)], written out on PDF p. 49, where "
        "E_i(1) is the number of order-1 clusters found for node i by the "
        "propagation method. Fig. 47 checks it against the section-3.3 "
        "coefficient on four 100-node networks and finds them identical."
    ),
    reading=(
        "The denominator vanishes for k < 2, and a ValueError is raised; "
        "local_cluster_coefficient_order1 gives such nodes the value 0, as in "
        "section 3.3."
    ),
    tags=("chapter-5", "original-method"),
)
def cluster_coefficient_order1(network: Network, node: int) -> float:
    """:math:`C_m(1)`, the order-1 cluster coefficient of one node.

    Raises
    ------
    ValueError
        When the node has fewer than two neighbours, where the formula divides
        by zero.

    Examples
    --------
    Every node of a triangle is fully clustered:

    >>> from netmotion import Network
    >>> triangle = Network.from_edges(3, [(0, 1), (1, 2), (2, 0)])
    >>> cluster_coefficient_order1(triangle, 0)
    1.0

    The hypercube of fig. 44 has no triangles, which is why the thesis finds no
    cluster of order 1 there:

    >>> q3 = Network.from_edges(8, [(0, 1), (1, 2), (2, 3), (3, 0),
    ...                             (4, 5), (5, 6), (6, 7), (7, 4),
    ...                             (0, 4), (1, 5), (2, 6), (3, 7)])
    >>> cluster_coefficient_order1(q3, 3)
    0.0
    """
    k = int(network.degrees()[node])
    if k < 2:
        raise ValueError(
            f"C({node}, 1) is undefined: degree {k} makes k(k-1) zero; use "
            "local_cluster_coefficient_order1 for every node"
        )
    counts = cluster_orders(network, node, max_order=1).counts
    return float(2.0 * counts[0] / (k * (k - 1)))


@sourced(
    Origin.THESIS_2011,
    pdf_pages="49-51",
    section="5.4.2 Aplicaciones del concepto de propagacion de una infeccion a traves de la red. El concepto Cluster C de orden n",
    note=(
        "The blue curves of fig. 47: C_n(1) computed by the propagation "
        "method for every node of a 100-node network, plotted against the "
        "node index. The thesis's conclusion (PDF p. 51) is that these agree "
        "with the section-3.3 curves, validating the method."
    ),
    reading=(
        "Nodes of degree < 2 take the value 0, as under the guard 'if k[i] > 1' "
        "of section 3.3; low_degree='nan' leaves them out instead."
    ),
    tags=("chapter-5", "convention"),
)
def local_cluster_coefficient_order1(
    network: Network,
    *,
    low_degree: str = "zero",
) -> np.ndarray:
    """:math:`C_i(1)` for every node, the curve fig. 47 plots.

    Parameters
    ----------
    low_degree:
        What to record for a node with fewer than two neighbours.  ``"zero"``
        matches the convention of :func:`~netmotion.metrics.local_clustering`,
        which is the one the thesis's own section-3.3 code uses; ``"nan"``
        leaves the node out of any subsequent mean.

    Examples
    --------
    The ordered network of fig. 47.d -- a ring in which every node joins its
    four nearest neighbours -- has :math:`C_i = 0.5` at every node, the flat
    line the thesis prints:

    >>> from netmotion.networks import ring_lattice
    >>> c = local_cluster_coefficient_order1(ring_lattice(100, 4))
    >>> sorted(set(c.tolist())), float(c.mean())
    ([0.5], 0.5)
    """
    if low_degree not in ("zero", "nan"):
        raise ValueError(f"low_degree must be 'zero' or 'nan', got {low_degree!r}")

    fill = 0.0 if low_degree == "zero" else float("nan")
    k = network.degrees()
    out = np.full(network.n, fill, dtype=float)
    for node in range(network.n):
        if k[node] >= 2:
            out[node] = cluster_coefficient_order1(network, node)
    return out


@sourced(
    Origin.REPRODUCTION,
    pdf_pages="48-51",
    section="5.4.2 Aplicaciones del concepto de propagacion de una infeccion a traves de la red. El concepto Cluster C de orden n",
    reproduces="netmotion.propagation.propagation_clusters.cluster_orders",
    note=(
        "The network-wide sweep the section is written for: run the infection "
        "from every node in turn and tabulate E_m(n) -- 'la busqueda de clusters de "
        "gran tamano dentro de la red'. Fig. 47 is this sweep at order 1."
    ),
    tags=("chapter-5",),
)
def cluster_order_matrix(network: Network, max_order: int) -> np.ndarray:
    """:math:`\\mathcal{E}_m(n)` for every node *m* and order :math:`n \\le` *max_order*.

    Returns an ``(N, max_order)`` integer array; ``matrix[m, n - 1]`` is the
    number of clusters of order *n* containing *m*.  Seeds whose infection dies
    out before *max_order* are padded with zeros.

    Examples
    --------
    The hypercube is vertex-transitive, so every node sees the same profile as
    the corner the thesis started from:

    >>> from netmotion import Network
    >>> q3 = Network.from_edges(8, [(0, 1), (1, 2), (2, 3), (3, 0),
    ...                             (4, 5), (5, 6), (6, 7), (7, 4),
    ...                             (0, 4), (1, 5), (2, 6), (3, 7)])
    >>> cluster_order_matrix(q3, 3).tolist()[:2]
    [[0, 3, 1], [0, 3, 1]]
    """
    if max_order < 1:
        raise ValueError(f"max_order must be at least 1, got {max_order}")

    out = np.zeros((network.n, max_order), dtype=int)
    for node in range(network.n):
        counts = cluster_orders(network, node, max_order=max_order).counts
        out[node, : counts.size] = counts
    return out


@sourced(
    Origin.EXTENSION,
    pdf_pages="48-51",
    section="5.4.2 Aplicaciones del concepto de propagacion de una infeccion a traves de la red. El concepto Cluster C de orden n",
    reproduces="netmotion.propagation.propagation_clusters.cluster_order_matrix",
    note=(
        "2026 extension: the nodes ranked by the number of order-n clusters "
        "that contain them."
    ),
    reading=(
        "An order-n cluster is defined relative to its seed, so the same "
        "conglomerate is found from every node lying on it; this function ranks "
        "seeds."
    ),
    tags=("chapter-5", "extension"),
)
def nodes_in_largest_clusters(
    network: Network,
    order: int,
    *,
    top: int | None = None,
) -> np.ndarray:
    """Nodes ranked by how many clusters of the given order contain them.

    Nodes with no cluster of that order are omitted.  Ties keep ascending node
    order, so the result is reproducible.

    Examples
    --------
    A triangle with a tail: only the three nodes of the triangle take part in
    an order-1 cluster.

    >>> from netmotion import Network
    >>> g = Network.from_edges(4, [(0, 1), (1, 2), (2, 0), (2, 3)])
    >>> nodes_in_largest_clusters(g, 1).tolist()
    [0, 1, 2]

    At order 3 the six-cycle closes at the node opposite each seed, so every
    node is equally implicated:

    >>> c6 = Network.from_edges(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)])
    >>> nodes_in_largest_clusters(c6, 3, top=2).tolist()
    [0, 1]
    """
    counts = cluster_order_matrix(network, order)[:, order - 1]
    present = np.flatnonzero(counts > 0)
    ranked = present[np.argsort(-counts[present], kind="stable")]
    return ranked if top is None else ranked[:top]
