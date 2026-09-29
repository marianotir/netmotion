"""Shortest paths and the mean distance of a network.

Thesis section 3.1, *Distancia media de una red compleja* (PDF pp. 7-9).

The thesis defines the distance between two nodes as the number of links along
the shortest path joining them, and the mean distance as the average of that
over all pairs.  It then develops its own matrix-based variant of Dijkstra's
algorithm to compute it, works a full example by hand on an eight-node ring
where each node joins its four nearest neighbours, and obtains

.. math:: \\ell = 10/7 \\approx 1.42

Two conventions of the thesis's algorithm are kept:

**Unreachable pairs count as zero.**  The distance matrix is initialised to
``100``, and every ``100`` left at the end is replaced by ``0``, the sum still
being divided by :math:`N(N-1)`.  :func:`mean_distance` does the same by
default and offers other conventions as options.

**The denominator is** :math:`N(N-1)`.  The sum runs over ordered pairs, so
each unordered pair is counted twice, and the normalisation matches.

The listing, transliterated, returns the distance matrix printed on PDF p. 9
element for element, and a mean distance of 1.42.
"""

from __future__ import annotations

import heapq
from collections import deque

import numpy as np

from ..graph import Network
from ..provenance import Origin, sourced

__all__ = [
    "shortest_path_lengths",
    "all_pairs_distances",
    "mean_distance",
    "diameter",
    "efficiency",
]


@sourced(
    Origin.THESIS_2011,
    pdf_pages="7-9",
    section="3.1 Distancia media de una red compleja",
    note=(
        "Single-source shortest paths. The thesis develops its own variant of "
        "Dijkstra's algorithm over the adjacency matrix; this implements the "
        "same computation with a binary heap, and falls back to breadth-first "
        "search when every link has weight 1, which is the case for every "
        "network in chapters 3-5."
    ),
    reading=(
        "The listing, transliterated, returns the distance matrix printed on "
        "PDF p. 9 element for element."
    ),
)
def shortest_path_lengths(network: Network, source: int) -> np.ndarray:
    """Distance from *source* to every node.

    Returns a length-``N`` float array; unreachable nodes are ``np.inf`` and
    the source itself is ``0``.  Use :func:`mean_distance` rather than reading
    the infinities directly.

    >>> from netmotion import Network
    >>> path = Network.from_edges(4, [(0, 1), (1, 2), (2, 3)])
    >>> shortest_path_lengths(path, 0)
    array([0., 1., 2., 3.])
    """
    n = network.n
    if not 0 <= source < n:
        raise IndexError(f"source {source} outside 0..{n - 1}")

    matrix = network.matrix
    dist = np.full(n, np.inf)
    dist[source] = 0.0

    if not network.weighted:
        # Every link has weight 1, so breadth-first search gives the same answer
        # as Dijkstra and does it in O(N + E).
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in np.flatnonzero(matrix[u]):
                if dist[v] == np.inf:
                    dist[v] = dist[u] + 1.0
                    queue.append(int(v))
        return dist

    visited = np.zeros(n, dtype=bool)
    heap: list[tuple[float, int]] = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        for v in np.flatnonzero(matrix[u]):
            alt = d + matrix[u, v]
            if alt < dist[v]:
                dist[v] = alt
                heapq.heappush(heap, (float(alt), int(v)))
    return dist


@sourced(
    Origin.THESIS_2011,
    pdf_pages="7-9",
    section="3.1 Distancia media de una red compleja",
    note=(
        "The matrix the thesis calls Ld: entry (i, j) is the length of the "
        "shortest path from i to j."
    ),
)
def all_pairs_distances(network: Network) -> np.ndarray:
    """The full ``(N, N)`` distance matrix, ``np.inf`` where no path exists."""
    return np.vstack([shortest_path_lengths(network, i) for i in range(network.n)])


@sourced(
    Origin.THESIS_2011,
    pdf_pages="7-9",
    section="3.1 Distancia media de una red compleja",
    note=(
        "Mean over ordered pairs, normalised by N(N-1). The default handling of "
        "unreachable pairs reproduces the thesis convention: they contribute 0 "
        "to the numerator while remaining in the denominator."
    ),
    reading=(
        "Unreachable pairs contribute 0 and the sum is divided by N(N-1), as in "
        "the thesis: unreachable='zero' (default). 'exclude' and "
        "'largest_component' give other conventions."
    ),
    tags=("convention", "disconnected"),
)
def mean_distance(network: Network, unreachable: str = "zero") -> float:
    """Mean shortest-path length :math:`\\ell` of *network*.

    Parameters
    ----------
    unreachable:
        What to do with pairs that no path joins.

        ``"zero"``
            The thesis convention (PDF p. 9): such pairs add 0 to the sum and
            are still counted in the :math:`N(N-1)` denominator.  The default.
        ``"exclude"``
            Average over connected pairs only.
        ``"largest_component"``
            Mean distance within the largest connected component.
        ``"inf"``
            Return ``inf`` if the network is disconnected.

    Returns
    -------
    float
        ``0.0`` for a network of fewer than two nodes.

    Examples
    --------
    The eight-node ring worked by hand on PDF p. 9, where each node joins its
    four nearest neighbours: four nodes at distance 1 and three at distance 2,
    so :math:`\\ell = 10/7 = 1.4286`.  The thesis reports 1.42.

    >>> from netmotion.networks import ring_lattice
    >>> round(mean_distance(ring_lattice(8, 4)), 4)
    1.4286

    The conventions differ only when the network is disconnected.  Two
    disjoint links over four nodes: four ordered pairs are at distance 1 and
    eight are unreachable, so the thesis convention gives 4/12 while excluding
    the unreachable pairs gives 4/4.

    >>> from netmotion import Network
    >>> split = Network.from_edges(4, [(0, 1), (2, 3)])
    >>> round(mean_distance(split, "zero"), 4), mean_distance(split, "exclude")
    (0.3333, 1.0)
    """
    n = network.n
    if n < 2:
        return 0.0

    dist = all_pairs_distances(network)
    finite = np.isfinite(dist)
    np.fill_diagonal(finite, False)

    if unreachable == "inf" and not finite.sum() == n * (n - 1):
        return float("inf")

    if unreachable == "largest_component":
        component = largest_component(network)
        if len(component) < 2:
            return 0.0
        sub = dist[np.ix_(component, component)]
        mask = np.isfinite(sub)
        np.fill_diagonal(mask, False)
        return float(sub[mask].sum() / (len(component) * (len(component) - 1)))

    total = float(dist[finite].sum())

    if unreachable in ("zero", "inf"):
        # The thesis divides by every ordered pair, reachable or not.
        return total / (n * (n - 1))
    if unreachable == "exclude":
        count = int(finite.sum())
        return total / count if count else 0.0

    raise ValueError(
        f"unreachable must be 'zero', 'exclude', 'largest_component' or 'inf', got {unreachable!r}"
    )


@sourced(
    Origin.REPRODUCTION,
    pdf_pages="7-9",
    section="3.1 Distancia media de una red compleja",
    note=(
        "Not named in the thesis, which reports only the mean. Provided because "
        "the propagation chapters need the eccentricity of the network: the "
        "diameter bounds the number of propagation steps."
    ),
)
def diameter(network: Network, on_largest_component: bool = True) -> float:
    """The longest shortest path in the network.

    ``inf`` if the network is disconnected and *on_largest_component* is False.
    """
    dist = all_pairs_distances(network)
    if on_largest_component:
        component = largest_component(network)
        if len(component) < 2:
            return 0.0
        dist = dist[np.ix_(component, component)]
    finite = np.isfinite(dist)
    if not finite.all():
        return float("inf")
    return float(dist.max())


@sourced(
    Origin.EXTENSION,
    pdf_pages="7-9",
    section="3.1 Distancia media de una red compleja",
    note=(
        "2026 extension. Global efficiency, the mean of 1/d over pairs, "
        "offered alongside mean_distance for networks that may be disconnected."
    ),
    tags=("modern-comparison",),
)
def efficiency(network: Network) -> float:
    """Global efficiency :math:`E = \\frac{1}{N(N-1)}\\sum_{i\\neq j} 1/d_{ij}`."""
    n = network.n
    if n < 2:
        return 0.0
    dist = all_pairs_distances(network)
    with np.errstate(divide="ignore"):
        inv = np.where(dist > 0, 1.0 / dist, 0.0)
    np.fill_diagonal(inv, 0.0)
    return float(inv.sum() / (n * (n - 1)))


@sourced(
    Origin.REPRODUCTION,
    pdf_pages="14",
    section="4.1.2 Algoritmo",
    note=(
        "Connectivity is the quantity the thesis checks against the "
        "Erdos-Renyi threshold p_c = ln(N)/N on PDF p. 14. Implemented here "
        "because mean_distance and the propagation chapters both need it."
    ),
)
def connected_components(network: Network) -> list[np.ndarray]:
    """The connected components, as arrays of node indices, largest first."""
    n = network.n
    seen = np.zeros(n, dtype=bool)
    components: list[np.ndarray] = []
    for start in range(n):
        if seen[start]:
            continue
        stack, group = [start], []
        seen[start] = True
        while stack:
            u = stack.pop()
            group.append(u)
            for v in np.flatnonzero(network.matrix[u]):
                if not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
        components.append(np.array(sorted(group)))
    components.sort(key=len, reverse=True)
    return components


@sourced(
    Origin.REPRODUCTION,
    pdf_pages="14",
    section="4.1.2 Algoritmo",
    reproduces="netmotion.metrics.distance.connected_components",
    note="Convenience wrapper; the giant component is what the thesis's connectivity figures show.",
)
def largest_component(network: Network) -> np.ndarray:
    """Node indices of the largest connected component."""
    components = connected_components(network)
    return components[0] if components else np.array([], dtype=int)


@sourced(
    Origin.REPRODUCTION,
    pdf_pages="14",
    section="4.1.2 Algoritmo",
    reproduces="netmotion.metrics.distance.connected_components",
    note=(
        "The predicate behind the thesis's check of the Erdos-Renyi threshold: "
        "'para p > p_c todos los nodos de la red estan conectados', with p_c = 0.23."
    ),
)
def is_connected(network: Network) -> bool:
    """True when every node can reach every other."""
    return network.n <= 1 or len(connected_components(network)) == 1
