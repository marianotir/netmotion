"""The ordered ring and the Watts-Strogatz small-world network.

Thesis section 4.2 (PDF pp. 16-23).

The construction has two stages, and the thesis prints a Maple listing for
each.  First an **ordered network** in which every node joins its ``k`` nearest
neighbours around a ring (PDF pp. 17-18: ``aconectar := 4``, ``dim := 20``).
Then the network is traversed, and each link of the ring is examined: with
probability :math:`p` it is broken and the node is joined instead to another
node chosen at random, provided the two were not already connected.

Sweeping :math:`p` from 0 to 1 takes the network from ordered -- high
clustering, long paths -- to random -- low clustering, short paths.  Between
them lies the regime Watts and Strogatz identified in 1998, where the mean
distance has already collapsed while the clustering has not: the small-world
effect.  The thesis reproduces it at :math:`N = 30, 50, 100` (PDF p. 23) and
explains why it sharpens as :math:`N` grows.

The printed rewiring loop
-------------------------
The printed rewiring loop (PDF pp. 18-19) conserves links by construction,
without ever counting them.  It keeps a guard matrix ``h``, initialised to all
ones, and a fresh adjacency ``M`` initialised to all zeros.  Each rewiring
event does three things at once: it writes one pair
``{x, l}`` into ``M``, it sets ``h[x, l] := 0`` -- the thesis annotates this
line *indica que los nodos x, l se han conectado* -- and it sets
``h[x, y] := 0``, *indica que la conexion entre los nodos x, y se ha roto*.  A
second sweep then walks the ring and adds every link whose ``h`` entry is still
non-zero, that is, every ring link that was not broken.

The two sweeps cannot overlap.  The replacement partner is required to be a
node ``x`` was not joined to already -- the text states the condition on PDF
p. 18, *teniendo en cuenta que esta nueva union se debe hacer con un nodo con
el cual no estuviese conectado anteriormente* -- and ``h[x, l] := 0`` then
blocks that pair from being chosen again.  So the pairs written by the first
sweep are distinct from each other and disjoint from the ring links the second
sweep restores, and the total is :math:`(E - R) + R = E` for any number ``R``
of rewiring events: the mean degree stays at :math:`k` for every :math:`p`, which
is what the thesis plots in fig. 7.b.

``l`` is drawn once per link examined, and if the guard rejects it -- ``l`` is
``x`` itself, or a node ``x`` is already joined to -- the loop moves on and the
link keeps its place in the ring.  The thesis explains the consequence on PDF
p. 21:

    *aun existe algun nodo que esta conectado a su vecino mas proximo, a pesar
    de que la probabilidad de romper la conexion con el vecino mas proximo es
    maxima*

so links to nearest neighbours survive even at :math:`p = 1`; at :math:`N = 20, k = 4`
:func:`watts_strogatz` leaves about 11% of them in place.
``variant="standard"`` gives the common textbook rewiring instead, drawing the partner
from the admissible nodes directly, so every examined link is rewired and none
of the ring survives at :math:`p = 1`.  Both variants conserve the edge count.
"""

from __future__ import annotations

import numpy as np

from ..graph import Network
from ..provenance import Origin, sourced
from ..rng import SeedLike, as_rng

__all__ = ["ring_lattice", "watts_strogatz"]


@sourced(
    Origin.THESIS_2011,
    pdf_pages="17-18",
    section="4.2.2 Algoritmo",
    note=(
        "Every node joins its k nearest neighbours around a ring, k/2 in each "
        "direction, wrapping at the end. The thesis uses N = 20, k = 4."
    ),
    reading=(
        "The inner loop runs over nod = aconectar/2 neighbours in each "
        "direction."
    ),
)
def ring_lattice(n_nodes: int, neighbours: int, *, name: str = "") -> Network:
    """The ordered network: *n_nodes* nodes on a ring, each joined to its *neighbours* nearest nodes.

    In the book these are :math:`N` and :math:`k`.

    Parameters
    ----------
    n_nodes:
        Number of nodes.
    neighbours:
        Number of neighbours per node; must be even, since ``k / 2`` are taken
        in each direction.  The thesis uses ``k = 4``.

    Examples
    --------
    Every node has exactly *neighbours* links, so :math:`P(k)` is a delta -- the thesis
    makes this point on PDF p. 18:

    >>> ring = ring_lattice(20, 4)
    >>> sorted(set(ring.degrees().tolist()))
    [4]
    >>> ring.n_edges
    40
    """
    n, k = n_nodes, neighbours
    if k % 2:
        raise ValueError(f"k must be even (k/2 neighbours each side), got {k}")
    if k >= n and n > 0:
        raise ValueError(f"k={k} needs at least {k + 1} nodes, got n={n}")

    matrix = np.zeros((n, n))
    for offset in range(1, k // 2 + 1):
        idx = np.arange(n)
        partner = (idx + offset) % n
        matrix[idx, partner] = 1.0
        matrix[partner, idx] = 1.0
    return Network(matrix, name=name or f"ring N={n} k={k}")


@sourced(
    Origin.THESIS_2011,
    pdf_pages="18-21",
    section="4.2.2 Algoritmo",
    note=(
        "Traverse the ordered ring; break each link with probability p and "
        "rejoin the node to a randomly chosen partner it was not already "
        "joined to, recording both in the guard matrix h. Ring links that were "
        "not broken are added in a second sweep, so the edge count is conserved."
    ),
    reading=(
        "The new partner l is drawn once per link examined, from all nodes, and "
        "must be one x is not joined to already (stated in the text on PDF "
        "p. 18; checked here against the ring matrix CO and the guard matrix h). A rejected draw is not "
        "retried: the loop moves on and the link stays in the ring, as the "
        "listing is written. It is what leaves nearest-neighbour links in place "
        "even at p = 1, the effect the thesis reports on PDF p. 21. Either way "
        "the edge count is conserved. variant='standard' (not the default) "
        "draws from the admissible nodes directly, so no link goes unrewired."
    ),
    tags=("original-behaviour",),
)
def watts_strogatz(
    n_nodes: int,
    neighbours: int,
    rewiring_probability: float,
    *,
    rng: SeedLike = None,
    variant: str = "thesis",
    name: str = "",
) -> Network:
    """A Watts-Strogatz small-world network.

    Parameters
    ----------
    n, k:
        Size and degree of the underlying ring; see :func:`ring_lattice`.
    rewiring_probability:
        Rewiring probability.  ``p = 0`` returns the ring unchanged; ``p = 1``
        rewires (or attempts to rewire) every link.
    variant:
        ``"thesis"``
            Reproduces the printed algorithm: one draw per link examined, and
            the link is left alone if that draw lands on a node ``x`` is joined
            to already.  This is what the thesis's own curves measured, and it
            is why some nearest-neighbour links survive at ``p = 1``.
        ``"standard"``
            The conventional rewiring: the partner is drawn from the nodes
            ``x`` is *not* currently joined to, so every examined link is in
            fact rewired.

    Examples
    --------
    At ``p = 0`` nothing is rewired and the ring is returned exactly:

    >>> ring = watts_strogatz(20, 4, 0.0, rng=0)
    >>> sorted(set(ring.degrees().tolist())), ring.n_edges
    ([4], 40)

    Both variants conserve links, so the mean degree stays at *neighbours* for every
    *rewiring_probability* -- the flat line of fig. 7.b:

    >>> watts_strogatz(20, 4, 1.0, rng=3).n_edges
    40
    >>> watts_strogatz(50, 4, 1.0, rng=3, variant="standard").n_edges
    100
    """
    n, k, p = n_nodes, neighbours, rewiring_probability
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must lie in [0, 1], got {p}")
    if variant not in ("thesis", "standard"):
        raise ValueError(f"variant must be 'thesis' or 'standard', got {variant!r}")

    generator = as_rng(rng)
    ordered = ring_lattice(n, k)
    co = ordered.adjacency  # the ring, kept fixed: the listing never updates CO

    if p == 0.0:
        return Network(co.copy(), name=name or f"WS N={n} k={k} p=0")

    matrix = np.zeros((n, n))
    # h is the listing's guard matrix, all ones to start. h[i, j] = 0 carries
    # both of the meanings the listing annotates: "these two have been joined"
    # and "this ring link has been broken". One entry can never have to carry
    # both at once, because a pair joined by a rewiring is never a ring pair,
    # which is what keeps the two sweeps below disjoint.
    h = np.ones((n, n))

    for x in range(n):
        for y in range(n):
            if co[x, y] == 0:
                continue
            if generator.random() > p:
                continue

            if variant == "thesis":
                # One draw per link examined, no retry: if the guard rejects l
                # the listing falls through and the link stays as it is.
                # The thesis's "esta nueva union se debe hacer con un nodo con el
                # cual no estuviese conectado anteriormente", checked here against
                # the ring matrix CO and the guard matrix h: not
                # x itself, not a ring neighbour of x, and not a node already
                # joined to x by an earlier rewiring (h[x, l] = 0). The last two
                # conditions require the ring link x-y to still be unbroken.
                partner = int(generator.integers(0, n))
                usable = (
                    partner != x
                    and co[x, partner] == 0
                    and h[x, partner] != 0
                    and h[partner, x] != 0
                    and h[x, y] != 0
                    and h[y, x] != 0
                )
            else:
                unavailable = matrix[x] != 0
                unavailable |= co[x] != 0
                unavailable[x] = True
                choices = np.flatnonzero(~unavailable)
                if choices.size == 0 or h[x, y] == 0:
                    continue
                partner = int(generator.choice(choices))
                usable = True

            if not usable:
                continue

            matrix[x, partner] = matrix[partner, x] = 1.0
            h[x, partner] = h[partner, x] = 0.0
            h[x, y] = h[y, x] = 0.0  # the ring link x-y is now broken

    # Second sweep: keep every ring link that was not broken.
    survivors = (h != 0) & (h.T != 0) & (co != 0)
    matrix[survivors] = 1.0
    np.fill_diagonal(matrix, 0.0)
    matrix = np.maximum(matrix, matrix.T)

    return Network(matrix, name=name or f"WS N={n} k={k} p={p:g}")
