"""The Newman-Watts small-world network.

Thesis section 4.3 (PDF pp. 23-26).

The objection the model answers is stated plainly in the thesis: in the
Watts-Strogatz construction, making a new link requires breaking an old one,
and friendships do not work that way.

    *Supongamos un modelo en el cual cada persona es amigo de sus dos vecinos
    mas proximos, dentro del bloque de viviendas donde vive. Un dia por
    coincidencia se hacen amigas dos personas del bloque que no son vecinas.
    Esto no implica que la relacion de amistad con uno de sus vecinos mas
    proximos ha de romperse.*

So the lattice is kept and shortcuts are *added* on top of it.

The thesis makes the point in the most economical way available to it: the
Newman-Watts generator is the Watts-Strogatz generator with a single
instruction deleted (PDF p. 23),

.. code-block:: maple

    h(r)[x, y] := 0 :  h(r)[y, x] := 0 :

which is the line that marked a ring link as broken -- the same line already
annotated in the Watts-Strogatz listing with *(Esto no se tendra en cuenta en
el modelo NW)*.  This implementation keeps that relationship visible: its loop is the
Watts-Strogatz loop without the ring-breaking assignment.

Because links are only ever added, the mean degree grows with :math:`p` instead
of staying fixed, which is the clearest observable difference from
Watts-Strogatz.
"""

from __future__ import annotations

import numpy as np

from ..graph import Network
from ..provenance import Origin, sourced
from ..rng import SeedLike, as_rng
from .small_world import ring_lattice

__all__ = ["newman_watts"]


@sourced(
    Origin.THESIS_2011,
    pdf_pages="23-26",
    section="4.3 Modelo small world de Newman Watts",
    note=(
        "The Watts-Strogatz algorithm of section 4.2.2 with the link-breaking "
        "instruction removed, as the thesis specifies on PDF p. 23. "
        "The ring is preserved and shortcuts are added on top of it."
    ),
)
def newman_watts(
    n_nodes: int,
    neighbours: int,
    shortcut_probability: float,
    *,
    rng: SeedLike = None,
    name: str = "",
) -> Network:
    """A Newman-Watts small-world network: a ring plus random shortcuts.

    Parameters
    ----------
    n, k:
        Size and degree of the underlying ring.
    shortcut_probability:
        Probability, per ring link examined, of adding one shortcut.

    Examples
    --------
    Nothing is ever removed, so the ring is always a subgraph and the number of
    links only grows:

    >>> ring = ring_lattice(50, 4)
    >>> nw = newman_watts(50, 4, 0.3, rng=7)
    >>> nw.n_edges >= ring.n_edges
    True
    >>> bool(((ring.adjacency == 1) <= (nw.adjacency == 1)).all())
    True

    At ``p = 0`` it is exactly the ring:

    >>> newman_watts(20, 4, 0.0, rng=0).n_edges
    40
    """
    n, k, p = n_nodes, neighbours, shortcut_probability
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must lie in [0, 1], got {p}")

    generator = as_rng(rng)
    ordered = ring_lattice(n, k)
    co = ordered.adjacency
    matrix = co.copy()

    if p == 0.0:
        return Network(matrix, name=name or f"NW N={n} k={k} p=0")

    # h still records which pairs have already been joined by a shortcut; what
    # is gone, relative to Watts-Strogatz, is the marking of the ring link as
    # broken.
    h = np.ones((n, n))

    for x in range(n):
        for y in range(n):
            if co[x, y] == 0:
                continue
            if generator.random() > p:
                continue
            partner = int(generator.integers(0, n))
            if partner == x or h[x, partner] == 0 or h[partner, x] == 0:
                continue
            matrix[x, partner] = matrix[partner, x] = 1.0
            h[x, partner] = h[partner, x] = 0.0

    np.fill_diagonal(matrix, 0.0)
    return Network(matrix, name=name or f"NW N={n} k={k} p={p:g}")
