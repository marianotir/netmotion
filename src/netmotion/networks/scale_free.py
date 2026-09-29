"""Scale-free networks by growth and preferential attachment.

Thesis section 4.4, *Barabasi model- Scale free model* (PDF pp. 26-31).

The network grows: it starts from a small seed network and at each step one new
node arrives and attaches to ``m`` existing nodes, choosing them with
probability proportional to their degree,

.. math:: \\Pi_i = \\frac{k_i}{\\sum_j k_j}

The printed Maple (PDF pp. 27-29) implements that by rejection sampling -- pick
an existing node at random, draw :math:`u \\sim U(0,1]`, accept if
:math:`u \\le \\Pi_i` -- and sweeps once over the existing nodes.  One sweep can
fail to find ``m`` partners, because :math:`\\Pi_i` is small once the network is
large, and the listing then gives three options.

Three named options, printed inside the listing
-----------------------------------------------
The thesis writes out three alternatives and comments on each:

``"none"`` (*Opcion 1: No se hace nada*)
    Nothing is done.  The thesis's comment:
    *se reproduce el fenomeno rich gets richer ya que solo los nodos con un alto
    grado k recibiran nuevos nodos.*  An arrival that makes no link does not
    join the network: 400 growth steps give the network of 179 nodes of
    PDF pp. 29-30.

``"random"`` (*Opcion 2*)
    Join the new node to uniformly random partners until it has ``m``.  The
    thesis notes that this gives low-degree and newly arrived nodes a chance to
    grow.

``"nearest"`` (*Opcion 3*)
    Join the new node to its nearest partners *in the plane*.  The generator
    places each new node at random coordinates and this option connects to the
    geometrically closest unconnected node.  It makes the model spatial, which
    the other two are not.

The three options give different tails of :math:`P(k)`: the thesis analyses
options 2 and 3 in figs. 17-19.  All three are implemented and the caller
chooses; ``"random"`` is the default.
"""

from __future__ import annotations

import numpy as np

from ..graph import Network
from ..provenance import Origin, sourced
from ..rng import SeedLike, as_rng

__all__ = ["barabasi_albert", "scale_free_seed"]

_FALLBACKS = ("random", "none", "nearest")


@sourced(
    Origin.THESIS_2011,
    pdf_pages="27",
    section="4.4.2 Algoritmo",
    note=(
        "The seed network the growth starts from, 'una red "
        "formada por un numero pequeno de nodos usando los algoritmos "
        "anteriores'. A complete graph on m0 nodes is used here, which "
        "guarantees every seed node has non-zero degree."
    ),
    reading="The seed is a complete graph on m0 nodes.",
)
def scale_free_seed(initial_nodes: int) -> Network:
    """The initial network: a complete graph on *initial_nodes* nodes (the book's :math:`m_0`).

    Preferential attachment needs every seed node to have non-zero degree,
    otherwise :math:`\\Pi_i = 0` and the node can never be chosen.
    """
    m0 = initial_nodes
    matrix = np.ones((m0, m0)) - np.eye(m0)
    return Network(matrix, name=f"seed K{m0}")


@sourced(
    Origin.THESIS_2011,
    pdf_pages="26-34",
    section="4.4 Barabasi model- Scale free model",
    note=(
        "Growth plus preferential attachment with Pi_i = k_i / sum_j k_j, by "
        "rejection sampling over one sweep of the existing nodes, then one of "
        "the three completion rules the thesis prints as Opcion 1, 2 and 3."
    ),
    reading=(
        "The thesis prints three completion rules, Opcion 1, 2 and 3, and shows "
        "a network for each (figs. 13-15). All three are implemented; the caller "
        "chooses."
    ),
    tags=("original-behaviour", "three-variants"),
)
def barabasi_albert(
    n_nodes: int,
    links_per_new_node: int = 2,
    *,
    initial_nodes: int | None = None,
    fallback: str = "random",
    rng: SeedLike = None,
    name: str = "",
) -> Network:
    """A scale-free network grown by preferential attachment.

    Parameters
    ----------
    n_nodes:
        Final number of nodes.
    links_per_new_node:
        Links each arriving node tries to make, the book's :math:`m`.  The thesis uses
        ``nodosKonectar := 2``.
    initial_nodes:
        Size of the seed network, the book's :math:`m_0`; defaults to
        ``links_per_new_node + 1``.
    fallback:
        What to do when one rejection-sampling sweep leaves the new node with
        fewer than *links_per_new_node* links: ``"none"``, ``"random"`` or ``"nearest"``.  See
        the module docstring -- the three are the thesis's own Opcion 1, 2 and 3
        and they give materially different degree distributions.

    Returns
    -------
    Network
        For ``fallback="nearest"`` the node coordinates used are attached as
        ``network.positions``, since that variant is genuinely spatial.

    Examples
    --------
    With the ``"random"`` completion every arrival makes exactly *links_per_new_node* links, so
    the link count is exact:

    >>> g = barabasi_albert(200, links_per_new_node=2, rng=11)
    >>> g.n_edges == scale_free_seed(3).n_edges + 2 * (200 - 3)
    True

    The hub structure is what makes it scale-free -- the largest degree grows
    far beyond the mean:

    >>> k = g.degrees()
    >>> bool(k.max() > 4 * k.mean())
    True

    Leaving the node under-connected instead ("Opcion 1") produces the
    rich-get-richer regime the thesis describes, with far fewer links:

    >>> option_1 = barabasi_albert(200, links_per_new_node=2, fallback="none", rng=11)
    >>> option_1.n_edges < g.n_edges
    True
    """
    n, m, m0 = n_nodes, links_per_new_node, initial_nodes
    if fallback not in _FALLBACKS:
        raise ValueError(f"fallback must be one of {_FALLBACKS}, got {fallback!r}")
    m0 = m0 if m0 is not None else m + 1
    if m0 < 2 or m > m0:
        raise ValueError(f"need 2 <= links_per_new_node <= initial_nodes, got {m} and {m0}")
    if n < m0:
        raise ValueError(f"n_nodes={n} is smaller than the seed network, initial_nodes={m0}")

    generator = as_rng(rng)
    matrix = np.zeros((n, n))
    seed = scale_free_seed(m0)
    matrix[:m0, :m0] = seed.matrix

    # The listing draws coordinates for every new node; only the "nearest"
    # completion reads them, but they are drawn either way so that the three
    # options consume the random stream identically and stay comparable.
    positions = np.zeros((n, 2))
    angles = 2.0 * np.pi * np.arange(m0) / m0
    positions[:m0] = np.column_stack([np.cos(angles), np.sin(angles)])

    for new in range(m0, n):
        existing = np.arange(new)
        k = matrix[:new, :new].sum(axis=1)
        ksum = k.sum()
        pi = k / ksum if ksum > 0 else np.full(new, 1.0 / new)

        span = float(new)
        positions[new] = generator.uniform(-span, span, size=2)

        made = 0
        # One sweep of rejection sampling, exactly as printed: for each pass a
        # candidate is drawn uniformly and accepted with probability Pi.
        for _ in existing:
            if made >= m:
                break
            candidate = int(generator.integers(0, new))
            if generator.random() <= pi[candidate] and matrix[candidate, new] == 0:
                matrix[candidate, new] = matrix[new, candidate] = 1.0
                made += 1

        if made >= m or fallback == "none":
            continue

        if fallback == "random":
            while made < m:
                candidate = int(generator.integers(0, new))
                if matrix[candidate, new] == 0:
                    matrix[candidate, new] = matrix[new, candidate] = 1.0
                    made += 1
        else:  # "nearest": join the geometrically closest unconnected node
            while made < m:
                free = np.flatnonzero(matrix[:new, new] == 0)
                d = np.linalg.norm(positions[free] - positions[new], axis=1)
                free, d = free[d > 0], d[d > 0]
                if free.size == 0:
                    break
                # The listing keeps a candidate when dist <= distmin, scanning
                # j upwards, so of two equally near nodes the later one is kept.
                closest = int(free[np.flatnonzero(d == d.min())[-1]])
                matrix[closest, new] = matrix[new, closest] = 1.0
                made += 1

    network = Network(matrix, name=name or f"BA N={n} m={m} ({fallback})")
    network.positions = positions
    return network
