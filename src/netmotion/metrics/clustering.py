"""The clustering coefficient.

Thesis section 3.3, *Clustering - Conglomerado* (PDF pp. 10-11), following the
definition of Watts and Strogatz: if node *i* has :math:`k_i` links, its
neighbours could be joined by at most :math:`k_i(k_i-1)/2` links; writing
:math:`E_i` for the number that actually exist,

.. math:: C_i = \\frac{2E_i}{k_i(k_i - 1)}

and the clustering coefficient of the network is the mean of :math:`C_i` over
its nodes.

Two details of the printed Maple (PDF p. 11):

* the inner sweep runs ``for a from j to dim``, so each pair of neighbours is
  counted once -- which is why the formula carries the factor 2;
* the guard is ``if k[i] > 1``, so a node of degree 0 or 1 keeps
  :math:`C_i=0` and counts in the mean over the :math:`N` nodes.
"""

from __future__ import annotations

import numpy as np

from ..graph import Network
from ..provenance import Origin, sourced

__all__ = ["local_clustering", "clustering_coefficient", "neighbour_links"]


@sourced(
    Origin.THESIS_2011,
    pdf_pages="11",
    section="3.3 Clustering - Conglomerado",
    note=(
        "E_i, the number of links among the neighbours of i. The printed "
        "listing counts each such pair once ('for a from j to dim')."
    ),
)
def neighbour_links(network: Network) -> np.ndarray:
    """:math:`E_i`: links present among the neighbours of each node.

    >>> from netmotion import Network
    >>> triangle_plus_tail = Network.from_edges(4, [(0,1),(1,2),(2,0),(2,3)])
    >>> neighbour_links(triangle_plus_tail)
    array([1, 1, 1, 0])
    """
    a = network.adjacency
    # (A @ A) * A sums, for each pair (i, j) that is linked, the number of
    # common neighbours; the row sum then counts every triangle at i twice,
    # which is exactly 2 E_i.
    triangles_twice = ((a @ a) * a).sum(axis=1)
    return (triangles_twice / 2).astype(int)


@sourced(
    Origin.THESIS_2011,
    pdf_pages="10-11",
    section="3.3 Clustering - Conglomerado",
    note=(
        "C_i = 2 E_i / [k_i (k_i - 1)]. Following the printed guard "
        "'if k[i] > 1', nodes of degree 0 or 1 are assigned C_i = 0 rather "
        "than being left undefined."
    ),
    reading=(
        "Degree-0 and degree-1 nodes keep C_i = 0 and count in the mean, as in "
        "the thesis. clustering_coefficient(..., skip_low_degree=True) averages "
        "over nodes of degree two or more instead."
    ),
    tags=("convention",),
)
def local_clustering(network: Network) -> np.ndarray:
    """:math:`C_i` for every node, with :math:`C_i = 0` when :math:`k_i < 2`.

    >>> from netmotion import Network
    >>> triangle = Network.from_edges(3, [(0, 1), (1, 2), (2, 0)])
    >>> local_clustering(triangle)
    array([1., 1., 1.])
    """
    k = network.degrees().astype(float)
    e = neighbour_links(network).astype(float)
    possible = k * (k - 1.0)
    return np.divide(2.0 * e, possible, out=np.zeros_like(possible), where=possible > 0)


@sourced(
    Origin.THESIS_2011,
    pdf_pages="11",
    section="3.3 Clustering - Conglomerado",
    note=(
        "The network clustering coefficient: the mean of C_i over all N nodes, "
        "Csum/N."
    ),
)
def clustering_coefficient(network: Network, *, skip_low_degree: bool = False) -> float:
    """Mean clustering coefficient :math:`C` of the network.

    Parameters
    ----------
    skip_low_degree:
        ``False`` (the default), as in the thesis: nodes with fewer than two
        neighbours contribute :math:`C_i = 0` to the mean.  ``True`` averages
        only over nodes that could have a triangle.

    Examples
    --------
    A fully connected network has :math:`C = 1`, which is what the thesis
    reports for the Erdos-Renyi network at :math:`p = 1` (PDF p. 16):

    >>> from netmotion.networks import erdos_renyi
    >>> clustering_coefficient(erdos_renyi(10, 1.0, rng=0))
    1.0

    A ring where every node joins its two nearest neighbours has no triangles:

    >>> from netmotion.networks import ring_lattice
    >>> clustering_coefficient(ring_lattice(10, 2))
    0.0
    """
    if network.n == 0:
        return 0.0
    c = local_clustering(network)
    if skip_low_degree:
        eligible = network.degrees() > 1
        return float(c[eligible].mean()) if eligible.any() else 0.0
    return float(c.mean())
