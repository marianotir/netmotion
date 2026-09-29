"""Degree and the degree distribution.

Thesis section 3.2, *Degree Distribution - Distribucion de grado*
(PDF pp. 9-10), where the degree of a node is defined as the number of links
leaving it and :math:`P(k)` as the probability that a node drawn at random has
degree *k*.

The printed Maple counts degrees by sweeping the whole adjacency matrix and
incrementing ``k[i]`` for every non-zero entry of row *i* -- a row sum, which is
what :meth:`~netmotion.graph.Network.degrees` does in one vectorised call.

:func:`degree_distribution` gives :math:`P(k)` normalised over all :math:`N`
nodes, isolated ones included.
"""

from __future__ import annotations

import numpy as np

from ..graph import Network
from ..provenance import Origin, sourced

__all__ = [
    "degrees",
    "mean_degree",
    "degree_distribution",
    "degree_counts",
    "power_law_exponent",
]


@sourced(
    Origin.THESIS_2011,
    pdf_pages="9-10",
    section="3.2 Degree Distribution - Distribucion de grado",
    note="Row sums of the adjacency matrix, as in the printed listing.",
)
def degrees(network: Network) -> np.ndarray:
    """Degree :math:`k_i` of every node, as an integer array."""
    return network.degrees()


@sourced(
    Origin.THESIS_2011,
    pdf_pages="14",
    section="4.1.3 Caracteristicas",
    note=(
        "The quantity the thesis plots against p for Erdos-Renyi networks "
        "(fig. 5.b), with the theoretical <k> = pN printed on PDF p. 14."
    ),
)
def mean_degree(network: Network) -> float:
    """Mean degree :math:`\\langle k \\rangle`, averaged over all nodes."""
    return float(network.degrees().mean()) if network.n else 0.0


@sourced(
    Origin.THESIS_2011,
    pdf_pages="9-10",
    section="3.2 Degree Distribution - Distribucion de grado",
    note=(
        "P(k) normalised over all N nodes, isolated nodes included."
    ),
)
def degree_distribution(network: Network, *, max_degree: int | None = None) -> np.ndarray:
    """:math:`P(k)` for :math:`k = 0, 1, \\ldots`, summing to one.

    Parameters
    ----------
    max_degree:
        Highest degree in the support.  Defaults to the largest degree present,
        which keeps the array short; pass ``network.n - 1`` for the full
        support.

    Examples
    --------
    A ring where every node has degree 4 gives a delta at 4, which is the
    :math:`P(k)` the thesis reports for the ordered network of section 4.2.2:

    >>> from netmotion.networks import ring_lattice
    >>> p = degree_distribution(ring_lattice(20, 4))
    >>> float(p[4]), float(p.sum())
    (1.0, 1.0)
    """
    kmax = max_degree
    k = network.degrees()
    top = int(k.max()) if k.size and kmax is None else (kmax or 0)
    counts = np.bincount(k, minlength=top + 1).astype(float)
    return counts / network.n if network.n else counts


@sourced(
    Origin.REPRODUCTION,
    pdf_pages="9-10",
    section="3.2 Degree Distribution - Distribucion de grado",
    reproduces="netmotion.metrics.degree.degree_distribution",
    note=(
        "The same histogram without the empty bins, for the log-log plots of "
        "the scale-free networks in section 4.4, where a zero count cannot be "
        "drawn."
    ),
)
def degree_counts(network: Network) -> tuple[np.ndarray, np.ndarray]:
    """``(k, count)`` for the degrees actually present, ascending.

    Convenient for a log-log plot, where the zero-count bins of
    :func:`degree_distribution` would have to be masked away anyway.
    """
    k = network.degrees()
    values, counts = np.unique(k, return_counts=True)
    return values, counts


@sourced(
    Origin.REPRODUCTION,
    pdf_pages="32-34",
    section="4.4.3 Caracteristicas",
    note=(
        "The maximum-likelihood estimator for a discrete power law. The thesis's "
        "factors gamma (PDF pp. 32-34) are regressions on logarithmic axes."
    ),
)
def power_law_exponent(network: Network, min_degree: int = 1) -> float:
    """Maximum-likelihood exponent :math:`\\gamma` of :math:`P(k)\\sim k^{-\\gamma}`.

    Uses the discrete-distribution estimator

    .. math::
        \\hat{\\gamma} = 1 + n \\left[
            \\sum_i \\ln \\frac{k_i}{k_{\\min} - 1/2} \\right]^{-1}

    over nodes with :math:`k_i \\ge k_{\\min}`.
    """
    kmin = min_degree
    k = network.degrees()
    tail = k[k >= kmin].astype(float)
    if tail.size < 2:
        return float("nan")
    return 1.0 + tail.size / np.sum(np.log(tail / (kmin - 0.5)))
