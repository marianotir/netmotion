"""Erdos-Renyi random graphs.

Thesis section 4.1 (PDF pp. 12-16).  Starting from :math:`N` unconnected nodes,
join every pair with probability :math:`p`.

The thesis works at :math:`N = 10` and sweeps :math:`p` in hundredths from 0 to
1, checking three things against theory: that the clustering coefficient equals
:math:`p`, that the mean degree equals :math:`pN`, and that the network
becomes connected above

.. math:: p_c = \\frac{\\ln N}{N}

which for :math:`N = 10` gives :math:`p_c \\approx 0.23`.  Figures on PDF p. 14
show the network disconnected just below that value and connected just above
it.

The printed Maple (PDF pp. 12-13) sweeps both orderings of every pair, using a
guard matrix ``h`` so that each unordered pair gets exactly one Bernoulli trial
at probability :math:`p` -- so the model is the usual :math:`G(N, p)`.
"""

from __future__ import annotations

import numpy as np

from ..graph import Network
from ..provenance import Origin, sourced
from ..rng import SeedLike, as_rng

__all__ = ["erdos_renyi", "connectivity_threshold"]


@sourced(
    Origin.THESIS_2011,
    pdf_pages="12-16",
    section="4.1 Random Graph Model",
    note=(
        "G(N, p): one Bernoulli trial per unordered pair. The printed listing "
        "reaches the same model through a double loop over ordered pairs with a "
        "guard matrix."
    ),
    reading="Each unordered pair is tried once at probability p.",
)
def erdos_renyi(
    n_nodes: int,
    link_probability: float,
    *,
    rng: SeedLike = None,
    name: str = "",
) -> Network:
    """A random graph :math:`G(N, p)`.

    Parameters
    ----------
    n_nodes:
        Number of nodes.  The thesis uses :math:`N = 10` throughout section 4.1.
    link_probability:
        Probability that any given pair is linked, in :math:`[0, 1]`.
    rng:
        Seed or generator.  Pass one for a reproducible run.

    Examples
    --------
    >>> erdos_renyi(10, 0.0, rng=1).n_edges
    0
    >>> erdos_renyi(10, 1.0, rng=1).n_edges          # every pair joined
    45

    Above the threshold the network is connected; below it, typically not:

    >>> from netmotion.metrics.distance import is_connected
    >>> is_connected(erdos_renyi(10, 0.9, rng=1))
    True
    """
    n, p = n_nodes, link_probability
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must lie in [0, 1], got {p}")

    generator = as_rng(rng)
    matrix = np.zeros((n, n))

    if n > 1 and p > 0.0:
        upper = np.triu_indices(n, k=1)
        linked = generator.random(upper[0].size) < p
        matrix[upper[0][linked], upper[1][linked]] = 1.0
        matrix = matrix + matrix.T

    return Network(matrix, name=name or f"ER N={n} p={p:g}")


@sourced(
    Origin.THESIS_2011,
    pdf_pages="14",
    section="4.1.2 Algoritmo",
    note=(
        "p_c = ln(N)/N, quoted by the thesis from Erdos and Renyi and verified "
        "for N = 10, where p_c is approximately 0.23."
    ),
)
def connectivity_threshold(n_nodes: int) -> float:
    """The probability above which the network is almost surely connected.

    >>> round(connectivity_threshold(10), 3)
    0.23
    """
    n = n_nodes
    if n < 2:
        return 0.0
    return float(np.log(n) / n)
