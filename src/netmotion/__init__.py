"""netmotion -- structure, propagation and contagion in complex networks.

A Python implementation of the models and dynamics of

    Mariano Tirado Alonso, *Dinamica y evolucion de redes complejas*,
    Trabajo Fin de Master, Master en Fisica de Sistemas Complejos,
    UNED, Departamento de Fisica Fundamental, 30 September 2011.

and of the interbank model of

    Mariano Tirado, *Complex Network for a Crisis Contagion on an Interbank
    System*, Int. J. Mod. Phys. C **23**(9), 1250058 (2012).
    doi:10.1142/S0129183112500581

The package is organised the way the thesis is:

``netmotion.graph``
    The adjacency-matrix representation everything else is built on.
``netmotion.networks``
    Network generators: Erdos-Renyi, Watts-Strogatz, Newman-Watts,
    Barabasi-Albert, and the community networks of chapter 5.6.
``netmotion.metrics``
    Mean distance, degree distribution, clustering, and the thesis's own
    higher-order degree and propagation-time centrality.
``netmotion.dynamics``
    Hopfield-type dynamics and opinion dynamics, weighted and unweighted.
``netmotion.propagation``
    Infection propagation, propagation speed, the place-based infection model,
    propagation-derived clusters, and immunisation.
``netmotion.finance``
    The weighted interbank network and its crisis-contagion cascade.

Every scientific object carries a :class:`~netmotion.provenance.Source` saying
whether it is 2011 original work, 2012 published work, a 2026 reproduction or a
2026 extension.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .graph import Network
from .provenance import Origin, Source, get_source, sourced
from .rng import as_rng

__all__ = [
    "__version__",
    "Network",
    "Origin",
    "Source",
    "sourced",
    "get_source",
    "as_rng",
]
