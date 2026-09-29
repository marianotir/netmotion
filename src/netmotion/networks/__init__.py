"""Network generators.

The four models of chapter 4 of the thesis, in the order it presents them --
each one answering an objection to the one before it -- plus the ordered ring
the small-world models are built on.

============================  ==============================================
:func:`erdos_renyi`           Join every pair with probability *p*.
:func:`ring_lattice`          Join each node to its *k* nearest neighbours.
:func:`watts_strogatz`        Ring, then rewire each link with probability *p*.
:func:`newman_watts`          Ring, then *add* shortcuts without breaking it.
:func:`barabasi_albert`       Grow, attaching in proportion to degree.
============================  ==============================================
"""

from __future__ import annotations

from .erdos_renyi import connectivity_threshold, erdos_renyi
from .newman_watts import newman_watts
from .scale_free import barabasi_albert, scale_free_seed
from .small_world import ring_lattice, watts_strogatz

__all__ = [
    "erdos_renyi",
    "connectivity_threshold",
    "ring_lattice",
    "watts_strogatz",
    "newman_watts",
    "barabasi_albert",
    "scale_free_seed",
]
