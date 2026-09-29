"""Propagation across a network, and what it reveals about the network.

Chapter 5.4-5.5 of the thesis.  Propagation is not treated as an application
here: how fast something crosses a network is a measurement *of* the network,
and running the process from every node in turn is how the thesis builds its
own centrality, its own cluster concept and its immunisation strategies.
"""

from __future__ import annotations

from .immunity import (
    CRITERION_ORIGIN,
    IMMUNISATION_CRITERIA,
    THESIS_BEST_ORDER_BY_NETWORK,
    THESIS_SCALE_FREE_100,
    ImmunisationSweep,
    ThesisImmunisationResult,
    beta_factor,
    compare_immunisation_criteria,
    compare_orders,
    immunisation_ranking,
    immunisation_sweep,
    nodes_to_extinction,
)
from .infection import (
    PropagationRun,
    alpha_max_size_law,
    infected_fraction_curve,
    mean_propagation_speed,
    propagate,
    propagation_speed,
    propagation_time,
)
from .place_infection import (
    THESIS_CONTAGION_PCT,
    THESIS_PLACE_COUNTS,
    City,
    PlaceInfectionRun,
    build_city,
    place_infection_curves,
    propagate_places,
)
from .propagation_clusters import (
    ClusterOrders,
    cluster_coefficient_order1,
    cluster_order_matrix,
    cluster_orders,
    infection_levels,
    local_cluster_coefficient_order1,
    nodes_in_largest_clusters,
)

__all__ = [
    "PropagationRun",
    "propagate",
    "infected_fraction_curve",
    "propagation_speed",
    "mean_propagation_speed",
    "propagation_time",
    "alpha_max_size_law",
    # 5.4.3 the place model
    "THESIS_PLACE_COUNTS",
    "THESIS_CONTAGION_PCT",
    "City",
    "build_city",
    "PlaceInfectionRun",
    "propagate_places",
    "place_infection_curves",
    "ClusterOrders",
    "infection_levels",
    "cluster_orders",
    "cluster_coefficient_order1",
    "local_cluster_coefficient_order1",
    "cluster_order_matrix",
    "nodes_in_largest_clusters",
    # 5.5.2 immunisation
    "IMMUNISATION_CRITERIA",
    "CRITERION_ORIGIN",
    "immunisation_ranking",
    "ImmunisationSweep",
    "immunisation_sweep",
    "nodes_to_extinction",
    "beta_factor",
    "compare_orders",
    "compare_immunisation_criteria",
    "ThesisImmunisationResult",
    "THESIS_SCALE_FREE_100",
    "THESIS_BEST_ORDER_BY_NETWORK",
]
