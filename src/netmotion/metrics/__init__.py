"""Network characteristics.

Chapter 3 of the thesis defines three: how far apart nodes are, how many links
they have, and how tightly their neighbourhoods are knit.  Chapter 5 adds two
more of its own -- a higher-order degree and a centrality built out of
propagation time -- which live in :mod:`netmotion.metrics.higher_order_degree`
and :mod:`netmotion.metrics.propagation_centrality`.
"""

from __future__ import annotations

from .clustering import clustering_coefficient, local_clustering, neighbour_links
from .degree import (
    degree_counts,
    degree_distribution,
    degrees,
    mean_degree,
    power_law_exponent,
)
from .distance import (
    all_pairs_distances,
    connected_components,
    diameter,
    efficiency,
    is_connected,
    largest_component,
    mean_distance,
    shortest_path_lengths,
)
from .higher_order_degree import (
    central_node,
    k_order,
    k_order_all,
    k_order_profile,
    k_order_ranking,
    saturation_order,
)
from .propagation_centrality import (
    CENTRALITY_CRITERIA,
    Criterion,
    NodeRanking,
    PeelingResult,
    central_node_by_removal,
    central_nodes_by_min_time,
    compare_centrality_criteria,
    degree_order_n_criterion,
    estrada_hatano_network,
    min_time_scores,
    node_removal_scores,
    peel_by_minimum_degree,
    rank_nodes,
)

__all__ = [
    # distance
    "shortest_path_lengths",
    "all_pairs_distances",
    "mean_distance",
    "diameter",
    "efficiency",
    "connected_components",
    "largest_component",
    "is_connected",
    # degree
    "degrees",
    "mean_degree",
    "degree_distribution",
    "degree_counts",
    "power_law_exponent",
    # clustering
    "local_clustering",
    "clustering_coefficient",
    "neighbour_links",
    # higher-order degree (chapter 5)
    "k_order",
    "k_order_profile",
    "k_order_all",
    "saturation_order",
    "central_node",
    "k_order_ranking",
    # vulnerability / centrality criteria (chapter 5.5.1)
    "estrada_hatano_network",
    "PeelingResult",
    "peel_by_minimum_degree",
    "node_removal_scores",
    "central_node_by_removal",
    "min_time_scores",
    "central_nodes_by_min_time",
    "NodeRanking",
    "Criterion",
    "CENTRALITY_CRITERIA",
    "degree_order_n_criterion",
    "rank_nodes",
    "compare_centrality_criteria",
]
