"""Figure style, network drawing and animation for the book and the repository."""

from __future__ import annotations

from .animation import (
    animate_cascade,
    animate_frames,
    animate_growth,
    animate_propagation,
    animate_states,
    cascade_frames,
    community_layout,
    growth_frames,
    propagation_frames,
    save_animation,
    to_html,
)
from .draw import circular_layout, draw_adjacency, draw_network, spring_layout
from .style import (
    INK,
    SEQUENTIAL,
    SERIES,
    annotate_source,
    label_line_ends,
    save_figure,
    series_color,
    use_style,
)

__all__ = [
    "SERIES",
    "SEQUENTIAL",
    "INK",
    "use_style",
    "series_color",
    "save_figure",
    "label_line_ends",
    "annotate_source",
    "draw_network",
    "draw_adjacency",
    "circular_layout",
    "spring_layout",
    "community_layout",
    "animate_frames",
    "animate_growth",
    "animate_propagation",
    "animate_states",
    "animate_cascade",
    "growth_frames",
    "propagation_frames",
    "cascade_frames",
    "to_html",
    "save_animation",
]
