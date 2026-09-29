"""Drawing networks.

The thesis draws its networks the way Maple's ``pointplot`` did in 2011: nodes
on a circle, links as straight chords.  That layout is not decoration -- for the
ring-based models of section 4.2 it *is* the model, and a force-directed layout
would hide the very structure the figures are about.  So :func:`draw_network`
keeps the circular layout as its default and offers a spring layout for the
grown networks of section 4.4, where there is no ring to respect.

What changes from 2011 is everything else: hairline links, filled nodes sized
by degree, no bounding box, and a palette that stays legible in print.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from ..graph import Network
from ..provenance import Origin, sourced
from ..rng import SeedLike, as_rng
from .style import INK, SEQUENTIAL, SERIES

__all__ = ["circular_layout", "spring_layout", "draw_network", "draw_adjacency"]


@sourced(
    Origin.THESIS_2011,
    pdf_pages="12",
    section="4.1.2 Algoritmo",
    note=(
        "Nodes placed on the unit circle at angles 2*pi*i/N, which is exactly "
        "what the printed Maple does: 'l := i*2*Pi/dim ; s[i] := [cos(l), "
        "sin(l)]'."
    ),
)
def circular_layout(n_nodes: int) -> np.ndarray:
    """``(n, 2)`` coordinates on the unit circle, as the thesis places them."""
    n = n_nodes
    angles = 2.0 * np.pi * np.arange(1, n + 1) / n
    return np.column_stack([np.cos(angles), np.sin(angles)])


@sourced(
    Origin.REPRODUCTION,
    pdf_pages="30-31",
    section="4.4.3 Caracteristicas",
    note=(
        "A force-directed layout for the grown networks of section 4.4, where "
        "no ring geometry needs preserving and a circular layout would hide the "
        "hubs. The 2011 figures used Maple's own placement."
    ),
)
def spring_layout(network: Network, *, rng: SeedLike = None, iterations: int = 60) -> np.ndarray:
    """Fruchterman-Reingold coordinates, via NetworkX (layout only)."""
    import networkx as nx

    seed = int(as_rng(rng).integers(0, 2**31 - 1))
    pos = nx.spring_layout(network.to_networkx(), iterations=iterations, seed=seed, weight=None)
    return np.array([pos[i] for i in range(network.n)])


def draw_network(
    network: Network,
    ax: Axes,
    *,
    layout: np.ndarray | str = "circular",
    node_color: str | np.ndarray = SERIES[0],
    node_size: float | np.ndarray | None = None,
    edge_color: str = INK["muted"],
    edge_width: float = 0.7,
    edge_alpha: float = 0.85,
    scale_by_degree: bool = False,
    title: str | None = None,
    rng: SeedLike = None,
) -> np.ndarray:
    """Draw *network* on *ax* and return the coordinates used.

    Parameters
    ----------
    layout:
        ``"circular"`` (default), ``"spring"``, or an ``(N, 2)`` array.  A
        network carrying ``.positions`` -- the spatial scale-free variant --
        can pass those in directly.
    scale_by_degree:
        Size each node by its degree.  For the scale-free networks this is the
        point of the figure, since the hubs are the whole story.
    node_color:
        A single colour, or a length-``N`` array of values that will be mapped
        through the sequential ramp (useful for infection times later).
    """
    if isinstance(layout, str):
        if layout == "circular":
            pos = circular_layout(network.n)
        elif layout == "spring":
            pos = spring_layout(network, rng=rng)
        else:
            raise ValueError(f"layout must be 'circular', 'spring' or an array, got {layout!r}")
    else:
        pos = np.asarray(layout, dtype=float)
        if pos.shape != (network.n, 2):
            raise ValueError(f"layout must have shape ({network.n}, 2), got {pos.shape}")

    segments = [(pos[i], pos[j]) for i, j, _ in network.edges()]
    if segments:
        ax.add_collection(
            LineCollection(
                segments,
                colors=edge_color,
                linewidths=edge_width,
                alpha=edge_alpha,
                zorder=1,
            )
        )

    k = network.degrees()
    if node_size is None:
        base = max(6.0, 260.0 / max(network.n, 1) ** 0.55)
        if scale_by_degree:
            # Area proportional to degree, with a floor, so a hub of degree 20
            # in a mean-degree-4 network reads as one rather than as a slightly
            # larger dot.
            span = max(int(k.max()), 1)
            sizes = base * (0.55 + 5.5 * (k / span))
        else:
            sizes = base
    else:
        sizes = node_size

    colors = node_color
    # A 1-D array of *numbers* is a value per node to map through the sequential
    # ramp; a 1-D array of strings is already one colour per node.
    if (
        isinstance(node_color, np.ndarray)
        and node_color.ndim == 1
        and np.issubdtype(node_color.dtype, np.number)
    ):
        from matplotlib.colors import LinearSegmentedColormap, Normalize

        cmap = LinearSegmentedColormap.from_list("nm-seq", SEQUENTIAL)
        norm = Normalize(vmin=float(np.nanmin(node_color)), vmax=float(np.nanmax(node_color)))
        colors = cmap(norm(node_color))

    ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c=colors, zorder=2, linewidths=0)

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.margins(0.06)
    if title:
        ax.set_title(title, fontsize=9)
    return pos


def draw_adjacency(network: Network, ax: Axes, *, title: str | None = None):
    """Show the adjacency matrix itself -- the object chapter 2 is about."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("nm-seq", ["#ffffff", SEQUENTIAL[5]])
    im = ax.imshow(network.adjacency, cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
    ax.set_xlabel("$j$")
    ax.set_ylabel("$i$")
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=9)
    return im
