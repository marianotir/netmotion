"""Networks in motion, animated: growth, infection, opinions and bank cascades.

Every ``animate_*`` function returns a :class:`matplotlib.animation.FuncAnimation`.  In a Jupyter
notebook, :func:`to_html` turns it into a player with a play button; :func:`save_animation` writes
it to a GIF, an MP4 or a stand-alone HTML file.

The frames are computed first, by the ``*_frames`` functions, as plain arrays with one row per
frame, so they can be inspected, tested or drawn some other way.  :func:`animate_frames` is the
engine underneath: give it a colour for every node in every frame, and optionally a size and a
link colour, and it animates any process on a network.

The colours are the book's: healthy nodes green and infected ones red (Section 5.4); opinions
:math:`+1` blue and :math:`-1` red (Sections 5.1-5.3); banks brown when they owe more than they
lend, green when they lend more than they owe, purple when the two are equal, and red once fallen
(Section 5.6.4).

Examples
--------
An infection crossing a small-world network, from node 0:

>>> from netmotion.networks import watts_strogatz
>>> g = watts_strogatz(40, 4, 0.1, rng=1)
>>> anim = animate_propagation(g, 0, layout="circular")
>>> anim.frame_count
7
>>> close(anim)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from ..graph import Network
from ..rng import SeedLike
from .draw import circular_layout, spring_layout
from .style import INK, SERIES

if TYPE_CHECKING:  # pragma: no cover
    from ..dynamics import HopfieldRun, OpinionRun
    from ..finance import CascadeRun, InterbankNetwork
    from ..propagation import PropagationRun

__all__ = [
    "community_layout",
    "growth_frames",
    "propagation_frames",
    "cascade_frames",
    "animate_frames",
    "animate_growth",
    "animate_propagation",
    "animate_states",
    "animate_cascade",
    "to_html",
    "save_animation",
    "close",
]

#: The book's colours for the processes it animates.
HEALTHY, INFECTED, JUST_INFECTED, IMMUNE = SERIES[5], SERIES[7], SERIES[1], INK["muted"]
PLUS, MINUS, NEUTRAL = SERIES[0], SERIES[7], INK["muted"]
OWES, LENDS, BALANCED, FALLEN = "#8c5a2b", "#1baf7a", "#8e44ad", SERIES[7]
LINK, LINK_HOT, LINK_EXPOSED, LINK_STANDING = "#bcd8f6", SERIES[7], SERIES[0], "#b9dfc9"
NODE, NEW_NODE = SERIES[0], SERIES[1]


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------


def community_layout(community: np.ndarray | list[int], *, radius: float = 1.0) -> np.ndarray:
    """``(N, 2)`` coordinates with each community gathered on a small circle of its own.

    The communities sit on a large circle of the given *radius*; the members of each sit on a
    small circle around its centre, so the countries of an interbank network, or the communities
    of Section 5.6, read as separate groups with the links between them in the middle.

    Examples
    --------
    >>> pos = community_layout([0, 0, 0, 1, 1, 1])
    >>> pos.shape
    (6, 2)
    >>> bool(np.linalg.norm(pos[0] - pos[1]) < np.linalg.norm(pos[0] - pos[3]))
    True
    """
    community = np.asarray(community)
    groups = np.unique(community)
    centres = radius * circular_layout(len(groups)) if len(groups) > 1 else np.zeros((1, 2))
    largest = max(int(np.sum(community == g)) for g in groups)
    gap = np.sin(np.pi / max(len(groups), 2)) * radius
    pos = np.zeros((community.size, 2))
    for g, centre in zip(groups, centres, strict=True):
        members = np.flatnonzero(community == g)
        small = 0.8 * gap * np.sqrt(members.size / largest)
        pos[members] = centre + small * circular_layout(members.size)
    return pos


def _layout(network: Network, layout: np.ndarray | str, rng: SeedLike) -> np.ndarray:
    if isinstance(layout, str):
        if layout == "circular":
            return circular_layout(network.n)
        if layout == "spring":
            return spring_layout(network, rng=rng)
        raise ValueError(f"layout must be 'circular', 'spring' or an (N, 2) array, got {layout!r}")
    pos = np.asarray(layout, dtype=float)
    if pos.shape != (network.n, 2):
        raise ValueError(f"layout must have shape ({network.n}, 2), got {pos.shape}")
    return pos


# ---------------------------------------------------------------------------
# Frames: plain arrays, one row per frame
# ---------------------------------------------------------------------------


def growth_frames(n_nodes: int, *, initial_nodes: int = 1) -> np.ndarray:
    """Which nodes exist in each frame of a growth, taking nodes to arrive in index order.

    Frame 0 holds the first *initial_nodes* nodes (the seed network); each later frame adds one node.
    The generators of the package number their nodes in order of arrival, so for a grown network
    this is its history.

    Examples
    --------
    >>> growth_frames(4, initial_nodes=2).astype(int).tolist()
    [[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]
    """
    start = initial_nodes
    n = n_nodes
    start = max(1, min(start, n))
    return np.arange(n)[None, :] < np.arange(start, n + 1)[:, None]


def propagation_frames(run: PropagationRun) -> np.ndarray:
    """The state of every node at each instant of an infection.

    ``0`` healthy, ``1`` infected at an earlier instant, ``2`` infected at this instant.  Frame
    *t* is the instant :math:`t + 1` of the thesis, whose seed is infected at :math:`t = 1`.

    Examples
    --------
    >>> from netmotion import Network
    >>> from netmotion.propagation import propagate
    >>> path = Network.from_edges(3, [(0, 1), (1, 2)])
    >>> propagation_frames(propagate(path, 0)).tolist()
    [[2, 0, 0], [1, 2, 0], [1, 1, 2]]
    """
    at = np.asarray(run.infected_at, dtype=float)
    instants = np.arange(1, int(np.nanmax(np.where(np.isfinite(at), at, 1))) + 1)
    frames = np.zeros((instants.size, at.size), dtype=int)
    for k, t in enumerate(instants):
        frames[k, at < t] = 1
        frames[k, at == t] = 2
    return frames


def cascade_frames(run: CascadeRun) -> np.ndarray:
    """Which banks have fallen at each step of a cascade, from its recorded history."""
    return np.array([np.asarray(c, dtype=bool) for c in run.crashed_history])


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


def _sizes(value, frames: int, n: int) -> np.ndarray:
    """One size, a size per node, or a size per node and frame, as a (frames, n) array."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full((frames, n), float(arr))
    if arr.ndim == 1:
        return np.broadcast_to(arr, (frames, n))
    return arr


def animate_frames(
    network: Network,
    colors,
    *,
    layout: np.ndarray | str = "spring",
    sizes=None,
    edge_colors=None,
    titles: list[str] | None = None,
    interval: int = 500,
    figsize: tuple[float, float] = (5.0, 5.0),
    dpi: int = 80,
    rng: SeedLike = 0,
) -> FuncAnimation:
    """Animate any process on a network, given what each frame looks like.

    Parameters
    ----------
    colors:
        A ``(frames, N)`` array of node colours, one row per frame.  Its length sets the number
        of frames.
    sizes:
        One size, a size per node, or a ``(frames, N)`` array.  A size of 0 hides a node.
    edge_colors:
        ``None`` for pale links throughout, or a ``(frames, L)`` array with a colour for each link
        of ``network.edges()`` in each frame; ``"none"`` hides a link.
    titles:
        A title for each frame.
    layout:
        ``"spring"`` (default), ``"circular"`` or an ``(N, 2)`` array.
    """
    colors = np.asarray(colors, dtype=object)
    frames, n = colors.shape
    if n != network.n:
        raise ValueError(f"colors must have one column per node ({network.n}), got {n}")
    pos = _layout(network, layout, rng)
    if sizes is None:
        sizes = max(8.0, 300.0 / max(n, 1) ** 0.55)
    sizes = _sizes(sizes, frames, n)
    edges = [(i, j) for i, j, _ in network.edges()]
    segments = np.array([(pos[i], pos[j]) for i, j in edges]).reshape(-1, 2, 2)
    if edge_colors is None:
        edge_colors = np.full((frames, len(edges)), LINK, dtype=object)
    edge_colors = np.asarray(edge_colors, dtype=object)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal")
    ax.set_axis_off()
    span = float(np.ptp(pos)) or 1.0
    pad = 0.08 * span
    ax.set_xlim(pos[:, 0].min() - pad, pos[:, 0].max() + pad)
    ax.set_ylim(pos[:, 1].min() - pad, pos[:, 1].max() + pad)
    # Hairlines once the links are many, so that the nodes stay readable through them.
    width = 0.8 if len(edges) < 150 else 0.45
    lines = LineCollection(segments, colors=list(edge_colors[0]) or LINK, linewidths=width, zorder=1)
    ax.add_collection(lines)
    dots = ax.scatter(pos[:, 0], pos[:, 1], s=sizes[0], c=list(colors[0]), zorder=2,
                      edgecolors="white", linewidths=0.5)
    title = ax.set_title(titles[0] if titles else "", fontsize=10, color=INK["primary"])

    def update(k: int):
        if len(edges):
            lines.set_color(list(edge_colors[k]))
        dots.set_facecolor(list(colors[k]))
        dots.set_sizes(np.asarray(sizes[k], dtype=float))
        if titles:
            title.set_text(titles[k])
        return lines, dots, title

    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    anim.frame_count = frames
    return anim


# ---------------------------------------------------------------------------
# The four processes of the book
# ---------------------------------------------------------------------------


def animate_growth(
    network: Network,
    *,
    initial_nodes: int = 1,
    layout: np.ndarray | str = "spring",
    interval: int = 150,
    rng: SeedLike = 0,
    **kwargs,
) -> FuncAnimation:
    """A network growing node by node, each node sized by its degree at that moment.

    Nodes arrive in index order, which is the order in which the generators of the package create
    them (Barabasi-Albert, the networks of communities, the interbank growth by strategies).  The
    newest node is drawn orange; a link appears when both of its ends exist.

    Parameters
    ----------
    initial_nodes:
        The number of nodes of the seed network, shown together in the first frame.
    """
    start = initial_nodes
    present = growth_frames(network.n, initial_nodes=start)
    edges = np.array([(i, j) for i, j, _ in network.edges()], dtype=int).reshape(-1, 2)
    newest = np.minimum(np.arange(present.shape[0]) + max(start, 1) - 1, network.n - 1)
    colors = np.where(present, NODE, "none").astype(object)
    colors[np.arange(present.shape[0]), newest] = NEW_NODE
    live = present[:, edges[:, 0]] & present[:, edges[:, 1]] if len(edges) else np.zeros((present.shape[0], 0), bool)
    degree = np.zeros(present.shape, dtype=float)
    for k in range(present.shape[0]):
        np.add.at(degree[k], edges[live[k]].ravel(), 1.0)
    base = max(6.0, 200.0 / max(network.n, 1) ** 0.55)
    top = max(float(degree.max()), 1.0)
    sizes = np.where(present, base * (0.6 + 5.0 * degree / top), 0.0)
    edge_colors = np.where(live, SERIES[0], "none").astype(object)
    titles = [f"{int(p.sum())} nodes, {int(lv.sum())} links" for p, lv in zip(present, live, strict=True)]
    return animate_frames(network, colors, layout=layout, sizes=sizes, edge_colors=edge_colors,
                          titles=titles, interval=interval, rng=rng, **kwargs)


def animate_propagation(
    network: Network,
    run: PropagationRun | int,
    *,
    layout: np.ndarray | str = "spring",
    interval: int = 700,
    rng: SeedLike = 0,
    **kwargs,
) -> FuncAnimation:
    """An infection spreading instant by instant (Section 5.4).

    *run* is a :class:`~netmotion.propagation.PropagationRun`, or a seed node, from which the
    infection of Section 5.4.1 is run here.  Healthy nodes are green, infected ones red, those
    infected at this very instant orange, immune ones grey; a link turns red once both of its
    ends are infected.
    """
    if not hasattr(run, "infected_at"):
        from ..propagation import propagate

        run = propagate(network, int(run))
    states = propagation_frames(run)
    palette = np.array([HEALTHY, INFECTED, JUST_INFECTED], dtype=object)
    colors = palette[states]
    immune = np.asarray(getattr(run, "immune", []), dtype=int)
    if immune.size:
        colors[:, immune] = IMMUNE
    edges = np.array([(i, j) for i, j, _ in network.edges()], dtype=int).reshape(-1, 2)
    hot = (states[:, edges[:, 0]] > 0) & (states[:, edges[:, 1]] > 0)
    edge_colors = np.where(hot, LINK_HOT, LINK).astype(object)
    total = network.n
    titles = [f"t = {k + 1}: {int((s > 0).sum())} of {total} infected" for k, s in enumerate(states)]
    return animate_frames(network, colors, layout=layout, edge_colors=edge_colors, titles=titles,
                          interval=interval, rng=rng, **kwargs)


def animate_states(
    network: Network,
    run: HopfieldRun | OpinionRun | np.ndarray,
    *,
    layout: np.ndarray | str = "circular",
    interval: int = 400,
    rng: SeedLike = 0,
    **kwargs,
) -> FuncAnimation:
    """Opinions changing step by step (Sections 5.1-5.3): :math:`+1` blue, :math:`-1` red.

    *run* is a :class:`~netmotion.dynamics.HopfieldRun` or
    :class:`~netmotion.dynamics.OpinionRun`, or a ``(steps, N)`` array of states.
    """
    states = np.asarray(getattr(run, "states", run))
    colors = np.where(states > 0, PLUS, np.where(states < 0, MINUS, NEUTRAL)).astype(object)
    titles = [f"t = {t}: {int((s > 0).sum())} at +1, {int((s < 0).sum())} at -1" for t, s in enumerate(states)]
    return animate_frames(network, colors, layout=layout, titles=titles, interval=interval,
                          rng=rng, **kwargs)


def animate_cascade(
    banks: InterbankNetwork,
    run: CascadeRun,
    *,
    layout: np.ndarray | str = "countries",
    interval: int = 800,
    rng: SeedLike = 0,
    **kwargs,
) -> FuncAnimation:
    """A bank failure travelling through an interbank network (Section 5.6.4).

    Standing banks keep the colour of their balance -- brown if they owe more than they lend,
    green if they lend more than they owe, purple if the two are equal -- and turn red when they
    fall.  Each bank is sized by the size of its balance.  A link is red when both of its banks
    have fallen, blue when one has, and pale green while both stand.

    Parameters
    ----------
    banks:
        The :class:`~netmotion.finance.InterbankNetwork` the cascade ran on.
    run:
        The :class:`~netmotion.finance.CascadeRun`, from
        :func:`~netmotion.finance.four_state_cascade` or :func:`~netmotion.finance.two_state_cascade`.
    layout:
        ``"countries"`` (default) groups the banks by country with :func:`community_layout`;
        ``"spring"``, ``"circular"`` or an array also work.
    """
    network = banks.network
    if isinstance(layout, str) and layout == "countries":
        layout = community_layout(banks.country)
    fallen = cascade_frames(run)
    balance = np.asarray(run.initial_capital, dtype=float)
    standing = np.where(balance > 0, LENDS, np.where(balance < 0, OWES, BALANCED)).astype(object)
    colors = np.where(fallen, FALLEN, standing[None, :]).astype(object)
    scale = np.abs(balance) / max(float(np.abs(balance).max()), 1.0)
    sizes = 10.0 + 260.0 * np.sqrt(scale)
    edges = np.array([(i, j) for i, j, _ in network.edges()], dtype=int).reshape(-1, 2)
    down = fallen[:, edges[:, 0]].astype(int) + fallen[:, edges[:, 1]].astype(int)
    edge_colors = np.select([down == 2, down == 1], [LINK_HOT, LINK_EXPOSED], LINK_STANDING).astype(object)
    total = network.n
    titles = [f"step {t}: {int(f.sum())} of {total} banks fallen ({100 * f.mean():.0f}%)"
              for t, f in enumerate(fallen)]
    return animate_frames(network, colors, layout=layout, sizes=sizes, edge_colors=edge_colors,
                          titles=titles, interval=interval, rng=rng, **kwargs)


# ---------------------------------------------------------------------------
# Showing and saving
# ---------------------------------------------------------------------------


def close(anim: FuncAnimation) -> None:
    """Close the figure behind *anim*, once it has been shown or saved (or is no longer wanted)."""
    anim._draw_was_started = True  # closing on purpose: no "deleted without rendering" warning
    plt.close(anim._fig)


def to_html(anim: FuncAnimation, *, default_mode: str = "once"):
    """The animation as a player with a play button, for a Jupyter notebook.

    Returns an :class:`IPython.display.HTML` object; put it last in a cell to show it.  The
    figure behind the animation is closed, so the notebook shows the player only.  Where
    IPython is not installed (a plain script), the player's HTML is returned as a string.
    """
    html = anim.to_jshtml(default_mode=default_mode)
    close(anim)
    try:
        from IPython.display import HTML
    except ImportError:
        return html
    return HTML(html)


def save_animation(anim: FuncAnimation, path: str | Path, *, fps: float = 2.0, dpi: int | None = None) -> Path:
    """Write *anim* to *path*: ``.gif`` (Pillow), ``.mp4`` (needs ffmpeg) or ``.html``."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".html":
        path.write_text(anim.to_jshtml(fps=fps), encoding="utf-8")
    elif suffix == ".gif":
        anim.save(path, writer="pillow", fps=fps, dpi=dpi)
    elif suffix == ".mp4":
        anim.save(path, writer="ffmpeg", fps=fps, dpi=dpi)
    else:
        raise ValueError(f"save as .gif, .mp4 or .html, got {suffix!r}")
    return path
