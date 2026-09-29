"""Chart primitives, so that every plot of a given kind looks the same.

Three shapes account for most of the plots in this book, and each was being
drawn a little differently every time it appeared.  They are collected here so
that the decisions are made once.

The decisions themselves are worth stating, because they are the difference
between a figure that reads and one that merely contains the data.

**Colour encodes something or it is not used.**  A chart of seven strategies
drawn in seven hues invites the reader to look for meaning in the hues, and
there is none: the categories are not ordered and the colours are arbitrary.
One hue, with the category under discussion picked out, says what the figure
means.  Colour is spent only where it carries identity that the reader needs.

**A per-node series is not a bar chart.**  Sixty bars separated by white
slivers is a barcode: the eye reads the gaps as much as the marks, and the
shape of the distribution disappears. Thin stems on a common baseline show the
same numbers and let the profile through.

**A histogram's bars touch.**  Gaps between them imply the categories are
separate when the variable is continuous. The bars carry a hairline of the page
colour instead, which separates them without opening a channel.

**A reference line is annotated at its end, not across its middle.**  Writing
the mean on top of the line that shows it makes both harder to read.

**A swept curve is an ensemble, not a run.**  At the network sizes of this
book a single realisation per parameter value travels by about as much between
neighbouring points as the trend does across the sweep, so the quantity drawn
is the mean over enough realisations to resolve it.  Where the spread between
them is itself worth knowing, :func:`mean_with_band` draws it as a light band
of the same hue behind the line.
"""

from __future__ import annotations

import warnings

import matplotlib.patheffects as pe
import numpy as np
from matplotlib.axes import Axes

from .style import INK, SERIES

__all__ = [
    "mean_with_band",
    "per_node",
    "histogram",
    "panel_letter",
    "panel_title",
    "panel_rows",
    "ranked_bars",
    "reference_line",
]


def panel_rows(fig, count: int, *, per_row: int = 3, top: float = 0.0,
               gap: float = 0.12, title: float = 0.3, ratio: float = 1.0) -> list[Axes]:
    """*count* panels in rows of *per_row*, a short last row centred.

    A sequence of five network drawings in a single row prints each network
    about an inch across, too small to follow; three above and two below,
    centred, prints each at twice that.  Every panel gets the same box, square
    unless *ratio* (height over width) makes it shorter for a network wider
    than tall, placed by hand (sizes in inches), so the snapshots of one
    network print at one scale.  A constrained-layout grid of staggered
    two-column spans gave each panel a different width.  *top* is kept free
    above the first row for a key set at the top of the figure.
    """
    fig.set_layout_engine("none")
    width, height = fig.get_size_inches()
    side = (width - (per_row + 1) * gap) / per_row
    tall = side * ratio
    axes = []
    for k in range(count):
        row, col = divmod(k, per_row)
        in_row = min(per_row, count - row * per_row)
        x0 = gap + (per_row - in_row) * (side + gap) / 2 + col * (side + gap)
        y1 = height - top - title - row * (tall + title)
        axes.append(fig.add_axes((x0 / width, (y1 - tall) / height, side / width, tall / height)))
    return axes


def panel_letter(ax: Axes, letter: str) -> None:
    """The a/b/c the captions cite, printed in the corner of a panel.

    The captions of this book name their panels -- *"Fig. 57.a shows the
    calculation of the degree k of order 2 of node 1"* -- so every panel of a
    figure whose caption does that carries its letter.  It is set small, in the
    muted ink and outside the frame, where it cannot land on the data.
    """
    ax.annotate(
        letter,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(-3, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK["muted"],
    )


def panel_title(ax: Axes, text: str, *, fontsize: float = 9.5, **kwargs) -> None:
    """Title one panel of a multi-panel figure, wrapped to the panel's width.

    Matplotlib centres a title on its axes and lets it run as wide as it likes.
    On a single plot that is harmless.  On a row of panels the titles grow
    towards each other until they meet, and what prints is two sentences fused
    at the seam -- "Option 2: pick at rando\\ **O**\\ ption 3: pick the nearest".
    Every such collision in this book came from that one habit, and shortening
    them one at a time only fixes the ones somebody happened to look at.

    So the width is measured and the text is folded to it.  The estimate of how
    many characters fit is deliberately conservative -- the average glyph in
    this face runs a little over half its point size, and a title is better
    broken a word early than a word late.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # A little is kept back so that two titles on neighbouring panels have a
    # gutter between them rather than meeting exactly.
    available = ax.get_window_extent().width * 0.94

    def too_wide(candidate: str) -> bool:
        artist = ax.set_title(candidate, fontsize=fontsize, **kwargs)
        return artist.get_window_extent(renderer).width > available

    if too_wide(text):
        # Guessing the width from a character count is what let these titles
        # through in the first place: the face is semibold, mathtext sets its
        # own widths, and neither is the average the guess assumed.  So each
        # candidate is measured as it will actually be drawn.
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}" if current else word
            if current and too_wide(candidate):
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        text = "\n".join(lines)

    ax.set_title(text, fontsize=fontsize, **kwargs)


def mean_with_band(
    ax: Axes,
    x,
    samples,
    *,
    color: str = SERIES[0],
    lw: float = 1.8,
    alpha: float = 0.15,
    quantiles: tuple[float, float] = (0.25, 0.75),
    band: bool = True,
    zorder: float = 2.0,
    **line_kwargs,
) -> np.ndarray:
    """A swept curve as the mean of its realisations, over their spread.

    Parameters
    ----------
    x:
        The swept parameter, one value per point.
    samples:
        Measurements, shaped ``(realisations, len(x))``.
    quantiles:
        The band's lower and upper quantile.  The default is the interquartile
        range, which is what the spread between realisations usually means
        here: half the runs fall inside it, and it does not chase the one run
        that came out strange.
    band:
        Set ``False`` where the panel is already crowded -- several series on
        one pair of axes -- and the mean alone is what the reader needs.

    Returns
    -------
    ndarray
        The mean curve, so the caller can label it or measure from it.

    Notes
    -----
    The band takes the line's own hue at low opacity rather than a second
    colour: it is the same quantity, drawn at less certainty, and a new hue
    would read as a new series.  It is drawn without an edge, and below the
    line, so the mean stays the thing the eye follows.
    """
    x = np.asarray(x, dtype=float)
    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    if samples.shape[1] != x.size:
        raise ValueError(f"samples must be shaped (realisations, {x.size}), got {samples.shape}")

    # A column that is entirely NaN is a point the caller deliberately did not
    # measure -- a density past its cap, a size out of regime -- and it should
    # come out as a gap in the line, not as a warning on the console.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(samples, axis=0)
        if band and samples.shape[0] > 1:
            lo = np.nanquantile(samples, quantiles[0], axis=0)
            hi = np.nanquantile(samples, quantiles[1], axis=0)

    if band and samples.shape[0] > 1:
        ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0, zorder=zorder - 1.0)
    ax.plot(x, mean, color=color, lw=lw, zorder=zorder, **line_kwargs)
    return mean


def reference_line(
    ax: Axes,
    value: float,
    label: str,
    *,
    axis: str = "y",
    color: str | None = None,
    side: str = "right",
    label_align: str = "center",
) -> None:
    """A mean or threshold, drawn faintly and labelled clear of the data.

    The line is a hairline in the muted ink rather than a heavy dashed rule:
    it is context, not a series, and it should not compete with the marks it is
    there to be compared against.

    The label sits at the line's own height, just above it, so that it is
    always read as belonging to that line and two reference lines on the same
    axes never stack on one spot.  A line within a tenth of the top of the
    axes takes its label underneath instead, where there is room for it.

    label_align:
        Horizontal anchor of the axis="x" label relative to ``value``
        (matplotlib's ``ha``). Centred by default; a panel whose value sits
        close to one of its own axes' edges, next to a neighbouring panel's
        title, can pass "left" or "right" to pull the label clear of it.
    """
    color = color or INK["muted"]
    drawer = ax.axhline if axis == "y" else ax.axvline
    drawer(value, color=color, lw=0.9, ls=(0, (4, 3)), zorder=1)

    if axis == "y":
        # At the line's own height rather than in the corner of the axes: a
        # label in the corner floats away from the rule it names as soon as the
        # data moves, and two of them on one axes land on each other.
        lo, hi = ax.get_ylim()
        fraction = (value - lo) / (hi - lo) if hi != lo else 0.5
        above = fraction < 0.90
        ax.annotate(
            label,
            xy=(0.98 if side == "right" else 0.02, value),
            xycoords=ax.get_yaxis_transform(),
            xytext=(0, 3 if above else -3),
            textcoords="offset points",
            ha="right" if side == "right" else "left",
            va="bottom" if above else "top",
            fontsize=7.5,
            color=INK["secondary"],
            # A stroke around the glyphs, never a panel behind them: the height
            # of the line is often the height of the data as well, and a filled
            # box there erases a band out of the stems it is sitting on.
            path_effects=[pe.withStroke(linewidth=2.4, foreground=INK["surface"])],
        )
    else:
        ax.annotate(
            label,
            xy=(value, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 4),
            textcoords="offset points",
            ha=label_align,
            va="bottom",
            fontsize=7.5,
            color=INK["secondary"],
            annotation_clip=False,
        )


def per_node(
    ax: Axes,
    values,
    *,
    color: str = SERIES[0],
    mean_label: str | None = None,
    start: int = 1,
    highlight=None,
) -> None:
    """One value per node, against node index.

    Drawn as stems from the baseline: for the sixty- to hundred-node networks
    of this book that keeps the profile legible where bars would not.

    Parameters
    ----------
    start:
        Index of the first node.  Nodes are numbered from one throughout.
    highlight:
        Node indices to pick out in the accent colour, for the occasions when
        one node is the subject of the paragraph.
    """
    values = np.asarray(values, dtype=float)
    x = np.arange(start, start + values.size)

    ax.vlines(x, 0, values, color=color, lw=1.15, alpha=0.9, zorder=2)

    if highlight is not None and len(highlight):
        idx = np.asarray(highlight, dtype=int)
        ax.vlines(x[idx], 0, values[idx], color=SERIES[1], lw=2.0, zorder=3)
        ax.plot(x[idx], values[idx], "o", ms=4.0, color=SERIES[1], zorder=4)

    if mean_label is not None:
        reference_line(ax, float(values.mean()), mean_label)

    ax.set_xlim(start - 1, start + values.size)
    ax.set_ylim(bottom=0)
    ax.margins(y=0.12)


def histogram(
    ax: Axes,
    values,
    *,
    bins=None,
    color: str = SERIES[0],
    density: bool = False,
    label: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """A histogram whose bars touch, separated by a hairline of the page.

    Returns ``(counts, edges)`` so a caller can annotate the peak.
    """
    values = np.asarray(values, dtype=float)
    if bins is None:
        lo, hi = np.floor(values.min()), np.ceil(values.max())
        bins = np.arange(lo - 0.5, hi + 1.5)

    counts, edges = np.histogram(values, bins=bins, density=density)
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    ax.bar(
        centres,
        counts,
        width=widths,
        color=color,
        edgecolor="white",
        linewidth=0.8,
        label=label,
        zorder=2,
    )
    ax.set_ylim(bottom=0)
    return counts, edges


def ranked_bars(
    ax: Axes,
    labels,
    values,
    *,
    highlight: int | None = None,
    color: str = SERIES[0],
    accent: str = SERIES[1],
    horizontal: bool = True,
    value_format: str = "{:.2f}",
) -> None:
    """Categories ranked by value, in one hue, with the values written on.

    One category may be picked out with *highlight*; everything else shares a
    single colour, because the categories are names rather than an ordering and
    a different hue for each would imply a structure that is not there.

    The value is written at the end of each bar, so the axis carrying it can be
    dropped: a bar chart with its numbers on it does not also need a ruler.
    """
    labels = list(labels)
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = values[order]

    colors = [color] * len(values)
    if highlight is not None:
        moved = list(order).index(highlight)
        colors[moved] = accent

    positions = np.arange(len(values))
    span = values.max() - min(0.0, values.min())
    pad = 0.02 * (span or 1.0)

    if horizontal:
        ax.barh(positions, values, height=0.62, color=colors, zorder=2)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        for y, v in zip(positions, values, strict=True):
            ax.annotate(
                value_format.format(v),
                xy=(v, y),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.5,
                color=INK["secondary"],
                bbox=dict(facecolor=INK["surface"], edgecolor="none", pad=0.3),
            )
        ax.set_xlim(0, values.max() + 12 * pad)
        # The ruler goes, because every bar carries its own number -- but the
        # axis *label* stays: it is what says what those numbers count, and
        # ``xaxis.set_visible(False)`` was taking it down with the ticks, so a
        # caller that set one got two bars labelled with a bare figure.
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.grid(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_visible(False)
    else:
        ax.bar(positions, values, width=0.62, color=colors, zorder=2)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        for x, v in zip(positions, values, strict=True):
            ax.annotate(
                value_format.format(v),
                xy=(x, v),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=INK["secondary"],
            )
        ax.set_ylim(0, values.max() + 12 * pad)
        ax.grid(True, axis="y")
