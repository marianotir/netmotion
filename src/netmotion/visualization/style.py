"""The figure style shared by the book, the repository and the notebooks.

The brief asks for *the same experiment and scientific meaning, modern
visualisation* -- explicitly not a pastiche of the 2011 Maple plots.  So the
style here is chosen for a printed scientific book first and a GitHub README
second:

* **Light surface, vector first.**  Figures are written as ``.pdf`` and ``.svg``
  for the book and ``.png`` for the web.  Nothing depends on a font that a
  reader might not have -- the default face is DejaVu Sans, which ships with
  Matplotlib, so a clean clone reproduces byte-comparable output.
* **A validated categorical palette.**  The eight series colours below clear
  colour-vision-deficiency separation, a chroma floor and a lightness band
  against a white page.  They are assigned *in fixed order and never cycled*:
  a ninth series means the figure needs small multiples, not a ninth hue.
* **Recessive chrome.**  Hairline grid, no top/right spines, ticks outside.  The
  data is the darkest thing on the page.
* **Identity is never colour alone.**  Every multi-series figure carries a
  legend, and :func:`label_line_ends` adds direct labels where they fit, which
  is also the required relief for the three lighter hues.

Usage::

    from netmotion.visualization import use_style, save_figure

    use_style()
    fig, ax = plt.subplots()
    ...
    save_figure(fig, "figures/reproduction/ch04_smallworld_lp_cp", meta)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from ..rng import ExperimentMetadata

__all__ = [
    "SERIES",
    "SEQUENTIAL",
    "INK",
    "use_style",
    "series_color",
    "save_figure",
    "fit_to_print_width",
    "TEXT_WIDTH_IN",
    "label_line_ends",
    "annotate_source",
]

#: Categorical series colours, in fixed assignment order.  Validated against a
#: white surface: lightness band, chroma floor, worst adjacent CVD dE 9.1,
#: worst adjacent normal-vision dE 19.6.  Do not re-order and do not extend.
SERIES: tuple[str, ...] = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)

#: Single-hue ramp for continuous magnitude (heatmaps of the adjacency matrix,
#: infection maps).  Light -> dark; never a rainbow.
SEQUENTIAL: tuple[str, ...] = (
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)

#: Chrome and text.
INK = {
    "surface": "#ffffff",
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}

_BASE_RC: dict[str, object] = {
    # figure ------------------------------------------------------------
    "figure.figsize": (6.0, 3.9),  # fits a single column of the book
    "figure.dpi": 120,
    "figure.facecolor": INK["surface"],
    "figure.constrained_layout.use": True,
    "savefig.facecolor": INK["surface"],
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    # text --------------------------------------------------------------
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "mathtext.fontset": "dejavusans",
    "text.color": INK["primary"],
    # axes --------------------------------------------------------------
    "axes.facecolor": INK["surface"],
    "axes.edgecolor": INK["axis"],
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK["secondary"],
    "axes.titlecolor": INK["primary"],
    "axes.titlelocation": "left",
    "axes.titleweight": "semibold",
    "axes.titlepad": 8.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": mpl.cycler(color=list(SERIES)),
    "axes.axisbelow": True,
    # grid --------------------------------------------------------------
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": INK["grid"],
    "grid.linewidth": 0.7,
    "grid.alpha": 1.0,
    # ticks -------------------------------------------------------------
    "xtick.color": INK["muted"],
    "ytick.color": INK["muted"],
    "xtick.labelcolor": INK["secondary"],
    "ytick.labelcolor": INK["secondary"],
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    # marks -------------------------------------------------------------
    "lines.linewidth": 1.8,
    "lines.markersize": 4.5,
    "lines.markeredgewidth": 0.0,
    "lines.solid_capstyle": "round",
    "patch.linewidth": 0.0,
    "scatter.edgecolors": "none",
    # legend ------------------------------------------------------------
    "legend.frameon": False,
    "legend.labelcolor": INK["secondary"],
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.6,
    "legend.columnspacing": 1.4,
    "legend.borderaxespad": 0.2,
    # output ------------------------------------------------------------
    "svg.fonttype": "none",  # keep text as text in the SVG
    "pdf.fonttype": 42,  # TrueType, so the book's PDF has selectable text
    "ps.fonttype": 42,
}


def use_style(**overrides) -> None:
    """Apply the ``netmotion`` figure style globally.

    Call once at the top of a figure script.  ``overrides`` are passed straight
    to ``matplotlib.rcParams`` for the occasional figure that needs a different
    size or a grid on both axes.
    """
    mpl.rcParams.update(_BASE_RC)
    if overrides:
        mpl.rcParams.update(overrides)


def series_color(index: int) -> str:
    """Colour of series *i* (0-based), in fixed order.

    Raises rather than cycling.  A figure with more than eight series is a
    figure that should be small multiples -- silently reusing blue for the
    ninth curve is how a reader ends up mis-reading a plot.
    """
    i = index
    if not 0 <= i < len(SERIES):
        raise IndexError(
            f"series {i} is outside the {len(SERIES)}-colour categorical palette; "
            "use small multiples or fold the tail into an 'other' series"
        )
    return SERIES[i]


def label_line_ends(
    ax: plt.Axes,
    labels: Sequence[str] | None = None,
    *,
    x_offset: float = 1.01,
    fontsize: float | None = None,
) -> None:
    """Write each line's name at its right-hand end, in the line's colour.

    Direct labels beat a legend for four or fewer curves: the eye does not have
    to travel.  The label is drawn in the secondary ink with a colour swatch
    provided by the line itself, so identity never rests on hue alone.
    """
    lines = [ln for ln in ax.get_lines() if ln.get_label() and not ln.get_label().startswith("_")]
    names = list(labels) if labels is not None else [ln.get_label() for ln in lines]
    for line, name in zip(lines, names, strict=False):
        xs, ys = line.get_xdata(), line.get_ydata()
        if len(xs) == 0:
            continue
        ax.annotate(
            name,
            xy=(xs[-1], ys[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            color=line.get_color(),
            fontsize=fontsize or mpl.rcParams["legend.fontsize"],
            va="center",
            ha="left",
            annotation_clip=False,
        )
    ax.set_xlim(right=ax.get_xlim()[1] * x_offset if ax.get_xscale() == "linear" else None)


def _wrap_stamp(text: str, max_chars: int) -> str:
    """Fold the provenance line onto as many lines as the figure is wide.

    The stamp is written before the raster save, and ``savefig`` is set to crop
    to the figure's contents.  A stamp longer than the axes therefore *widens
    the saved image*, and every such figure came out with its plot pushed into
    the left half of a canvas sized by a line of small grey type.  Folding the
    stamp at its own ``|`` separators keeps the picture the shape it was drawn.
    """
    parts = [p.strip() for p in text.split("|")]
    lines: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current} | {part}" if current else part
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def annotate_source(fig: plt.Figure, text: str) -> None:
    """Stamp the provenance line along the bottom of a figure.

    Every reproduction figure says, on its face, which page of which document
    it reproduces and with which seed.  A figure that travels into a slide deck
    then carries its own citation.

    The line is folded to the figure's own width first: see :func:`_wrap_stamp`.
    """
    # 6.5pt DejaVu Sans averages a little over half its point size per glyph.
    max_chars = max(40, int(fig.get_figwidth() * 72.0 / (6.5 * 0.54)))
    fig.text(
        0.0,
        -0.015,
        _wrap_stamp(text, max_chars),
        ha="left",
        va="top",
        fontsize=6.5,
        color=INK["muted"],
        linespacing=1.5,
    )


#: Width of the book's text block, in inches.  ``\nmfigure`` fills it.
TEXT_WIDTH_IN = 6.0


def fit_to_print_width(fig: plt.Figure, width_in: float, *, passes: int = 10) -> float:
    """Redraw *fig* so that the file it saves is *width_in* inches across.

    This is the difference between lettering a reader can read and lettering
    they cannot, and it is worth being precise about why.

    Font sizes are absolute: a tick label set at 8.5pt is 8.5pt in the file
    however wide the figure is.  When the manuscript then prints that file at
    ``width=\\linewidth``, the whole thing -- lettering included -- is scaled by
    ``6 inches / file width``.  A figure drawn nine inches wide therefore
    reaches the page with 5.7pt ticks, and one drawn at four inches reaches it
    with 12.75pt ticks, larger than the body text it sits in.  Neither is what
    the script asked for, and the book had both.

    Drawing at the width the figure is printed at makes that scale factor one,
    so the type on the page is the type in the style sheet.  The axes take up
    the slack: ``constrained_layout`` reflows them into whatever room is left.

    Iterating is necessary because ``savefig(bbox_inches="tight")`` crops to the
    *contents*, and a label written past the axes -- a direct line label, a
    reference-line annotation -- widens the crop by an amount that is not known
    until the figure is drawn.  Narrowing the canvas moves those labels, so the
    correction is applied two or three times and damped to keep it from
    oscillating between a wide crop and a narrow one.

    Returns the width finally achieved, so a caller can check it.
    """
    if width_in <= 0:
        raise ValueError(f"print width must be positive, got {width_in}")

    original_w, height = fig.get_size_inches()

    def axes_share() -> float:
        """The widest axes, as a fraction of the canvas.

        This is the guard that matters.  When a figure's width is set by text
        rather than by its axes -- a note under a network drawing, a legend
        wider than the plot -- narrowing the canvas does not narrow the text,
        so the axes surrenders the width instead.  Pushed far enough the
        drawing disappears altogether and the file is a caption with a speck in
        it.  Watching the axes' share of the canvas catches that happening,
        which watching the canvas alone cannot.
        """
        fig.canvas.draw()
        widths = [ax.get_window_extent().width for ax in fig.axes]
        if not widths:
            return 1.0
        return max(widths) / (fig.get_size_inches()[0] * fig.dpi)

    start_share = axes_share()
    #: An axes may give up a little width to a label, but not most of it.
    floor_share = min(0.45, start_share * 0.75)
    # Some figures cannot be fitted by narrowing at all: the width of a legend,
    # or of a row of tick labels, does not fall when the canvas does, so the
    # correction asks for a narrower canvas, measures the same overflowing text
    # again, and asks for narrower still.  Left alone that runs the canvas down
    # to nothing.  So the best attempt is remembered and restored at the end,
    # and the search never goes below a fixed fraction of the size the figure
    # was drawn at -- a figure that will not fit is better left at its author's
    # size than squeezed into a sliver and blown back up by the page.
    best_w, best_error = original_w, float("inf")
    measured = float("nan")

    for attempt in range(passes):
        fig.canvas.draw()
        bbox = fig.get_tightbbox(fig.canvas.get_renderer())
        measured = bbox.width
        if measured <= 0:
            return measured

        error = width_in / measured
        if abs(error - 1.0) < abs(best_error - 1.0):
            best_error, best_w = error, fig.get_size_inches()[0]
        if abs(error - 1.0) < 0.01:
            return measured

        # Only the canvas width is corrected: the height carries the figure's
        # vertical room, and squeezing it too would undo the gain.  Later passes
        # are damped because the overflow from text does not scale with the
        # canvas, so a full correction overshoots and the sequence rings.
        damping = 1.0 if attempt == 0 else 0.5
        w = fig.get_size_inches()[0]
        proposed = w * (1.0 + damping * (error - 1.0))
        floor, ceiling = 0.62 * original_w, 1.60 * original_w
        clamped = min(max(proposed, floor), ceiling)
        if abs(clamped - w) < 0.01:
            break  # against a limit and going nowhere
        fig.set_size_inches(clamped, height)

        # Having applied the correction, check what it cost the axes.  If the
        # drawing is being squeezed rather than the margins, the fit is doing
        # more harm than the mis-scaling it was there to prevent, so it stops
        # and the figure keeps the size its author chose.
        if axes_share() < floor_share:
            fig.set_size_inches(original_w, height)
            fig.canvas.draw()
            return fig.get_tightbbox(fig.canvas.get_renderer()).width

    fig.set_size_inches(best_w, height)
    if axes_share() < floor_share:
        fig.set_size_inches(original_w, height)
    fig.canvas.draw()
    return fig.get_tightbbox(fig.canvas.get_renderer()).width


def save_figure(
    fig: plt.Figure,
    stem: str | Path,
    metadata: ExperimentMetadata | None = None,
    *,
    formats: Iterable[str] = ("pdf", "svg", "png"),
    dpi: int = 200,
    stamp: bool = True,
    print_width: float | None = None,
) -> list[Path]:
    """Save *fig* as ``stem.<ext>`` for each format, plus a metadata sidecar.

    Vector formats are for the book, the PNG for GitHub and for slides.

    The provenance line is stamped onto the **raster** output only.  A PNG
    travels on its own -- into a README, a talk, a message -- and needs to carry
    its own citation and seed.  The vector version is placed by
    :func:`~book.build`, which puts the same information in the LaTeX caption,
    so stamping it there would print it twice on the page.  Pass ``stamp=False``
    to suppress it everywhere.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    formats = list(formats)

    if print_width is not None:
        fit_to_print_width(fig, print_width)

    raster = {"png", "jpg", "jpeg", "webp"}
    written: list[Path] = []
    stamped = False

    # Vector first, while the figure is still clean.
    for ext in [f for f in formats if f not in raster]:
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=dpi)
        written.append(path)

    for ext in [f for f in formats if f in raster]:
        if metadata is not None and stamp and not stamped:
            annotate_source(fig, metadata.caption_suffix())
            stamped = True
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=dpi)
        written.append(path)

    if metadata is not None and written:
        metadata.write(written[0])

    return written
