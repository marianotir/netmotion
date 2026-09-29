"""The thesis's own figures, where the book shows them rather than re-running them.

Some figures of the thesis are results this package does not reproduce -- a
particular cascade, a particular network -- and the text reads its conclusions
off them.  For those the book shows the thesis's own data.  Where the thesis PDF
stores a figure as drawing instructions, ``book/tools/thesis_vectors.py`` reads
the data back exactly and the figure is drawn afresh.  Where it stores only a
picture, ``book/tools/thesis_rasters.py`` extracts the picture and the helpers
here set it in the book's layout: on the page with white made transparent, or,
for a chart, inside the book's own axes.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..visualization import SERIES

#: The book's red and blue, which the thesis's pure screen colours are redrawn in.
BOOK_COLOURS = {"red": SERIES[7], "blue": SERIES[0]}

#: Where the book keeps what it takes from the thesis PDF; the published package
#: carries the data its figures use beside this module.
TFM = Path(__file__).resolve().parents[3] / "book" / "figures" / "tfm"
if not TFM.exists():
    TFM = Path(__file__).with_name("thesis_data")
RASTERS = TFM / "rasters"


def data(ident: str) -> dict:
    """The data ``book/tools/thesis_vectors.py`` read out of fig. *ident*."""
    return json.loads((TFM / f"fig{ident}_data.json").read_text(encoding="utf-8"))


def picture(ident: str, k: int) -> np.ndarray:
    """The *k*-th picture of fig. *ident*, as the thesis embeds it."""
    return plt.imread(RASTERS / f"fig{ident}_{k}.png")


#: The colours of the thesis's network drawings, by hue in degrees, and the
#: book's colour for each.  The meaning is kept: links are green between banks
#: still standing, blue between one standing and one fallen, red between two
#: fallen (the text of fig. 82 names them so); banks are coloured by balance
#: as fig. 77's caption names them -- brown owes more, green lends more,
#: purple lends as much as it owes.
NETWORK_HUES = (
    (0.0, SERIES[7]),     # red links, fallen banks
    (40.0, "#8c5a2b"),    # orange banks: owe more -- brown
    (95.0, SERIES[5]),    # lime-green links
    (165.0, "#1baf7a"),   # aqua banks: lend more -- green
    (240.0, SERIES[0]),   # blue links
    (285.0, "#8e44ad"),   # magenta banks: balanced -- purple
)
#: Fig. 80 is a diagram in two colours of its own: green hubs and links, and
#: yellow outer banks.
DIAGRAM_HUES = ((90.0, SERIES[5]), (50.0, SERIES[3]))


def recolour_hues(img: np.ndarray, table, *, opacity: float = 1.0) -> np.ndarray:
    """Redraw every coloured pixel in the book's colour for its hue.

    Each pixel with colour in it is given the book colour whose reference hue
    is nearest its own, so a line and its paler anti-aliased edge stay one
    colour; how strongly coloured it was sets how opaque it is.
    """
    from matplotlib.colors import rgb_to_hsv, to_rgb

    out = img.copy()
    hsv = rgb_to_hsv(np.clip(out[:, :, :3], 0.0, 1.0))
    hue, sat = hsv[:, :, 0] * 360.0, hsv[:, :, 1] * hsv[:, :, 2]
    coloured = sat > 0.12
    refs = np.array([h for h, _ in table])
    dist = np.abs(hue[..., None] - refs[None, None, :])
    dist = np.minimum(dist, 360.0 - dist)
    nearest = dist.argmin(axis=-1)
    for i, (_, colour) in enumerate(table):
        mask = coloured & (nearest == i)
        out[mask, :3] = to_rgb(colour)
        out[mask, 3] = np.minimum(out[mask, 3], np.clip(sat[mask] * 1.5, 0.0, 1.0)) * opacity
    return out


def show_picture(ax, ident: str, k: int, *, hues=None, opacity: float = 1.0) -> None:
    """A drawing of the thesis's, set on the book's page, in the book's colours."""
    img = picture(ident, k)
    if hues:
        img = recolour_hues(img, hues, opacity=opacity)
    ax.imshow(img, interpolation="none")
    ax.set_aspect("equal")
    ax.axis("off")


def recolour(img: np.ndarray, palette: dict[str, str], *, opacity: float = 1.0) -> np.ndarray:
    """Draw a chart's red and blue marks in the book's red and blue.

    The thesis's charts are drawn in pure screen red and blue, which is what
    makes them read as 2011 beside the book's own.  Each pixel is classed by
    which of the two it is nearer, and redrawn in the book's colour for that
    class; how strongly it was coloured -- the anti-aliasing at a line's edge
    -- becomes how opaque it is, so edges stay soft.  *palette* maps ``"red"``
    and ``"blue"`` to colours; *opacity* lightens dense bundles so their depth
    shows.
    """
    from matplotlib.colors import to_rgb

    out = img.copy()
    rgb = out[:, :, :3]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    strength = np.clip(rgb.max(axis=2) - rgb.min(axis=2), 0.0, 1.0)
    for name, is_it in (("red", (r > b) & (r > g)), ("blue", (b > r) & (b > g))):
        if name not in palette:
            continue
        mask = is_it & (strength > 0.12)
        out[mask, :3] = to_rgb(palette[name])
        out[mask, 3] = np.minimum(out[mask, 3], np.clip(strength[mask] * 1.4, 0.0, 1.0)) * opacity
    return out


def picture_in_axes(
    ax, ident: str, k: int, *, crop, x_ticks, y_ticks, erase_ink=False, palette=None, opacity=1.0
) -> None:
    """A chart of the thesis's: its own plotted marks, inside the book's axes.

    *crop* is the plot area of the picture, ``(left, top, right, bottom)`` in
    pixels, taken just outside the frame so that a line lying on an axis is
    kept.  *x_ticks* and *y_ticks* each give two ticks of the picture as
    ``((pixel, value), (pixel, value))``, read off the picture itself; they
    place the cut-out so that its marks land at the values they plot.

    With *erase_ink*, what is black or grey inside the crop -- the thesis's own
    axis lines, tick marks and any stray lettering -- is made transparent, so
    the book's axes are the only ones drawn.  Only for charts whose plotted
    marks all have a colour.
    """
    img = picture(ident, k).copy()
    if erase_ink:
        rgb = img[:, :, :3]
        colourless = (rgb.max(axis=2) - rgb.min(axis=2)) < 0.24
        not_white = rgb.sum(axis=2) < 2.7
        img[colourless & not_white, 3] = 0.0
    if palette:
        img = recolour(img, palette, opacity=opacity)
    left, top, right, bottom = crop
    (px0, vx0), (px1, vx1) = x_ticks
    (py0, vy0), (py1, vy1) = y_ticks
    x_at = lambda p: vx0 + (p - px0) * (vx1 - vx0) / (px1 - px0)  # noqa: E731
    y_at = lambda p: vy0 + (p - py0) * (vy1 - vy0) / (py1 - py0)  # noqa: E731
    ax.imshow(
        img[top:bottom, left:right],
        extent=(x_at(left), x_at(right), y_at(bottom), y_at(top)),
        aspect="auto",
        interpolation="none",
        zorder=3,
    )
    ax.set_xlim(x_at(left), x_at(right))
    ax.set_ylim(y_at(bottom), y_at(top))
