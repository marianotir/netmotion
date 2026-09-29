"""The figure registry: every book figure is a registered, seeded function.

The brief asks that each figure-generation script be deterministic, state its
seed and parameters, save publication-quality output, be callable
independently, and link back to the thesis figure it reproduces.  Rather than
leaving that to the discipline of each script, the requirements are the
signature:

    @figure(
        "F04-03",
        chapter=4,
        title="Small-world: L(p) and C(p)",
        source=Source(Origin.THESIS_2011, pdf_pages="20", section="4.2.3"),
        seed=20110930,
    )
    def smallworld_lp_cp(ctx: FigureContext) -> Figure:
        ...

``ctx`` hands the function its seeded generator, its output stem and a
half-filled :class:`~netmotion.rng.ExperimentMetadata`; the registry saves the
figure, writes the metadata sidecar and stamps the provenance line.  A figure
therefore cannot be produced without a recorded seed, and cannot be saved to
the wrong directory: reproductions go to ``figures/reproduction/``, extensions
to ``figures/extensions/``, decided by the declared origin rather than by the
author's memory.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..provenance import Origin, Source
from ..provenance import registry as provenance_registry
from ..rng import ExperimentMetadata

__all__ = [
    "FigureContext",
    "FigureSpec",
    "figure",
    "FIGURES",
    "run_figure",
    "select",
    "print_width_for",
]

#: Where ``book/build.py`` records how wide the manuscript prints each figure; the
#: published package carries a copy beside this module.
_PRINT_WIDTHS = Path(__file__).resolve().parents[3] / "book" / "figures" / "print_widths.json"
if not _PRINT_WIDTHS.exists():
    _PRINT_WIDTHS = Path(__file__).with_name("print_widths.json")

#: Used when the manuscript has not placed a figure yet.  A figure that nothing
#: cites is still generated -- for the README, for a notebook -- and the text
#: block is the sensible width to draw it at.
DEFAULT_PRINT_WIDTH_IN = 5.05

_print_widths: dict[str, float] | None = None


def print_width_for(ident: str) -> float:
    """How wide, in inches, the book prints figure *ident*.

    Figures are drawn at the width they are printed at, so that the lettering
    on the page is the lettering the style sheet specifies -- see
    :func:`netmotion.visualization.fit_to_print_width` for why that matters.
    The widths themselves come from the manuscript, because the manuscript is
    what decides them: a figure placed with ``\\nmfigurewide`` spills into the
    margin, one placed with ``\\nmfiguresmall`` takes well under half the page.

    The file is written by ``book/build.py``.  When it is absent -- a fresh
    clone generating figures before the book has ever been built -- every
    figure is drawn at the text width, which is what most of them use.
    """
    global _print_widths
    if _print_widths is None:
        if _PRINT_WIDTHS.exists():
            import json

            _print_widths = {
                k: float(v) for k, v in json.loads(_PRINT_WIDTHS.read_text("utf-8")).items()
            }
        else:
            _print_widths = {}
    return _print_widths.get(ident, DEFAULT_PRINT_WIDTH_IN)


def _repo_root() -> Path:
    """The repository root, whether installed editable or run from a checkout."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


@dataclass
class FigureContext:
    """What a figure function is given.

    Attributes
    ----------
    rng:
        A generator seeded from the figure's declared seed.  Use it for
        everything stochastic; do not call ``np.random`` directly.
    meta:
        Metadata pre-filled with the figure name, seed and provenance.  Add the
        model parameters to ``meta.parameters`` -- they end up in the sidecar
        file and in the stamped caption.
    stem:
        Output path without extension, already in the right directory.
    """

    rng: np.random.Generator
    meta: ExperimentMetadata
    stem: Path

    def parameters(self, **kwargs: Any) -> None:
        """Record the model parameters of this figure."""
        self.meta.parameters.update(kwargs)

    def realisations(self, n: int) -> int:
        """Record how many stochastic realisations were averaged, and return n."""
        self.meta.n_realisations = n
        return n


@dataclass
class FigureSpec:
    """A registered figure."""

    ident: str
    chapter: int
    title: str
    source: Source
    seed: int
    func: Callable[[FigureContext], Any]
    slow: bool = False
    reproduces_figure: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def subdir(self) -> str:
        """Reproductions and extensions never share a directory."""
        return "extensions" if self.source.origin is Origin.EXTENSION else "core"

    @property
    def stem(self) -> Path:
        return _repo_root() / "figures" / self.subdir / self.slug

    @property
    def slug(self) -> str:
        return f"{self.ident.lower()}_{self.func.__name__}"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        mark = " [slow]" if self.slow else ""
        return f"{self.ident}  ch{self.chapter}  {self.title}{mark}"


#: Every registered figure, keyed by identifier.
FIGURES: dict[str, FigureSpec] = {}


def figure(
    ident: str,
    *,
    chapter: int,
    title: str,
    source: Source,
    seed: int,
    slow: bool = False,
    reproduces_figure: str = "",
    tags: Iterable[str] = (),
) -> Callable[[Callable[[FigureContext], Any]], Callable[[FigureContext], Any]]:
    """Register a figure-generation function.

    Parameters
    ----------
    ident:
        Stable identifier, matching the reproduction inventory, e.g. ``F04-03``.
        The book cites this, so it must not change once published.
    chapter:
        Chapter of the book / thesis this figure belongs to.
    source:
        Provenance.  Determines whether the output is a reproduction or an
        extension, and supplies the caption's citation.
    seed:
        The figure's seed.  Required even for deterministic figures, so that a
        later stochastic addition cannot silently become irreproducible.
    reproduces_figure:
        The label of the original figure, e.g. ``"thesis fig. p. 20 (top)"``.
    """

    def decorate(func):
        if ident in FIGURES:
            raise ValueError(f"figure identifier {ident!r} is already registered")
        spec = FigureSpec(
            ident=ident,
            chapter=chapter,
            title=title,
            source=source,
            seed=seed,
            func=func,
            slow=slow,
            reproduces_figure=reproduces_figure,
            tags=tuple(tags),
        )
        FIGURES[ident] = spec
        func.figure_spec = spec
        func.source = source
        provenance_registry.append((f"{func.__module__}.{func.__qualname__}", source))
        return func

    return decorate


def select(
    chapter: int | None = None,
    idents: Iterable[str] | None = None,
    include_slow: bool = True,
) -> list[FigureSpec]:
    """Figures matching the given filters, in identifier order."""
    specs = sorted(FIGURES.values(), key=lambda s: (s.chapter, s.ident))
    if chapter is not None:
        specs = [s for s in specs if s.chapter == chapter]
    if idents:
        wanted = {i.upper() for i in idents}
        specs = [s for s in specs if s.ident.upper() in wanted]
    if not include_slow:
        specs = [s for s in specs if not s.slow]
    return specs


def run_figure(
    spec: FigureSpec,
    *,
    formats: Iterable[str] = ("pdf", "svg", "png"),
    stem: Path | None = None,
) -> list[Path]:
    """Run one figure and write its output and metadata sidecar."""
    import matplotlib.pyplot as plt

    from ..visualization import save_figure, use_style

    use_style()

    meta = ExperimentMetadata(
        name=spec.ident,
        seed=spec.seed,
        source=spec.source.citation,
        description=spec.title,
    )
    ctx = FigureContext(
        rng=np.random.default_rng(spec.seed),
        meta=meta,
        stem=stem or spec.stem,
    )

    fig = spec.func(ctx)
    if fig is None:
        raise RuntimeError(f"figure {spec.ident} returned None; it must return a Figure")

    written = save_figure(
        fig, ctx.stem, meta, formats=formats, print_width=print_width_for(spec.ident)
    )
    plt.close(fig)
    return written
