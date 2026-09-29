"""Figure-generation experiments, one module per chapter.

Import :mod:`netmotion.experiments.registry` to register a figure; run them
with ``python -m netmotion.experiments.cli``.
"""

from __future__ import annotations

from .registry import FIGURES, FigureContext, FigureSpec, figure, run_figure, select

__all__ = ["figure", "FigureContext", "FigureSpec", "FIGURES", "run_figure", "select"]
