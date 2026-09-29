"""Random-state handling and the record that travels with a figure.

``as_rng`` converts whatever a caller passed (``None``, an ``int`` seed, an
existing ``Generator``) into a ``numpy.random.Generator``.  Every stochastic
function in ``netmotion`` takes ``rng=`` and funnels it through this, so a seed
given at the top of a run reaches the bottom of the call stack.

``ExperimentMetadata`` holds the seed, the parameters and the package versions a
drawn figure was made with, and is written beside it as a JSON sidecar.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["as_rng", "SeedLike", "ExperimentMetadata", "package_versions"]

SeedLike = int | np.random.Generator | np.random.SeedSequence | None


def as_rng(rng: SeedLike = None) -> np.random.Generator:
    """Normalise *rng* to a ``numpy.random.Generator``.

    ``None`` produces a fresh, unseeded generator.  Passing an ``int`` gives a
    reproducible run.  Passing an existing ``Generator`` returns it unchanged,
    so a caller can drive a whole simulation from one stream.

    >>> int(as_rng(0).integers(10))
    8
    >>> g = np.random.default_rng(1)
    >>> as_rng(g) is g
    True
    """
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None or isinstance(rng, (int, np.integer, np.random.SeedSequence)):
        return np.random.default_rng(rng)
    raise TypeError(
        f"rng must be None, an int seed, a SeedSequence or a Generator, got {type(rng)!r}"
    )


def package_versions() -> dict[str, str]:
    """Versions of everything that can change a numerical result."""
    import matplotlib
    import networkx

    try:
        import scipy

        scipy_version = scipy.__version__
    except ImportError:  # SciPy is not a dependency of the package
        scipy_version = "not installed"

    from netmotion import __version__

    return {
        "netmotion": __version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy_version,
        "networkx": networkx.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
    }


@dataclass
class ExperimentMetadata:
    """Everything needed to draw a figure again and get the same picture.

    ``parameters`` holds the model parameters (``N``, ``p``, ``k``, ``m``, the
    propagation parameters, ...) and ``source`` the section of the book the
    figure belongs to.
    """

    name: str
    seed: int
    parameters: dict[str, Any] = field(default_factory=dict)
    n_realisations: int = 1
    source: str = ""
    description: str = ""
    versions: dict[str, str] = field(default_factory=package_versions)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, default=str)

    def write(self, path: str | Path) -> Path:
        """Write the metadata beside a figure, as ``<figure>.meta.json``."""
        path = Path(path)
        target = path.with_suffix(path.suffix + ".meta.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    def caption_suffix(self) -> str:
        """A one-line parameter string, for a figure that travels on its own."""
        params = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        bits = [b for b in (self.source, params, f"seed={self.seed}") if b]
        if self.n_realisations > 1:
            bits.append(f"{self.n_realisations} realisations")
        return " | ".join(bits)
