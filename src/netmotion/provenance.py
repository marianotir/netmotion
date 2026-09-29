"""Machine-readable provenance for every scientific object in ``netmotion``.

Four kinds of material are kept distinguishable:

===================  ==========================================================
``THESIS_2011``      A method or result that appears in the Master's thesis
                     *Dinamica y evolucion de redes complejas* (UNED, 2011).
``PAPER_2012``       A method or result that appears in M. Tirado, *Complex
                     Network for a Crisis Contagion on an Interbank System*,
                     Int. J. Mod. Phys. C **23**(9), 1250058 (2012).
``REPRODUCTION``     A 2026 Python reproduction of an original result.  The
                     science is the original author's; only the code is new.
``EXTENSION``        A genuinely new 2026 experiment, analysis or result that
                     is *not* present in the original work.
===================  ==========================================================

A comment saying "from the thesis, page 42" cannot be checked by a test.  A
:class:`Source` attached to the function can be, so provenance here is data:

>>> from netmotion.provenance import Origin, sourced
>>> @sourced(Origin.THESIS_2011, pdf_pages="12-13", section="4.1.2 Algoritmo")
... def example():
...     ...
>>> example.source.origin is Origin.THESIS_2011
True

Everything decorated is collected in a global registry, which
``tests/test_provenance.py`` uses to assert that no public scientific
entry point ships without a citation.

Page numbers always refer to the **PDF page** of the source document.  For the
thesis the printed page number is the PDF page number minus three.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

__all__ = [
    "Origin",
    "Source",
    "sourced",
    "get_source",
    "registry",
    "iter_registry",
]


class Origin(enum.Enum):
    """Where a piece of science came from.

    The string values are the printed labels.
    """

    THESIS_2011 = "2011 ORIGINAL"
    PAPER_2012 = "2012 PUBLISHED"
    REPRODUCTION = "2026 REPRODUCTION"
    EXTENSION = "2026 EXTENSION"

    @property
    def is_original(self) -> bool:
        """True for material authored in 2011/2012, false for 2026 work."""
        return self in (Origin.THESIS_2011, Origin.PAPER_2012)


@dataclass(frozen=True)
class Source:
    """A citation attached to a callable, class or module-level object.

    Parameters
    ----------
    origin:
        Which of the four provenance classes this object belongs to.
    pdf_pages:
        Page or page range in the source PDF, e.g. ``"12"`` or ``"45-50"``.
        Required for :attr:`Origin.THESIS_2011` and :attr:`Origin.PAPER_2012`,
        and expected for :attr:`Origin.REPRODUCTION` (the page of the result
        being reproduced).
    section:
        Section number and title as printed in the source, in the original
        language.  Keeping the Spanish makes the citation checkable against the
        thesis without a translation step.
    reproduces:
        For a ``REPRODUCTION`` or ``EXTENSION``, the dotted name of the
        original-implementation object it is derived from, if any.
    note:
        Anything a reader needs in order to use the citation, such as a
        transcription choice.
    reading:
        The reading of the text that the implementation takes, where an
        implementation detail changes the numbers.
    """

    origin: Origin
    pdf_pages: str = ""
    section: str = ""
    document: str = "thesis"
    reproduces: str = ""
    note: str = ""
    reading: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.origin.is_original and not self.pdf_pages:
            raise ValueError(f"{self.origin.value} sources must cite a page: pass pdf_pages=...")
        if self.document not in ("thesis", "paper", "both", ""):
            raise ValueError(f"document must be 'thesis', 'paper' or 'both', got {self.document!r}")

    @property
    def citation(self) -> str:
        """A short human-readable citation."""
        doc = {
            "thesis": "Tirado Alonso (2011)",
            "paper": "Tirado (2012), IJMPC 23(9):1250058",
            "both": "Tirado Alonso (2011); Tirado (2012)",
            "": "",
        }[self.document]
        bits = [b for b in (doc, f"PDF p. {self.pdf_pages}" if self.pdf_pages else "") if b]
        head = ", ".join(bits)
        return f"[{self.origin.value}] {head}" if head else f"[{self.origin.value}]"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.citation


#: Every object decorated with :func:`sourced`, in decoration order.
registry: list[tuple[str, Source]] = []

F = TypeVar("F", bound=Callable[..., Any] | type)


def sourced(
    origin: Origin,
    *,
    pdf_pages: str = "",
    section: str = "",
    document: str = "thesis",
    reproduces: str = "",
    note: str = "",
    reading: str = "",
    tags: tuple[str, ...] = (),
) -> Callable[[F], F]:
    """Attach a :class:`Source` to a function or class.

    The decorated object gains a ``.source`` attribute and is added to
    :data:`registry`.  It is otherwise untouched -- this never wraps the
    callable, so it costs nothing at call time and does not disturb
    introspection or pickling.
    """

    src = Source(
        origin=origin,
        pdf_pages=pdf_pages,
        section=section,
        document=document,
        reproduces=reproduces,
        note=note,
        reading=reading,
        tags=tuple(tags),
    )

    def decorate(obj: F) -> F:
        obj.source = src  # type: ignore[attr-defined]
        qualname = getattr(obj, "__qualname__", repr(obj))
        module = getattr(obj, "__module__", "?")
        registry.append((f"{module}.{qualname}", src))
        return obj

    return decorate


def get_source(obj: Any) -> Source | None:
    """Return the :class:`Source` attached to *obj*, or ``None``."""
    return getattr(obj, "source", None)


def iter_registry(origin: Origin | None = None):
    """Iterate ``(dotted_name, Source)`` pairs, optionally filtered by origin."""
    for name, src in registry:
        if origin is None or src.origin is origin:
            yield name, src
