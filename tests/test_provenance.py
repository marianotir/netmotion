"""The provenance rules, as executable checks.

Original work, reproductions and extensions must stay clearly distinguishable.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

import netmotion
from netmotion.provenance import Origin, Source, registry


def _import_all():
    for mod in pkgutil.walk_packages(netmotion.__path__, "netmotion."):
        importlib.import_module(mod.name)


_import_all()


def test_registry_is_populated():
    assert registry, "no object carries a Source; the provenance system is not wired up"


def test_original_sources_cite_a_page():
    """2011/2012 material must name the page it came from."""
    for name, src in registry:
        if src.origin.is_original:
            assert src.pdf_pages, f"{name} claims {src.origin.value} without a page citation"


def test_original_sources_name_a_section():
    for name, src in registry:
        if src.origin.is_original:
            assert src.section, f"{name} claims {src.origin.value} without a section"


def test_page_citations_are_in_range():
    """The thesis has 79 PDF pages, the IJMPC article 20."""
    limits = {"thesis": 79, "paper": 20, "both": 79}
    for name, src in registry:
        if not src.pdf_pages:
            continue
        limit = limits.get(src.document)
        if limit is None:
            continue
        for part in src.pdf_pages.replace(" ", "").split(","):
            for page in part.split("-"):
                if not page:
                    continue
                assert page.isdigit(), f"{name}: unparsable page reference {src.pdf_pages!r}"
                assert 1 <= int(page) <= limit, (
                    f"{name}: PDF page {page} is outside the {src.document} ({limit} pages)"
                )


def test_reproductions_point_at_what_they_reproduce():
    """A reproduction that cites nothing is indistinguishable from an invention."""
    for name, src in registry:
        if src.origin is Origin.REPRODUCTION:
            assert src.pdf_pages or src.reproduces, (
                f"{name} is a 2026 REPRODUCTION but names neither a source page "
                "nor the object it reproduces"
            )


def test_extensions_are_not_dressed_up_as_originals():
    """An extension may cite a page for context, but must say it is an extension."""
    for name, src in registry:
        if src.origin is Origin.EXTENSION:
            assert not src.origin.is_original
            assert src.note, (
                f"{name} is a 2026 EXTENSION and must carry a note saying what is "
                "new about it, so it is never mistaken for original work"
            )


def test_original_sources_reject_missing_pages():
    with pytest.raises(ValueError, match="must cite a page"):
        Source(origin=Origin.THESIS_2011)


def test_citation_string_names_the_document():
    src = Source(origin=Origin.PAPER_2012, pdf_pages="4", document="paper")
    assert "2012 PUBLISHED" in src.citation
    assert "IJMPC" in src.citation


PUBLIC_SCIENTIFIC_MODULES = [
    "netmotion.networks",
    "netmotion.metrics",
    "netmotion.dynamics",
    "netmotion.propagation",
    "netmotion.finance",
]


@pytest.mark.parametrize("module_name", PUBLIC_SCIENTIFIC_MODULES)
def test_every_public_scientific_callable_is_sourced(module_name):
    """No scientific entry point ships without a citation.

    This is the test that keeps provenance complete as the package grows:
    adding a model or a metric to a public ``__all__`` without saying
    where it came from fails the suite.
    """
    module = importlib.import_module(module_name)
    exported = getattr(module, "__all__", None)
    if not exported:
        pytest.skip(f"{module_name} exports nothing yet")

    missing = []
    for name in exported:
        obj = getattr(module, name)
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        if getattr(obj, "source", None) is None:
            missing.append(name)

    assert not missing, (
        f"{module_name} exports {missing} without a @sourced(...) citation; "
        "every public scientific object must say where it came from"
    )
