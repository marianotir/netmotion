"""Command line for regenerating the book's figures.

    python -m netmotion.experiments.cli --list
    python -m netmotion.experiments.cli --all
    python -m netmotion.experiments.cli --chapter 4
    python -m netmotion.experiments.cli F04-03 F04-05 --formats png

Every figure in the book comes out of this command.  Nothing is drawn by hand,
and nothing is drawn without a seed.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
import time
from pathlib import Path

from .registry import FIGURES, run_figure, select


def _load_figure_modules() -> None:
    """Import every ``netmotion.experiments.*`` module so figures register."""
    import netmotion.experiments as pkg

    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name in ("cli", "registry"):
            continue
        importlib.import_module(f"netmotion.experiments.{mod.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="netmotion-figures",
        description="Regenerate the figures of 'Networks in Motion' from code.",
    )
    parser.add_argument("idents", nargs="*", help="figure identifiers, e.g. F04-03")
    parser.add_argument("--all", action="store_true", help="regenerate every figure")
    parser.add_argument("--chapter", type=int, help="only figures of this chapter")
    parser.add_argument("--list", action="store_true", help="list figures and exit")
    parser.add_argument(
        "--formats",
        default="pdf,svg,png",
        help="comma-separated output formats (default: pdf,svg,png)",
    )
    parser.add_argument("--skip-slow", action="store_true", help="skip long simulations")
    parser.add_argument("--outdir", type=Path, help="override the output directory")
    args = parser.parse_args(argv)

    _load_figure_modules()

    if not FIGURES:
        print("No figures are registered yet.", file=sys.stderr)
        return 0

    if args.list:
        for spec in select():
            print(f"  {spec}")
            print(f"        {spec.source.citation} -> figures/{spec.subdir}/{spec.slug}")
        return 0

    if not (args.all or args.chapter or args.idents):
        parser.error("choose --all, --chapter N, or one or more figure identifiers")

    specs = select(
        chapter=args.chapter,
        idents=args.idents or None,
        include_slow=not args.skip_slow,
    )
    if not specs:
        print("nothing matched", file=sys.stderr)
        return 1

    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    failures: list[tuple[str, Exception]] = []

    for spec in specs:
        started = time.perf_counter()
        print(f"[{spec.ident}] {spec.title} ... ", end="", flush=True)
        try:
            stem = (args.outdir / spec.slug) if args.outdir else None
            written = run_figure(spec, formats=formats, stem=stem)
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append((spec.ident, exc))
            print(f"FAILED: {exc}")
            continue
        elapsed = time.perf_counter() - started
        print(f"{elapsed:5.1f}s -> {written[0].parent}/{written[0].stem}.{{{','.join(formats)}}}")

    if failures:
        print(f"\n{len(failures)} figure(s) failed:", file=sys.stderr)
        for ident, exc in failures:
            print(f"  {ident}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"\n{len(specs)} figure(s) regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
