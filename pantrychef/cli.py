"""Command-line entrypoint (placeholder).

Wired in pyproject as `pantrychef`. Subcommands are added per phase
(e.g. `pantrychef cook --have eggs,flour,milk`). Phase 0: prints status only.
"""

from __future__ import annotations

import sys

from pantrychef import __version__


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    print(f"PantryChef v{__version__} — Phase 0 scaffold. No ML implemented yet.")
    print("See ROADMAP.md for what's coming.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
