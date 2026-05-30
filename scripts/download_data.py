"""Dataset download/prepare entrypoint (placeholder — Phase 1).

Will fetch RecipeNLG (HuggingFace) into data/raw/. See docs/DATASET.md for
sources and the reproducibility contract.
"""

from __future__ import annotations

from pantrychef.common import get_logger

log = get_logger(__name__)


def main() -> int:
    log.info("download_data: not implemented until Phase 1. See docs/DATASET.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
