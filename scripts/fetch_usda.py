"""Download USDA FoodData Central SR Legacy + Foundation CSV bundles and build
the compact artifact at data/processed/usda.json.

Manual download is fine too: unzip the FDC CSV bundles into a folder and pass
--csv-dir. Record the FDC release date in docs/DATASET.md after running."""

from __future__ import annotations

import argparse
from pathlib import Path

from pantrychef.config import get_settings
from pantrychef.nutrition.usda import build_artifact, save_artifact


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", required=True, help="folder with FDC *.csv files")
    args = ap.parse_args()
    s = get_settings()
    table = build_artifact(Path(args.csv_dir))
    out = s.processed_dir / "usda.json"
    save_artifact(table, out)
    print(f"wrote {out} ({len(table)} foods)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
