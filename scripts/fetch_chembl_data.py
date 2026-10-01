#!/usr/bin/env python
"""Fetch and curate HSP90 bioactivity data, writing a tidy CSV.

Examples
--------
    # Pull HSP90-alpha from ChEMBL
    python scripts/fetch_chembl_data.py --source chembl --out data/hsp90.csv

    # Curate a CSV you already have (canonicalize + label)
    python scripts/fetch_chembl_data.py --source csv --csv raw.csv \\
        --smiles-col SMILES --activity-col pChEMBL --out data/hsp90.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hsp_xai.data import DataConfig, load_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["chembl", "csv"], default="chembl")
    parser.add_argument("--csv", dest="csv_path", default=None, help="Input CSV for --source csv")
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--activity-col", default="pchembl")
    parser.add_argument("--target", default=None, help="Override ChEMBL target id (default HSP90-alpha)")
    parser.add_argument("--threshold", type=float, default=6.0, help="pChEMBL active cutoff")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = DataConfig(
        source=args.source,
        csv_path=args.csv_path,
        smiles_col=args.smiles_col,
        activity_col=args.activity_col,
        activity_threshold=args.threshold,
    )
    if args.target:
        cfg.target_chembl_id = args.target

    df = load_dataset(cfg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} molecules to {args.out}")
    print(df["active"].value_counts().rename({0: "inactive", 1: "active"}).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
