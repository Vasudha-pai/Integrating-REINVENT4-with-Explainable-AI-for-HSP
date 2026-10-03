#!/usr/bin/env python
"""Train the HSP90 QSAR model from a curated CSV and save it.

Example
-------
    python scripts/train_qsar.py --data data/hsp90.csv --out results/qsar_model.joblib
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hsp_xai.qsar import QSARConfig, QSARModel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Curated CSV with 'smiles' and 'active' columns")
    parser.add_argument("--out", required=True, help="Output .joblib model path")
    parser.add_argument("--model-type", choices=["random_forest", "gradient_boosting"], default="random_forest")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE oversampling")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    df = pd.read_csv(args.data)
    for col in ("smiles", "active"):
        if col not in df.columns:
            parser.error(f"Input CSV must contain a '{col}' column.")

    cfg = QSARConfig(
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        use_smote=not args.no_smote,
    )
    model = QSARModel(cfg).fit(df["smiles"].tolist(), df["active"].to_numpy())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)

    metrics_path = Path(args.out).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(model.metrics, indent=2))
    print(f"Saved model to {args.out}")
    print(f"Held-out metrics: {json.dumps(model.metrics, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
