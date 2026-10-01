#!/usr/bin/env python
"""End-to-end driver for the XAI-guided HSP90 pipeline (data -> model -> actives).

This runs the parts that execute locally (no GPU required):

  1. Fetch/curate the HSP90 dataset        -> data/hsp90.csv
  2. Train the QSAR model                   -> results/qsar_model.joblib
  3. Export actives for REINVENT4 TL        -> data/hsp90_actives.smi

The GPU-bound REINVENT4 steps (transfer learning, RL, sampling) are driven
separately with the TOML files in configs/ - see the README. After sampling,
use scripts/explain_molecule.py on generated hits.

Example
-------
    python scripts/run_pipeline.py --source chembl
    python scripts/run_pipeline.py --source csv --csv mydata.csv --smiles-col SMILES
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hsp_xai.data import DataConfig, load_dataset  # noqa: E402
from hsp_xai.qsar import QSARConfig, QSARModel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["chembl", "csv"], default="chembl")
    parser.add_argument("--csv", dest="csv_path", default=None)
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--activity-col", default="pchembl")
    parser.add_argument("--threshold", type=float, default=6.0)
    parser.add_argument("--data-out", default=str(ROOT / "data" / "hsp90.csv"))
    parser.add_argument("--model-out", default=str(ROOT / "results" / "qsar_model.joblib"))
    parser.add_argument("--actives-out", default=str(ROOT / "data" / "hsp90_actives.smi"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # 1. Data
    cfg = DataConfig(
        source=args.source,
        csv_path=args.csv_path,
        smiles_col=args.smiles_col,
        activity_col=args.activity_col,
        activity_threshold=args.threshold,
    )
    df = load_dataset(cfg)
    Path(args.data_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.data_out, index=False)
    logging.info("Step 1/3: wrote %d molecules -> %s", len(df), args.data_out)

    # 2. QSAR model
    model = QSARModel(QSARConfig()).fit(df["smiles"].tolist(), df["active"].to_numpy())
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)
    logging.info("Step 2/3: trained QSAR model (%s) -> %s", model.metrics, args.model_out)

    # 3. Export actives for transfer learning
    actives = df.loc[df["active"] == 1, "smiles"]
    Path(args.actives_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.actives_out).write_text("\n".join(actives) + "\n")
    logging.info("Step 3/3: exported %d actives -> %s", len(actives), args.actives_out)

    print("\nLocal pipeline complete. Next (on a GPU box with REINVENT4 installed):")
    print("  reinvent -l tl.log     configs/transfer_learning.toml")
    print("  reinvent -l rl.log     configs/staged_learning.toml")
    print("  reinvent -l sample.log configs/sampling.toml")
    print("  python scripts/explain_molecule.py --model", args.model_out,
          "--smiles '<generated_smiles>' --background", args.data_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
