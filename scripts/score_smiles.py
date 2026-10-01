#!/usr/bin/env python
"""REINVENT4 ExternalProcess scorer: read SMILES, print QSAR scores as JSON.

REINVENT4's ``ExternalProcess`` scoring component invokes an external command,
passing candidate SMILES and reading back a JSON payload of scores. This script
implements that contract:

- SMILES arrive either as CLI args or, by default, one per line on stdin.
- Output is a JSON object ``{"version": 1, "payload": {"predictions": [...]}}``
  with one float score per input SMILES, in order.

Point the REINVENT config at it (see configs/staged_learning.toml):

    [[stage.scoring.component]]
    type = "ExternalProcess"
    name = "HSP90_QSAR"
    [stage.scoring.component.ExternalProcess.endpoint]
    params.executable = "python"
    params.args = "scripts/score_smiles.py --model results/qsar_model.joblib"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hsp_xai.scoring_component import load_component  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to trained QSAR .joblib model")
    parser.add_argument("smiles", nargs="*", help="SMILES to score (default: read stdin, one per line)")
    args = parser.parse_args()

    if args.smiles:
        smiles = args.smiles
    else:
        smiles = [line.strip() for line in sys.stdin if line.strip()]

    component = load_component(args.model)
    scores = component.score(smiles)

    payload = {"version": 1, "payload": {"predictions": [float(s) for s in scores]}}
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
