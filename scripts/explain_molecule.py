#!/usr/bin/env python
"""Explain a molecule's HSP90 activity prediction with SHAP + counterfactuals.

Prints the prediction, top atom attributions, top global features, and
counterfactual suggestions; optionally renders an atom-level heatmap PNG.

Example
-------
    python scripts/explain_molecule.py --model results/qsar_model.joblib \\
        --smiles "O=C(Nc1ccccc1)c1ccc(O)cc1" --background data/hsp90.csv \\
        --heatmap results/attribution.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hsp_xai.counterfactual import explain_by_masking, suggest_substitutions  # noqa: E402
from hsp_xai.explain import ShapExplainer  # noqa: E402
from hsp_xai.qsar import QSARModel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--background", default=None, help="CSV of SMILES for global importance")
    parser.add_argument("--heatmap", default=None, help="Output PNG path for the atom heatmap")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    model = QSARModel.load(args.model)
    explainer = ShapExplainer(model)

    attribution = explainer.explain_atoms(args.smiles)
    print(f"SMILES      : {args.smiles}")
    print(f"P(active)   : {attribution.prediction:.3f}")
    print(f"SHAP base   : {attribution.base_value:.3f}")

    order = np.argsort(np.abs(attribution.atom_weights))[::-1][: args.top_k]
    print("\nTop atom attributions (atom_idx: signed SHAP):")
    for idx in order:
        print(f"  atom {int(idx):>3}: {attribution.atom_weights[idx]:+.4f}")

    if args.background and Path(args.background).exists():
        bg = pd.read_csv(args.background)["smiles"].tolist()
        print("\nTop global features (bit: mean|SHAP|):")
        for bit, val in explainer.global_feature_importance(bg, top_k=args.top_k):
            print(f"  bit {bit:>5}: {val:.4f}")

    print("\nCounterfactual - fragment ablation (what carries activity):")
    original, masks = explain_by_masking(model, args.smiles, top_k=args.top_k)
    for m in masks:
        print(f"  remove {m.removed_fragment:<20} -> P={m.remaining_proba:.3f} (drop {m.delta:+.3f})")

    print("\nCounterfactual - substitutions that would raise activity:")
    _, subs = suggest_substitutions(model, args.smiles, top_k=args.top_k)
    if not subs:
        print("  (no single-atom substitution improved the prediction)")
    for s in subs:
        print(f"  +{s.rgroup:<12} -> {s.new_smiles}  (P={s.new_proba:.3f}, gain {s.delta:+.3f})")

    if args.heatmap:
        explainer.render_atom_heatmap(args.smiles, args.heatmap)
        print(f"\nSaved atom heatmap to {args.heatmap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
