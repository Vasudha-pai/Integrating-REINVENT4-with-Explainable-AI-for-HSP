"""Counterfactual masking for actionable structural suggestions.

Given a generated molecule, we ask: *what minimal structural change would most
improve (or explain) its predicted HSP90 activity?* We answer it by
enumerating local edits, re-scoring each with the QSAR model, and reporting the
edits that move the probability the most.

Two complementary strategies are provided:

- **Fragment masking** (``explain_by_masking``): break each acyclic bond,
  remove the smaller fragment, and measure the drop in predicted activity. A
  large drop means that fragment was *carrying* the activity - a SHAP-style
  attribution via ablation, and a natural counterfactual ("remove this and you
  lose potency").
- **R-group substitution** (``suggest_substitutions``): replace attachment-point
  substituents with a small library of common medicinal-chemistry R-groups and
  keep those that raise predicted activity - actionable "try this instead"
  suggestions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .qsar import QSARModel

LOGGER = logging.getLogger(__name__)

# A compact, drug-like R-group library for substitution scans.
DEFAULT_RGROUPS: tuple[str, ...] = (
    "F", "Cl", "Br", "C", "CC", "O", "OC", "N", "NC", "C#N",
    "C(F)(F)F", "S(=O)(=O)N", "c1ccccc1", "C(=O)O", "C(=O)N", "OCC",
)


@dataclass
class MaskingResult:
    """One fragment-ablation counterfactual."""

    removed_fragment: str   # SMILES of the removed fragment
    remaining: str          # SMILES of what was left
    delta: float            # original P(active) - remaining P(active)
    remaining_proba: float


@dataclass
class SubstitutionResult:
    """One R-group substitution counterfactual."""

    new_smiles: str
    rgroup: str
    delta: float            # new P(active) - original P(active)
    new_proba: float


def explain_by_masking(
    model: QSARModel, smiles: str, top_k: int = 5
) -> tuple[float, list[MaskingResult]]:
    """Ablate single-bond-separated fragments and rank by activity loss.

    Returns ``(original_proba, results)`` with ``results`` sorted by the
    largest activity drop first (the fragments most responsible for activity).
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    original = float(model.predict_proba([smiles])[0])
    results: list[MaskingResult] = []
    seen: set[str] = set()

    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        frag_mol = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=False)
        pieces = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)
        if len(pieces) != 2:
            continue

        pieces = sorted(pieces, key=lambda m: m.GetNumAtoms())
        removed, remaining = pieces[0], pieces[1]
        try:
            removed_smi = Chem.MolToSmiles(removed)
            remaining_smi = Chem.MolToSmiles(remaining)
        except Exception:  # pragma: no cover - sanitization edge cases
            continue
        if not remaining_smi or remaining_smi in seen:
            continue
        seen.add(remaining_smi)

        remaining_proba = float(model.predict_proba([remaining_smi])[0])
        results.append(
            MaskingResult(
                removed_fragment=removed_smi,
                remaining=remaining_smi,
                delta=original - remaining_proba,
                remaining_proba=remaining_proba,
            )
        )

    results.sort(key=lambda r: r.delta, reverse=True)
    return original, results[:top_k]


def suggest_substitutions(
    model: QSARModel,
    smiles: str,
    rgroups: Optional[tuple[str, ...]] = None,
    top_k: int = 5,
    min_gain: float = 0.01,
) -> tuple[float, list[SubstitutionResult]]:
    """Scan single-atom substitutions and keep those that raise predicted activity.

    We replace each carbon/heteroatom bearing at least one hydrogen with a
    library R-group (attached via RWMol surgery) and re-score. Only edits with
    a predicted-activity gain >= ``min_gain`` are returned, best first.
    """
    from rdkit import Chem

    rgroups = rgroups or DEFAULT_RGROUPS
    base_mol = Chem.MolFromSmiles(smiles)
    if base_mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    original = float(model.predict_proba([smiles])[0])
    candidates: dict[str, SubstitutionResult] = {}

    for atom in base_mol.GetAtoms():
        if atom.GetTotalNumHs() < 1:
            continue
        for rgroup in rgroups:
            new_smi = _attach_rgroup(smiles, atom.GetIdx(), rgroup)
            if new_smi is None or new_smi == smiles:
                continue
            new_proba = float(model.predict_proba([new_smi])[0])
            gain = new_proba - original
            if gain < min_gain:
                continue
            # Keep the best-scoring route to each distinct product.
            prev = candidates.get(new_smi)
            if prev is None or gain > prev.delta:
                candidates[new_smi] = SubstitutionResult(
                    new_smiles=new_smi, rgroup=rgroup, delta=gain, new_proba=new_proba
                )

    ranked = sorted(candidates.values(), key=lambda r: r.delta, reverse=True)
    return original, ranked[:top_k]


def _attach_rgroup(smiles: str, atom_idx: int, rgroup_smiles: str) -> Optional[str]:
    """Attach ``rgroup_smiles`` to ``atom_idx`` of ``smiles`` via a single bond.

    Returns a canonical SMILES for the product, or ``None`` if the result is
    chemically invalid.
    """
    from rdkit import Chem

    base = Chem.MolFromSmiles(smiles)
    frag = Chem.MolFromSmiles(rgroup_smiles)
    if base is None or frag is None:
        return None

    combo = Chem.RWMol(Chem.CombineMols(base, frag))
    frag_first_atom = base.GetNumAtoms()  # fragment atoms are appended after base atoms
    try:
        combo.AddBond(atom_idx, frag_first_atom, Chem.BondType.SINGLE)
        product = combo.GetMol()
        Chem.SanitizeMol(product)
    except Exception:
        return None
    return Chem.MolToSmiles(product)
