"""SHAP-based explainability for the HSP90 QSAR model.

Two levels of explanation are produced:

1. **Bit-level** - SHAP values over the Morgan fingerprint bits, telling us
   which substructural features push a prediction toward "active".
2. **Atom-level** - those per-bit SHAP values are projected back onto atoms via
   RDKit's ``bitInfo`` (which bit was set by which atom environment), giving an
   interpretable per-atom attribution that can be rendered as a heatmap on the
   2D structure.

This turns the QSAR model from a black box into a chemistry-readable map of
"what the model likes / dislikes about this molecule".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .featurization import FingerprintConfig, featurize_one, featurize_smiles
from .qsar import QSARModel

LOGGER = logging.getLogger(__name__)


@dataclass
class AtomAttribution:
    """Per-atom explanation for a single molecule."""

    smiles: str
    atom_weights: np.ndarray  # shape (n_atoms,), signed SHAP contribution per atom
    base_value: float         # SHAP expected value (model's average output)
    prediction: float         # model P(active) for this molecule


class ShapExplainer:
    """Wraps a SHAP TreeExplainer around a fitted :class:`QSARModel`."""

    def __init__(self, model: QSARModel, background_smiles: Optional[list[str]] = None):
        self.model = model
        self.fp_config: FingerprintConfig = model.config.fingerprint
        self._explainer = None
        self._background = background_smiles

    def _ensure_explainer(self):
        if self._explainer is not None:
            return
        import shap

        # TreeExplainer is exact and fast for the tree ensembles we use.
        self._explainer = shap.TreeExplainer(self.model.estimator)

    def explain_bits(self, smiles: str) -> tuple[np.ndarray, float]:
        """Return ``(bit_shap_values, base_value)`` for the positive class."""
        self._ensure_explainer()
        features, _, _ = featurize_one(smiles, self.fp_config)
        if features is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")

        shap_values = self._explainer.shap_values(features.reshape(1, -1))
        values, base = _select_positive_class(shap_values, self._explainer.expected_value)
        return values.reshape(-1), float(base)

    def explain_atoms(self, smiles: str) -> AtomAttribution:
        """Project bit-level SHAP values onto atoms for ``smiles``.

        Each set bit's SHAP value is distributed over the atoms that make up
        its circular environment; an atom's weight is the sum over all bits it
        participates in. This mirrors the standard RDKit similarity-map
        projection but driven by SHAP rather than raw bit presence.
        """
        self._ensure_explainer()
        features, bit_info, mol = featurize_one(smiles, self.fp_config)
        if features is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")

        from rdkit.Chem import AllChem

        bit_values, base = self.explain_bits(smiles)
        n_atoms = mol.GetNumAtoms()
        atom_weights = np.zeros(n_atoms, dtype=np.float64)

        for bit, environments in bit_info.items():
            shap_val = float(bit_values[bit])
            if shap_val == 0.0:
                continue
            for atom_idx, radius in environments:
                atoms_in_env = _atoms_in_environment(mol, atom_idx, radius, AllChem)
                if not atoms_in_env:
                    continue
                share = shap_val / len(atoms_in_env)
                for a in atoms_in_env:
                    atom_weights[a] += share

        prediction = float(self.model.predict_proba_features(features.reshape(1, -1))[0])
        return AtomAttribution(
            smiles=smiles,
            atom_weights=atom_weights,
            base_value=base,
            prediction=prediction,
        )

    def global_feature_importance(self, smiles_list: list[str], top_k: int = 20):
        """Mean absolute SHAP value per bit across a set of molecules.

        Returns a list of ``(bit_index, mean_abs_shap)`` sorted descending -
        the model's globally most influential substructural features.
        """
        self._ensure_explainer()
        batch = featurize_smiles(smiles_list, self.fp_config)
        if len(batch.valid_smiles) == 0:
            return []
        shap_values = self._explainer.shap_values(batch.features)
        values, _ = _select_positive_class(shap_values, self._explainer.expected_value)
        mean_abs = np.abs(values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:top_k]
        return [(int(b), float(mean_abs[b])) for b in order]

    def render_atom_heatmap(self, smiles: str, out_path: str):
        """Render a 2D similarity-style map coloured by atom SHAP weights.

        Saves a PNG to ``out_path``. Red = pushes toward active,
        blue = pushes toward inactive.
        """
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import SimilarityMaps
        import matplotlib.pyplot as plt

        attribution = self.explain_atoms(smiles)
        _, _, mol = featurize_one(smiles, self.fp_config)
        weights = list(attribution.atom_weights)

        fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, weights, colorMap="bwr")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return out_path


def _atoms_in_environment(mol, atom_idx, radius, AllChem) -> list[int]:
    """Return the atom indices making up a Morgan environment of given radius."""
    if radius == 0:
        return [atom_idx]
    env = AllChem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    atoms = set()
    for bond_idx in env:
        bond = mol.GetBondWithIdx(bond_idx)
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())
    atoms.add(atom_idx)
    return sorted(atoms)


def _select_positive_class(shap_values, expected_value):
    """Normalize SHAP output across shap/sklearn versions to the positive class.

    Depending on versions, ``shap_values`` may be a list ``[neg, pos]`` or a
    single array; ``expected_value`` likewise may be scalar or per-class.
    """
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[1])
        base = expected_value[1] if np.ndim(expected_value) else expected_value
        return values, float(base)
    values = np.asarray(shap_values)
    if values.ndim == 3:  # (n_samples, n_features, n_classes)
        values = values[..., 1]
        base = expected_value[1] if np.ndim(expected_value) else expected_value
        return values, float(base)
    base = expected_value if np.ndim(expected_value) == 0 else expected_value[-1]
    return values, float(base)
