"""Molecular featurization: SMILES -> numeric features.

The QSAR model and the SHAP explainer share a single featurizer so that the
feature axis means the same thing everywhere. We use Morgan (ECFP-like)
fingerprints because each bit is traceable back to the atoms that set it,
which is exactly what the atom-level attribution in :mod:`hsp_xai.explain`
needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FingerprintConfig:
    """Morgan fingerprint settings."""

    radius: int = 2
    n_bits: int = 2048


@dataclass
class FeaturizedBatch:
    """Container holding features plus the bit->atom provenance per molecule."""

    features: np.ndarray  # shape (n_molecules, n_bits)
    valid_smiles: list[str]
    bit_infos: list[dict] = field(default_factory=list)  # one RDKit bitInfo per molecule


def _mol_from_smiles(smiles: str):
    from rdkit import Chem

    return Chem.MolFromSmiles(smiles)


def featurize_smiles(
    smiles_list: list[str],
    config: Optional[FingerprintConfig] = None,
    keep_bit_info: bool = False,
) -> FeaturizedBatch:
    """Featurize a list of SMILES into a dense fingerprint matrix.

    Invalid SMILES are dropped; ``valid_smiles`` records what survived so the
    caller can realign labels.
    """
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    config = config or FingerprintConfig()

    vectors: list[np.ndarray] = []
    valid: list[str] = []
    bit_infos: list[dict] = []

    for smi in smiles_list:
        mol = _mol_from_smiles(smi)
        if mol is None:
            continue
        bit_info: dict = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, config.radius, nBits=config.n_bits, bitInfo=bit_info
        )
        arr = np.zeros((config.n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        vectors.append(arr)
        valid.append(smi)
        if keep_bit_info:
            bit_infos.append(bit_info)

    features = (
        np.vstack(vectors) if vectors else np.empty((0, config.n_bits), dtype=np.float32)
    )
    return FeaturizedBatch(features=features, valid_smiles=valid, bit_infos=bit_infos)


def featurize_one(smiles: str, config: Optional[FingerprintConfig] = None):
    """Featurize a single molecule, returning ``(features, bit_info, mol)``.

    ``bit_info`` maps each set fingerprint bit to the ``(atom_idx, radius)``
    environments that produced it - the bridge used by atom-level SHAP.
    Returns ``(None, None, None)`` if the SMILES is invalid.
    """
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    config = config or FingerprintConfig()
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None, None, None

    bit_info: dict = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, config.radius, nBits=config.n_bits, bitInfo=bit_info
    )
    arr = np.zeros((config.n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr, bit_info, mol
