"""HSP90 bioactivity data loading and curation.

Two interchangeable sources are supported:

1. **ChEMBL** (default) - pulled live for HSP90-alpha (``CHEMBL3880``) using the
   official ``chembl_webresource_client``. Requires network access.
2. **Local CSV** - a user-provided file with at least a SMILES column and an
   activity column. Use this when you have a curated dataset.

In both cases the output is a tidy :class:`pandas.DataFrame` with canonical
columns so the rest of the pipeline does not care where the data came from.

Canonical output columns
-------------------------
``smiles``        : canonical SMILES (RDKit)
``pchembl``       : pChEMBL-style potency (``-log10(IC50 in M)``), float, may be NaN
``active``        : int {0, 1}, thresholded on ``pchembl`` (see ``activity_threshold``)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# HSP90-alpha (Heat shock protein HSP 90-alpha, gene HSP90AA1) in ChEMBL.
HSP90_ALPHA_CHEMBL_ID = "CHEMBL3880"

# A pChEMBL value of 6.0 corresponds to IC50 = 1 uM; a common activity cutoff.
DEFAULT_ACTIVITY_THRESHOLD = 6.0


@dataclass
class DataConfig:
    """Configuration for assembling the HSP90 dataset."""

    source: str = "chembl"  # "chembl" or "csv"
    csv_path: Optional[str] = None
    smiles_col: str = "smiles"
    activity_col: str = "pchembl"
    target_chembl_id: str = HSP90_ALPHA_CHEMBL_ID
    activity_threshold: float = DEFAULT_ACTIVITY_THRESHOLD
    # Only keep assays reported in these standard types when pulling from ChEMBL.
    standard_types: tuple[str, ...] = ("IC50", "Ki", "Kd")


def load_dataset(config: DataConfig) -> pd.DataFrame:
    """Load and curate the HSP90 dataset according to ``config``."""
    if config.source == "chembl":
        raw = _load_from_chembl(config)
    elif config.source == "csv":
        raw = _load_from_csv(config)
    else:
        raise ValueError(f"Unknown data source: {config.source!r} (use 'chembl' or 'csv')")

    curated = _curate(raw, config)
    LOGGER.info(
        "Curated dataset: %d molecules (%d active / %d inactive)",
        len(curated),
        int(curated["active"].sum()),
        int((curated["active"] == 0).sum()),
    )
    return curated


def _load_from_chembl(config: DataConfig) -> pd.DataFrame:
    """Query ChEMBL for bioactivities against the configured target."""
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "chembl_webresource_client is required for source='chembl'. "
            "Install it with `pip install chembl_webresource_client`."
        ) from exc

    LOGGER.info("Querying ChEMBL for target %s ...", config.target_chembl_id)
    activity = new_client.activity
    records = activity.filter(
        target_chembl_id=config.target_chembl_id,
        standard_type__in=list(config.standard_types),
    ).only(
        [
            "canonical_smiles",
            "standard_value",
            "standard_units",
            "standard_type",
            "pchembl_value",
        ]
    )

    rows = []
    for rec in records:
        smiles = rec.get("canonical_smiles")
        if not smiles:
            continue
        pchembl = rec.get("pchembl_value")
        if pchembl is None:
            # Fall back to deriving pChEMBL from nM/uM potency when possible.
            pchembl = _derive_pchembl(rec.get("standard_value"), rec.get("standard_units"))
        rows.append({"smiles": smiles, "pchembl": pchembl})

    if not rows:
        raise RuntimeError(
            f"No activities returned from ChEMBL for {config.target_chembl_id}. "
            "Check the target id and network access."
        )
    return pd.DataFrame(rows)


def _derive_pchembl(value, units) -> Optional[float]:
    """Convert a raw potency value to a pChEMBL-style ``-log10(M)`` number."""
    if value is None or units is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    unit_to_molar = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9, "pM": 1e-12}
    factor = unit_to_molar.get(str(units))
    if factor is None:
        return None
    return float(-np.log10(value * factor))


def _load_from_csv(config: DataConfig) -> pd.DataFrame:
    """Read a user-provided CSV and rename columns to the canonical schema."""
    if not config.csv_path:
        raise ValueError("source='csv' requires config.csv_path to be set.")
    df = pd.read_csv(config.csv_path)
    missing = {config.smiles_col} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {missing}")
    out = pd.DataFrame({"smiles": df[config.smiles_col]})
    out["pchembl"] = df[config.activity_col] if config.activity_col in df.columns else np.nan
    return out


def _curate(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Canonicalize SMILES, deduplicate, and label actives.

    Deduplication averages the pChEMBL of repeated molecules, which smooths
    over inter-assay noise before thresholding.
    """
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")  # silence RDKit parse warnings

    canonical = []
    for smi in df["smiles"].astype(str):
        mol = Chem.MolFromSmiles(smi)
        canonical.append(Chem.MolToSmiles(mol) if mol is not None else None)
    df = df.assign(smiles=canonical).dropna(subset=["smiles"])

    agg = (
        df.groupby("smiles", as_index=False)["pchembl"]
        .mean()
        .reset_index(drop=True)
    )
    agg["active"] = (agg["pchembl"] >= config.activity_threshold).astype("Int64")
    # Rows with no potency value cannot be labelled; drop them for supervised use.
    agg = agg.dropna(subset=["pchembl"]).reset_index(drop=True)
    agg["active"] = agg["active"].astype(int)
    return agg
