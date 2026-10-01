"""REINVENT4 scoring plugin that scores molecules with the HSP90 QSAR model.

REINVENT4 discovers external scoring components as subclasses registered in its
``scoring`` plugin system. During reinforcement learning, REINVENT passes a list
of SMILES per step; this component returns a per-molecule score in ``[0, 1]``
(the QSAR P(active)), which REINVENT then folds into the aggregated reward.

Usage from a REINVENT4 TOML config (see ``configs/staged_learning.toml``):

    [[stage.scoring.component]]
    type = "ExternalProcess"            # or a registered custom component
    name = "HSP90_QSAR"
    ...

The cleanest integration is REINVENT4's ``ExternalProcess`` component calling
``scripts/score_smiles.py`` (which wraps this class), so no edits to the
REINVENT4 source tree are required. The :class:`HSP90QSARComponent` class below
is also importable directly if you register it as a native plugin.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from .qsar import QSARModel

LOGGER = logging.getLogger(__name__)


class HSP90QSARComponent:
    """Callable scorer wrapping a trained :class:`QSARModel`.

    Parameters
    ----------
    model_path:
        Path to a ``.joblib`` model saved by :meth:`QSARModel.save`.
    """

    def __init__(self, model_path: str | Path):
        self.model = QSARModel.load(model_path)

    def score(self, smiles: Sequence[str]) -> np.ndarray:
        """Return an array of P(active) in ``[0, 1]`` aligned to ``smiles``.

        Invalid SMILES score 0.0 so REINVENT learns to avoid them.
        """
        return self.model.predict_proba(list(smiles))

    # REINVENT4's native component interface expects a ``__call__`` returning a
    # structure with per-endpoint scores. We keep a simple contract: a plain
    # ndarray, which both the CLI wrapper and a thin native adapter can consume.
    def __call__(self, smiles: Sequence[str]) -> np.ndarray:
        return self.score(smiles)


def load_component(model_path: str | Path) -> HSP90QSARComponent:
    """Factory used by the CLI scorer and any native plugin adapter."""
    return HSP90QSARComponent(model_path)
