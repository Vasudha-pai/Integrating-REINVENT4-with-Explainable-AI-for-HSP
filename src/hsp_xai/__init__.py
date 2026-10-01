"""hsp_xai: Explainable AI-guided de novo design of HSP90 inhibitors.

A small, self-contained toolkit that sits alongside REINVENT4 and provides:

- ``data``            : load HSP90 bioactivity data from ChEMBL or a local CSV
- ``featurization``   : convert SMILES to Morgan fingerprints / descriptors
- ``qsar``            : train and apply a QSAR activity model
- ``explain``         : SHAP-based feature and atom-level attribution
- ``counterfactual``  : masking-based structural-edit suggestions
- ``scoring_component``: wire the QSAR model into REINVENT4 as a scorer
"""

__version__ = "0.1.0"

__all__ = [
    "data",
    "featurization",
    "qsar",
    "explain",
    "counterfactual",
    "scoring_component",
]
