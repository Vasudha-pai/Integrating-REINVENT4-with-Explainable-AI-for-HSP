# Explainable AI-Guided De Novo Design of HSP90 Inhibitors Using REINVENT4

This repository contains a complete, runnable computational pipeline for the
**interpretable de novo design of HSP90 inhibitors**. It couples
[REINVENT4](https://github.com/MolecularAI/REINVENT4) generative chemistry with
a **SHAP-based explainable-AI (XAI)** layer, turning the generative loop from a
black box into a chemistry-readable discovery tool.

The pipeline integrates:

- **REINVENT4** — transfer learning + reinforcement-learning molecule generation
- **QSAR activity model** — a fingerprint-based classifier of HSP90 inhibition
- **SHAP** — bit-level *and* atom-level feature attribution on each prediction
- **Counterfactual masking** — actionable "remove this / try this instead" edits
- **Molecular docking & MD** — downstream validation of candidate hits *(external, not scripted here)*

---

## How the pieces fit together

```
                 ChEMBL (HSP90-alpha)  OR  your own CSV
                             │
                             ▼
                  ┌──────────────────────┐
                  │  hsp_xai.data         │  curate: canonicalize, dedup, label
                  └──────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │  hsp_xai.qsar         │  QSAR model: P(active) from Morgan FP
                  └───┬───────────────┬───┘
         actives.smi  │               │  results/qsar_model.joblib
                      ▼               ▼
            ┌───────────────┐   ┌──────────────────────────────┐
            │  REINVENT4 TL │   │  hsp_xai.scoring_component    │
            │  (focus prior)│   │  = RL reward for REINVENT4    │
            └───────┬───────┘   └───────────────┬──────────────┘
                    └──────────────┬────────────┘
                                   ▼
                        ┌──────────────────────┐
                        │  REINVENT4 Staged RL  │  generate HSP90-optimized molecules
                        └──────────┬───────────┘
                                   ▼  sampled molecules
             ┌─────────────────────┴─────────────────────┐
             ▼                                            ▼
   ┌──────────────────────┐                   ┌──────────────────────┐
   │  hsp_xai.explain      │  SHAP atom map    │  hsp_xai.counterfactual│  edit suggestions
   └──────────────────────┘                   └──────────────────────┘
                                   │
                                   ▼
                        docking / MD validation (external)
```

---

## Repository layout

```
.
├── environment.yml              # conda environment (reinvent4-hsp-xai)
├── requirements.txt             # pip-only alternative
├── pyproject.toml               # installs the hsp_xai package (pip install -e .)
├── configs/
│   ├── transfer_learning.toml   # REINVENT4: focus the prior on HSP90 actives
│   ├── staged_learning.toml     # REINVENT4: RL with the QSAR model as reward
│   └── sampling.toml            # REINVENT4: sample molecules from a model
├── src/hsp_xai/
│   ├── data.py                  # ChEMBL / CSV loading + curation
│   ├── featurization.py         # SMILES -> Morgan fingerprints (+ atom provenance)
│   ├── qsar.py                  # train / save / apply the QSAR model
│   ├── explain.py               # SHAP bit- and atom-level attribution
│   ├── counterfactual.py        # fragment-masking & substitution counterfactuals
│   └── scoring_component.py     # QSAR-as-REINVENT4-scorer
├── scripts/
│   ├── fetch_chembl_data.py     # build data/hsp90.csv
│   ├── train_qsar.py            # train results/qsar_model.joblib
│   ├── run_pipeline.py          # data -> model -> actives, in one command
│   ├── score_smiles.py          # REINVENT4 ExternalProcess scorer (stdin/CLI -> JSON)
│   └── explain_molecule.py      # SHAP + counterfactual report for one molecule
├── data/                        # datasets (git-ignored; see data/README.md)
└── results/                     # models, logs, figures (git-ignored)
```

---

## 1. Environment setup

Create the conda environment, then install REINVENT4 (which pins its own deps)
and this repo's `hsp_xai` package:

```bash
# This repo's environment (RDKit, scikit-learn, SHAP, imbalanced-learn, ...)
conda env create -f environment.yml
conda activate reinvent4-hsp-xai

# REINVENT4 itself, from its upstream repo
git clone https://github.com/MolecularAI/REINVENT4.git
cd REINVENT4 && pip install -e . && cd ..

# Make `import hsp_xai` work everywhere
pip install -e .
```

> A GPU is strongly recommended for the REINVENT4 stages. The data/QSAR/XAI
> steps run fine on CPU.

---

## 2. Build the data and QSAR model (CPU)

One command runs curation → QSAR training → actives export:

```bash
python scripts/run_pipeline.py --source chembl
```

This writes `data/hsp90.csv`, `results/qsar_model.joblib` (+ `*.metrics.json`),
and `data/hsp90_actives.smi`. To use your own curated file instead of ChEMBL:

```bash
python scripts/run_pipeline.py --source csv --csv my_raw.csv \
    --smiles-col SMILES --activity-col pChEMBL
```

The individual steps are also available separately
(`scripts/fetch_chembl_data.py`, `scripts/train_qsar.py`).

**Activity labelling.** A molecule is labelled active when its pChEMBL
(`-log10(IC50 in M)`) meets `--threshold` (default **6.0**, i.e. IC50 ≤ 1 µM).
Duplicates are averaged before thresholding to smooth inter-assay noise.

---

## 3. Generate molecules with REINVENT4 (GPU)

```bash
# 3a. Transfer learning: bias the prior toward HSP90 chemical space
reinvent -l tl.log configs/transfer_learning.toml

# 3b. Reinforcement learning: optimize toward activity + drug-likeness
#     (the QSAR model is the reward, via scripts/score_smiles.py)
reinvent -l rl.log configs/staged_learning.toml

# 3c. Sample candidates from the optimized agent
reinvent -l sample.log configs/sampling.toml
```

The RL reward is wired through REINVENT4's `ExternalProcess` component — it
calls `scripts/score_smiles.py`, which loads the trained model and returns
`P(active)` per molecule as JSON. **No edits to the REINVENT4 source tree are
needed.** Review the `TODO` paths in the TOML files before running.

---

## 4. Explain and refine candidates (CPU)

For any generated (or known) molecule, produce a full interpretability report:

```bash
python scripts/explain_molecule.py \
    --model results/qsar_model.joblib \
    --smiles "O=C(Nc1ccccc1)c1ccc(O)cc1" \
    --background data/hsp90.csv \
    --heatmap results/attribution.png
```

It prints:

- **P(active)** and the SHAP base value,
- **top atom attributions** — signed per-atom SHAP contributions (red/blue map in the PNG),
- **top global features** — the model's most influential fingerprint bits,
- **counterfactuals** — (a) fragment ablations that *carry* the activity, and
  (b) single-atom substitutions predicted to *raise* activity.

This is the XAI feedback loop: the atom map shows *why* a molecule scores as it
does, and the counterfactuals suggest *what to change*.

---

## Components in detail

### `hsp_xai.data`
Loads HSP90 bioactivity from **ChEMBL** (target HSP90-alpha, `CHEMBL3880`, via
`chembl_webresource_client`) or a **local CSV**. Canonicalizes SMILES with
RDKit, deduplicates by averaging pChEMBL, and labels actives. Output schema is
identical regardless of source: `smiles`, `pchembl`, `active`.

### `hsp_xai.featurization`
Converts SMILES to **Morgan fingerprints** (radius 2, 2048 bits by default).
Morgan bits are used specifically because each bit is traceable to the atoms
that set it (RDKit `bitInfo`), which is what enables atom-level SHAP.

### `hsp_xai.qsar`
A scikit-learn classifier (random forest by default; `HistGradientBoosting`
optional) over fingerprints, predicting `P(active)`. Handles class imbalance
with balanced class weights and optional **SMOTE**. Reports held-out
accuracy / F1 / MCC / ROC-AUC, and serializes with `joblib`.

### `hsp_xai.explain`
Wraps SHAP's exact `TreeExplainer`. `explain_bits` gives per-bit SHAP values;
`explain_atoms` projects them onto atoms via the fingerprint bit→atom map;
`global_feature_importance` ranks substructural features across a set; and
`render_atom_heatmap` draws an RDKit similarity-map PNG (red = toward active).

### `hsp_xai.counterfactual`
`explain_by_masking` breaks acyclic bonds, removes the smaller fragment, and
ranks fragments by the activity drop they cause (ablation-based attribution).
`suggest_substitutions` scans a drug-like R-group library at substitutable
atoms and keeps edits that increase predicted activity — actionable, minimal
structural changes.

### `hsp_xai.scoring_component`
Loads the trained QSAR model and scores SMILES in `[0, 1]`. Exposed to
REINVENT4 through the CLI wrapper `scripts/score_smiles.py` using the
`ExternalProcess` contract.

---

## Validation (external)

Top-ranked, interpretable candidates are intended to flow into **molecular
docking** (e.g. against an HSP90 N-terminal ATP-pocket structure) and **MD
simulations** for binding-stability assessment. Those steps use external tools
(e.g. AutoDock Vina, GROMACS/OpenMM) and are not scripted in this repository.

## Notes & caveats

- A fingerprint QSAR model is a **ligand-based surrogate**, not a physics-based
  affinity predictor; treat its scores as a prioritization signal, always
  confirmed by docking/MD and, ultimately, assay.
- ChEMBL pulls require network access; results depend on the current ChEMBL
  release.
- Counterfactual edits are proposals from the model's own view of SAR — they
  need synthetic-feasibility and medicinal-chemistry review.

## License

See [`LICENSE`](LICENSE).
