# data/

Datasets live here. These files are **git-ignored** by default (see
`.gitignore`) because they are regenerable and can be large.

Expected files (created by the pipeline):

| File                  | Produced by                       | Description                                   |
|-----------------------|-----------------------------------|-----------------------------------------------|
| `hsp90.csv`           | `scripts/fetch_chembl_data.py`    | Curated set: `smiles`, `pchembl`, `active`    |
| `hsp90_actives.smi`   | `scripts/run_pipeline.py`         | Actives only, one SMILES per line (REINVENT TL) |

## Bringing your own data

If you have a curated CSV, point the pipeline at it instead of ChEMBL:

```bash
python scripts/fetch_chembl_data.py --source csv --csv my_raw.csv \
    --smiles-col SMILES --activity-col pChEMBL --out data/hsp90.csv
```

The only hard requirement is a SMILES column. If an activity column is present
it is used to label actives at the `--threshold` pChEMBL cutoff (default 6.0,
i.e. IC50 <= 1 uM); rows without a potency value are dropped for supervised
training.
