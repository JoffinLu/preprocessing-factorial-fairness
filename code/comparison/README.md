# Comparison arm

This directory replaces the C stage of the main grid with one of three published fairness methods:

- **AIM** (KDD 2024), `third_party/AIM`
- **TabFairGDT** (ICDM 2025), installed from PyPI as `tabfairgdt`
- **APW** (arXiv:2309.08375), reimplemented from the published update rule

The family contains 90 runs: 3 methods x 3 datasets (Adult, COMPAS, Diabetes) x 2 missingness mechanisms (MCAR, MAR at 30%) x 5 blocks (seeds 0--4). Every arm uses the same splits, masks and A1 reference imputation as the main grid.

## Run

From the repository root, after the public datasets have been downloaded:

```powershell
python code/comparison/run_comparison.py --out results/revised/comparison/comparison.csv
```

The runner appends rows incrementally and writes a provenance manifest next to the output CSV. Use `--limit` for a smoke test and `--method`, `--dataset`, `--mechanism`, `--seeds` to restrict the grid.

## Implementation constraints

- AIM builds a full proximity matrix. Adult and Diabetes are therefore reduced to 5,000 stratified training rows; COMPAS is used in full.
- APW has no finite fixed point when a group receives no positive prediction. Its four fairness multipliers are clipped to `[1e-3, 1e3]` and renormalised each epoch. Clipping was triggered in seven of the ten COMPAS runs and in none of the Adult or Diabetes runs.
- All methods are complete-data learners. Each arm first receives A1 mean/mode imputation, so no arm sees a different amount of missing information.

## Outputs

- `results/revised/comparison/comparison.csv`
- `results/revised/comparison/comparison.manifest.json`
- `results/revised/comparison/comparison_summary.csv`
- `results/revised/comparison/tables/` (manuscript tables)
