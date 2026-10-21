# Preprocessing Factorial Fairness

Replication package for *How Preprocessing Stages Compose: A Factorial Study of Imputation, Imbalance Treatment and Fairness Interventions* (manuscript under review, Knowledge and Information Systems).

The package regenerates every number reported in the paper from a fixed random seed. It contains the preprocessing pipeline, the experiment runner, the statistical analysis, the verification script and the complete result table.

## What the experiment does

A $3 \times 3 \times 3 \times 2$ full factorial crosses four preprocessing factors, in the fixed order $A \rightarrow B \rightarrow C$:

| Factor | Levels |
|---|---|
| A: imputation | A1 mean/mode, A2 MICE (10 iterations), A3 KNN ($k=5$) |
| B: imbalance treatment | B1 none, B2 class weights, B3 SMOTE ($k=5$, training set only) |
| C: fairness intervention | C1 none, C2 Kamiran–Calders reweighing, C3 disparate impact remover (repair level 1.0) |
| M: missingness mechanism | M1 MCAR 30%, M2 MAR 30% |

Five datasets (Adult, German Credit, Bank Marketing, Diabetes 130-US, COMPAS), two model families (logistic regression, LightGBM) and three seeds give 891 preprocessing runs and 1,782 training–evaluation runs: 1,620 in the main design (810 preprocessing units $\times$ 2 models) and 162 in the natural-missingness arm on Adult (81 preprocessing units $\times$ 2 models). All runs are analysed as 50 dataset $\times$ model $\times$ metric ANOVAs with 162 observations each.

## Repository layout

```
code/
  00_verify_deps.py            dependency and algorithm self-check
  01_download_and_audit.py     downloads the five public datasets, audits them
  02_run_experiment.py         runs the factorial design (resumable, parallel)
  03_statistical_analysis.py   Type III ANOVA, partial omega^2, cluster bootstrap,
                               Tukey-Kramer, Benjamini-Hochberg
  04_verify_reproducibility.py re-runs sampled cells and compares bit-for-bit
  fairness_metrics.py          metric implementations
  pipeline/                    data, missingness, imputation, imbalance,
                               intervention, model, metrics, run_cell
configs/mvp.yaml               frozen configuration (no free parameters at run time)
results/full_results.csv       1,620 rows, main design
results/nat_results.csv        162 rows, natural-missingness arm
results/stats/                 five statistical tables
environment/                   full development-environment freeze
reproduce.ps1                  one-command end-to-end reproduction
```

The five datasets are public (UCI Machine Learning Repository and the ProPublica COMPAS release) and are downloaded by `code/01_download_and_audit.py`. No data are redistributed here; `data/` is excluded by `.gitignore`.

## Requirements

- Python 3.13
- Windows PowerShell (the driver script) or any shell for the individual commands
- No GPU

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

`environment/requirements-dev-freeze.txt` records the exact 36-package environment used for the reported run.

## Reproducing the results

Full run (about 133 minutes on one machine, resumable):

```powershell
.\reproduce.ps1 -Jobs 4
```

Reuse existing results and only recompute statistics and the reproducibility check:

```powershell
.\reproduce.ps1 -SkipRun
```

Smoke test (two preprocessing units per dataset):

```powershell
.\reproduce.ps1 -Quick
```

On Linux or macOS, run the same steps directly:

```bash
python code/00_verify_deps.py
python code/01_download_and_audit.py
for m in config data missing features impute imbalance intervention model metrics run_cell; do
  python code/pipeline/$m.py > /dev/null
done
python code/02_run_experiment.py --jobs 4 --out results/full_results.csv --resume
python code/02_run_experiment.py --datasets adult --mechanisms NAT --natural-missing \
  --jobs 4 --out results/nat_results.csv --resume
python code/03_statistical_analysis.py
python code/04_verify_reproducibility.py --n 10
```

## Expected output

| File | Expected size |
|---|---|
| `results/full_results.csv` | 1,620 rows, `status` = `ok` for every row |
| `results/nat_results.csv` | 162 rows (Adult only, `mechanism` = `NAT`) |
| `results/stats/anova_results.csv` | 900 rows (60 analyses $\times$ 15 terms) |
| `results/stats/interaction_cells.csv` | cell means for every two-way interaction |
| `results/stats/main_effects.csv` | marginal means for every factor level |
| `results/stats/tukey_pairs.csv` | pairwise level comparisons |
| `results/stats/gate_summary.csv` | pre-specified analysis gates |

Reproduction is deterministic: all randomness is derived from the seed, LightGBM runs without subsampling on one thread, and `04_verify_reproducibility.py` re-runs sampled cells at zero tolerance.

## Headline results

| Quantity | Value |
|---|---|
| Significant main effects (of 50 analyses) | B 30, M 29, C 22, A 13 |
| Significant two-way terms | B$\times$C 10, M$\times$A 6, M$\times$B 6, A$\times$C 4, M$\times$C 3, A$\times$B 2 |
| Significant three-way or higher terms | 0 of 200 |
| Largest partial $\omega^2$ | 0.993 (B, Diabetes/LightGBM/F1) |
| Wall-clock time | 133 minutes, no GPU |

## Citation

If you use this package, please cite the manuscript:

> Lu C (2026) How preprocessing stages compose: a factorial study of imputation, imbalance treatment and fairness interventions. Manuscript under review, Knowledge and Information Systems.

A machine-readable record is provided in `CITATION.cff`.

## License

MIT. See `LICENSE`.
