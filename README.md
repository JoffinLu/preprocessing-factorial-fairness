# Preprocessing Factorial Fairness

Replication package for *Interactions among Imputation, Class-Imbalance Treatment and Fairness Interventions: A Controlled Factorial Study* (manuscript in preparation, Knowledge and Information Systems).

The package regenerates every number reported in the paper from a fixed configuration. It contains the preprocessing pipeline, the factorial runner, the paired analysis, the verification script, the complete result tables and the generated manuscript assets.

## Design

A $3 \times 3 \times 3 \times 2$ full factorial crosses four preprocessing factors in the fixed order $A \rightarrow B \rightarrow C$:

| Factor | Levels |
|---|---|
| A: imputation | A1 training mean, A2 iterative multivariate single imputation (`IterativeImputer`, 10 rounds), A3 KNN ($k=5$) |
| B: imbalance treatment | B1 none, B2 class weights $n/(2n_y)$, B3 within-group minority SMOTE ($k \leq 5$) |
| C: fairness intervention | C1 none, C2 native-group reweighing, C3 training-fitted numerical quantile mapping |
| M: missingness mechanism | M1 MCAR, M2 MAR with always-observed drivers |

Five datasets (Adult, Statlog German Credit, Bank Marketing, Diabetes 130-US, COMPAS), two classifiers (logistic regression, LightGBM) and 20 paired randomisation blocks give **10,800 runs** in the main grid (54 treatments $\times$ 20 blocks $\times$ 5 datasets $\times$ 2 models). Each block fixes the train/test split and the missingness draws for all 54 treatments.

Protocol details that differ from the earlier version of this package:

- imputation statistics and encoding vocabularies are fitted on training rows only, after the split;
- MAR probabilities depend only on always-observed numerical drivers, and one training-fitted plan is reused on test rows with independent mask draws;
- SMOTE interpolates only between minority-class rows of the same protected group, so the group label is never interpolated;
- the repair map is fitted on training rows and frozen for test rows; the transductive DIR is a separately named supplementary arm;
- Diabetes is split by patient, so no patient appears in both partitions;
- Bank Marketing excludes call duration, which is unavailable before a call is completed;
- German Credit uses an age grouping because the legacy status field does not reliably identify sex.

## Repository layout

```
code/
  00_verify_deps.py            dependency self-check
  01_download_and_audit.py     downloads and audits the five public datasets
  05_run_revision.py           resumable factorial runner with provenance manifests
  paired_inference.py          paired seed-block contrasts, BH/BY control, Hotelling tests
  03_statistical_analysis.py   entry point for the paired analysis
  06_verify_revision.py        manifest audit and sampled treatment replay
  07_build_paper_assets.py     manuscript macros and tables from frozen results
  fairness_metrics.py          metric implementations
  pipeline/                    data, missingness, imputation, imbalance,
                               intervention, model, metrics, run_cell
configs/mvp.yaml               frozen configuration
results/revised/               main and supplementary runs, manifests, statistics
environment/                   full development-environment freeze
reproduce.ps1                  one-command reproduction
```

The five datasets are public (UCI Machine Learning Repository and the ProPublica COMPAS release). No data are redistributed here; `data/` is excluded by `.gitignore`.

## Requirements

- Python 3.13
- Windows PowerShell (driver script) or any shell for individual commands
- No GPU

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Reproducing the results

Main grid (about 144 minutes on a 16-core machine with four to six workers, resumable):

```powershell
.\reproduce.ps1 -Jobs 4
```

Reuse the archived main results, recompute statistics and run the sampled replay:

```powershell
.\reproduce.ps1 -SkipRun
```

Supplementary arms (10% and 50% missingness, random oversampling, fixed repair reference, independent seeds 20--24, transductive DIR) are driven by `revision/run_supplements.ps1` in the manuscript workspace; the archived outputs are included under `results/revised/` so the reported checks can be recomputed without rerunning them.

## Expected output

| File | Expected content |
|---|---|
| `results/revised/main.csv` | 10,800 rows, `status` = `ok` for every row |
| `results/revised/main.manifest.json` | configuration, source and data hashes, output checksum, timing |
| `results/revised/stats/paired_contrasts.csv` | 120 primary fairness contrasts plus exploratory families |
| `results/revised/stats/term_tests.csv` | omnibus Hotelling tests |
| `results/revised/stats/cell_means.csv` | cell means for every treatment combination |
| `results/revised/stats/main_effects.csv` | marginal means for every factor level |
| `results/revised/stats/summary.json` | inference target, families, seeds, limitations |
| `results/revised/reproduction_check.json` | 10/10 sampled replay report |

Reproduction is deterministic: all randomness derives from the block seed, LightGBM runs without subsampling on one thread, and `06_verify_revision.py` replays ten representative treatments at absolute tolerance $10^{-12}$.

## Headline results

| Quantity | Value |
|---|---|
| Primary fairness contrasts | 120 (four $B\!:\!C$ pairs $\times$ 5 datasets $\times$ 2 models $\times$ 3 metrics) |
| Passing Benjamini--Hochberg at 0.05 | 46 |
| Passing Benjamini--Yekutieli at 0.05 | 39 |
| Range of primary contrasts | $-0.168$ to $+0.089$ |
| Largest contrast | Adult, logistic regression, EOdd, B2$\times$C2: $-0.168$ (95% interval $-0.188$ to $-0.147$) |
| Stable Adult EO effect across ablations | about $-0.12$ |
| Main-grid wall-clock time | 144 minutes, no GPU |

## Verification status

- All 10,800 main runs completed with status `ok`.
- All 5,400 logistic-regression fits converged.
- Zero undefined subgroup rates and zero train/test patient overlap.
- Source and data hashes are recorded before and checked after each run.
- Sampled replay: 10/10 pass at absolute tolerance $10^{-12}$.

## Citation

If you use this package, please cite the manuscript:

> Lu C (2026) Interactions among imputation, class-imbalance treatment and fairness interventions: a controlled factorial study. Manuscript in preparation, Knowledge and Information Systems.

A machine-readable record is provided in `CITATION.cff`.

## License

MIT. See `LICENSE`.
