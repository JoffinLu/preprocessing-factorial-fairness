"""Generate manuscript numbers/tables from complete revised results only."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from pipeline.data import load_dataset

DATASETS = ("adult", "german", "bank", "diabetes", "compas")
DISPLAY = {"adult": "Adult", "german": "German", "bank": "Bank", "diabetes": "Diabetes", "compas": "COMPAS"}
OUT = ROOT / "paper/latex/generated"


def tex(value):
    return str(value).replace("_", r"\_").replace("%", r"\%")


def dataset_table():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = {"adult": "sex", "german": r"age $>25$", "bank": "median age", "diabetes": "race", "compas": "race"}
    rows = []
    records = []
    for name in DATASETS:
        d = load_dataset(name)
        n = len(d.y)
        counts = d.sensitive.value_counts().to_dict()
        ratio = min(int((d.y == 0).sum()), int(d.y.sum())) / max(int((d.y == 0).sum()), int(d.y.sum()))
        group_labels = {"german": {"1": "Age $>25$", "0": "Age $\\leq25$"},
                        "bank": {"1": "Age $>$ median", "0": "Age $\\leq$ median"}}
        groups = "; ".join(f"{group_labels.get(name, {}).get(str(g), tex(g))}: {int(k):,}"
                           for g, k in counts.items())
        rows.append(f"{DISPLAY[name]} & {n:,} & {ratio:.3f} & {labels[name]} & {groups} " + r"\\")
        records.append({"dataset": name, "n": n, "n_positive": int(d.y.sum()), "ratio": ratio,
                        "groups": {str(k): int(v) for k,v in counts.items()},
                        "n_patients": None if d.groups is None else len(np.unique(d.groups)),
                        "n_natural_missing": int(d.X_raw.isna().sum().sum())})
    text = r"""\begin{table}[tb]
\centering\small
\caption{Fixed analysis cohorts. Ratio is minority/majority class count. Group sizes refer to the full analysis cohort, before a random split.}\label{tab:data}
\begin{tabular}{@{}lrrp{1.7cm}p{6.3cm}@{}}
\toprule
Dataset & Rows & Ratio & Audit group & Group sizes\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "datasets_table.tex").write_text(text, encoding="utf-8")
    (OUT / "dataset_facts.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def read_complete(name, expected):
    path = ROOT / f"results/revised/{name}.csv"
    mp = path.with_suffix(".manifest.json")
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    if manifest['status'] != 'complete':
        raise ValueError(f"{name} is not complete")
    if manifest['output_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError(f"{name} checksum mismatch")
    d = pd.read_csv(path, float_precision='round_trip')
    if len(d) != expected or not d.status.eq('ok').all():
        raise ValueError(f"{name}: incorrect run count/status")
    if d.duplicated(['dataset','mechanism','A','B','C','seed','model']).any():
        raise ValueError(f"{name}: duplicate runs")
    return d, manifest


def number(value, digits=3):
    return f"{float(value):.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets-only', action='store_true')
    args = parser.parse_args()
    dataset_table()
    if args.datasets_only:
        return
    d, manifest = read_complete('main', 10800)
    stats = ROOT / 'results/revised/stats'
    p = pd.read_csv(stats / 'paired_contrasts.csv')
    primary = p[p.family.eq('primary_BC_fairness')].copy()
    if len(primary) != 120 or set(primary.n_seeds) != {20}:
        raise ValueError('Expected exactly 120 primary tests with 20 shared blocks')
    # Recompute the main estimand independently of the inference implementation.
    for r in primary.itertuples():
        z = d[d.dataset.eq(r.dataset)&d.model.eq(r.model_name)]
        means = z.groupby(['B','C'])[r.metric].mean()
        effect = means.loc[(r.B_to,r.C_to)] - means.loc[(r.B_to,'C1')] - means.loc[('B1',r.C_to)] + means.loc[('B1','C1')]
        if not np.isclose(effect, r.effect, atol=1e-12, rtol=0):
            raise ArithmeticError('Manuscript contrast does not match run table')
    macros = {}
    def macro(key, value):
        words = {'1':'One', '2':'Two', '3':'Three', '10':'Ten', '30':'Thirty', '50':'Fifty'}
        key = re.sub(r'\d+', lambda m: words[m.group()], key)
        macros[key] = str(value)
    macro('MainRuns', f'{len(d):,}')
    macro('MainBlocks', d.seed.nunique())
    macro('PrimaryTests', len(primary))
    macro('PrimaryBH', int(primary.q_value.lt(.05).sum()))
    macro('PrimaryBY', int(primary.q_by.lt(.05).sum()))
    from datetime import datetime
    elapsed = (datetime.fromisoformat(manifest['finished_utc'])
               - datetime.fromisoformat(manifest['started_utc'])).total_seconds()
    macro('MainMinutes', f"{elapsed/60:.0f}")
    macro('MainHours', f"{elapsed/3600:.1f}")
    macro('UndefinedGroups', int(d.undefined_tpr_groups.max()))
    macro('PatientOverlap', int(d.patient_overlap.max()))
    macro('LRConverged', int(d[d.model.eq('LR')].lr_converged.fillna(False).astype(bool).sum()))
    macro('LRRuns', int(d.model.eq('LR').sum()))
    for metric in ('DP','EO','EOdd'):
        macro('BH'+metric, int(primary[primary.metric.eq(metric)].q_value.lt(.05).sum()))
        macro('BY'+metric, int(primary[primary.metric.eq(metric)].q_by.lt(.05).sum()))
    raw_means = d.groupby(['dataset','model','B','C'])[['DP','EO','EOdd','F1','AUC','AP','n_pred_pos']].mean()
    for ds in ('adult','diabetes','compas'):
        for b,c in [('B1','C1'),('B1','C2'),('B1','C3'),('B2','C1'),('B2','C2'),('B2','C3'),('B3','C1'),('B3','C2'),('B3','C3')]:
            for metric in ('DP','EO','F1','AUC','AP'):
                macro(ds.capitalize()+b+c+metric, number(raw_means.loc[(ds,'LR',b,c),metric]))
    rows = []
    for ds in DATASETS:
        for model in ('LR','LGBM'):
            for b,c in [('B2','C2'),('B3','C3')]:
                r = primary[primary.dataset.eq(ds)&primary.model_name.eq(model)&primary.metric.eq('EO')&primary.B_to.eq(b)&primary.C_to.eq(c)].iloc[0]
                q = r.q_value
                qtext = '--' if not np.isfinite(q) else '$<0.001$' if q < .001 else number(q)
                rows.append(f"{DISPLAY[ds]} & {model} & {b}$\\times${c} & {r.effect:+.3f} & [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & {qtext} " + r"\\")
    (OUT/'contrasts_table.tex').write_text(r"""\begin{table}[tb]
\centering\small
\caption{Two prespecified EO interaction contrasts for every dataset and classifier. Contrasts are relative to B1 and C1, averaged equally over A and M. Intervals are pointwise 95\% conditional Monte Carlo $t$ intervals; BH adjustment uses all 120 primary fairness contrasts.}\label{tab:contrasts}
\begin{tabular}{@{}lllrrr@{}}
\toprule Dataset & Model & Contrast & DID & 95\% interval & $p_{\rm BH}$\\\midrule
"""+'\n'.join(rows)+r"""
\bottomrule\end{tabular}
\end{table}
""",encoding='utf-8')
    rows=[]
    for ds in ('adult','diabetes','compas'):
        for b,c in [('B1','C1'),('B1','C2'),('B2','C1'),('B2','C2'),('B3','C1'),('B3','C3')]:
            r=raw_means.loc[(ds,'LR',b,c)]
            rows.append(f"{DISPLAY[ds]} & {b}{c} & {r.DP:.3f} & {r.EO:.3f} & {r.F1:.3f} & {r.AP:.3f} " + r"\\")
    (OUT/'case_means_table.tex').write_text(r"""\begin{table}[tb]
\centering\small
\caption{Logistic-regression case means, equally averaged over A, M and the 20 blocks. These are bounded case comparisons, not rules inferred from class-imbalance ratios. AP is average precision.}\label{tab:cases}
\begin{tabular}{@{}llrrrr@{}}
\toprule Dataset & Configuration & DP & EO & F1 & AP\\\midrule
"""+'\n'.join(rows)+r"""
\bottomrule\end{tabular}
\end{table}
""",encoding='utf-8')
    # Supplement status is a hard gate before final manuscript numbers exist.
    for name,n in [('validation30',2700),('rate10',2700),('rate50',2700),('random_over',900),('fixed_reference',900),('transductive_dir',160)]:
        supp, sm = read_complete(name,n)
        macro('Runs'+name.replace('_','').capitalize(), len(supp))

    def arm_counts(stats_name, key):
        q = pd.read_csv(ROOT / f'results/revised/{stats_name}/paired_contrasts.csv')
        q = q[q.family.eq('primary_BC_fairness')]
        macro('BH'+key, int(q.q_value.lt(.05).sum()))
        macro('BY'+key, int(q.q_by.lt(.05).sum()))
        macro('N'+key, len(q))
        macro('Min'+key, f'{q.effect.min():.3f}')
        macro('Max'+key, f'{q.effect.max():.3f}')
    for stats_name, key in [('stats_main_a1','MainAone'), ('stats_validation30','Validation'),
                            ('stats_validation_a1','ValidationAone'), ('stats_rate10','RateTen'),
                            ('stats_rate50','RateFifty'), ('stats_random_over','RandomOver'),
                            ('stats_fixed_reference','FixedReference'),
                            ('stats_transductive','Transductive')]:
        arm_counts(stats_name, key)
    for filename in ('results_macros.tex','paper_facts.json'):
        path=OUT/filename
        if filename.endswith('.tex'):
            path.write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n',encoding='utf-8')
        else:
            path.write_text(json.dumps({'macros':macros,'main_manifest':manifest,'primary_contrasts':primary.to_dict(orient='records')},indent=2),encoding='utf-8')
    print('Generated datasets, checked primary contrasts, manuscript macros and tables')


if __name__ == '__main__':
    main()
