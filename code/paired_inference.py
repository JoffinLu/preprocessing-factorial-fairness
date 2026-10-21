"""Paired inference conditional on fixed benchmark datasets.

Target: E_seed[metric(run)], or a prespecified linear contrast thereof, with
datasets and factor levels fixed. Each seed is one joint randomisation block
containing every treatment cell. Different seed blocks are assumed independent
draws of the specified split/mask/algorithm randomisation, conditional on data.
Repeated overlapping splits create no independent patients or population-level
or causal generalisation.

Scalar inference uses a one-sample t approximation on block-level contrasts,
df=n_seeds-1. One bootstrap index matrix resamples complete seed blocks jointly
across metrics and contrasts. Percentile intervals and sign fractions describe
stability, not posterior probabilities or another hypothesis test. Neither
method remedies too few blocks. Intervals are pointwise, not simultaneous.

The primary BH family is four M/A-marginal B:C differences-in-differences x all
included dataset/model pairs x DP, EO, EOdd (120 tests in the main experiment).
It is not described as confirmatory: earlier results have been inspected.
Other contrasts have explicit exploratory families. BY adjustment is supplied
as a sensitivity check for arbitrary dependence; BH requires suitable null
p-value independence/positive dependence.

Fixed-grid orthogonal SS are descriptive. eta2_total partitions observed-run
variation; eta2_design partitions estimated treatment-cell-mean variation.
No partial omega-squared or pooled independent-error ANOVA tests are used.
Secondary main/two-way Hotelling T2 tests require independent multivariate-
normal seed-contrast vectors; insufficient/rank-deficient cases are skipped.
No inference of absent higher-order interactions is made.
"""
from __future__ import annotations
import itertools
import json
import math
from functools import reduce
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist

FACTORS = ('M', 'A', 'B', 'C')
LEVELS = {'M': ('M1', 'M2'), 'A': ('A1', 'A2', 'A3'),
          'B': ('B1', 'B2', 'B3'), 'C': ('C1', 'C2', 'C3')}
PRIMARY_METRICS = ('DP', 'EO', 'EOdd')
SECONDARY_METRICS = ('AUC', 'F1')
METRICS = PRIMARY_METRICS + SECONDARY_METRICS
TARGET = 'fixed_dataset_expectation_over_joint_seed_randomisation'
IDENTIFIERS = ['dataset', 'model_name', 'metric']
CONTRAST_IDENTIFIERS = ['mechanism_scope', 'contrast_type', 'contrast_id',
                        'B_from', 'B_to', 'C_from', 'C_to']
SCALAR_COLUMNS = ['effect', 'n_seeds', 'sd', 'se', 'df', 'ci_lo', 'ci_hi',
                  't_stat', 'p_value', 'boot_ci_lo', 'boot_ci_hi',
                  'bootstrap_positive_fraction', 'bootstrap_negative_fraction',
                  'bootstrap_sign_fraction', 'bootstrap_runs', 'precision',
                  'test_status', 'inference_target']


def load_blocks(path: Path, expected_seeds: int = 20,
                allow_partial_grid: bool = False) -> tuple[pd.DataFrame, dict, int]:
    """Reject failed runs, duplicate cells, or incomplete shared seed blocks."""
    if expected_seeds < 2:
        raise ValueError('At least two seed blocks are required')
    df = pd.read_csv(path)
    if 'M' not in df and 'mechanism' in df:
        df = df.rename(columns={'mechanism': 'M'})
    if 'model_name' not in df and 'model' in df:
        df = df.rename(columns={'model': 'model_name'})
    excluded = 0
    if 'natural_missing' in df:
        values = df.natural_missing.astype(str).str.lower().str.strip()
        if not values.isin(['true', 'false', '1', '0']).all():
            raise ValueError('natural_missing contains unknown values')
        nat = values.isin(['true', '1'])
        excluded = int(nat.sum())
        df = df.loc[~nat].copy()
    required = {'dataset', 'model_name', 'seed', *FACTORS, *METRICS}
    if required - set(df):
        raise ValueError(f'Missing input columns: {sorted(required - set(df))}')
    if df.empty:
        raise ValueError('No controlled experiment rows remain')
    if 'status' in df and not df.status.eq('ok').all():
        raise ValueError('Failed/non-ok runs must be rerun before inference')
    for column in ('undefined_tpr_groups', 'undefined_sel_groups'):
        if column in df and (pd.to_numeric(df[column], errors='raise') != 0).any():
            raise ValueError(f'Undefined subgroup rates in {column}; complete-group inference is unavailable')
    if 'protocol_version' in df and not df.protocol_version.eq('revision_v2').all():
        raise ValueError('Input is not uniformly from revision_v2')
    if df[list(required)].isna().any().any():
        raise ValueError('Missing identifiers or undefined metrics; cannot drop blocks')
    for col in FACTORS:
        df[col] = df[col].astype(str)
    for col in METRICS:
        df[col] = pd.to_numeric(df[col], errors='raise')
        if not np.isfinite(df[col]).all():
            raise ValueError(f'Non-finite metric: {col}')
    if df.duplicated(['dataset', 'model_name', 'seed', *FACTORS]).any():
        raise ValueError('Duplicated seed/treatment cells')
    all_seeds = sorted(df.seed.unique().tolist())
    if len(all_seeds) != expected_seeds:
        raise ValueError(f'Expected {expected_seeds} seed blocks, found {len(all_seeds)}')
    group_levels = {}
    for key, sub in df.groupby(['dataset', 'model_name'], sort=True):
        observed = {fac: tuple(sorted(sub[fac].unique())) for fac in FACTORS}
        if not allow_partial_grid and observed != LEVELS:
            raise ValueError(f'{key}: expected the full 2 x 3 x 3 x 3 factor grid')
        for fac in FACTORS:
            if fac != 'M' and not set(observed[fac]).issubset(LEVELS[fac]):
                raise ValueError(f'{key}: unknown {fac} levels {observed[fac]}')
        grid = set(itertools.product(*(observed[fac] for fac in FACTORS)))
        if sorted(sub.seed.unique()) != all_seeds:
            raise ValueError(f'{key}: seed sets must agree across dataset/model pairs')
        for seed, block in sub.groupby('seed', sort=True):
            actual = set(block[list(FACTORS)].itertuples(index=False, name=None))
            if actual != grid or len(block) != len(grid):
                raise ValueError(f'{key}, seed {seed}: incomplete Cartesian treatment block')
        group_levels[key] = observed
    return df, group_levels, excluded


def adjusted_pvalues(values: np.ndarray, arbitrary_dependence: bool = False) -> np.ndarray:
    """BH/BY adjustment preserving NaN for skipped tests."""
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    p = values[finite]
    if not len(p):
        return result
    order = np.argsort(p, kind='stable')
    m = len(p)
    multiplier = sum(1.0 / k for k in range(1, m + 1)) if arbitrary_dependence else 1.0
    ranked = p[order] * m * multiplier / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    restored = np.empty(m)
    restored[order] = ranked
    result[finite] = restored
    return result


def correct_families(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    out = table.copy()
    out['q_value'] = np.nan
    out['q_by'] = np.nan
    out['family_size'] = 0
    out['family_tested'] = 0
    for _, indices in out.groupby('family', sort=True).groups.items():
        p = out.loc[indices, 'p_value'].to_numpy(dtype=float)
        # Skipped tests still count in the declared family, conservatively p=1.
        for column, by in [('q_value', False), ('q_by', True)]:
            adjusted = adjusted_pvalues(np.where(np.isfinite(p), p, 1.0), by)
            adjusted[~np.isfinite(p)] = np.nan
            out.loc[indices, column] = adjusted
        out.loc[indices, 'family_size'] = len(indices)
        out.loc[indices, 'family_tested'] = int(np.isfinite(p).sum())
    return out


def scalar_summary(values: np.ndarray, bootstrap_indices: np.ndarray,
                   alpha: float = 0.05) -> dict:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2 or not np.isfinite(values).all():
        raise ValueError('Require >=2 finite seed-block values')
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    se = sd / math.sqrt(n)
    tolerance = 1e-13 * max(1.0, float(np.abs(values).max()))
    if sd <= tolerance:
        statistic = np.nan
        p_value = 1.0 if abs(mean) <= tolerance else np.nan
        status = 'zero_sample_variance'
    else:
        statistic = mean / se
        p_value = float(2 * t_dist.sf(abs(statistic), n - 1))
        status = 'ok_t_approximation'
    halfwidth = float(t_dist.ppf(1 - alpha / 2, n - 1) * se)
    boot = values[bootstrap_indices].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    positive, negative = float(np.mean(boot > 0)), float(np.mean(boot < 0))
    sign_fraction = positive if mean > 0 else negative if mean < 0 else np.nan
    return {'effect': mean, 'n_seeds': n, 'sd': sd, 'se': se, 'df': n - 1,
            'ci_lo': mean - halfwidth, 'ci_hi': mean + halfwidth,
            't_stat': statistic, 'p_value': p_value,
            'boot_ci_lo': float(lo), 'boot_ci_hi': float(hi),
            'bootstrap_positive_fraction': positive,
            'bootstrap_negative_fraction': negative,
            'bootstrap_sign_fraction': sign_fraction,
            'bootstrap_runs': len(bootstrap_indices),
            'precision': 'limited_seed_replication' if n < 20 else '20_or_more_seed_blocks',
            'test_status': status, 'inference_target': TARGET}


def _basis(levels: dict, term: tuple[str, ...]) -> np.ndarray:
    pieces = []
    for factor in FACTORS:
        n = len(levels[factor])
        pieces.append(helmert(n, full=False).T if factor in term
                      else np.ones((n, 1)) / math.sqrt(n))
    return reduce(np.kron, pieces)


def hotelling(values: np.ndarray) -> dict:
    """Do not replace deficient covariance rank with a pseudoinverse."""
    values = np.asarray(values, dtype=float)
    n, dimension = values.shape
    result = {'n_seeds': n, 'dimension': dimension, 'T2': np.nan, 'F': np.nan,
              'df_num': dimension, 'df_den': n - dimension, 'p_value': np.nan,
              'covariance_condition': np.nan, 'test_status': 'insufficient_blocks',
              'assumption': 'independent_multivariate_normal_seed_contrast_vectors',
              'inference_target': TARGET,
              'precision': 'limited_seed_replication' if n < 20 else '20_or_more_seed_blocks'}
    if dimension == 0 or n <= dimension:
        return result
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    tolerance = (1e-13 * max(1.0, float(np.abs(values).max()))) ** 2
    if np.linalg.eigvalsh(covariance).min() <= tolerance:
        result['test_status'] = 'singular_covariance'
        return result
    if np.linalg.matrix_rank(covariance) < dimension:
        result['test_status'] = 'singular_covariance'
        return result
    condition = float(np.linalg.cond(covariance))
    result['covariance_condition'] = condition
    if not np.isfinite(condition) or condition > 1e12:
        result['test_status'] = 'ill_conditioned_covariance'
        return result
    mean = values.mean(axis=0)
    t2 = float(n * mean @ np.linalg.solve(covariance, mean))
    statistic = (n - dimension) * t2 / (dimension * (n - 1))
    result.update(T2=t2, F=statistic,
                  p_value=float(f_dist.sf(statistic, dimension, n - dimension)),
                  test_status='ok_normal_vector_approximation')
    return result


def decompose(values: np.ndarray, levels: dict) -> tuple[list[dict], list[dict]]:
    """Exact descriptive SS partition of a balanced seed-by-treatment array."""
    n, n_cells = values.shape
    grand, means = float(values.mean()), values.mean(axis=0)
    total = float(np.square(values - grand).sum())
    between = float(n * np.square(means - grand).sum())
    rows, tests = [], []
    active = [fac for fac in FACTORS if len(levels[fac]) > 1]
    for order in range(1, len(active) + 1):
        for term in itertools.combinations(active, order):
            coefficients = values @ _basis(levels, term)
            ss = float(n * np.square(coefficients.mean(axis=0)).sum())
            label = ':'.join(term)
            rows.append({'term': label, 'term_order': order, 'component': 'fixed_treatment',
                         'sum_sq': ss, 'df': coefficients.shape[1],
                         'eta2_total': ss / total if total > 0 else np.nan,
                         'eta2_design': ss / between if between > 0 else np.nan,
                         'ss_total': total, 'ss_design': between,
                         'n_seeds': n, 'n_cells': n_cells})
            if order <= 2:
                tests.append({'term': label, **hotelling(coefficients)})
    common = float(n_cells * np.square(values.mean(axis=1) - grand).sum())
    interaction = values - means[None, :] - values.mean(axis=1)[:, None] + grand
    residual = float(np.square(interaction).sum())
    for term, ss, dof in [('seed_common', common, n - 1),
                           ('seed_by_cell', residual, (n - 1) * (n_cells - 1))]:
        rows.append({'term': term, 'term_order': 0, 'component': 'seed_variation',
                     'sum_sq': ss, 'df': dof,
                     'eta2_total': ss / total if total > 0 else np.nan,
                     'eta2_design': np.nan, 'ss_total': total, 'ss_design': between,
                     'n_seeds': n, 'n_cells': n_cells})
    if not np.isclose(sum(row['sum_sq'] for row in rows), total, atol=1e-10, rtol=1e-9):
        raise ArithmeticError('Orthogonal SS partition failed')
    return rows, tests


def _family(metric: str, kind: str, scope: str) -> str:
    fairness = metric in PRIMARY_METRICS
    if kind == 'BC_difference_in_differences' and scope == 'marginal':
        return 'primary_BC_fairness' if fairness else 'exploratory_BC_performance'
    if kind == 'BC_difference_in_differences':
        return ('exploratory_BC_stratified_fairness' if fairness
                else 'exploratory_BC_stratified_performance')
    suffix = 'fairness' if fairness else 'performance'
    return f'exploratory_{kind}_{suffix}'


def analyse(input_path: Path, expected_seeds: int = 20, n_boot: int = 5000,
            bootstrap_seed: int = 20261021, alpha: float = 0.05,
            allow_partial_grid: bool = False) -> dict[str, Any]:
    if n_boot < 100:
        raise ValueError('Use at least 100 joint bootstrap resamples')
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie between zero and one')
    data, group_levels, excluded = load_blocks(input_path, expected_seeds, allow_partial_grid)
    seeds = sorted(data.seed.unique().tolist())
    boot_idx = np.random.default_rng(bootstrap_seed).integers(
        0, len(seeds), size=(n_boot, len(seeds)))
    contrasts, block_rows, cell_rows, main_rows, descriptive_rows, term_rows = [], [], [], [], [], []
    for (dataset, model), subset in data.groupby(['dataset', 'model_name'], sort=True):
        levels = group_levels[(dataset, model)]
        grid = list(itertools.product(*(levels[fac] for fac in FACTORS)))
        coords = {fac: np.array([cell[i] for cell in grid]) for i, fac in enumerate(FACTORS)}
        index = pd.MultiIndex.from_tuples(grid, names=FACTORS)
        for metric in METRICS:
            values = np.stack([subset[subset.seed == seed].set_index(list(FACTORS))[metric]
                               .reindex(index).to_numpy(dtype=float) for seed in seeds])
            ids = {'dataset': dataset, 'model_name': model, 'metric': metric}
            for cell_index, cell in enumerate(grid):
                stat = scalar_summary(values[:, cell_index], boot_idx, alpha)
                # A confidence interval for a cell mean is useful; a test of
                # that mean against zero is not one of the research questions.
                stat.pop('p_value')
                stat.pop('t_stat')
                cell_rows.append({**ids, **dict(zip(FACTORS, cell)), **stat, 'mean': stat['effect']})
            for factor in FACTORS:
                for level in levels[factor]:
                    pick = coords[factor] == level
                    stat = scalar_summary(values[:, pick].mean(axis=1), boot_idx, alpha)
                    stat.pop('p_value')
                    stat.pop('t_stat')
                    main_rows.append({**ids, 'factor': factor, 'level': level,
                                      'n_cells': int(pick.sum()),
                                      'n_obs': int(pick.sum() * len(seeds)),
                                      **stat, 'mean': stat['effect']})
            decomposition, tests = decompose(values, levels)
            descriptive_rows.extend({**ids, **row} for row in decomposition)
            for row in tests:
                family = ('secondary_omnibus_fairness' if metric in PRIMARY_METRICS
                          else 'secondary_omnibus_performance')
                term_rows.append({**ids, **row, 'family': family})

            def add(block_values: np.ndarray, scope: str, kind: str, identifier: str,
                    b_from: str = '', b_to: str = '', c_from: str = '', c_to: str = '') -> None:
                labels = {**ids, 'mechanism_scope': scope, 'contrast_type': kind,
                          'contrast_id': identifier, 'B_from': b_from, 'B_to': b_to,
                          'C_from': c_from, 'C_to': c_to}
                contrasts.append({**labels, **scalar_summary(block_values, boot_idx, alpha),
                                  'family': _family(metric, kind, scope)})
                block_rows.extend({**labels, 'seed': seed, 'value': float(value)}
                                  for seed, value in zip(seeds, block_values))

            dids = {}
            for scope in ['marginal', *levels['M']]:
                mask = (np.ones(len(grid), dtype=bool) if scope == 'marginal' else coords['M'] == scope)

                def mean_bc(b: str, c: str) -> np.ndarray:
                    pick = mask & (coords['B'] == b) & (coords['C'] == c)
                    if not pick.any():
                        raise ValueError('Attempted to estimate an absent B:C cell')
                    return values[:, pick].mean(axis=1)

                if 'C1' in levels['C']:
                    for b in levels['B']:
                        for c in ('C2', 'C3'):
                            if c in levels['C']:
                                add(mean_bc(b, c) - mean_bc(b, 'C1'), scope,
                                    'C_simple_effect', f'C_{c}vC1_at_{b}', b, b, 'C1', c)
                if 'B1' not in levels['B'] or 'C1' not in levels['C']:
                    continue
                for b, c in itertools.product(('B2', 'B3'), ('C2', 'C3')):
                    if b not in levels['B'] or c not in levels['C']:
                        continue
                    did = mean_bc(b, c) - mean_bc(b, 'C1') - mean_bc('B1', c) + mean_bc('B1', 'C1')
                    dids[(scope, b, c)] = did
                    add(did, scope, 'BC_difference_in_differences', f'BC_{b}vB1_{c}vC1', 'B1', b, 'C1', c)
            if {'M1', 'M2'}.issubset(levels['M']):
                mdiff = values[:, coords['M'] == 'M2'].mean(axis=1) - values[:, coords['M'] == 'M1'].mean(axis=1)
                add(mdiff, 'M2_minus_M1', 'M_main_contrast', 'M2vM1')
                for b, c in itertools.product(('B2', 'B3'), ('C2', 'C3')):
                    if ('M1', b, c) in dids and ('M2', b, c) in dids:
                        add(dids[('M2', b, c)] - dids[('M1', b, c)], 'M2_minus_M1',
                            'BC_mechanism_moderation', f'M2vM1_BC_{b}vB1_{c}vC1', 'B1', b, 'C1', c)

    paired = correct_families(pd.DataFrame(contrasts, columns=[
        *IDENTIFIERS, *CONTRAST_IDENTIFIERS, *SCALAR_COLUMNS, 'family']))
    if paired.empty:
        for col in ['q_value', 'q_by', 'family_size', 'family_tested']:
            paired[col] = pd.Series(dtype=float)
    term_tests = correct_families(pd.DataFrame(term_rows, columns=[
        *IDENTIFIERS, 'term', 'n_seeds', 'dimension', 'T2', 'F', 'df_num', 'df_den',
        'p_value', 'covariance_condition', 'test_status', 'assumption',
        'inference_target', 'precision', 'family']))
    if term_tests.empty:
        for col in ['q_value', 'q_by', 'family_size', 'family_tested']:
            term_tests[col] = pd.Series(dtype=float)
    families = paired.groupby('family').size().astype(int).to_dict() if not paired.empty else {}
    summary = {
        'schema_version': 2, 'input': str(Path(input_path).resolve()),
        'inference_target': TARGET,
        'target_description': 'Expected run metric over split/mask/algorithm randomisation, '
                              'conditional on fixed datasets and the finite grid; metrics '
                              'are averaged after evaluation, not computed from pooled predictions.',
        'sampling_unit': 'complete joint seed block', 'n_rows': len(data),
        'n_seed_blocks': len(seeds), 'seeds': [int(seed) for seed in seeds],
        'n_dataset_model_pairs': len(group_levels), 'excluded_natural_missing_rows': excluded,
        'allow_partial_grid': allow_partial_grid,
        'primary_metrics': list(PRIMARY_METRICS), 'secondary_metrics': list(SECONDARY_METRICS),
        'excluded_redundant_metric': 'TPR_gap equals EO and is not analysed twice',
        'primary_family': 'primary_BC_fairness',
        'primary_family_n_tests': int(families.get('primary_BC_fairness', 0)),
        'primary_family_definition': 'all estimable four reference-coded B:C DID marginal over M/A '
                                     'across included dataset/model and DP/EO/EOdd',
        'multiplicity_families': families,
        'multiplicity': 'BH within declared cross-dataset/model/metric families; BY sensitivity q_by; '
                        'skipped tests remain counted in family size',
        'alpha': alpha, 'intervals': 'pointwise t approximation and seed-block percentile bootstrap',
        'bootstrap_runs': n_boot, 'bootstrap_seed': bootstrap_seed,
        'bootstrap_joint': 'one common seed-index matrix across every metric and contrast',
        'bootstrap_sign_fraction': 'fraction with observed estimate sign; stability, not probability of truth',
        'descriptive_SS': 'orthogonal observed-run SS; eta2_total includes seed variation; '
                          'eta2_design uses treatment means only; no partial omega2',
        'secondary_omnibus': 'Hotelling T2 on main/two-way vectors, independent multivariate-normal '
                             'blocks assumed; dimension>=n or deficient covariance causes skip',
        'limitations': ['Overlapping splits create no new independent patients.',
                        'No population-level or causal generalisation.',
                        'Fewer than 20 blocks are flagged as limited replication.',
                        'Pointwise intervals are not simultaneous.',
                        'Previously inspected results preclude a preregistered confirmatory label.',
                        'No absence of higher-order interactions is inferred.']}
    return {'paired_contrasts': paired, 'block_contrasts': pd.DataFrame(
                block_rows, columns=[*IDENTIFIERS, *CONTRAST_IDENTIFIERS, 'seed', 'value']),
            'cell_means': pd.DataFrame(cell_rows), 'main_effects': pd.DataFrame(main_rows),
            'descriptive_decomposition': pd.DataFrame(descriptive_rows),
            'term_tests': term_tests, 'summary': summary}


def write_results(result: dict[str, Any], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in result.items():
        if name != 'summary':
            table.to_csv(out_dir / f'{name}.csv', index=False, encoding='utf-8-sig')
    (out_dir / 'summary.json').write_text(
        json.dumps(result['summary'], ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
