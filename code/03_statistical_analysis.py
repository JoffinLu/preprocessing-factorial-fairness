"""阶段 5：统计分析（四因子 ANOVA + 效应量 + Bootstrap CI + Tukey/FDR）

输入
    results/full_results.csv（阶段 4 主设计：5 数据集 × M×A×B×C × 3 种子 × 2 模型）

分析口径（与范围说明书 §4.4–§4.5、E1 一致）
    1. 每个 (数据集, 模型) 单独拟合一个四因子全因子模型
           y ~ M + A + B + C + 全部二阶交互 + 三阶 + 四阶
       Type Ⅲ 平方和（sum-to-zero 对照），种子作为重复（replicate）。
       划分只依赖 (dataset, seed)（E1），故同一 seed 下的 54 个单元格构成配对块，
       种子进入残差即等价于「配对差值」的方差，而不是把划分噪声算进因子效应。
    2. 效应量：η²（经典，SS_效应/SS_总）、partial η²、partial ω²。
       主报告 partial ω²（无偏、可跨表比较），η² 仅作参考。
    3. 多重比较：对每个因子的水平对做 Tukey–Kramer（用全模型 MSE 与残差 df），
       再在 (数据集, 模型, 指标) 内对全部水平对做 Benjamini–Hochberg FDR。
    4. 边际均值与交互单元均值的 95% CI：**簇 bootstrap**（以单元格为簇，
       簇内 3 个种子一并重抽），避免把同一种子的 6 个单元格当独立观测。
    5. 指标：公平 DP / EO / EOdd / TPR_gap；性能 AUC / F1。

输出（results/stats/）
    anova_results.csv      全部 (数据集, 模型, 指标, 效应项) 的 SS/df/F/p/η²/ω²
    main_effects.csv       各因子水平的边际均值 + 95% CI
    interaction_cells.csv  A×B / A×C / B×C / A×M 的单元均值 + 95% CI（Figure 4）
    interaction_tests.csv  二阶交互的 F/p/ω² 与「Δ 效应」（Figure 5）
    tukey_pairs.csv        水平对差异 + Tukey p + BH-FDR
    gate_summary.csv       Gate 5 判定：每个 (数据集, 模型, 指标) 有几个显著二阶交互
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import studentized_range
from statsmodels.stats.multitest import multipletests

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
IN_CSV = ROOT / "results" / "full_results.csv"
OUT_DIR = ROOT / "results" / "stats"

FACTORS: tuple[str, ...] = ("M", "A", "B", "C")
LEVELS: dict[str, tuple[str, ...]] = {
    "M": ("M1", "M2"), "A": ("A1", "A2", "A3"),
    "B": ("B1", "B2", "B3"), "C": ("C1", "C2", "C3"),
}
# 公式里不能直接用名为 C 的列（会与 patsy 的 C() 冲突），统一加 f 前缀
F_COL: dict[str, str] = {f: f"f{f}" for f in FACTORS}
COL_F: dict[str, str] = {v: k for k, v in F_COL.items()}
FAIR_METRICS: tuple[str, ...] = ("DP", "EO", "EOdd", "TPR_gap")
PERF_METRICS: tuple[str, ...] = ("AUC", "F1")
METRICS: tuple[str, ...] = FAIR_METRICS + PERF_METRICS
INTERACTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("A", "B"), ("A", "C"), ("B", "C"), ("A", "M"), ("B", "M"), ("C", "M"),
)

N_BOOT = 2000
BOOT_SEED = 20261020
ALPHA = 0.05


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------
def load_results(path: Path = IN_CSV) -> pd.DataFrame:
    """读入并校验主设计完整性（缺格直接报错，不静默丢弃）"""
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    if "natural_missing" in df.columns:
        df = df[df["natural_missing"] == False].copy()  # noqa: E712
    df.rename(columns={"mechanism": "M"}, inplace=True)
    need = set(FACTORS) | {"dataset", "model", "seed"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"结果文件缺列：{sorted(missing)}")
    for f, col in F_COL.items():
        df[col] = df[f].astype(str)
    # 必须删掉原名（尤其 C）：patsy 会优先把公式里的 C(...) 解析成该列
    df.drop(columns=list(FACTORS), inplace=True)

    for (ds, mdl), sub in df.groupby(["dataset", "model"]):
        expect = np.prod([len(LEVELS[f]) for f in FACTORS]) * df["seed"].nunique()
        if len(sub) != expect:
            raise ValueError(f"{ds}/{mdl}: 期望 {expect} 行，实际 {len(sub)} 行"
                             f"（存在缺失单元格，不能直接做平衡 ANOVA）")
        dup = sub.duplicated(subset=[F_COL[f] for f in FACTORS] + ["seed"]).sum()
        if dup:
            raise ValueError(f"{ds}/{mdl}: 存在 {dup} 个重复单元格")
    return df


# ---------------------------------------------------------------------------
# ANOVA + 效应量
# ---------------------------------------------------------------------------
def clean_term(term: str) -> str:
    """把 patsy 项名 C(fA, Sum):C(fB, Sum) 转成 A:B"""
    out = term
    for f, col in F_COL.items():
        out = out.replace(f"C({col}, Sum)", f)
    return out


def fit_anova(d: pd.DataFrame, metric: str) -> tuple[object, pd.DataFrame]:
    """四因子全因子 Type Ⅲ ANOVA"""
    formula = (f"Q('{metric}') ~ " + " * ".join(f"C({F_COL[f]}, Sum)"
                                                  for f in ("M", "A", "B", "C")))
    model = smf.ols(formula, data=d).fit()
    tab = sm.stats.anova_lm(model, typ=3)
    tab = tab[tab.index != "Intercept"].copy()
    return model, tab


def effect_sizes(tab: pd.DataFrame, model) -> pd.DataFrame:
    """补 η² / partial η² / partial ω²

    partial ω² = (SS_e - df_e·MS_err) / (SS_e + (N - df_e)·MS_err)
    （与 Olejnik & Algina 2003 的 partial omega squared 一致；负值截断为 0）
    """
    ss_err = float(tab.loc["Residual", "sum_sq"])
    df_err = float(tab.loc["Residual", "df"])
    ms_err = ss_err / df_err
    n = float(model.nobs)
    ss_total = ss_err + float(tab.drop(index="Residual")["sum_sq"].sum())
    out = tab.drop(index="Residual").copy()
    out["eta2"] = out["sum_sq"] / ss_total
    out["partial_eta2"] = out["sum_sq"] / (out["sum_sq"] + ss_err)
    omega = (out["sum_sq"] - out["df"] * ms_err) / (out["sum_sq"] + (n - out["df"]) * ms_err)
    out["partial_omega2"] = omega.clip(lower=0.0)
    out = out.reset_index().rename(columns={"index": "term"})
    out["term"] = out["term"].map(clean_term)
    return out


def fdr_within(tab: pd.DataFrame) -> pd.DataFrame:
    """BH-FDR（族 = 同一 ANOVA 表的全部效应项）"""
    p = tab["p_value"].to_numpy(dtype=float) if "p_value" in tab else tab["PR(>F)"].to_numpy()
    tab = tab.copy()
    tab["p_value"] = p
    tab["p_fdr"] = multipletests(p, method="fdr_bh")[1] if np.all(np.isfinite(p)) else p
    return tab


# ---------------------------------------------------------------------------
# 边际均值 / 单元均值 + 簇 bootstrap CI
# ---------------------------------------------------------------------------
def cluster_boot_ci(cells: list[np.ndarray], n_boot: int = N_BOOT,
                    seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """以单元格为簇重抽样的均值 95% CI

    cells: 每个簇（= 固定其余因子后的一个因子组合）内的观测向量，
           通常每个簇含 3 个种子。平衡设计（各簇大小相同）时向量化：
           簇 bootstrap 的均值等价于「对簇均值做有放回重抽」。
    """
    k = len(cells)
    sizes = np.array([len(c) for c in cells], dtype=int)
    sums = np.array([c.sum() for c in cells], dtype=float)
    point = float(sums.sum() / sizes.sum())
    rng = np.random.default_rng(seed)
    if np.all(sizes == sizes[0]):
        cell_mean = sums / sizes[0]
        idx = rng.integers(0, k, size=(n_boot, k))
        means = cell_mean[idx].mean(axis=1)
    else:  # 非平衡兜底：按重抽到的簇逐次拼接
        means = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.integers(0, k, size=k)
            means[b] = np.concatenate([cells[i] for i in pick]).mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return point, float(lo), float(hi)


def marginal_effects(df: pd.DataFrame) -> pd.DataFrame:
    """各因子各水平的边际均值（对其余因子全部水平等权平均）

    逐 (数据集, 模型) 计算：数据集是分块属性而非因子（§5 规则），
    跨数据集直接平均会把「不同基准公平水平」混成一个数。
    """
    rows: list[dict] = []
    for (ds, mdl), sub_ds in df.groupby(["dataset", "model"]):
        for metric in METRICS:
            d = sub_ds.dropna(subset=[metric])
            for factor in FACTORS:
                for level in LEVELS[factor]:
                    sub = d[d[F_COL[factor]] == level]
                    cells = [g[metric].to_numpy(dtype=float)
                             for _, g in sub.groupby([F_COL[f] for f in FACTORS
                                                      if f != factor])]
                    point, lo, hi = cluster_boot_ci(cells)
                    rows.append({"dataset": ds, "model_name": mdl, "metric": metric,
                                 "factor": factor, "level": level,
                                 "n_obs": len(sub), "n_cells": len(cells),
                                 "mean": point, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def interaction_cells(df: pd.DataFrame) -> pd.DataFrame:
    """二阶交互的单元均值 + 95% CI（画 Figure 4 用），逐 (数据集, 模型)"""
    rows: list[dict] = []
    for (ds, mdl), sub_ds in df.groupby(["dataset", "model"]):
        for metric in METRICS:
            d = sub_ds.dropna(subset=[metric])
            for f1, f2 in INTERACTION_PAIRS:
                others = [F_COL[f] for f in FACTORS if f not in (f1, f2)]
                for l1 in LEVELS[f1]:
                    for l2 in LEVELS[f2]:
                        sub = d[(d[F_COL[f1]] == l1) & (d[F_COL[f2]] == l2)]
                        cells = [g[metric].to_numpy(dtype=float)
                                 for _, g in sub.groupby(others)]
                        point, lo, hi = cluster_boot_ci(cells)
                        rows.append({"dataset": ds, "model_name": mdl, "metric": metric,
                                     "f1": f1, "f2": f2, "level1": l1, "level2": l2,
                                     "n_obs": len(sub), "mean": point,
                                     "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def interaction_delta(cells: pd.DataFrame) -> pd.DataFrame:
    """把单元均值转成「相对总均值的离差」与「交互离差」

    interaction_delta = cell_mean - level1_mean - level2_mean + grand_mean
    （即该单元格超出两个主效应可加预测的部分）
    """
    out = []
    for (ds, mdl, metric, f1, f2), g in cells.groupby(
            ["dataset", "model_name", "metric", "f1", "f2"]):
        grand = g["mean"].mean()
        m1 = g.groupby("level1")["mean"].mean()
        m2 = g.groupby("level2")["mean"].mean()
        for _, r in g.iterrows():
            delta = r["mean"] - m1[r["level1"]] - m2[r["level2"]] + grand
            out.append({**r.to_dict(), "grand_mean": grand,
                        "interaction_delta": delta})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Tukey–Kramer + BH-FDR
# ---------------------------------------------------------------------------
def tukey_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """每个因子水平对的两两比较（用全模型 MSE 与残差 df 的 Tukey–Kramer）

    逐 (数据集, 模型, 指标) 计算，并在该族内对全部水平对做 BH-FDR。
    """
    rows: list[dict] = []
    for (ds, mdl), sub_ds in df.groupby(["dataset", "model"]):
        for metric in METRICS:
            d = sub_ds.dropna(subset=[metric])
            model, tab = fit_anova(d, metric)
            ms_err = float(tab.loc["Residual", "sum_sq"]) / float(
                tab.loc["Residual", "df"])
            df_err = float(tab.loc["Residual", "df"])
            for factor in FACTORS:
                levels = list(LEVELS[factor])
                k = len(levels)
                means = {lv: d.loc[d[F_COL[factor]] == lv, metric].mean()
                         for lv in levels}
                ns = {lv: int((d[F_COL[factor]] == lv).sum()) for lv in levels}
                for l1, l2 in itertools.combinations(levels, 2):
                    diff = means[l1] - means[l2]
                    se = np.sqrt(ms_err / 2 * (1 / ns[l1] + 1 / ns[l2]))
                    q = abs(diff) / se
                    p = float(studentized_range.sf(q, k, df_err))
                    rows.append({"dataset": ds, "model_name": mdl, "metric": metric,
                                 "factor": factor, "level1": l1, "level2": l2,
                                 "diff": diff, "se": se, "q": q, "p_tukey": p,
                                 "df_err": df_err, "ms_err": ms_err,
                                 "n1": ns[l1], "n2": ns[l2]})
    out = pd.DataFrame(rows)
    out["p_fdr"] = np.nan
    for _, idx in out.groupby(["dataset", "model_name", "metric"]).groups.items():
        p = out.loc[idx, "p_tukey"].to_numpy()
        out.loc[idx, "p_fdr"] = multipletests(p, method="fdr_bh")[1]
    out["sig_tukey"] = out["p_tukey"] < ALPHA
    out["sig_fdr"] = out["p_fdr"] < ALPHA
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(in_csv: Path = IN_CSV, out_dir: Path = OUT_DIR) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(in_csv)
    print(f"读入 {len(df)} 行，数据集 {df['dataset'].nunique()} 个，"
          f"模型 {sorted(df['model'].unique())}，种子 {sorted(df['seed'].unique())}")

    anova_rows: list[pd.DataFrame] = []
    gate_rows: list[dict] = []
    for (ds, mdl), sub in df.groupby(["dataset", "model"]):
        for metric in METRICS:
            d = sub.dropna(subset=[metric])
            n_drop = len(sub) - len(d)
            model, tab = fit_anova(d, metric)
            eff = fdr_within(effect_sizes(tab, model))
            eff.insert(0, "model_name", mdl)
            eff.insert(0, "dataset", ds)
            eff.insert(2, "metric", metric)
            eff["n_used"] = int(model.nobs)
            eff["n_dropped_nan"] = int(n_drop)
            anova_rows.append(eff)

            inter = eff[eff["term"].str.contains(":", regex=False)]
            sig = inter[inter["p_fdr"] < ALPHA]
            gate_rows.append({
                "dataset": ds, "model_name": mdl, "metric": metric,
                "n_two_way": int(sum(1 for t in inter["term"] if t.count(":") == 1)),
                "n_sig_two_way": int(sum(1 for t in sig["term"] if t.count(":") == 1)),
                "sig_two_way_terms": "; ".join(
                    sorted(t for t in sig["term"] if t.count(":") == 1)) or "-",
                "max_partial_omega2_two_way": float(
                    inter[inter["term"].map(lambda t: t.count(":") == 1)]
                    ["partial_omega2"].max()) if len(inter) else np.nan,
                "n_three_way": int(sum(1 for t in inter["term"] if t.count(":") == 2)),
                "n_sig_three_way": int(sum(1 for t in sig["term"] if t.count(":") == 2)),
            })

    anova = pd.concat(anova_rows, ignore_index=True)
    gate = pd.DataFrame(gate_rows)

    main = marginal_effects(df)
    cells = interaction_cells(df)
    inter_delta = interaction_delta(cells)
    tukey = tukey_pairs(df)

    # D 项目标表：att_ 前缀避免与已有列冲突
    anova.to_csv(out_dir / "anova_results.csv", index=False, encoding="utf-8-sig")
    main.to_csv(out_dir / "main_effects.csv", index=False, encoding="utf-8-sig")
    inter_delta.to_csv(out_dir / "interaction_cells.csv", index=False,
                       encoding="utf-8-sig")
    gate.to_csv(out_dir / "gate_summary.csv", index=False, encoding="utf-8-sig")
    tukey.to_csv(out_dir / "tukey_pairs.csv", index=False, encoding="utf-8-sig")

    print(f"\n写出：{out_dir}")
    for f in ("anova_results.csv", "main_effects.csv", "interaction_cells.csv",
              "gate_summary.csv", "tukey_pairs.csv"):
        print(f"  {f:24s} {len(pd.read_csv(out_dir / f)):6d} 行")
    return {"anova": anova, "main": main, "cells": inter_delta,
            "gate": gate, "tukey": tukey}


# ---------------------------------------------------------------------------
# 控制台摘要
# ---------------------------------------------------------------------------
def print_summary(res: dict[str, pd.DataFrame]) -> None:
    anova, gate = res["anova"], res["gate"]
    print("\n" + "=" * 92)
    print("一、主效应（各数据集 × 模型 × 指标 中显著的因子数，FDR<0.05）")
    print("=" * 92)
    main_terms = anova[~anova["term"].str.contains(":", regex=False)]
    for (ds, mdl, metric), g in main_terms.groupby(["dataset", "model_name", "metric"]):
        sig = g[g["p_fdr"] < ALPHA]
        parts = [f"{r['term'].split('[')[0]}:ω²={r['partial_omega2']:.3f}"
                 for _, r in sig.iterrows()]
        print(f"{ds:9s} {mdl:5s} {metric:8s} 显著主效应 {len(sig):2d} "
              f"{'  '.join(parts) if parts else '-'}")

    print("\n" + "=" * 92)
    print("二、二阶交互（Gate 5 判定依据）")
    print("=" * 92)
    for (ds, mdl, metric), g in gate.groupby(["dataset", "model_name", "metric"]):
        print(f"{ds:9s} {mdl:5s} {metric:8s} 显著二阶交互 "
              f"{int(g['n_sig_two_way'].iloc[0])}/{int(g['n_two_way'].iloc[0])}"
              f"  最大ω²={g['max_partial_omega2_two_way'].iloc[0]:.3f}  "
              f"{g['sig_two_way_terms'].iloc[0]}")

    print("\n" + "=" * 92)
    print("三、Gate 5 汇总：显著二阶交互的覆盖率")
    print("=" * 92)
    cov = gate.assign(has=gate["n_sig_two_way"] > 0).groupby("metric")["has"].agg(
        ["sum", "size"])
    for metric, r in cov.iterrows():
        print(f"{metric:8s} {int(r['sum'])}/{int(r['size'])} 个 (数据集×模型) "
              f"至少有一个显著二阶交互")

    print("\n" + "=" * 92)
    print("四、Tukey 两两比较（BH-FDR<0.05 的显著对数 / 总对数）")
    print("=" * 92)
    tk = res["tukey"]
    piv = tk.groupby(["dataset", "model_name", "metric"])["sig_fdr"].agg(["sum", "size"])
    for (ds, mdl, metric), r in piv.iterrows():
        print(f"{ds:9s} {mdl:5s} {metric:8s} {int(r['sum'])}/{int(r['size'])}")


if __name__ == "__main__":
    print("=" * 92)
    print("阶段 5：四因子 ANOVA（Type Ⅲ）+ 效应量 + 簇 Bootstrap CI + Tukey/FDR")
    print("=" * 92)
    res = run()
    print_summary(res)
    sys.exit(0)
