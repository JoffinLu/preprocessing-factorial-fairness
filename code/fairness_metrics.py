"""公平性指标实现（T3 项目）

设计要点
  - 只依赖 numpy / pandas，便于公开与复现
  - 明确区分「有定义的率」与「无定义的率」：某敏感组没有正类样本时 TPR 无定义，
    此时**不是**当作 0，而是从极差计算中剔除并计数（`undefined_groups`），
    否则会把「样本不足」误报成「公平性很好」
  - 所有极差型指标（DP / EO / EOdd / TPR-gap）均取「组间最大值 − 最小值」，
    与 fairlearn 的 difference 口径一致
  - 已通过已知值测试（见 `python code/fairness_metrics.py` 的自检）

指标定义
  selection_rate(d) = P(Ŷ=1 | D=d)
  TPR(d)            = P(Ŷ=1 | Y=1, D=d)
  FPR(d)            = P(Ŷ=1 | Y=0, D=d)
  DP                = max_d sel − min_d sel
  EO                = max_d TPR − min_d TPR
  EOdd              = max( max_d TPR − min_d TPR , max_d FPR − min_d FPR )
  TPR-gap           = EO（单独提供，便于论文分别报告）
  DI                = min_d sel / max_d sel（取值 (0,1]，1 表示无差异）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "group_rates",
    "demographic_parity_diff",
    "equal_opportunity_diff",
    "equalized_odds_diff",
    "tpr_gap",
    "disparate_impact",
    "fairness_report",
]


def _as_array(x) -> np.ndarray:
    if isinstance(x, (pd.Series, pd.Index)):
        return x.to_numpy()
    return np.asarray(x)


def group_rates(y_true, y_pred, groups, min_pos: int = 1, min_neg: int = 1) -> pd.DataFrame:
    """按敏感组计算各项率值

    参数
        y_true, y_pred : 真实标签与预测标签（0/1）
        groups         : 敏感属性取值（每个样本一个组标签）
        min_pos, min_neg : 计算 TPR / FPR 所需的最小正类 / 负类样本数；
                          低于该阈值的组，对应率值记为 NaN（无定义）

    返回
        DataFrame，每行一个组，列为 n / n_pos / n_neg / sel / tpr / fpr / tnr / ppv
    """
    yt, yp, g = _as_array(y_true).astype(int), _as_array(y_pred).astype(int), _as_array(groups)
    rows = []
    for d in pd.unique(g):
        m = g == d
        yt_d, yp_d = yt[m], yp[m]
        n, n_pos, n_neg = len(yt_d), int((yt_d == 1).sum()), int((yt_d == 0).sum())

        sel = float(yp_d.mean()) if n else np.nan
        if n_pos >= min_pos:
            tpr = float(yp_d[yt_d == 1].mean())
        else:
            tpr = np.nan
        if n_neg >= min_neg:
            fpr = float(yp_d[yt_d == 0].mean())
            tnr = 1.0 - fpr
        else:
            fpr = tnr = np.nan
        n_pred_pos = int((yp_d == 1).sum())
        ppv = float(yt_d[yp_d == 1].mean()) if n_pred_pos > 0 else np.nan

        rows.append({"group": d, "n": n, "n_pos": n_pos, "n_neg": n_neg,
                     "sel": sel, "tpr": tpr, "fpr": fpr, "tnr": tnr, "ppv": ppv})
    return pd.DataFrame(rows)


def _spread(series: pd.Series) -> tuple[float, int]:
    """组间极差（max − min），剔除 NaN 并返回被剔除的组数"""
    v = series.dropna()
    n_undef = int(len(series) - len(v))
    if len(v) < 2:
        return np.nan, n_undef
    return float(v.max() - v.min()), n_undef


def demographic_parity_diff(y_true, y_pred, groups) -> float:
    """人口均等差（DP）：各组预测正类率的极差"""
    r = group_rates(y_true, y_pred, groups)
    return _spread(r["sel"])[0]


def equal_opportunity_diff(y_true, y_pred, groups) -> float:
    """机会均等差（EO）：各组 TPR 的极差"""
    r = group_rates(y_true, y_pred, groups)
    return _spread(r["tpr"])[0]


def tpr_gap(y_true, y_pred, groups) -> float:
    """TPR 极差，与 EO 数值相同，单独提供以便论文分别报告"""
    return equal_opportunity_diff(y_true, y_pred, groups)


def equalized_odds_diff(y_true, y_pred, groups) -> float:
    """均等赔率差（EOdd）：TPR 极差与 FPR 极差中的较大者

    降级规则：若 TPR 极差因「只有一个组有正类样本」而无定义，则退回 FPR 极差；
    两侧都无定义时返回 NaN。无定义组的个数由 `fairness_report` 的
    `undefined_tpr_groups` 字段记录，论文附录须一并报告。
    """
    r = group_rates(y_true, y_pred, groups)
    g_tpr, _ = _spread(r["tpr"])
    g_fpr, _ = _spread(r["fpr"])
    if np.isnan(g_tpr) and np.isnan(g_fpr):
        return np.nan
    return float(np.nanmax([g_tpr, g_fpr]))


def disparate_impact(y_true, y_pred, groups) -> float:
    """差异影响（DI）：min(sel) / max(sel)，取值 (0,1]，1 为无差异"""
    r = group_rates(y_true, y_pred, groups)
    v = r["sel"].dropna()
    if len(v) < 2 or v.max() == 0:
        return np.nan
    return float(v.min() / v.max())


def fairness_report(y_true, y_pred, groups, model_name: str = "",
                    dataset: str = "", **meta) -> dict:
    """一次性计算全部指标，并附带无定义组的计数（供 QA 与论文附录使用）"""
    r = group_rates(y_true, y_pred, groups)
    dp_g, dp_u = _spread(r["sel"])
    eo_g, eo_u = _spread(r["tpr"])
    ed_g = equalized_odds_diff(y_true, y_pred, groups)
    out = {
        "dataset": dataset, "model": model_name,
        "DP": dp_g, "EO": eo_g, "EOdd": ed_g, "TPR_gap": eo_g,
        "DI": disparate_impact(y_true, y_pred, groups),
        "n_groups": len(r), "undefined_tpr_groups": eo_u,
        "undefined_sel_groups": dp_u,
        "min_group_n": int(r["n"].min()),
        "min_group_n_pos": int(r["n_pos"].min()),
    }
    out.update(meta)
    return out


# ---------------------------------------------------------------------------
# 自检（已知值测试）：python code/fairness_metrics.py
# ---------------------------------------------------------------------------
def _self_test() -> None:
    # 场景：两组各 5 人、各 3 正类 / 2 负类
    #   组0 全对：sel=0.6  TPR=1.0  FPR=0.0
    #   组1 漏检 2 个正类并产生 1 个假正类：sel=0.4  TPR=1/3  FPR=0.5
    y_true = np.array([1, 1, 1, 0, 0, 1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 1, 0, 0, 1, 0])
    grp = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    exp_sel, exp_tpr, exp_fpr = (0.6, 0.4), (1.0, 1 / 3), (0.0, 0.5)

    r = group_rates(y_true, y_pred, grp)
    assert np.allclose(r["sel"].to_numpy(), exp_sel), r
    assert np.allclose(r["tpr"].to_numpy(), exp_tpr), r
    assert np.allclose(r["fpr"].to_numpy(), exp_fpr), r

    assert abs(demographic_parity_diff(y_true, y_pred, grp) - 0.2) < 1e-12
    assert abs(equal_opportunity_diff(y_true, y_pred, grp) - 2 / 3) < 1e-12
    assert abs(equalized_odds_diff(y_true, y_pred, grp) - 2 / 3) < 1e-12
    # DI = min(sel)/max(sel) = 0.4/0.6 = 2/3
    assert abs(disparate_impact(y_true, y_pred, grp) - 2 / 3) < 1e-12

    # 边界：单组 → 极差无定义（返回 NaN）
    assert np.isnan(demographic_parity_diff(y_true, y_pred, np.zeros_like(grp)))
    # 边界：某组无正类样本 → TPR 无定义，且被计数而非当作 0
    y2 = np.array([1, 1, 0, 0, 0, 0])
    p2 = np.array([1, 1, 0, 0, 0, 1])
    g2 = np.array([0, 0, 0, 1, 1, 1])
    rep = fairness_report(y2, p2, g2)
    assert rep["undefined_tpr_groups"] == 1, rep
    # 只剩一组有定义 TPR 时，「组间极差」本身无定义 → 必须返回 NaN，不能返回 0
    # （返回 0 会把「样本不足」误报成「完全公平」）
    assert np.isnan(rep["EO"]) and np.isnan(rep["TPR_gap"]), rep
    # EOdd 的降级规则：TPR 极差无定义时退回 FPR 极差（此处组1 FPR=1/3，组0 FPR=0）
    assert abs(rep["EOdd"] - 1 / 3) < 1e-12, rep
    # 边界：完全无差异的完美预测 → 全零
    rep2 = fairness_report(np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0]), np.array([0, 0, 1, 1]))
    assert rep2["DP"] == 0.0 and rep2["EO"] == 0.0 and rep2["EOdd"] == 0.0, rep2

    print("fairness_metrics 自检全部通过")


if __name__ == "__main__":
    _self_test()
