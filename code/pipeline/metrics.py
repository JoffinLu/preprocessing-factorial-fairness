"""评估模块：性能指标 + 公平性指标（T3 项目）

输出
    性能：AUC、F1、accuracy、balanced accuracy、precision、recall、specificity
    公平：DP、EO、EOdd、TPR-gap、DI（调用 code/fairness_metrics.py，
          与阶段 2 前置验证中已做已知值测试的实现完全同一份代码）
    QA  ：undefined_tpr_groups / undefined_sel_groups / min_group_n / min_group_n_pos

口径要点（写入论文 §3.4）
    - 公平指标只用**测试集**、按**原始敏感属性取值**计算；C2 的再加权只作用于
      训练集权重，不改标签也不改测试集，因此公平指标的改善只能通过模型行为体现。
    - 某组正类样本不足时 TPR 无定义，返回 NaN 而非 0，并从极差计算中剔除并计数
      （否则会把「样本不足」误报成「完全公平」）。
    - 只有 1 个敏感组、或测试集只有一个真实类别时，相应指标无定义，一律返回 NaN。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from fairness_metrics import fairness_report, group_rates  # noqa: E402

METRIC_COLS: tuple[str, ...] = (
    "AUC", "F1", "accuracy", "balanced_accuracy", "precision", "recall",
    "specificity", "n_test", "n_pos_true", "n_neg_true", "n_pred_pos",
    "DP", "EO", "EOdd", "TPR_gap", "DI",
    "n_groups", "undefined_tpr_groups", "undefined_sel_groups",
    "min_group_n", "min_group_n_pos",
)


def evaluate(y_true, y_pred, y_prob, sensitive, dataset: str = "",
             model: str = "", **meta) -> dict:
    """一次性计算全部性能与公平指标

    参数
        y_true, y_pred : 真实标签与 0.5 阈值下的预测标签（0/1）
        y_prob         : 正类预测概率（用于 AUC）
        sensitive      : 敏感属性取值（**原始多类别口径**）
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    sensitive = np.asarray(sensitive)
    if not (len(y_true) == len(y_pred) == len(y_prob) == len(sensitive)):
        raise ValueError("y_true / y_pred / y_prob / sensitive 长度必须一致")
    if len(y_true) == 0:
        raise ValueError("测试集为空，无法评估")

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    n_pos, n_neg = tp + fn, fp + tn

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / n_pos if n_pos > 0 else np.nan
    specificity = tn / n_neg if n_neg > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if np.isfinite(precision) and np.isfinite(recall)
          and (precision + recall) > 0 else 0.0)
    # sklearn 口径：无正类预测时 F1 = 0
    if (tp + fp) == 0:
        f1 = 0.0

    accuracy = (tp + tn) / len(y_true)
    if n_pos > 0 and n_neg > 0:
        balanced_accuracy = 0.5 * (recall + specificity)
    else:
        balanced_accuracy = np.nan

    # AUC：测试集只有一个真实类别时无定义
    if n_pos > 0 and n_neg > 0:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, y_prob))
    else:
        auc = np.nan

    fair = fairness_report(y_true, y_pred, sensitive, model_name=model,
                           dataset=dataset)
    out = {
        "n_test": int(len(y_true)), "n_pos_true": n_pos, "n_neg_true": n_neg,
        "n_pred_pos": int(tp + fp),
        "AUC": auc, "F1": float(f1), "accuracy": float(accuracy),
        "balanced_accuracy": (float(balanced_accuracy)
                              if np.isfinite(balanced_accuracy) else np.nan),
        "precision": float(precision) if np.isfinite(precision) else np.nan,
        "recall": float(recall) if np.isfinite(recall) else np.nan,
        "specificity": (float(specificity) if np.isfinite(specificity) else np.nan),
        "DP": fair["DP"], "EO": fair["EO"], "EOdd": fair["EOdd"],
        "TPR_gap": fair["TPR_gap"], "DI": fair["DI"],
        "n_groups": fair["n_groups"],
        "undefined_tpr_groups": fair["undefined_tpr_groups"],
        "undefined_sel_groups": fair["undefined_sel_groups"],
        "min_group_n": fair["min_group_n"],
        "min_group_n_pos": fair["min_group_n_pos"],
    }
    out.update(meta)
    return out


def group_table(y_true, y_pred, sensitive, **extra) -> pd.DataFrame:
    """各敏感组的率值表（供论文附录导出）"""
    r = group_rates(y_true, y_pred, sensitive)
    for k, v in extra.items():
        r[k] = v
    return r


__all__ = ["evaluate", "group_table", "METRIC_COLS"]


# ---------------------------------------------------------------------------
# 自检：python code/pipeline/metrics.py
# ---------------------------------------------------------------------------
def _self_test() -> None:
    # ---- 场景 A：完美预测、两组无差异 -> 全部公平指标为 0，AUC/F1 = 1 ----
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    m = evaluate(y, y, y.astype(float), g)
    assert m["AUC"] == 1.0 and m["F1"] == 1.0 and m["accuracy"] == 1.0
    assert m["DP"] == 0.0 and m["EO"] == 0.0 and m["EOdd"] == 0.0, m
    assert m["DI"] == 1.0
    print(f"[完美预测] AUC={m['AUC']} F1={m['F1']} DP={m['DP']} EO={m['EO']} DI={m['DI']}")

    # ---- 场景 B：手算已知值（与 fairness_metrics 的自检同一场景）----
    #   组0（前 5 人）：sel=0.6 TPR=1.0 FPR=0.0
    #   组1（后 5 人）：sel=0.4 TPR=1/3 FPR=0.5
    #   -> DP = 0.6-0.4 = 0.2；EO = 1-1/3 = 2/3；EOdd = max(2/3, 0.5) = 2/3
    #      DI = 0.4/0.6 = 2/3
    #   混淆矩阵：TP=4 FP=1 FN=2 TN=3 -> precision=4/5 recall=4/6
    #      F1 = 2*0.8*(2/3)/(0.8+2/3) = 8/11；accuracy = 7/10
    yt = np.array([1, 1, 1, 0, 0, 1, 1, 1, 0, 0])
    yp = np.array([1, 1, 1, 0, 0, 1, 0, 0, 1, 0])
    grp = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    mb = evaluate(yt, yp, yp.astype(float), grp)
    assert abs(mb["DP"] - 0.2) < 1e-12, mb["DP"]
    assert abs(mb["EO"] - 2 / 3) < 1e-12, mb["EO"]
    assert abs(mb["EOdd"] - 2 / 3) < 1e-12, mb["EOdd"]
    assert abs(mb["DI"] - 2 / 3) < 1e-12, mb["DI"]
    assert abs(mb["F1"] - 8 / 11) < 1e-12, mb["F1"]
    assert abs(mb["accuracy"] - 0.7) < 1e-12
    assert mb["n_pred_pos"] == 5 and mb["n_pos_true"] == 6
    # AUC 手算：6 正 × 4 负 = 24 对；正类 prob∈{1,1,1,1,0,0}，负类∈{0,0,1,0}
    #   正 prob=1 的 4 个：各 3 胜 1 平 -> 3.5；正 prob=0 的 2 个：各 3 平 1 负 -> 1.5
    #   AUC = (4*3.5 + 2*1.5)/24 = 17/24
    assert abs(mb["AUC"] - 17 / 24) < 1e-12, mb["AUC"]
    print(f"[手算已知值] DP={mb['DP']:.4f} EO={mb['EO']:.4f} EOdd={mb['EOdd']:.4f} "
          f"DI={mb['DI']:.4f} F1={mb['F1']:.4f}(期望 8/11={8/11:.4f}) "
          f"AUC={mb['AUC']:.4f}(期望 17/24={17/24:.4f})")

    # ---- 场景 C：无正类预测 -> F1 按 sklearn 口径取 0 ----
    mc = evaluate(yt, np.zeros_like(yp), np.zeros_like(yt, dtype=float), grp)
    assert mc["F1"] == 0.0 and mc["n_pred_pos"] == 0
    assert np.isnan(mc["precision"]), mc["precision"]
    print(f"[无正类预测] F1={mc['F1']} precision=NaN DI={mc['DI']}（DI 因 max sel=0 无定义）")

    # ---- 场景 D：测试集只有一个真实类别 -> AUC 与 balanced accuracy 无定义 ----
    md = evaluate(np.ones(6, dtype=int), np.ones(6, dtype=int),
                  np.full(6, 0.9), np.array([0, 0, 0, 1, 1, 1]))
    assert np.isnan(md["AUC"]) and np.isnan(md["balanced_accuracy"]), md
    print(f"[单一真实类别] AUC=NaN balanced_accuracy=NaN（按无定义处理，未记为 0）")

    # ---- 场景 E：某组正类样本不足 -> TPR 无定义，计数而非当作 0 ----
    me = evaluate(np.array([1, 1, 0, 0, 0, 0]), np.array([1, 1, 0, 0, 0, 1]),
                  np.array([0.9, 0.8, 0.2, 0.1, 0.3, 0.6]),
                  np.array([0, 0, 0, 1, 1, 1]))
    assert me["undefined_tpr_groups"] == 1, me
    assert np.isnan(me["EO"]) and np.isnan(me["TPR_gap"]), me
    print(f"[组内正类不足] undefined_tpr_groups={me['undefined_tpr_groups']} "
          f"EO=NaN（未误报为 0）；EOdd 降级为 FPR 极差={me['EOdd']:.6f}")

    # ---- 场景 F：单一敏感组 -> 组间极差无定义 ----
    mf = evaluate(yt, yp, yp.astype(float), np.zeros_like(grp))
    assert mf["n_groups"] == 1 and np.isnan(mf["DP"]), mf
    print(f"[单一敏感组] n_groups=1 DP=NaN")

    # ---- 长度校验与分组表 ----
    try:
        evaluate(yt, yp, yp[:5], grp)
        raise AssertionError("长度不一致应报错")
    except ValueError as e:
        assert "长度" in str(e)
    gt = group_table(yt, yp, grp, dataset="toy")
    assert list(gt["group"]) == [0, 1] and gt.shape[0] == 2
    assert np.allclose(gt["sel"].to_numpy(), [0.6, 0.4])
    print(f"[分组表] {gt.to_dict(orient='records')}")
    print("metrics 自检全部通过")


if __name__ == "__main__":
    _self_test()
