"""阶段 2 前置：依赖与关键算法的端到端验证（含已知值测试）

验证对象
  1. AIF360 Reweighing —— 不只看能否 import，而是跑通 fit_transform 并用解析权重做已知值测试
  2. AIF360 DisparateImpactRemover（C 因子候选替换水平）
  3. sklearn 插补器（均值 / MICE / KNN）能否处理含缺失数据
  4. imbalanced-learn SMOTE
  5. LightGBM 训练
  6. fairlearn 公平性指标与手算值的一致性
  7. 自带公平性指标实现（DP / EO / EOdd / TPR-gap）与手算值的一致性

运行：python code/00_verify_deps.py
输出：results/dep_verification.md（逐项 PASS/FAIL）
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

REPORT: list[tuple[str, str, str]] = []   # (项目, 结果, 说明)


def check(name: str, fn):
    """执行一项验证，捕获异常并记录"""
    try:
        detail = fn()
        REPORT.append((name, "PASS", detail or ""))
        print(f"[PASS] {name} :: {detail or ''}")
        return True
    except Exception as e:  # noqa: BLE001
        REPORT.append((name, "FAIL", f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {name} :: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        return False


# ---------------- 构造已知结构的小数据集（便于手算校验） ----------------
def toy_frame(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """已知单元格计数：sex0=60/sex1=40；y0=70/y1=30；(0,0)=50 (0,1)=10 (1,0)=20 (1,1)=20"""
    rng = np.random.default_rng(seed)
    cells = [((0, 0), 50), ((0, 1), 10), ((1, 0), 20), ((1, 1), 20)]
    rows = []
    for (d, y), k in cells:
        for _ in range(k):
            rows.append({"sex": d, "y": y,
                         "x1": float(rng.normal(d + y, 1.0)),
                         "x2": float(rng.normal(0, 1.0))})
    return pd.DataFrame(rows)


# Kamiran–Calders 再加权的解析期望权重 w(d,y) = P(D=d)P(Y=y) / P(D=d,Y=y)
EXPECTED_W = {(0, 0): 0.84, (0, 1): 1.80, (1, 0): 1.40, (1, 1): 0.60}


# ---------------- 各验证项 ----------------
def v_aif360_reweighing():
    """AIF360 Reweighing 端到端 + 已知值测试"""
    from aif360.algorithms.preprocessing import Reweighing
    from aif360.datasets import BinaryLabelDataset

    df = toy_frame()
    ds = BinaryLabelDataset(df=df, label_names=["y"], protected_attribute_names=["sex"])
    rw = Reweighing(unprivileged_groups=[{"sex": 0}], privileged_groups=[{"sex": 1}])
    out = rw.fit_transform(ds)
    w = np.asarray(out.instance_weights, dtype=float)

    got = {}
    for (d, y), exp in EXPECTED_W.items():
        m = (df["sex"] == d) & (df["y"] == y)
        got[(d, y)] = (float(w[m.values].mean()), exp)

    # 归一化系数：AIF360 常把权重整体缩放，使均值=1
    scale = float(np.mean(w))
    rel = {k: round(v[0] / scale, 4) for k, v in got.items()}
    exp_rel = {k: round(v[1], 4) for k, v in got.items()}

    ok = all(abs(rel[k] - exp_rel[k]) < 1e-3 for k in exp_rel)
    detail = (f"权重均值={scale:.4f}；相对权重 观测={rel} 期望={exp_rel}；"
              f"已知值一致={'是' if ok else '否'}")
    assert ok, detail
    return detail


def v_aif360_dir():
    """AIF360 DisparateImpactRemover（C 因子候选替换水平）"""
    from aif360.algorithms.preprocessing import DisparateImpactRemover
    from aif360.datasets import BinaryLabelDataset

    df = toy_frame()
    ds = BinaryLabelDataset(df=df, label_names=["y"], protected_attribute_names=["sex"],
                            favorable_label=1.0, unfavorable_label=0.0)
    dirr = DisparateImpactRemover(repair_level=1.0)
    out = dirr.fit_transform(ds)
    feat = np.asarray(out.features, dtype=float)
    return f"输出特征形状={feat.shape}，特征是否被修改={not np.allclose(feat, ds.features)}"


def v_sklearn_imputers():
    """三种插补器能否处理含缺失数据，且插补值符合解析期望"""
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(200, 3))
    X[rng.random(X.shape) < 0.3] = np.nan

    # 已知值校验：单列常数数据缺失 30% 后，均值插补必须还原该常数
    Z = np.full((100, 2), 5.0)
    Z[rng.random(Z.shape) < 0.3] = np.nan
    simple = SimpleImputer(strategy="mean").fit_transform(Z)
    assert np.allclose(simple, 5.0), "均值插补未还原常数"

    n_missing = int(np.isnan(X).sum())
    a = SimpleImputer(strategy="mean").fit_transform(X)
    b = IterativeImputer(random_state=0, max_iter=10).fit_transform(X)
    c = KNNImputer(n_neighbors=5).fit_transform(X)
    assert not np.isnan(a).any() and not np.isnan(b).any() and not np.isnan(c).any(), "插补后仍有缺失"
    return (f"缺失单元 {n_missing}；均值/MICE/KNN 三者插补后均无残留缺失；"
            f"均值插补常数还原校验通过")


def v_smote():
    """imbalanced-learn SMOTE"""
    from imblearn.over_sampling import SMOTE

    df = toy_frame()
    X, y = df[["x1", "x2"]].values, df["y"].values
    Xr, yr = SMOTE(random_state=0, k_neighbors=5).fit_resample(X, y)
    before = np.bincount(y) / len(y)
    after = np.bincount(yr) / len(yr)
    return f"过采样前类别占比={np.round(before, 3)}，后={np.round(after, 3)}，样本 {len(y)}→{len(yr)}"


def v_lightgbm():
    """LightGBM 训练与预测"""
    import lightgbm as lgb

    df = toy_frame(n=100, seed=2)
    X, y = df[["x1", "x2"]].values, df["y"].values
    clf = lgb.LGBMClassifier(n_estimators=20, verbose=-1, random_state=0)
    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    return f"lightgbm {lgb.__version__} 训练成功，预测概率区间=[{p.min():.4f}, {p.max():.4f}]"


def v_fairlearn_metrics():
    """fairlearn 指标与手算值一致（已知值测试）"""
    from fairlearn.metrics import (demographic_parity_difference,
                                   equalized_odds_difference)

    # 手算场景：两组各 4 人，A 组全预测 1，B 组全预测 0 → DP 差 = 1.0
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    grp = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    dp = demographic_parity_difference(y_true, y_pred, sensitive_features=grp)
    eo = equalized_odds_difference(y_true, y_pred, sensitive_features=grp)
    assert abs(dp - 1.0) < 1e-9, f"DP 期望 1.0，实得 {dp}"
    return f"DP 手算=1.0 实得={dp:.6f}；EOdd 实得={eo:.6f}"


def v_own_metrics():
    """自带公平性指标实现与手算值一致"""
    sys.path.insert(0, str(BASE / "code"))
    from fairness_metrics import (demographic_parity_diff, equal_opportunity_diff,
                                  equalized_odds_diff, tpr_gap)

    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1])       # 两组各 3 正类 / 2 负类
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 0])       # 组0 全对；组1 漏检 2 个正类
    grp = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    # 手算：组0 预测正率 = 3/5 = 0.6；组1 预测正率 = 1/5 = 0.2 → DP = 0.4
    # 手算：组0 TPR = 3/3 = 1.0；组1 TPR = 1/3 ≈ 0.3333 → TPR-gap ≈ 0.6667
    # 手算：组0 FPR = 0/2 = 0；组1 FPR = 0/2 = 0 → EOdd = max(0.6667, 0) = 0.6667
    dp = demographic_parity_diff(y_true, y_pred, grp)
    eo = equal_opportunity_diff(y_true, y_pred, grp)
    ed = equalized_odds_diff(y_true, y_pred, grp)
    tg = tpr_gap(y_true, y_pred, grp)
    assert abs(dp - 0.4) < 1e-9, f"DP 期望 0.4，实得 {dp}"
    assert abs(eo - 2 / 3) < 1e-9, f"EO 期望 0.6667，实得 {eo}"
    assert abs(ed - 2 / 3) < 1e-9, f"EOdd 期望 0.6667，实得 {ed}"
    assert abs(tg - 2 / 3) < 1e-9, f"TPR-gap 期望 0.6667，实得 {tg}"
    return f"DP={dp:.6f}(期望0.4) EO={eo:.6f}(期望0.6667) EOdd={ed:.6f} TPR-gap={tg:.6f} 全部一致"


def main() -> None:
    print("=" * 60)
    print("环境版本")
    for m in ["numpy", "pandas", "sklearn", "lightgbm", "imblearn", "fairlearn", "aif360", "statsmodels"]:
        try:
            mod = __import__(m)
            print(f"  {m:12s} {getattr(mod, '__version__', 'n/a')}")
        except Exception as e:  # noqa: BLE001
            print(f"  {m:12s} 不可用 :: {type(e).__name__}")
    print("=" * 60)

    check("AIF360 Reweighing 端到端 + 已知值", v_aif360_reweighing)
    check("AIF360 DisparateImpactRemover", v_aif360_dir)
    check("sklearn 三种插补器", v_sklearn_imputers)
    check("imbalanced-learn SMOTE", v_smote)
    check("LightGBM 训练", v_lightgbm)
    check("fairlearn 指标已知值", v_fairlearn_metrics)
    check("自带公平性指标已知值", v_own_metrics)

    md = ["# 依赖与算法验证报告（阶段 2 前置）", "",
          f"生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M}", "",
          "| 验证项 | 结果 | 说明 |", "|---|---|---|"]
    for name, res, detail in REPORT:
        md.append(f"| {name} | **{res}** | {detail} |")
    n_fail = sum(1 for _, r, _ in REPORT if r == "FAIL")
    md += ["", f"**汇总：{len(REPORT) - n_fail} / {len(REPORT)} 通过**", ""]
    (RESULTS / "dep_verification.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n已写出 {RESULTS / 'dep_verification.md'}")


if __name__ == "__main__":
    main()
