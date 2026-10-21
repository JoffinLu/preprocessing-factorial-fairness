"""模型训练与预测（T3 项目）

冻结选型（范围说明书 §4.4）
    M1  Logistic Regression
    M2  LightGBM（MLP 移至附录）

关键口径（写入论文 §3.5）
    - 预测阈值**固定 0.5**，不做任何逐细胞阈值调优：本研究的自变量是预处理因子，
      若同时调阈值，观察到的公平性差异将无法归因。
      B2（类别权重）通过 `class_weight` 影响训练目标，而非改动预测阈值。
    - C2（Reweighing）通过 `sample_weight` 传入；B2 与 C2 可同时生效
      （sklearn 逐样本把两者相乘，口径已在此写明）。
    - LightGBM 固定 `n_jobs=1` / `deterministic=True` / `force_row_wise=True`
      且不做行/列子采样（subsample=1.0, colsample_bytree=1.0），以保证同种子可复现。
    - 输入矩阵**不允许含 NaN**（插补已在 A 因子完成）；含 NaN 直接报错，
      避免把「静默丢弃样本」当成正常结果。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

MODELS: tuple[str, ...] = ("LR", "LGBM")
DECISION_THRESHOLD: float = 0.5
LGBM_PARAMS: dict = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 1.0,
    "subsample_freq": 0,
    "colsample_bytree": 1.0,
    "n_jobs": 1,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
LR_PARAMS: dict = {"max_iter": 2000, "solver": "lbfgs"}


def _make_estimator(model_name: str, seed: int, class_weight: dict | None):
    if model_name == "LR":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(random_state=seed, class_weight=class_weight,
                                  **LR_PARAMS)
    if model_name == "LGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, class_weight=class_weight,
                              **LGBM_PARAMS)
    raise ValueError(f"未知模型 {model_name!r}，可选 {MODELS}")


def fit_predict(model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, sample_weight: np.ndarray | None = None,
                class_weight: dict | None = None, seed: int = 0) -> dict:
    """训练并预测（仅返回测试集预测）

    返回
        {"y_pred", "y_prob", "train_time", "meta"}
    """
    if model_name not in MODELS:
        raise ValueError(f"未知模型 {model_name!r}，可选 {MODELS}")
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train).astype(int)

    if np.isnan(X_train).any() or np.isnan(X_test).any():
        raise ValueError("输入含 NaN：插补（因子 A）应先于模型训练完成")
    classes = np.unique(y_train)
    if len(classes) < 2:
        raise ValueError(f"训练集只有单一类别 {classes.tolist()}，无法训练；"
                         f"该单元格无法评估，须记入失败清单而非静默跳过")

    clf = _make_estimator(model_name, seed, class_weight)
    t0 = time.perf_counter()
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=float)
    clf.fit(X_train, y_train, **fit_kwargs)
    train_time = time.perf_counter() - t0

    prob = np.asarray(clf.predict_proba(X_test)[:, 1], dtype=float)
    pred = (prob >= DECISION_THRESHOLD).astype(int)

    meta = {"decision_threshold": DECISION_THRESHOLD,
            "class_weight_used": class_weight,
            "sample_weight_used": sample_weight is not None,
            "n_train": int(len(y_train)), "n_test": int(len(X_test))}
    if model_name == "LR":
        n_iter = int(np.max(clf.n_iter_))
        meta["lr_n_iter"] = n_iter
        meta["lr_converged"] = bool(n_iter < LR_PARAMS["max_iter"])
    else:
        meta["lgbm_params"] = {k: v for k, v in LGBM_PARAMS.items()
                               if k in ("n_estimators", "learning_rate", "num_leaves")}
    return {"y_pred": pred, "y_prob": prob, "train_time": train_time, "meta": meta}


# ---------------------------------------------------------------------------
# 自检：python code/pipeline/model.py
# ---------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(0)

    # ---- 已知值：线性可分数据上 LR 应达到完美分类 ----
    n = 200
    X = np.vstack([rng.normal(-3.0, 0.5, size=(n, 2)),
                   rng.normal(+3.0, 0.5, size=(n, 2))])
    y = np.array([0] * n + [1] * n)
    r = fit_predict("LR", X[:300], y[:300], X[300:], seed=0)
    acc = float((r["y_pred"] == y[300:]).mean())
    assert acc == 1.0, f"线性可分数据 LR 准确率应为 1.0，实得 {acc}"
    print(f"[已知值] 线性可分：LR 测试准确率={acc:.4f}，"
          f"概率区间=[{r['y_prob'].min():.4f}, {r['y_prob'].max():.4f}]")
    assert r["meta"]["lr_converged"] is True, r["meta"]

    # ---- 确定性：同种子结果逐位一致；不同种子对 LGBM 应产生差异 ----
    Xb = rng.normal(size=(400, 5))
    yb = (Xb[:, 0] + 0.5 * Xb[:, 1] + rng.normal(0, 0.5, 400) > 0).astype(int)
    p_lr1 = fit_predict("LR", Xb[:300], yb[:300], Xb[300:], seed=0)["y_prob"]
    p_lr2 = fit_predict("LR", Xb[:300], yb[:300], Xb[300:], seed=1)["y_prob"]
    assert np.array_equal(p_lr1, p_lr2), "LR(lbfgs) 应无随机性，不同种子结果应一致"
    p_g1 = fit_predict("LGBM", Xb[:300], yb[:300], Xb[300:], seed=0)["y_prob"]
    p_g2 = fit_predict("LGBM", Xb[:300], yb[:300], Xb[300:], seed=0)["y_prob"]
    p_g3 = fit_predict("LGBM", Xb[:300], yb[:300], Xb[300:], seed=7)["y_prob"]
    assert np.array_equal(p_g1, p_g2), "LGBM 同种子必须逐位一致"
    # 冻结设置下 LGBM 无任何随机来源（不做行/列子采样 + deterministic=True），
    # 故 random_state 不改变输出。这是**有意为之**：模型侧噪声为 0，
    # 种子效应只通过训练/测试划分进入残差，避免把模型随机性混入因子效应。
    assert np.array_equal(p_g1, p_g3), (
        "冻结设置下 LGBM 应对 random_state 不敏感；若此断言失败说明设置被改动")
    print(f"[确定性] LR 跨种子一致；LGBM 同种子一致且跨种子最大差="
          f"{np.abs(p_g1 - p_g3).max():.2e}（冻结设置下模型侧无随机性）")

    # ---- 对照：若打开子采样，种子才会生效 —— 证明上一条是「设置」导致而非 LightGBM 忽略种子 ----
    from lightgbm import LGBMClassifier
    rng_l = LGBMClassifier(random_state=0, n_estimators=100, n_jobs=1, verbose=-1,
                           subsample=0.7, subsample_freq=1, colsample_bytree=0.7)
    rng_l.fit(Xb[:300], yb[:300])
    rng_l2 = LGBMClassifier(random_state=9, n_estimators=100, n_jobs=1, verbose=-1,
                            subsample=0.7, subsample_freq=1, colsample_bytree=0.7)
    rng_l2.fit(Xb[:300], yb[:300])
    d = np.abs(rng_l.predict_proba(Xb[300:])[:, 1]
               - rng_l2.predict_proba(Xb[300:])[:, 1]).max()
    assert d > 0, "开启子采样后不同种子应产生差异"
    print(f"[对照] 开启子采样后不同种子最大概率差={d:.6f} -> 冻结设置确为随机性来源")

    # ---- B2（class_weight）应提高少数类被预测为正类的比例 ----
    n_min, n_maj = 60, 700
    Xw = np.vstack([rng.normal(0.3, 1.0, size=(n_min, 4)),
                    rng.normal(-0.3, 1.0, size=(n_maj, 4))])
    yw = np.array([1] * n_min + [0] * n_maj)
    base = fit_predict("LR", Xw, yw, Xw, seed=0)
    cw = {0: len(yw) / (2 * n_maj), 1: len(yw) / (2 * n_min)}
    weighted = fit_predict("LR", Xw, yw, Xw, class_weight=cw, seed=0)
    print(f"[B2] 无权重预测正类率={base['y_pred'].mean():.4f}，"
          f"class_weight 后={weighted['y_pred'].mean():.4f}，权重={cw}")
    assert weighted["y_pred"].mean() > base["y_pred"].mean(), \
        "class_weight 应提高正类预测率"

    # ---- C2（sample_weight）应改变预测（作用点与 B2 不同）----
    sw = np.where(yw == 1, 5.0, 1.0)
    sw_pred = fit_predict("LR", Xw, yw, Xw, sample_weight=sw, seed=0)
    print(f"[C2] sample_weight（少数类×5）后正类率={sw_pred['y_pred'].mean():.4f}")
    assert not np.array_equal(sw_pred["y_pred"], base["y_pred"]), \
        "sample_weight 应改变预测"
    assert sw_pred["meta"]["sample_weight_used"] is True

    # ---- 边界：输入含 NaN 必须显式报错（不得静默处理）----
    Xn = Xb.copy()
    Xn[0, 0] = np.nan
    try:
        fit_predict("LR", Xn[:300], yb[:300], Xn[300:], seed=0)
        raise AssertionError("含 NaN 输入应报错")
    except ValueError as e:
        assert "NaN" in str(e), e
    print("[边界] 含 NaN 输入已显式报错（不静默丢弃样本）")

    # ---- 边界：训练集单一类别必须报错并给出可读原因 ----
    try:
        fit_predict("LR", Xb[:20], np.zeros(20, dtype=int), Xb[20:40], seed=0)
        raise AssertionError("单一类别应报错")
    except ValueError as e:
        assert "单一类别" in str(e), e
    print("[边界] 训练集单一类别已显式报错")

    # ---- 边界：样本数小于特征数仍可运行（LR 需正则兜底）----
    tiny = fit_predict("LR", rng.normal(size=(6, 12)),
                       np.array([0, 1, 0, 1, 0, 1]),
                       rng.normal(size=(4, 12)), seed=0)
    assert np.isfinite(tiny["y_prob"]).all()
    print(f"[边界] n<p 场景 LR 仍给出有限概率（{np.round(tiny['y_prob'], 3)}）")
    print("model 自检全部通过")


if __name__ == "__main__":
    _self_test()
