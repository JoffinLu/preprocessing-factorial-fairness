"""APW 对比臂适配器（preprint, commit 807d4b0）

APW = Adaptive Prioritization Weights
原始代码：https://github.com/che2198/APW @ 807d4b0

方法概述
  迭代训练 LogisticRegression，每轮根据：
    (a) 当前模型在边界附近的预测概率 → sample weights
    (b) Equalized Odds 的四个 group 权重 → fairness multipliers
  200 epochs 收敛到 Equalized Odds 解。

适配要点
  1. APW 不处理缺失 → 先做主管线 A1 均值插补（pipeline.impute.impute）
  2. APW 假设敏感属性是二值化的 → 用 common.binary_sensitive()
  3. APW 用自己的评估函数 → 替换成主管线 pipeline.metrics.evaluate
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from common import (build_result_row, evaluate_method,  # noqa: E402
                     prepare_comparison_data)


# ---------------------------------------------------------------------------
# APW 核心算法（从 equalized_odds.py 移植，按协议 S3 §2.3 固定配置）
# ---------------------------------------------------------------------------
def _compute_sample_weights(y_true, y_prob, protected_attrs, multipliers, eta=0.0,
                             decision_boundary=0.5):
    """APW sample weights（官方默认 eta=0，边界感知关闭）"""
    n = len(y_true)
    exponential = np.exp(-eta * np.abs(y_prob - decision_boundary))
    weight_comp = np.zeros(n)
    for attr in protected_attrs:
        denom = np.sum(exponential * attr)
        if denom > 0:
            weight_comp += exponential * np.sum(attr) / denom * attr
        else:
            weight_comp += 0.0  # 无 protected 样本，权重贡献为 0
    combined = np.zeros(n)
    for i, mult in enumerate(multipliers):
        combined += mult * protected_attrs[i]
    sw = combined * weight_comp
    # 安全保护：NaN/inf → 0.0（该样本权重退化为不参与）
    sw = np.nan_to_num(sw, nan=0.0, posinf=0.0, neginf=0.0)
    return sw


def _compute_group_weights(y_pred, y_true, protected_attrs, alpha=0):
    """APW group fairness weights（官方默认 alpha=0，Laplace 平滑关闭）"""
    weights = []
    for attr in protected_attrs:
        pos_prot = np.where(np.logical_and(attr > 0, y_true > 0))
        neg_prot = np.where(np.logical_and(attr > 0, y_true <= 0))
        all_pos = np.where(y_true > 0)
        all_neg = np.where(y_true <= 0)

        denom1 = np.sum(y_true[all_pos]) * np.sum(y_pred[pos_prot]) + alpha
        if denom1 > 0:
            w1 = (np.sum(y_true[pos_prot]) * np.sum(y_pred[all_pos]) + alpha) / denom1
        else:
            w1 = 1.0  # 分母为 0 → 保持 neutral

        denom2 = np.sum(1 - y_true[all_neg]) * np.sum(1 - y_pred[neg_prot]) + alpha
        if denom2 > 0:
            w2 = (np.sum(1 - y_true[neg_prot]) * np.sum(1 - y_pred[all_neg]) + alpha) / denom2
        else:
            w2 = 1.0

        weights.extend([w1, w2])
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    return weights


def run_apw(dataset: str, mechanism: str, seed: int) -> dict:
    """运行 APW 一次，返回标准结果行"""
    # common.py 已经把 pipeline code 加入 sys.path
    from pipeline.impute import impute  # type: ignore

    t0 = time.perf_counter()

    # 1. 准备数据（主管线加载+划分+缺失注入）
    data = prepare_comparison_data(dataset, mechanism, seed)
    Xtr_m = data["X_train_missing"]
    ytr = data["y_train"]
    str_bin = data["sensitive_train_bin"]
    Xte_m = data["X_test_missing"]
    yte = data["y_test"]
    ste_bin = data["sensitive_test_bin"]
    ste_raw = data["sensitive_test_raw"]

    # 2. A1 均值插补（APW 不处理缺失 → 适配代价，协议 S3 §4）
    # 需要编码+标准化+插补（复用主管线 impute 模块的完整流程）
    from pipeline.features import encode, fit_category_spec, standardize  # type: ignore
    fv = None
    spec = fit_category_spec(Xtr_m, "as_category", 20, fv)
    tr_mat, names = encode(Xtr_m, "as_category", spec, fv)
    te_mat, _ = encode(Xte_m, "as_category", spec, fv)
    tr_s, te_s = standardize(tr_mat, te_mat)
    tr_a, te_a, _ = impute(tr_s, te_s, "A1", seed=seed)

    prep_secs = time.perf_counter() - t0
    t1 = time.perf_counter()

    # 3. APW 迭代训练（协议 S3 §2.3 官方默认配置：epoch=200, alpha=0, eta=0）
    #    数值稳定化保护（协议 S3 v1.1 §2.3）：alpha=0 时若某组某期预测正例数为 0，
    #    组权重比值的分母趋近 0，乘子在 200 轮内会指数发散（实测 COMPAS 达 1.1e7），
    #    此时模型退化为对少数样本过拟合，AUC 大幅下降，不是方法本身的公平性效果。
    #    因此每轮对乘子做幅度裁剪 + 均值归一化，并记录触发次数供审计。
    MULT_BOUND = 1e3
    ytr_arr = np.asarray(ytr).astype(int)
    protected_attrs = [str_bin, 1 - str_bin]
    label_combos = [str_bin * ytr_arr, str_bin * (1 - ytr_arr),
                    (1 - str_bin) * ytr_arr, (1 - str_bin) * (1 - ytr_arr)]
    multipliers = np.ones(4)
    sample_weights = np.ones(len(ytr_arr))
    n_clipped = 0

    n_epoch = 200
    for epoch in range(n_epoch):
        clf = LogisticRegression(max_iter=10000, random_state=seed, solver="lbfgs")
        clf.fit(tr_a, ytr_arr, sample_weight=sample_weights)

        yp_tr = clf.predict(tr_a)
        pp_tr = clf.predict_proba(tr_a)[:, 1].astype(np.float32)

        sample_weights = _compute_sample_weights(
            ytr_arr, pp_tr, label_combos, multipliers, eta=0.0
        )
        gw = np.asarray(_compute_group_weights(yp_tr, ytr_arr, protected_attrs, alpha=0))
        multipliers = multipliers * gw

        # 裁剪 + 归一化（保持相对结构，仅约束绝对量级）
        if np.any(multipliers > MULT_BOUND) or np.any(multipliers < 1.0 / MULT_BOUND):
            multipliers = np.clip(multipliers, 1.0 / MULT_BOUND, MULT_BOUND)
            multipliers = multipliers / np.mean(multipliers)
            n_clipped += 1

    train_secs = time.perf_counter() - t1

    # 4. 在测试集上预测
    yp_te = clf.predict(te_a)
    yp_prob_te = clf.predict_proba(te_a)[:, 1]

    # 5. 用主管线评估：公平性必须用**原始多类别**敏感属性（与主表口径一致），
    #    二值化只用于 APW 内部训练
    yte_arr = np.asarray(yte).astype(int)
    eval_dict = evaluate_method(yte_arr, yp_te, yp_prob_te, ste_raw, dataset,
                                "APW", sensitive_bin=ste_bin)

    row = build_result_row(
        method="APW",
        dataset=dataset,
        mechanism=mechanism,
        seed=seed,
        prep_secs=prep_secs,
        train_secs=train_secs,
        n_train=len(ytr),
        n_test=len(yte),
        sensitive_name=data["ds"].sensitive_name,
        n_groups=int(ste_raw.astype(str).nunique()),
        eval_dict=eval_dict,
    )
    # 诊断量：APW 权重与公平乘子的量级（alpha=0 时乘子可能发散，须留痕）
    row.update({
        "sw_mean": float(np.mean(sample_weights)),
        "sw_min": float(np.min(sample_weights)),
        "sw_max": float(np.max(sample_weights)),
        "apw_mult_max": float(np.max(np.abs(multipliers))),
        "apw_clip_epochs": int(n_clipped),
        "apw_capped": bool(n_clipped > 0),
    })
    return row
