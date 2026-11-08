"""S3 对比臂共享工具模块

提供：
1. 主管线路径覆盖（DATA_RAW → 根目录 data/raw）
2. 敏感属性二值化（APW 专用，COMPAS/Diabetes 多类别→二组）
3. 统一数据预处理管线（加载 + 划分 + 缺失注入）
4. 统一评估函数（用主管线 pipeline.metrics.evaluate）
5. 输出列定义（ORDER_COLS_COMPARISON，与主管线 ORDER_COLS 兼容）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 路径覆盖：把主管线 pipeline 模块的 DATA_RAW 指向根目录 data/raw
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent  # <repo>/code/comparison/
# _HERE.parents[0] = <repo>/code/comparison/
# _HERE.parents[1] = <repo>/  ← repository root (contains data/ and code/)
_REPO_ROOT = _HERE.parents[1]
_PIPELINE_CODE = _REPO_ROOT / "code"

if str(_PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_CODE))

# Override pipeline.config.DATA_RAW to point to root data/raw
import pipeline.config as _pipeline_cfg  # noqa: E402
_pipeline_cfg.DATA_RAW = _REPO_ROOT / "data" / "raw"

# Also import pipeline modules for reuse
from pipeline.data import load_dataset  # noqa: E402
from pipeline.missing import inject_missing  # noqa: E402
from pipeline.run_cell import split_indices  # noqa: E402
from pipeline.metrics import evaluate  # noqa: E402


# ---------------------------------------------------------------------------
# 敏感属性二值化（协议 S3 §3）
# ---------------------------------------------------------------------------
def reference_group(s) -> str:
    """返回占比最大的一组的规范化标签（小写）"""
    s_lower = pd.Series(s).astype(str).str.lower().reset_index(drop=True)
    counts = s_lower.value_counts()
    if len(counts) < 2:
        raise ValueError(f"敏感属性只有单一取值 {counts.index.tolist()}，无法二值化")
    return counts.index[0]


def binary_sensitive(s: pd.Series, dataset: str, ref: str | None = None) -> np.ndarray:
    """把敏感属性序列二值化（协议 S3 §3）

    规则：**取占比最大的一组为参考组（值=1）**，其余为 0。
    参考组应由 `reference_group()` 在全量数据上确定后传入，
    保证训练集与测试集使用同一个参考组，避免两集众数不同导致分组翻转。

    不硬编码标签名——各数据集实际标签不同（Diabetes 用 Caucasian 而非 White），
    硬编码会导致某组完全落空、n_groups 退化为 1、全部公平性指标失效。
    """
    if s is None or len(s) == 0:
        raise ValueError(f"空敏感属性序列（dataset={dataset}）")

    if ref is None:
        ref = reference_group(s)

    s_lower = pd.Series(s).astype(str).str.lower().reset_index(drop=True).to_numpy()
    binary = (s_lower == ref).astype(int)

    # 防御：二值化后必须两组都存在，否则公平性指标会静默退化为 NaN
    if binary.min() == binary.max():
        raise ValueError(
            f"二值化后只剩一组（dataset={dataset}, ref={ref}），"
            f"原始取值={pd.Series(s).astype(str).unique().tolist()}")
    return binary


# ---------------------------------------------------------------------------
# 统一数据预处理管线
# ---------------------------------------------------------------------------
def prepare_comparison_data(dataset: str, mechanism: str, seed: int,
                            natural_missing: bool = False) -> dict:
    """为对比臂方法准备统一的 (X_train_missing, y_train, s_train,
    X_test_missing, y_test, s_test) 元组

    返回的字典包含：
      X_train_raw : pd.DataFrame  训练集特征（已置零天然缺失，未注入缺失）
      X_train_missing : pd.DataFrame  训练集特征（已注入机制 M 的缺失）
      y_train     : np.ndarray     训练标签
      sensitive_train_raw : pd.Series  训练集敏感属性（原始，多类别）
      sensitive_train_bin : np.ndarray  训练集敏感属性（二值化）
      X_test_raw  : pd.DataFrame   测试集特征（已置零天然缺失）
      X_test_missing : pd.DataFrame  测试集特征（已注入机制 M 的缺失，seed+5000）
      y_test      : np.ndarray     测试标签
      sensitive_test_raw : pd.Series  测试集敏感属性（原始）
      sensitive_test_bin  : np.ndarray  测试集敏感属性（二值化）
      ds          : Dataset        主管线 Dataset 对象（含 meta）
      idx_tr, idx_te : np.ndarray  训练/测试索引（供配对验证）
    """
    ds = load_dataset(dataset, natural_missing=natural_missing)
    idx_tr, idx_te = split_indices(dataset, ds.y, seed, 0.30)

    Xtr_raw = ds.X_raw.iloc[idx_tr].reset_index(drop=True)
    Xte_raw = ds.X_raw.iloc[idx_te].reset_index(drop=True)
    ytr = ds.y[idx_tr]
    yte = ds.y[idx_te]
    str_raw = ds.sensitive.iloc[idx_tr].reset_index(drop=True)
    ste_raw = ds.sensitive.iloc[idx_te].reset_index(drop=True)

    # 注入缺失（M1 MCAR / M2 MAR；测试集用 seed + 5000 保证确定性）
    if mechanism == "NAT":
        Xtr_m = Xtr_raw
        Xte_m = Xte_raw
    else:
        mech_name = "MCAR" if mechanism == "M1" else "MAR"
        Xtr_m = inject_missing(Xtr_raw, mech_name, 0.30, seed=seed)
        Xte_m = inject_missing(Xte_raw, mech_name, 0.30, seed=seed + 5000)

    # 参考组在全量数据上确定，训练/测试集共用，避免两集众数不同导致分组翻转
    ref = reference_group(ds.sensitive)

    return {
        "X_train_raw": Xtr_raw,
        "X_train_missing": Xtr_m,
        "y_train": ytr,
        "sensitive_train_raw": str_raw,
        "sensitive_train_bin": binary_sensitive(str_raw, dataset, ref=ref),
        "X_test_raw": Xte_raw,
        "X_test_missing": Xte_m,
        "y_test": yte,
        "sensitive_test_raw": ste_raw,
        "sensitive_test_bin": binary_sensitive(ste_raw, dataset, ref=ref),
        "sensitive_ref_group": ref,
        "ds": ds,
        "idx_tr": idx_tr,
        "idx_te": idx_te,
    }


# ---------------------------------------------------------------------------
# 统一评估（复用主管线 pipeline.metrics.evaluate）
# ---------------------------------------------------------------------------
def evaluate_method(y_test: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray, sensitive_test: np.ndarray,
                    dataset: str, method_name: str,
                    sensitive_bin: np.ndarray | None = None) -> dict:
    """包装主管线的 evaluate()，返回扁平的指标字典

    主指标用**原始多类别**敏感属性计算，与主表口径一致（可直接比较）。
    若提供 sensitive_bin，额外返回 `*_bin` 后缀的公平性指标——
    这是各对比方法实际优化的二值目标口径，用于解释"方法优化二值公平、
    但多组公平未必改善"的差异（协议 S3 §3 适配约束）。
    """
    out = evaluate(y_test, y_pred, y_prob, sensitive_test,
                   dataset=dataset, model=method_name)

    if sensitive_bin is not None:
        bin_metrics = evaluate(y_test, y_pred, y_prob,
                               np.asarray(sensitive_bin),
                               dataset=dataset, model=method_name + "_bin")
        for k in ("DP", "EO", "EOdd", "TPR_gap", "DI"):
            if k in bin_metrics:
                out[f"{k}_bin"] = bin_metrics[k]
        out["n_groups_bin"] = bin_metrics.get("n_groups", np.nan)
    return out


# ---------------------------------------------------------------------------
# 输出列定义（与主管线 ORDER_COLS 兼容，新增 method 列）
# ---------------------------------------------------------------------------
ORDER_COLS_COMPARISON: tuple[str, ...] = (
    "method", "dataset", "mechanism", "seed",
    "n_train", "n_test", "sensitive_name", "n_groups",
    "AUC", "F1", "accuracy", "balanced_accuracy", "precision", "recall",
    "DP", "EO", "EOdd", "TPR_gap", "DI",
    "undefined_tpr_groups", "undefined_sel_groups",
    "min_group_n", "min_group_n_pos",
    "prep_secs", "train_secs", "total_secs",
    "status", "error", "traceback",
)


def build_result_row(method: str, dataset: str, mechanism: str, seed: int,
                     prep_secs: float, train_secs: float,
                     n_train: int, n_test: int,
                     sensitive_name: str, n_groups: int,
                     eval_dict: dict, status: str = "ok",
                     error: str | None = None,
                     traceback: str | None = None) -> dict:
    """构建标准输出行"""
    row = {
        "method": method,
        "dataset": dataset,
        "mechanism": mechanism,
        "seed": int(seed),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "sensitive_name": sensitive_name,
        "n_groups": int(n_groups),
        "prep_secs": float(prep_secs),
        "train_secs": float(train_secs),
        "total_secs": float(prep_secs + train_secs),
        "status": status,
        "error": error,
        "traceback": traceback,
    }
    row.update(eval_dict)
    return row
