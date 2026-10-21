"""因子 A：缺失值插补（T3 项目）

冻结水平
    A1：sklearn.impute.SimpleImputer(strategy="mean")
    A2：迭代多变量单次插补 IterativeImputer（不是多重插补）
    A3：sklearn.impute.KNNImputer(n_neighbors=5)

关键约束
    - 插补器只在**训练集** fit，测试集用同一插补器 transform（禁止用测试统计量）
    - 返回矩阵不得含 NaN（否则抛异常）
    - 同时返回插补耗时（秒）
    - 顺序：A 在 B/C 之前（固定 A → B → C）

注：分类列的缺失在编码阶段已由「缺失自成一类」（默认）或众数填充处理，
因此进入本模块的 NaN 只存在于数值列。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

METHODS: tuple[str, ...] = ("A1", "A2", "A3")


def _make_imputer(method: str, seed: int):
    if method == "A1":
        from sklearn.impute import SimpleImputer
        return SimpleImputer(strategy="mean")
    if method == "A2":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        # Training-complete columns are skipped for tractability. If a column
        # first becomes missing at prediction time, sklearn's documented path
        # uses its training mean. This is part of the declared single-imputation
        # algorithm, not a reason to reject otherwise valid natural gaps.
        return IterativeImputer(random_state=seed, max_iter=10, skip_complete=True,
                               sample_posterior=False)
    if method == "A3":
        from sklearn.impute import KNNImputer
        return KNNImputer(n_neighbors=5)
    raise ValueError(f"未知插补方法 {method!r}，可选 {METHODS}")


def impute(X_train: np.ndarray, X_test: np.ndarray, method: str,
           seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """用指定方法插补训练/测试矩阵

    返回
        (训练集插补结果, 测试集插补结果, 插补耗时秒)
    """
    if method not in METHODS:
        raise ValueError(f"未知插补方法 {method!r}，可选 {METHODS}")

    t0 = time.perf_counter()
    imp = _make_imputer(method, seed)
    tr = imp.fit_transform(X_train)          # 只在训练集 fit
    te = imp.transform(X_test)
    secs = time.perf_counter() - t0

    if np.isnan(tr).any() or np.isnan(te).any():
        raise ValueError(f"{method} 插补后仍存在 NaN（train={int(np.isnan(tr).sum())}, "
                         f"test={int(np.isnan(te).sum())}）")
    return np.asarray(tr, dtype=float), np.asarray(te, dtype=float), secs


# ---------------------------------------------------------------------------
# 自检：python code/pipeline/impute.py
# ---------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(0)

    # 已知值：某列除少量缺失外全为常数 5.0；均值插补必须还原 5.0
    Z = np.full((200, 3), 5.0)
    Z[:, 1] = rng.normal(0.0, 1.0, size=200)    # 第二列非常数
    Ztr = Z.copy()
    Zte = Z.copy()
    miss_tr = rng.random(Ztr.shape) < 0.3
    Ztr[miss_tr] = np.nan
    Zte[rng.random(Zte.shape) < 0.3] = np.nan

    tr1, te1, s1 = impute(Ztr, Zte, "A1", seed=0)
    # 第一列（常数 5）与第三列（常数 5）插补后必须精确还原 5.0
    assert np.allclose(tr1[:, 0], 5.0) and np.allclose(tr1[:, 2], 5.0), \
        "均值插补未还原常数列"
    assert np.allclose(te1[:, 0], 5.0) and np.allclose(te1[:, 2], 5.0)
    print(f"[A1] 均值插补还原常数列=5.0 校验通过；耗时={s1:.4f}s")

    # 三种方法插补后不得有 NaN
    Xtr = rng.normal(size=(300, 5))
    Xte = rng.normal(size=(120, 5))
    Xtr[rng.random(Xtr.shape) < 0.3] = np.nan
    Xte[rng.random(Xte.shape) < 0.3] = np.nan
    for m in METHODS:
        a, b, secs = impute(Xtr, Xte, m, seed=0)
        assert a.shape == Xtr.shape and b.shape == Xte.shape
        assert not np.isnan(a).any() and not np.isnan(b).any(), f"{m} 残留 NaN"
        print(f"[{m}] 插补后无 NaN：train={a.shape} test={b.shape} 耗时={secs:.4f}s")

    # 确定性：同种子同输入结果完全一致
    a1 = impute(Xtr, Xte, "A2", seed=0)[0]
    a2 = impute(Xtr, Xte, "A2", seed=0)[0]
    assert np.array_equal(a1, a2), "A2 结果不确定"
    # skip_complete=True 是纯优化：与 sklearn 默认设置结果必须逐位相同
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    Xtr_r, Xte_r = Xtr.copy(), Xte.copy()
    # 造出「部分列完整、部分列缺失」的形态（模拟 one-hot 列在 30% 注入下仍完整）
    Xtr_r[:, 3:] = np.nan_to_num(Xtr_r[:, 3:])
    Xte_r[:, 3:] = np.nan_to_num(Xte_r[:, 3:])
    ref = IterativeImputer(random_state=0, max_iter=10).fit_transform(Xtr_r)
    fast = impute(Xtr_r, Xte_r, "A2", seed=0)[0]
    assert np.array_equal(ref, fast), \
        f"skip_complete=True 改变了 A2 结果：最大绝对差={np.max(np.abs(ref - fast))}"
    print("impute 自检：A2 的 skip_complete=True 与 sklearn 默认设置逐位一致")
    # Newly missing test values in training-complete columns use training means.
    complete = np.nan_to_num(Xtr)
    _, newly_missing, _ = impute(complete, Xte, "A2", seed=0)
    rows, cols = np.where(np.isnan(Xte))
    assert np.allclose(newly_missing[rows, cols], complete.mean(axis=0)[cols])
    print("impute check: test-only missing columns use training means")
    # 测试集用训练集统计量：改变训练集后测试集插补值应变化
    Xtr2 = Xtr + 10.0
    te_changed = impute(Xtr2, Xte, "A1", seed=0)[1]
    te_same = impute(Xtr, Xte, "A1", seed=0)[1]
    assert not np.allclose(te_changed, te_same), "测试集未使用训练集统计量"
    print("impute 自检全部通过（含常数还原、无 NaN、确定性与训练集 fit）")


if __name__ == "__main__":
    _self_test()
