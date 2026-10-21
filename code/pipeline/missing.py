"""缺失注入（因子 M：M1 MCAR / M2 MAR）（T3 项目）

冻结要求
    - 缺失率固定 30%（由 Config.missing_rate 传入）
    - MCAR：对每个单元格独立地以 rate 概率置 NaN（np.random.default_rng(seed)）
    - MAR ：某列的缺失概率依赖于**其他观测到的列**
    - 注入后每列必须保留若干可观测值（不得整列缺失）

MAR 实现细节（务必与论文口径一致）
    1. 把所有列转成数值再标准化：
         数值列 -> 原值；分类列 -> pd.factorize 编码
         逐列做 z-score（nan-aware，缺失预测子按 0 处理即均值）
       得到 Z（N×P）。
    2. 为每一列 j 生成一组固定随机权重 W[j, k] ~ N(0,1)（W[j, j] = 0），
       线性打分 s_j(i) = Σ_{k≠j} W[j,k] · Z[i,k]。即「列 j 是否缺失」只依赖
       其他列（这些列在注入开始时都是观测到的），因此严格是 MAR 而非 MCAR。
    3. 概率 p_j(i) = sigmoid(s_j(i) + b_j)，其中截距 b_j 用**二分法**校准，
       使该列的期望缺失率≈ rate（从而整体缺失率≈ rate）。
    4. 按 p_j(i) 抽样置 NaN；p 截断到 [1e-4, 0.95] 以保证可观测样本量。
    5. 兜底：若某列置 NaN 后可观测样本 < MIN_OBSERVED，随机制换回一部分。

注意：本函数只「增加」缺失，不改变已有 NaN。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

MECHANISMS: tuple[str, ...] = ("MCAR", "MAR")
MIN_OBSERVED: int = 3          # 每列至少保留的可观测单元数（兜底）
_P_MIN, _P_MAX = 1e-4, 0.95    # 单列缺失概率截断区间


def _is_categorical(s: pd.Series) -> bool:
    return (not pd.api.types.is_numeric_dtype(s)
            and not pd.api.types.is_datetime64_any_dtype(s))


def _numeric_z(X: pd.DataFrame) -> np.ndarray:
    """把所有列转数值并逐列 z-score 标准化（nan-aware），返回 N×P 矩阵"""
    n, cols = len(X), list(X.columns)
    Z = np.zeros((n, len(cols)), dtype=float)
    for k, c in enumerate(cols):
        s = X[c]
        if _is_categorical(s):
            codes = pd.factorize(s, use_na_sentinel=True)[0].astype(float)
            v = np.where(codes < 0, np.nan, codes)
        else:
            v = s.to_numpy(dtype=float)
        mu, sd = np.nanmean(v), np.nanstd(v)
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        z = (v - (mu if np.isfinite(mu) else 0.0)) / sd
        Z[:, k] = np.nan_to_num(z, nan=0.0)
    return Z


def _mar_weights(n_cols: int, seed: int) -> np.ndarray:
    """每列一行的固定随机权重矩阵（对角为 0）"""
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 1.0, size=(n_cols, n_cols))
    np.fill_diagonal(W, 0.0)
    return W


def mar_scores(X: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """MAR 线性打分矩阵 s（N×P），供注入与自检共用同一公式

    s_j(i) = Σ_{k≠j} W[j,k] · Z[i,k]；列 j 的缺失概率 = sigmoid(s_j + b_j)。
    """
    Z = _numeric_z(X)
    W = _mar_weights(Z.shape[1], seed)
    return Z @ W.T


def _calibrate_intercept(scores: np.ndarray, rate: float,
                         lo: float = -30.0, hi: float = 30.0,
                         iters: int = 60) -> float:
    """二分法求截距 b 使 mean(sigmoid(scores + b)) == rate（sigmoid 关于 b 单调）"""
    s = np.clip(scores, -60.0, 60.0)

    def mean_p(b: float) -> float:
        return float(np.mean(1.0 / (1.0 + np.exp(-(s + b)))))

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if mean_p(mid) > rate:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def inject_missing(X: pd.DataFrame, mechanism: str, rate: float = 0.30,
                   seed: int = 0) -> pd.DataFrame:
    """按机制注入缺失，返回含 NaN 的新 DataFrame（不改动入参）

    参数
        X         : 输入特征表（可含既有天然缺失）
        mechanism : "MCAR" 或 "MAR"
        rate      : 目标缺失率（固定 0.30）
        seed      : 随机种子（保证确定性）
    """
    if mechanism not in MECHANISMS:
        raise ValueError(f"mechanism 须为 {MECHANISMS}，收到 {mechanism!r}")
    if not 0.0 < rate < 1.0:
        raise ValueError("rate 须在 (0,1) 内")

    X = X.reset_index(drop=True)
    n, p = X.shape
    out = X.copy()

    if mechanism == "MCAR":
        rng = np.random.default_rng(seed)
        mask = rng.random((n, p)) < rate
    else:  # MAR
        scores = mar_scores(X, seed)
        rng = np.random.default_rng(seed + 1000)
        mask = np.zeros((n, p), dtype=bool)
        for j in range(p):
            b = _calibrate_intercept(scores[:, j], rate)
            pj = 1.0 / (1.0 + np.exp(-(np.clip(scores[:, j], -60.0, 60.0) + b)))
            pj = np.clip(pj, _P_MIN, _P_MAX)
            mask[:, j] = rng.random(n) < pj

    # 兜底：保证每列可观测样本不少于 MIN_OBSERVED
    for j in range(p):
        missing_idx = np.flatnonzero(mask[:, j])
        n_observed = n - len(missing_idx)
        if n_observed < MIN_OBSERVED and len(missing_idx):
            need = MIN_OBSERVED - n_observed
            keep = rng.choice(missing_idx, size=min(need, len(missing_idx)),
                              replace=False)
            mask[keep, j] = False

    # 用 DataFrame.mask 逐列置 NaN：
    #   **不可** 用 `to_numpy()` 再重建 DataFrame——混合 dtype 会被压成 object，
    #   下游 features._is_categorical 会把数值列误判成分类列，导致因子 A 完全失效
    #   且 one-hot 按浮点取值爆炸。df.mask 保留原 dtype（int 列引入 NaN 时升为 float）。
    mask_df = pd.DataFrame(mask, columns=out.columns, index=out.index)
    return out.mask(mask_df)


# ---------------------------------------------------------------------------
# 自检：python code/pipeline/missing.py
# ---------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(7)
    n, p = 8000, 6
    X = pd.DataFrame(rng.normal(size=(n, p)),
                     columns=[f"x{i}" for i in range(p)])
    X[f"c"] = rng.choice(["a", "b", "c"], size=n)

    # ---- MCAR：实测缺失率与目标 rate 偏差 < 0.02（逐列与整体）----
    rate = 0.30
    Xm = inject_missing(X, "MCAR", rate, seed=0)
    overall = float(Xm.isna().to_numpy().mean())
    col_rate = Xm.isna().mean()
    print(f"[MCAR] 整体实测缺失率={overall:.4f}（目标 {rate}）"
          f"  偏差={abs(overall - rate):.4f}")
    print(f"[MCAR] 逐列缺失率 min={col_rate.min():.4f} max={col_rate.max():.4f}")
    assert abs(overall - rate) < 0.02, overall
    assert (col_rate - rate).abs().max() < 0.02, col_rate
    # 确定性
    Xm2 = inject_missing(X, "MCAR", rate, seed=0)
    assert Xm.equals(Xm2), "MCAR 同种子应完全一致"

    # ---- dtype 保持（回归测试）：数值列不得退化为 object ----
    # 历史缺陷：早期实现用 `to_numpy()` + 重建 DataFrame，混合 dtype 被压成 object，
    # 下游 features._is_categorical 会据此把数值列当作分类列，使因子 A 完全失效。
    from features import split_columns
    num_m, cat_m = split_columns(Xm)
    assert cat_m == ["c"], f"分类列应仅为 ['c']，实得 {cat_m}"
    assert len(num_m) == p, f"数值列数应为 {p}，实得 {len(num_m)}"
    assert all(pd.api.types.is_numeric_dtype(Xm[c]) for c in num_m), \
        f"注入后数值列 dtype 退化：{Xm.dtypes.to_dict()}"
    assert all(not pd.api.types.is_numeric_dtype(Xm[c]) for c in cat_m)
    print(f"[dtype] 注入后数值列仍为数值型：{ {c: str(Xm[c].dtype) for c in num_m[:2]} }"
          f"…；分类列={cat_m}")

    # ---- MAR：目标列缺失率随「用于打分的线性打分」单调递增 ----
    Xmar = inject_missing(X, "MAR", rate, seed=0)
    overall_mar = float(Xmar.isna().to_numpy().mean())
    print(f"[MAR ] 整体实测缺失率={overall_mar:.4f}（目标 {rate}）"
          f"  偏差={abs(overall_mar - rate):.4f}")
    assert abs(overall_mar - rate) < 0.02, overall_mar

    scores = mar_scores(X, seed=0)
    # 取数值列 x0，按打分四分位分箱，检查缺失率单调递增
    j = 0
    order = np.argsort(scores[:, j])
    bins = np.array_split(order, 4)
    rates = [float(Xmar.iloc[b, j].isna().mean()) for b in bins]
    print(f"[MAR ] 列 x0 按打分四分位的缺失率（低→高）={[round(r, 4) for r in rates]}")
    strictly_increasing = all(rates[i] < rates[i + 1] for i in range(len(rates) - 1))
    assert strictly_increasing, f"MAR 缺失率未随打分单调递增：{rates}"
    # 对照：MCAR 下同样分箱应基本持平
    rates_mcar = [float(Xm.iloc[b, j].isna().mean()) for b in bins]
    print(f"[MCAR] 列 x0 按同一打分的缺失率（低→高）={[round(r, 4) for r in rates_mcar]}")
    assert abs(rates_mcar[3] - rates_mcar[0]) < 0.05, "MCAR 不应与打分强相关"
    assert (rates[3] - rates[0]) > (rates_mcar[3] - rates_mcar[0]), "MAR 相关性应更强"

    # ---- 每列至少保留可观测值 ----
    for name, Xx in (("MCAR", Xm), ("MAR", Xmar)):
        nobs = (n - Xx.isna().sum()).to_numpy()
        assert nobs.min() >= MIN_OBSERVED, (name, nobs)
        assert not Xx.isna().all().any(), f"{name} 出现整列缺失"
    print(f"最少可观测列样本数：MCAR={int(n - Xm.isna().sum().min())}, "
          f"MAR={int(n - Xmar.isna().sum().min())}")
    print("missing 自检全部通过（MCAR 偏差<0.02；MAR 与打分严格单调且非 MCAR）")


if __name__ == "__main__":
    _self_test()
