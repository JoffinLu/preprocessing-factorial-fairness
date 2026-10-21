"""特征编码与标准化（因子管线中的编码阶段）（T3 项目）

关键顺序说明（务必与论文一致）
    注入缺失 → **编码** → 标准化 → 插补
    编码必须发生在插补之前：若先插补再 one-hot，缺失会变成「整组哑变量都是 0」的
    二义状态。因此：
      - 分类列的缺失在 one-hot 前按配置处理：
          "as_category"（默认）：缺失本身当作一个专门取值 -> "__MISSING__"，
              正常变成一列。于是因子 A（插补）只作用于**数值列**。
          "mode"：分类列缺失先用训练集众数填充再 one-hot。
      - 数值列的缺失保持 NaN 进入插补器（因子 A 直接作用于此）。

高基数截断（Gate 2b 决策）
    Diabetes 的 `diag_1/2/3` 是 ICD-9 原始码，单列水平数 700+；不截断时
    one-hot 后总列数达 2,343（实测），MICE / KNN 插补在 9.9 万行上不可行，
    且矩阵内存约 1.9 GB。故对分类列按**训练集频次**取前 `max_cardinality` 个
    取值，其余（含测试集新出现的取值）统一并入 `__OTHER__`。
    —— 本模块原实现在此不做截断；截断逻辑为 Gate 2b 新增，须在论文 §3.5 写明。

对齐契约（train / test 必须列名与列序完全一致）
    `fit_category_spec(train)` 得到 spec 后，**训练集与测试集都传入同一 spec**。
    spec 一旦给定，输出列名与列序即完全由 spec 决定，与传入表内容无关。
    —— 这是 train/test 对齐的唯一机制；不允许在测试集上重新 fit。

标准化
    仅用训练集统计量 fit（StandardScaler 内部 nan-aware：缺失单元 transform 后
    仍为 NaN，等待插补）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

MISSING_TOKEN: str = "__MISSING__"
OTHER_TOKEN: str = "__OTHER__"


def _is_categorical(s: pd.Series) -> bool:
    return (not pd.api.types.is_numeric_dtype(s)
            and not pd.api.types.is_datetime64_any_dtype(s))


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """返回 (数值列, 分类列) 列名列表"""
    num = [c for c in X.columns if not _is_categorical(X[c])]
    cat = [c for c in X.columns if _is_categorical(X[c])]
    return num, cat


def categorical_fill_values(X: pd.DataFrame) -> dict[str, str]:
    """计算分类列的众数（供 categorical_missing="mode" 时在训练集上 fit）"""
    num, cat = split_columns(X)
    out: dict[str, str] = {}
    for c in cat:
        modes = X[c].mode(dropna=True)
        out[c] = str(modes.iloc[0]) if len(modes) else MISSING_TOKEN
    return out


def prepare_categoricals(X: pd.DataFrame, categorical_missing: str = "as_category",
                        fill_values: dict[str, str] | None = None) -> pd.DataFrame:
    """把分类列的缺失按 `categorical_missing` 口径转成纯字符串值表

    - "as_category"：缺失 -> `__MISSING__`（自成一类，因子 A 不作用于分类列）
    - "mode"       ：缺失 -> 训练集众数
    """
    if categorical_missing not in ("as_category", "mode"):
        raise ValueError("categorical_missing 须为 'as_category' 或 'mode'")
    _, cat = split_columns(X)
    X = X.reset_index(drop=True)
    fv = fill_values if categorical_missing == "mode" else None
    if categorical_missing == "mode" and fv is None:
        fv = categorical_fill_values(X)
    cols: dict[str, pd.Series] = {}
    for c in cat:
        s = X[c].astype(object)
        if categorical_missing == "as_category":
            vals = s.map(lambda v: MISSING_TOKEN if pd.isna(v) else str(v))
        else:
            assert fv is not None
            vals = s.map(lambda v: fv[c] if pd.isna(v) else str(v))
        cols[c] = vals.astype(str)
    return pd.DataFrame(cols, index=X.index) if cols else pd.DataFrame(index=X.index)


def fit_category_spec(X: pd.DataFrame, categorical_missing: str = "as_category",
                      max_cardinality: int | None = None,
                      fill_values: dict[str, str] | None = None
                      ) -> dict[str, list[str]]:
    """在**训练集**上按频次降序确定每列保留的取值集合（可截断）

    返回
        {列名: [保留取值, ...]}；截断时被丢弃的取值统一并入 `__OTHER__` 列。
    """
    catdf = prepare_categoricals(X, categorical_missing, fill_values)
    spec: dict[str, list[str]] = {}
    for c in catdf.columns:
        vals = list(catdf[c].value_counts(dropna=False).index)
        if max_cardinality is not None and len(vals) > max_cardinality:
            vals = vals[:max_cardinality]
        spec[c] = vals
    return spec


def encode(X: pd.DataFrame, categorical_missing: str = "as_category",
           category_spec: dict[str, list[str]] | None = None,
           fill_values: dict[str, str] | None = None
           ) -> tuple[np.ndarray, list[str]]:
    """编码：数值列原样保留（含 NaN），分类列 one-hot（不丢参照列）

    参数
        X                   : 输入特征表
        categorical_missing : "as_category"（默认）或 "mode"
        category_spec       : 由 `fit_category_spec(训练集)` 得到；**Train 与 Test
                              都必须传入同一 spec**。为 None 时在本表上现拟合，
                              仅允许用于训练集。
        fill_values         : categorical_missing="mode" 时的训练集众数

    返回
        (矩阵 np.ndarray[float]（数值列可能含 NaN）, 列名 list[str]）
    """
    if categorical_missing not in ("as_category", "mode"):
        raise ValueError("categorical_missing 须为 'as_category' 或 'mode'")
    num_cols, cat_cols = split_columns(X)
    X = X.reset_index(drop=True)
    if category_spec is None:
        category_spec = fit_category_spec(X, categorical_missing, fill_values=fill_values)

    num = (X[num_cols].to_numpy(dtype=float) if num_cols
           else np.zeros((len(X), 0), dtype=float))

    blocks: list[np.ndarray] = []
    dummy_names: list[str] = []
    if cat_cols:
        catdf = prepare_categoricals(X, categorical_missing, fill_values)
        n_dummy = sum(len(category_spec[c]) + 1 for c in cat_cols)
        dummy = np.zeros((len(X), n_dummy), dtype=float)
        off = 0
        for c in cat_cols:
            kept = category_spec[c]
            idx = {v: i for i, v in enumerate(kept)}
            other_col = len(kept)                      # 最后一个位置留给 __OTHER__
            vals = catdf[c].to_numpy()
            for row, v in enumerate(vals):
                dummy[row, off + idx.get(v, other_col)] = 1.0
            dummy_names += [f"{c}_{v}" for v in kept] + [f"{c}_{OTHER_TOKEN}"]
            off += other_col + 1
        blocks.append(dummy)

    mat = np.hstack([num] + blocks) if blocks else num
    names = list(num_cols) + dummy_names
    return np.asarray(mat, dtype=float), names


def standardize(train: np.ndarray, test: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """在训练集上 fit StandardScaler，再 transform 训练/测试集（禁止用测试统计量）

    StandardScaler 支持 NaN：缺失单元 transform 后仍为 NaN，留给插补阶段处理。
    零方差列（如恒为 0 的 `__OTHER__` 列）由 sklearn 内部令 scale=1 处理。
    """
    sc = StandardScaler()
    tr = sc.fit_transform(train)
    te = sc.transform(test)
    return tr, te


# ---------------------------------------------------------------------------
# 自检：python code/pipeline/features.py
# ---------------------------------------------------------------------------
def _self_test() -> None:
    X = pd.DataFrame({
        "n1": [1.0, 2.0, np.nan, 4.0],
        "n2": [10.0, np.nan, 30.0, 40.0],
        "c1": ["a", "b", None, "a"],
        "c2": ["x", "x", "y", None],
    })

    # ---- as_category：分类缺失自成一类，数值 NaN 保留 ----
    mat_a, names_a = encode(X, "as_category")
    assert f"c1_{MISSING_TOKEN}" in names_a, names_a
    assert f"c2_{MISSING_TOKEN}" in names_a, names_a
    assert f"c1_{OTHER_TOKEN}" in names_a and f"c2_{OTHER_TOKEN}" in names_a
    num_pos = [names_a.index("n1"), names_a.index("n2")]
    assert np.isnan(mat_a[:, num_pos]).sum() == 2, "数值 NaN 应被保留"
    cat_pos = [i for i, nm in enumerate(names_a) if nm not in ("n1", "n2")]
    assert not np.isnan(mat_a[:, cat_pos]).any(), "分类 one-hot 不应含 NaN"
    # one-hot 完备性：共有 2 个分类列，故整个分类块每行应恰好有 2 个 1（每列 1 个）
    n_cat_cols = 2
    assert np.allclose(mat_a[:, cat_pos].sum(axis=1), float(n_cat_cols)), \
        "分类块每行 1 的个数应等于分类列数"
    print(f"[as_category] 列名({len(names_a)})={names_a}")
    print(f"[as_category] 数值列保留 NaN 数={int(np.isnan(mat_a[:, num_pos]).sum())}")

    # ---- mode：分类缺失用训练集众数填充 ----
    fv = categorical_fill_values(X)
    spec_m = fit_category_spec(X, "mode", fill_values=fv)
    mat_m, names_m = encode(X, "mode", category_spec=spec_m, fill_values=fv)
    assert f"c1_{MISSING_TOKEN}" not in names_m, names_m
    assert mat_m[2, names_m.index("c1_a")] == 1.0     # c1 众数为 a
    print(f"[mode] 分类众数填充值={fv}，填充后分类列无 NaN")

    # ---- 高基数截断：稀有取值须并入 __OTHER__，且每行仍恰有一个 1 ----
    Xh = pd.DataFrame({"c": ["a"] * 50 + ["b"] * 10 + ["c"] * 3 + ["d"] * 1})
    spec_h = fit_category_spec(Xh, "as_category", max_cardinality=2)
    assert spec_h["c"] == ["a", "b"], spec_h
    mat_h, names_h = encode(Xh, "as_category", category_spec=spec_h)
    assert names_h == ["c_a", "c_b", f"c_{OTHER_TOKEN}"], names_h
    other = mat_h[:, names_h.index(f"c_{OTHER_TOKEN}")]
    assert abs(other.sum() - 4.0) < 1e-12, other.sum()   # c 3 个 + d 1 个
    assert np.allclose(mat_h.sum(axis=1), 1.0), "截断后仍须 one-hot 完备"
    print(f"[截断] max_cardinality=2 -> 保留 {spec_h['c']}，4 行落入 __OTHER__，"
          f"每行仍恰 1 个 1")

    # ---- 对齐：训练与测试传入同一 spec -> 列名/列序完全一致，未见类别进 __OTHER__ ----
    tr = pd.DataFrame({"n1": [1.0, 2.0], "c1": ["a", "b"]})
    te = pd.DataFrame({"n1": [3.0, 4.0], "c1": ["a", "z"]})     # z 为训练集未见类别
    spec = fit_category_spec(tr, "as_category")
    mtr, cols = encode(tr, "as_category", category_spec=spec)
    mte, cols2 = encode(te, "as_category", category_spec=spec)
    assert cols2 == cols, (cols, cols2)
    assert mtr.shape[1] == mte.shape[1] == len(cols)
    assert mte[1, cols.index(f"c1_{OTHER_TOKEN}")] == 1.0, "未见类别须进 __OTHER__"
    assert mte[1, cols.index("c1_a")] == 0.0
    print(f"[对齐] 训练列={cols}；测试列名/列序一致，未见类别进 __OTHER__")

    # ---- 数值列不得被误判为分类列（dtype 回归测试）----
    num_c, cat_c = split_columns(pd.DataFrame({"f": [1.0, 2.0], "s": ["a", "b"]}))
    assert num_c == ["f"] and cat_c == ["s"], (num_c, cat_c)
    print(f"[dtype] 数值列={num_c} 分类列={cat_c}")

    # ---- standardize：只在训练集 fit，测试集用训练统计量 ----
    A = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    B = np.array([[100.0, 1.0]])
    At, Bt = standardize(A, B)
    mu, sd = A.mean(axis=0), A.std(axis=0)
    assert np.allclose(At, (A - mu) / sd), "训练集标准化口径不符"
    assert np.allclose(Bt, (B - mu) / sd), "测试集必须用训练集统计量"
    print("[standardize] 训练/测试均使用训练集 mean/std")

    # ---- 含 NaN 时 standardize 不报错且保留 NaN；零方差列不产生 inf ----
    Xn = np.array([[1.0, np.nan], [2.0, 3.0], [3.0, 5.0]])
    trn, ten = standardize(Xn, Xn)
    assert np.isnan(trn[0, 1]), "NaN 应被保留"
    Zv = np.zeros((5, 1))
    zt, _ = standardize(Zv, Zv)
    assert np.isfinite(zt).all(), "零方差列不应产生 inf"
    print(f"features 自检全部通过（含 5 项：缺失口径 / 截断 / 对齐 / dtype / 标准化）")


if __name__ == "__main__":
    _self_test()
