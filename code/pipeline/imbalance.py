"""因子 B：类别不平衡处理（T3 项目）

冻结水平
    B1：无处理（原样返回）
    B2：类别权重 —— 不做数据重采样，只在返回的 meta 里给出 class_weight
        （权重在模型阶段使用）
    B3：SMOTE —— imblearn.over_sampling.SMOTE(random_state=seed, k_neighbors=5)，
        只对**训练集**重采样

SMOTE 与敏感属性对齐方式（写明口径）
    SMOTE 会生成新样本，其敏感属性需与特征一致。本实现把敏感属性编码为
    数值指示列（pd.factorize 得到 0..K-1），与特征**拼接后一起**送入 SMOTE
    （即对「特征 + 敏感属性指示列」联合重采样），再拆回。
    这样 SMOTE 近邻插值会同时作用于敏感属性列，新样本的敏感属性由近邻决定，
    与生成的特征保持同一近邻来源，避免事后错配。

边界要求
    若某类样本数 < 6（SMOTE 的 k_neighbors 下限），自动把 k_neighbors 降到
    min(count) - 1（>=1）；若某类样本数 < 2 则跳过 SMOTE 并记入 meta。

处理顺序：B 在 A 之后、C 之前（固定 A → B → C）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

METHODS: tuple[str, ...] = ("B1", "B2", "B3")
SMOTE_K: int = 5


def _dist(y: np.ndarray) -> dict[str, int]:
    vals, cnts = np.unique(y, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnts)}


def class_weight_balanced(y: np.ndarray) -> dict[int, float]:
    """sklearn 口径的 balanced 类别权重：n_samples / (n_classes * count_c)"""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    w = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(v) for c, v in zip(classes, w)}


def resample(X_train: np.ndarray, y_train: np.ndarray,
             sensitive_train: np.ndarray, method: str, seed: int = 0
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """处理类别不平衡（仅训练集）

    返回
        (X_train', y_train', sensitive_train', meta)
        meta 记录重采样前后类别分布、样本数；B2 额外给出 class_weight。
    """
    if method not in METHODS:
        raise ValueError(f"未知不平衡方法 {method!r}，可选 {METHODS}")
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train).astype(int)
    s = np.asarray(sensitive_train)

    if method == "B1":
        meta = {"before_dist": _dist(y), "after_dist": _dist(y),
                "n_before": int(len(y)), "n_after": int(len(y)),
                "class_weight": None, "smote_k": None}
        return X, y, s, meta

    if method == "B2":
        cw = class_weight_balanced(y)
        meta = {"before_dist": _dist(y), "after_dist": _dist(y),
                "n_before": int(len(y)), "n_after": int(len(y)),
                "class_weight": cw, "smote_k": None}
        return X, y, s, meta

    # ---- B3：SMOTE（特征 + 敏感指示列 联合重采样）----
    from imblearn.over_sampling import SMOTE

    codes, uniques = _factorize_sensitive(s)
    X_aug = np.hstack([X, codes.reshape(-1, 1).astype(float)])

    counts = np.bincount(y)
    min_count = int(counts[counts > 0].min())
    if int((counts > 0).sum()) < 2 or min_count < 2:
        meta = {"before_dist": _dist(y), "after_dist": _dist(y),
                "n_before": int(len(y)), "n_after": int(len(y)),
                "class_weight": None, "smote_k": None,
                "note": "某类样本数<2，SMOTE 无法执行，已跳过（等价 B1）"}
        return X, y, s, meta

    k = int(min(SMOTE_K, min_count - 1))
    if k < 1:
        k = 1
    smote = SMOTE(random_state=seed, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X_aug, y)

    X_out = X_res[:, :-1]
    code_out = np.rint(X_res[:, -1]).astype(int)
    code_out = np.clip(code_out, 0, len(uniques) - 1)
    s_out = np.asarray(uniques)[code_out]

    meta = {"before_dist": _dist(y), "after_dist": _dist(y_res.astype(int)),
            "n_before": int(len(y)), "n_after": int(len(y_res)),
            "class_weight": None, "smote_k": k,
            "sensitive_alignment": "敏感属性以数值指示列与特征拼接后联合 SMOTE，"
                                   "新样本敏感属性取近邻插值后最近整数",
            "sensitive_before_dist": _dist_codes(codes),
            "sensitive_after_dist": _dist_codes(code_out)}
    return X_out, y_res.astype(int), s_out, meta


def _factorize_sensitive(s: np.ndarray):
    """把敏感属性编码为 0..K-1 整数码，返回 (codes, uniques)"""
    uniques, codes = np.unique(s, return_inverse=True)
    return codes.astype(int), uniques


def _dist_codes(codes: np.ndarray) -> dict[str, int]:
    vals, cnts = np.unique(codes, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals, cnts)}


# ---------------------------------------------------------------------------
# 自检：python code/pipeline/imbalance.py
# ---------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(0)
    n_min, n_maj = 30, 300
    X = rng.normal(size=(n_min + n_maj, 3))
    y = np.array([1] * n_min + [0] * n_maj)
    s = np.array(["male"] * (n_min // 2) + ["female"] * (n_min - n_min // 2)
                 + ["male"] * (n_maj // 2) + ["female"] * (n_maj - n_maj // 2))

    # B1：原样
    X1, y1, s1, m1 = resample(X, y, s, "B1")
    assert X1 is X or np.array_equal(X1, X)
    assert m1["class_weight"] is None and m1["n_before"] == m1["n_after"] == len(y)
    print(f"[B1] 原样返回，类分布={m1['after_dist']}")

    # B2：不重采样，只给 class_weight
    X2, y2, s2, m2 = resample(X, y, s, "B2")
    assert m2["n_after"] == len(y), "B2 不应改变样本数"
    exp_maj = len(y) / (2 * n_maj)
    exp_min = len(y) / (2 * n_min)
    assert abs(m2["class_weight"][0] - exp_maj) < 1e-9
    assert abs(m2["class_weight"][1] - exp_min) < 1e-9
    print(f"[B2] 未重采样，class_weight={m2['class_weight']}")

    # B3：SMOTE 平衡，敏感属性对齐长度一致
    X3, y3, s3, m3 = resample(X, y, s, "B3", seed=0)
    assert m3["after_dist"][0] == m3["after_dist"][1], m3["after_dist"]
    assert len(X3) == len(y3) == len(s3) == m3["n_after"]
    assert set(np.unique(s3)) <= {"male", "female"}
    print(f"[B3] SMOTE 前={m3['before_dist']} 后={m3['after_dist']} "
          f"k={m3['smote_k']} 敏感分布后={m3['sensitive_after_dist']}")

    # 边界：某类只有 3 个样本 -> 自动降 k
    yb = np.array([1] * 3 + [0] * 200)
    sb = np.array(["a", "b", "a"] + ["a"] * 100 + ["b"] * 100)
    Xb = rng.normal(size=(203, 3))
    Xb2, yb2, sb2, mb2 = resample(Xb, yb, sb, "B3", seed=0)
    assert mb2["smote_k"] == 2, mb2["smote_k"]
    assert mb2["after_dist"][0] == mb2["after_dist"][1]
    print(f"[B3 边界] 少数类=3 -> 自动 k_neighbors={mb2['smote_k']}，"
          f"后分布={mb2['after_dist']}")
    # 边界：某类只有 1 个样本 -> 跳过并给出可读说明
    yc = np.array([1] + [0] * 100)
    sc = np.array(["a"] + ["a"] * 50 + ["b"] * 50)
    Xc = rng.normal(size=(101, 3))
    _, _, _, mc = resample(Xc, yc, sc, "B3", seed=0)
    assert "note" in mc and mc["smote_k"] is None
    print(f"[B3 边界] 少数类=1 -> {mc['note']}")
    # 确定性
    assert np.array_equal(resample(X, y, s, "B3", seed=0)[0],
                          resample(X, y, s, "B3", seed=0)[0]), "B3 不确定"
    print("imbalance 自检全部通过")


if __name__ == "__main__":
    _self_test()
