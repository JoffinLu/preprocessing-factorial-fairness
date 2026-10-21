"""因子 C：公平性预处理干预（T3 项目）

冻结水平
    C1：无干预（原样返回，sample_weight=None）
    C2：样本再加权 —— aif360 Reweighing（Kamiran–Calders）
    C3：特征去相关 —— aif360 DisparateImpactRemover(repair_level=1.0)

处理顺序：C 在 A（插补）、B（不平衡）之后（固定 A → B → C）。
因此 C2 的权重与 C3 的修复都作用在「已插补、已按 B 处理」的训练集上。

------------------------------------------------------------------ C2 口径
aif360 的 BinaryLabelDataset 要求敏感属性为数值。本实现：
    - 2 类敏感属性：升序排列后 0=unprivileged、1=privileged
      （adult sex: Female<Male -> Female=0；german sex: female<male -> female=0；
       bank age_bin: 0=年轻组=unprivileged）
    - **多于 2 类**（diabetes race 5 类、compas race 6 类）：采用「一对多」，
      取**多数组为 privileged(=1)**、其余全部为 unprivileged(=0)，即把敏感属性
      二分化为「多数组 vs 其余」。原始多类别仍保留用于公平性指标计算
      （run_cell 中评估阶段始终传原始敏感取值，因此 DP/EO/EOdd 仍按原始多类别口径）。
    对训练集 fit_transform，取 instance_weights 作为 sample_weight。

------------------------------------------------------------------ C3 实测结论
在 aif360 0.6.1 中，DisparateImpactRemover 的公开 API **不支持**
「在训练集 fit、再 repair 测试集」这条路径，实测证据：
    1. `DisparateImpactRemover.transform` / `.predict` 直接 raise
       NotImplementedError（源码注释：「Perhaps you meant 'predict' or
       'fit_transform' instead?」）。
    2. `fit_transform` 内部把修复器建为**局部变量** `repairer`，并未保存到
       `self.repairer`，因此 fit 之后无法复用。
    3. 底层 `BlackBoxAuditing` 的 `Repairer.repair(data)` 是**无状态**的：
       它完全从传入的 data 重新推导分箱与中位数（`self.all_data` 从不参与
       repair），所以对测试集单独调用等价于「独立修复测试集」，并非
       「用训练集的口径修复测试集」。
    最小可行替代方案：把 [训练集; 测试集] 的（特征 + 敏感指示列）**合并**后
    调用一次 fit_transform，再按行数拆回。这样训练/测试共享同一次修复口径，
    且能保证「训练集与测试集都被变换」，与 aif360 官方文档的 Note
    （要求两集在受保护属性上的条件分布一致）自洽。代价是该修复是直推式
    （用到测试集特征分布），已在 meta.c3_path 中显式标注。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

METHODS: tuple[str, ...] = ("C1", "C2", "C3")
SENS_COL: str = "__sensitive__"


@dataclass
class InterventionResult:
    """C 因子输出：变换后的训练/测试特征，以及可选的样本权重"""
    X_train: np.ndarray
    X_test: np.ndarray
    sample_weight: np.ndarray | None
    meta: dict[str, Any] = field(default_factory=dict)


def _binarize_sensitive(s_train: np.ndarray, s_test: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """把敏感属性编码为 0/1（返回 (训练码, 测试码, 原始类别, 口径说明)）

    - 2 类：升序第 0 个 -> 0（unprivileged），第 1 个 -> 1（privileged）
    - >2 类：多数组 -> 1（privileged），其余 -> 0（unprivileged）
    """
    uniques, counts = np.unique(s_train, return_counts=True)
    if len(uniques) < 2:
        codes_tr = np.zeros(len(s_train), dtype=int)
        codes_te = np.zeros(len(s_test), dtype=int)
        return codes_tr, codes_te, uniques, "仅单一类别"

    if len(uniques) == 2:
        mapping = {uniques[0]: 0, uniques[1]: 1}
        note = (f"2 类：{uniques[0]!r}->0(unprivileged)、"
                f"{uniques[1]!r}->1(privileged)")
    else:
        majority = uniques[int(np.argmax(counts))]
        mapping = {u: (1 if u == majority else 0) for u in uniques}
        note = (f"{len(uniques)} 类：多数组 {majority!r}->1(privileged)，"
                f"其余->0(unprivileged)（一对多，原始类别保留用于公平指标）")
    codes_tr = np.array([mapping[v] for v in s_train], dtype=int)
    codes_te = np.array([mapping.get(v, 0) for v in s_test], dtype=int)
    return codes_tr, codes_te, uniques, note


def _reweighing(codes_tr: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """对训练集做 aif360 Reweighing，返回 instance_weights"""
    from aif360.algorithms.preprocessing import Reweighing
    from aif360.datasets import BinaryLabelDataset

    df = pd.DataFrame({SENS_COL: codes_tr.astype(float),
                       "label": np.asarray(y_train).astype(float)})
    ds = BinaryLabelDataset(df=df, label_names=["label"],
                            protected_attribute_names=[SENS_COL])
    rw = Reweighing(unprivileged_groups=[{SENS_COL: 0}],
                    privileged_groups=[{SENS_COL: 1}])
    out = rw.fit_transform(ds)
    return np.asarray(out.instance_weights, dtype=float)


def _dir_combined(X_train: np.ndarray, y_train: np.ndarray, codes_tr: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray, codes_te: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """C3 最小可行替代：合并 [train;test] 做一次 DIR，再拆回

    返回 (修复后训练特征, 修复后测试特征, 是否真的发生修改)
    """
    from aif360.algorithms.preprocessing import DisparateImpactRemover
    from aif360.datasets import BinaryLabelDataset

    p = X_train.shape[1]
    n_tr = len(X_train)
    feat_cols = [f"f{i}" for i in range(p)] + [SENS_COL]
    aug = np.hstack([np.vstack([X_train, X_test]),
                     np.concatenate([codes_tr, codes_te]).reshape(-1, 1).astype(float)])
    df = pd.DataFrame(aug, columns=feat_cols)
    df["label"] = np.concatenate([y_train, y_test]).astype(float)

    ds = BinaryLabelDataset(df=df, label_names=["label"],
                            protected_attribute_names=[SENS_COL])
    order = list(ds.feature_names)
    assert order == feat_cols, f"aif360 特征列顺序与预期不符：{order}"
    dirr = DisparateImpactRemover(repair_level=1.0)
    out = dirr.fit_transform(ds)
    repaired = np.asarray(out.features, dtype=float)[:, :p]   # 去掉敏感列
    changed = not np.allclose(repaired, np.vstack([X_train, X_test]))
    return repaired[:n_tr], repaired[n_tr:], changed


def apply_fair_intervention(X_train: np.ndarray, y_train: np.ndarray,
                            sensitive_train: np.ndarray, X_test: np.ndarray,
                            sensitive_test: np.ndarray, method: str,
                            seed: int = 0) -> InterventionResult:
    """应用公平性预处理干预（只作用于训练集的特征/权重；C3 同时变换测试集）

    参数
        X_train, y_train, sensitive_train : 训练集特征、标签、原始敏感取值
        X_test, sensitive_test            : 测试集特征、原始敏感取值
        method                            : "C1" / "C2" / "C3"
        seed                              : 随机种子（aif360 无随机性，仅透传）

    返回
        InterventionResult(X_train', X_test', sample_weight, meta)
    """
    if method not in METHODS:
        raise ValueError(f"未知干预方法 {method!r}，可选 {METHODS}")
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train).astype(int)
    sensitive_train = np.asarray(sensitive_train)
    sensitive_test = np.asarray(sensitive_test)

    if method == "C1":
        return InterventionResult(X_train.copy(), X_test.copy(), None,
                                  {"seed": seed, "method": "C1",
                                   "c3_path": None, "sample_weight_mean": None})

    codes_tr, codes_te, uniques, note = _binarize_sensitive(sensitive_train,
                                                            sensitive_test)

    if method == "C2":
        w = _reweighing(codes_tr, y_train)
        return InterventionResult(X_train.copy(), X_test.copy(), w,
                                  {"seed": seed, "method": "C2",
                                   "sensitive_binary_note": note,
                                   "n_sensitive_classes": int(len(uniques)),
                                   "unprivileged": "0", "privileged": "1",
                                   "sample_weight_mean": float(w.mean()),
                                   "sample_weight_min": float(w.min()),
                                   "sample_weight_max": float(w.max()),
                                   "c3_path": None})

    # ---- C3 ----
    tr, te, changed = _dir_combined(X_train, y_train, codes_tr,
                                    X_test, y_test=np.zeros(len(X_test), dtype=int),
                                    codes_te=codes_te)
    return InterventionResult(tr, te, None,
                              {"seed": seed, "method": "C3",
                               "sensitive_binary_note": note,
                               "n_sensitive_classes": int(len(uniques)),
                               "c3_path": "combined_fit_transform（直推式）",
                               "c3_features_changed": bool(changed),
                               "sample_weight_mean": None})


# ---------------------------------------------------------------------------
# 自检：python code/pipeline/intervention.py
# ---------------------------------------------------------------------------
def _toy_train() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """已知单元格计数：sex0=60/sex1=40；y0=70/y1=30；(0,0)=50 (0,1)=10 (1,0)=20 (1,1)=20"""
    rng = np.random.default_rng(0)
    cells = [((0, 0), 50), ((0, 1), 10), ((1, 0), 20), ((1, 1), 20)]
    rows_x, rows_y, rows_s = [], [], []
    for (d, y), k in cells:
        for _ in range(k):
            rows_x.append([rng.normal(d + y, 1.0), rng.normal(0, 1.0)])
            rows_y.append(y)
            rows_s.append(d)
    return np.array(rows_x), np.array(rows_y), np.array(rows_s)


EXPECTED_W = {(0, 0): 0.84, (0, 1): 1.80, (1, 0): 1.40, (1, 1): 0.60}


def _self_test() -> None:
    Xtr, ytr, str_ = _toy_train()
    Xte, yte, ste = Xtr.copy(), ytr.copy(), str_.copy()

    # ---- C1：原样 ----
    r1 = apply_fair_intervention(Xtr, ytr, str_, Xte, ste, "C1")
    assert r1.sample_weight is None and np.array_equal(r1.X_train, Xtr)
    print("[C1] 原样返回，sample_weight=None")

    # ---- C2：相对权重须与 Kamiran–Calders 解析权重一致（aif360 会整体缩放）----
    r2 = apply_fair_intervention(Xtr, ytr, str_, Xte, ste, "C2")
    w = r2.sample_weight
    assert w is not None and len(w) == len(Xtr)
    scale = float(np.mean(w))              # aif360 常令权重均值=1
    got, exp = {}, {}
    for (d, y) in EXPECTED_W:
        m = (str_ == d) & (ytr == y)
        got[(d, y)] = round(float(w[m].mean()) / scale, 4)
    for k, v in EXPECTED_W.items():
        exp[k] = round(v, 4)
    ok = all(abs(got[k] - exp[k]) < 1e-3 for k in exp)
    print(f"[C2] 权重均值={scale:.4f}；相对权重 观测={got}")
    print(f"[C2]                        期望={exp}；一致={'是' if ok else '否'}")
    assert ok, (got, exp)
    assert r2.meta["unprivileged"] == "0" and r2.meta["privileged"] == "1"

    # ---- C2 多类别（>2）：一对多，须能运行并给出非退化权重 ----
    s_multi = np.array(["A"] * 200 + ["B"] * 40 + ["C"] * 10)
    y_multi = np.array([1] * 120 + [0] * 80 + [1] * 20 + [0] * 20 + [1] * 5 + [0] * 5)
    X_multi = np.random.default_rng(1).normal(size=(250, 3))
    r2m = apply_fair_intervention(X_multi, y_multi, s_multi, X_multi, s_multi, "C2")
    assert r2m.sample_weight is not None and len(r2m.sample_weight) == 250
    assert "多数组" in r2m.meta["sensitive_binary_note"]
    assert r2m.sample_weight.max() - r2m.sample_weight.min() > 1e-6, "权重退化"
    print(f"[C2 多类] 3 类敏感 -> 一对多；权重区间="
          f"[{r2m.sample_weight.min():.3f}, {r2m.sample_weight.max():.3f}]；"
          f"口径={r2m.meta['sensitive_binary_note']}")

    # ---- C3：特征确实被修改，且训练/测试都被变换 ----
    rng = np.random.default_rng(2)
    Xc_tr = rng.normal(size=(200, 4))
    Xc_te = rng.normal(size=(60, 4))
    # 令部分特征与敏感属性相关，使 DIR 有可修复空间
    sc_tr = np.array([0] * 120 + [1] * 80)
    sc_te = np.array([0] * 36 + [1] * 24)
    Xc_tr[:, 0] += sc_tr * 3.0
    Xc_te[:, 0] += sc_te * 3.0
    yc_tr = rng.integers(0, 2, size=200)
    yc_te = rng.integers(0, 2, size=60)
    r3 = apply_fair_intervention(Xc_tr, yc_tr, sc_tr, Xc_te, sc_te, "C3")
    assert r3.X_train.shape == Xc_tr.shape and r3.X_test.shape == Xc_te.shape
    assert r3.X_train.shape[1] == Xc_tr.shape[1], "敏感指示列须被移除"
    tr_changed = not np.allclose(r3.X_train, Xc_tr)
    te_changed = not np.allclose(r3.X_test, Xc_te)
    print(f"[C3] c3_path={r3.meta['c3_path']}；"
          f"训练集被修改={tr_changed}，测试集被修改={te_changed}")
    assert tr_changed and te_changed, "DIR 必须同时变换训练集与测试集"
    assert r3.sample_weight is None
    # DIR 不改动敏感列之外的行内顺序
    assert len(r3.X_train) == len(Xc_tr) and len(r3.X_test) == len(Xc_te)
    # 确定性
    r3b = apply_fair_intervention(Xc_tr, yc_tr, sc_tr, Xc_te, sc_te, "C3")
    assert np.array_equal(r3.X_train, r3b.X_train)
    print("intervention 自检全部通过")


if __name__ == "__main__":
    _self_test()
