"""单单元格执行（T3 项目）

一个「单元格」= (dataset, mechanism, A, B, C, seed, model) 的唯一组合。
本模块把管线按**冻结顺序 A → B → C** 串起来，输出一行扁平结果。

固定顺序（写入论文 §3.3）
    加载清洗 → 分层 7:3 划分 → 缺失注入(M) → 编码 → 标准化 → 插补(A)
    → 不平衡处理(B) → 公平性干预(C) → 训练/预测 → 评估

划分口径（关键，影响统计功效）
    训练/测试划分**只依赖 (dataset, seed)**，与 M/A/B/C/model 无关。
    于是同一 (dataset, seed) 下 54 个单元格共享同一划分，比较退化为**配对比较**，
    正是因子实验应有的做法。

种子派生规则（写入论文，保证一键复现）
    - 数据划分      : random_state = seed
    - 缺失注入(训练): seed        ；缺失注入(测试): seed + 5000
    - 插补(A2 MICE) : seed        ；SMOTE(B3): seed
    - 模型          : seed
    —— 全部为确定性派生，无隐藏随机性。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from pipeline.config import MECHANISM_TO_NAME, Config  # noqa: E402
from pipeline.data import Dataset, load_dataset  # noqa: E402
from pipeline.features import (categorical_fill_values, encode,  # noqa: E402
                               fit_category_spec, standardize)
from pipeline.imbalance import resample  # noqa: E402
from pipeline.impute import impute  # noqa: E402
from pipeline.intervention import apply_fair_intervention  # noqa: E402
from pipeline.metrics import evaluate, group_table  # noqa: E402
from pipeline.missing import inject_missing  # noqa: E402
from pipeline.model import fit_predict  # noqa: E402

INJECT_SEED_OFFSET_TEST: int = 5000

_DATASET_CACHE: dict[tuple[str, bool], Dataset] = {}
_SPLIT_CACHE: dict[tuple[str, int, float], tuple[np.ndarray, np.ndarray]] = {}

ORDER_COLS: tuple[str, ...] = (
    "dataset", "mechanism", "A", "B", "C", "seed", "model", "natural_missing",
    "n_train", "n_test", "n_features_encoded",
    "imbalance_ratio", "minority_majority_ratio", "sensitive_name", "n_groups",
    "AUC", "F1", "accuracy", "balanced_accuracy", "precision", "recall",
    "specificity", "n_pos_true", "n_neg_true", "n_pred_pos",
    "DP", "EO", "EOdd", "TPR_gap", "DI",
    "undefined_tpr_groups", "undefined_sel_groups",
    "min_group_n", "min_group_n_pos",
    "impute_secs", "train_secs", "total_secs",
    "n_train_before_B", "smote_k", "resample_note",
    "c_note", "c3_path", "c3_changed", "sw_mean", "sw_min", "sw_max",
    "lr_converged", "inject_note", "status",
)


def get_dataset(name: str, natural_missing: bool = False) -> Dataset:
    """带缓存的加载（同一进程内复用，纯只读）"""
    key = (name, bool(natural_missing))
    if key not in _DATASET_CACHE:
        _DATASET_CACHE[key] = load_dataset(name, natural_missing)
    return _DATASET_CACHE[key]


def split_indices(dataset: str, y: np.ndarray, seed: int, test_size: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """分层 7:3 划分的索引（只依赖 dataset/seed/test_size，可缓存复用）"""
    key = (dataset, int(seed), round(float(test_size), 8))
    if key not in _SPLIT_CACHE:
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(np.arange(len(y)), test_size=test_size,
                                  random_state=int(seed), stratify=y)
        _SPLIT_CACHE[key] = (np.asarray(tr), np.asarray(te))
    return _SPLIT_CACHE[key]


def prepare_cell(dataset: str, mechanism: str, a: str, b: str, c: str, seed: int,
                 cfg: Config | None = None, natural_missing: bool = False) -> dict:
    """执行一个单元格的**预处理部分**（A → B → C），返回供建模使用的中间结果

    拆出该函数的原因：预处理与模型无关，同一 (dataset, mechanism, A, B, C, seed)
    下的 2 个模型共享同一份预处理结果，正好对应范围说明书 §4.6 的
    「810 次预处理 / 1,620 次训练-评估」——预处理只跑 810 次而非 1,620 次。
    """
    cfg = cfg or Config()
    t0 = time.perf_counter()

    if mechanism == "NAT" and not natural_missing:
        raise ValueError("mechanism=NAT 要求 natural_missing=True")

    ds = get_dataset(dataset, natural_missing)
    idx_tr, idx_te = split_indices(dataset, ds.y, seed, cfg.test_size)
    Xtr_raw = ds.X_raw.iloc[idx_tr].reset_index(drop=True)
    Xte_raw = ds.X_raw.iloc[idx_te].reset_index(drop=True)
    ytr, yte = ds.y[idx_tr], ds.y[idx_te]
    str_ = ds.sensitive.iloc[idx_tr].reset_index(drop=True)
    ste = ds.sensitive.iloc[idx_te].reset_index(drop=True)

    # ---- 因子 M：缺失注入 ----
    if mechanism == "NAT":
        Xtr_m, Xte_m = Xtr_raw, Xte_raw
        inject_note = "未注入（天然缺失对照）"
    else:
        mech_name = MECHANISM_TO_NAME[mechanism]
        Xtr_m = inject_missing(Xtr_raw, mech_name, cfg.missing_rate, seed=seed)
        Xte_m = inject_missing(Xte_raw, mech_name, cfg.missing_rate,
                               seed=seed + INJECT_SEED_OFFSET_TEST)
        inject_note = f"{mech_name} 注入 {cfg.missing_rate:.0%}"

    # ---- 编码（分类列缺失口径 + 高基数截断；train/test 共用同一 spec）----
    fv = (categorical_fill_values(Xtr_m)
          if cfg.categorical_missing == "mode" else None)
    spec = fit_category_spec(Xtr_m, cfg.categorical_missing, cfg.max_cardinality, fv)
    mat_tr, names = encode(Xtr_m, cfg.categorical_missing, spec, fv)
    mat_te, names_te = encode(Xte_m, cfg.categorical_missing, spec, fv)
    assert names_te == names, "训练/测试编码列名或列序不一致"

    # ---- 标准化（训练集统计量）----
    tr_s, te_s = standardize(mat_tr, mat_te)

    # ---- 因子 A：插补 ----
    tr_a, te_a, impute_secs = impute(tr_s, te_s, a, seed=seed)

    # ---- 因子 B：类别不平衡 ----
    tr_b, ytr_b, str_b, b_meta = resample(tr_a, ytr, str_, b, seed=seed)

    # ---- 因子 C：公平性预处理干预 ----
    iv = apply_fair_intervention(tr_b, ytr_b, str_b, te_a, ste, c, seed=seed)

    return {
        "ds": ds, "cfg": cfg, "dataset": dataset, "mechanism": mechanism,
        "A": a, "B": b, "C": c, "seed": int(seed),
        "natural_missing": bool(natural_missing),
        "X_train": iv.X_train, "y_train": ytr_b, "X_test": iv.X_test,
        "sample_weight": iv.sample_weight, "class_weight": b_meta["class_weight"],
        "y_test": yte, "sensitive_test": ste,
        "n_features_encoded": int(len(names)),
        "impute_secs": float(impute_secs),
        "n_train_before_B": b_meta["n_before"],
        "smote_k": b_meta.get("smote_k"),
        "resample_note": b_meta.get("note"),
        "c_note": iv.meta.get("sensitive_binary_note"),
        "c3_path": iv.meta.get("c3_path"),
        "c3_changed": iv.meta.get("c3_features_changed"),
        "sw_mean": iv.meta.get("sample_weight_mean"),
        "sw_min": iv.meta.get("sample_weight_min"),
        "sw_max": iv.meta.get("sample_weight_max"),
        "inject_note": inject_note,
        "prep_secs": float(time.perf_counter() - t0),
    }


def run_cell_from_prep(prep: dict, model_name: str,
                       collect_groups: bool = False) -> dict:
    """在已完成的预处理结果上训练+评估一个模型，返回结果行"""
    cfg = prep["cfg"]
    ds = prep["ds"]
    t0 = time.perf_counter()
    mr = fit_predict(model_name, prep["X_train"], prep["y_train"], prep["X_test"],
                     sample_weight=prep["sample_weight"],
                     class_weight=prep["class_weight"], seed=prep["seed"])
    train_secs = float(mr["train_time"])

    # ---- 评估（用测试集、原始多类别敏感属性口径）----
    m = evaluate(prep["y_test"], mr["y_pred"], mr["y_prob"], prep["sensitive_test"],
                 dataset=prep["dataset"], model=model_name)

    row = {
        "dataset": prep["dataset"], "mechanism": prep["mechanism"],
        "A": prep["A"], "B": prep["B"], "C": prep["C"], "seed": prep["seed"],
        "model": model_name, "natural_missing": prep["natural_missing"],
        "n_train": int(len(prep["y_train"])),
        "n_test": int(len(prep["y_test"])),
        "n_features_encoded": prep["n_features_encoded"],
        "imbalance_ratio": ds.meta["imbalance_ratio"],
        "minority_majority_ratio": ds.meta["minority_majority_ratio"],
        "sensitive_name": ds.sensitive_name,
        "impute_secs": prep["impute_secs"],
        "train_secs": train_secs,
        # 预处理与训练共同构成该单元格的墙钟耗时（同一预处理的第二个模型不重复计）
        "total_secs": prep["prep_secs"] + train_secs,
        "n_train_before_B": prep["n_train_before_B"],
        "smote_k": prep["smote_k"],
        "resample_note": prep["resample_note"],
        "c_note": prep["c_note"],
        "c3_path": prep["c3_path"],
        "c3_changed": prep["c3_changed"],
        "sw_mean": prep["sw_mean"],
        "sw_min": prep["sw_min"],
        "sw_max": prep["sw_max"],
        "lr_converged": mr["meta"].get("lr_converged"),
        "inject_note": prep["inject_note"],
        "status": "ok",
    }
    row.update(m)
    if collect_groups:
        row["_groups"] = group_table(prep["y_test"], mr["y_pred"],
                                     prep["sensitive_test"],
                                     dataset=prep["dataset"],
                                     mechanism=prep["mechanism"],
                                     A=prep["A"], B=prep["B"], C=prep["C"],
                                     seed=prep["seed"],
                                     model=model_name).to_dict(orient="records")
    return row


def run_cell(dataset: str, mechanism: str, a: str, b: str, c: str, seed: int,
             model_name: str, cfg: Config | None = None,
             natural_missing: bool = False,
             collect_groups: bool = False) -> dict:
    """执行一个完整单元格（预处理 + 单个模型），返回扁平结果行（键见 `ORDER_COLS`）

    参数
        mechanism       : "M1"(MCAR) / "M2"(MAR) / "NAT"（不注入，保留天然缺失）
        natural_missing : 是否让 data 层保留天然缺失（NAT 必须为 True）
        collect_groups  : True 时结果行内额外含 "_groups"（各敏感组率值表）

    批量运行时请用 `prepare_cell` + `run_cell_from_prep`，同一预处理可复用给多个模型。
    """
    prep = prepare_cell(dataset, mechanism, a, b, c, seed, cfg, natural_missing)
    return run_cell_from_prep(prep, model_name, collect_groups)


# ---------------------------------------------------------------------------
# 自检：python code/pipeline/run_cell.py
# ---------------------------------------------------------------------------
def _self_test() -> None:
    cfg = Config(seeds=(0, 1, 2))

    # ---- 1) 全链路可跑通（german 最小，纯受控注入）----
    r1 = run_cell("german", "M1", "A1", "B1", "C1", 0, "LR", cfg)
    assert r1["status"] == "ok"
    assert r1["n_test"] == 300 and r1["n_train"] == 700, (r1["n_train"], r1["n_test"])
    for k in ("AUC", "F1", "DP", "EO", "DI"):
        assert np.isfinite(r1[k]), (k, r1[k])
    print(f"[全链路] german M1/A1/B1/C1/LR seed0 -> AUC={r1['AUC']:.4f} "
          f"F1={r1['F1']:.4f} DP={r1['DP']:.4f} EO={r1['EO']:.4f} "
          f"DI={r1['DI']:.4f} 列数={r1['n_features_encoded']} "
          f"耗时={r1['total_secs']:.2f}s")

    # ---- 2) 确定性：同单元格两次运行，全部数值列逐位一致（计时列除外）----
    r1b = run_cell("german", "M1", "A1", "B1", "C1", 0, "LR", cfg)
    timing_cols = {"impute_secs", "train_secs", "total_secs"}
    for k, v in r1.items():
        if k == "_groups" or k in timing_cols:
            continue
        if isinstance(v, float):
            assert (v == r1b[k]) or (np.isnan(v) and np.isnan(r1b[k])), (k, v, r1b[k])
        else:
            assert v == r1b[k], (k, v, r1b[k])
    print("[确定性] 同单元格两次运行结果逐位一致（仅计时列豁免）")

    # ---- 3) 配对设计：不同机制/因子共享同一划分 ----
    tr_a, te_a = split_indices("german", get_dataset("german").y, 0, cfg.test_size)
    r_other = run_cell("german", "M2", "A3", "B3", "C2", 0, "LGBM", cfg)
    tr_b2, te_b2 = split_indices("german", get_dataset("german").y, 0, cfg.test_size)
    assert np.array_equal(tr_a, tr_b2) and np.array_equal(te_a, te_b2)
    assert r_other["n_test"] == r1["n_test"]
    print(f"[配对设计] M1/A1/B1/C1/LR 与 M2/A3/B3/C2/LGBM 共享同一划分"
          f"（测试集 n={r_other['n_test']}，完全相同索引）")

    # ---- 4) 因子 M 真的改变了数据：A1 在 M1 与 M2 下结果应不同 ----
    rM2 = run_cell("german", "M2", "A1", "B1", "C1", 0, "LR", cfg)
    diff = abs(rM2["AUC"] - r1["AUC"])
    assert diff > 1e-12, "MCAR 与 MAR 注入应产生不同结果（否则 M 因子无效）"
    print(f"[因子 M 有效] MCAR AUC={r1['AUC']:.4f} vs MAR AUC={rM2['AUC']:.4f}，"
          f"差={diff:.6f}")

    # ---- 5) 因子 A 真的起作用：A1 与 A3 结果应不同 ----
    rA3 = run_cell("german", "M1", "A3", "B1", "C1", 0, "LR", cfg)
    assert abs(rA3["AUC"] - r1["AUC"]) > 1e-12, "插补方法应产生影响"
    print(f"[因子 A 有效] A1 AUC={r1['AUC']:.4f} vs A3 AUC={rA3['AUC']:.4f}")

    # ---- 6) 因子 B、C 的作用路径：B2 必须给出 class_weight，C2 必须给出 sample_weight ----
    rB2 = run_cell("german", "M1", "A1", "B2", "C1", 0, "LR", cfg)
    rC2 = run_cell("german", "M1", "A1", "B1", "C2", 0, "LR", cfg)
    rC3 = run_cell("german", "M1", "A1", "B1", "C3", 0, "LR", cfg)
    assert rB2["n_train_before_B"] == rB2["n_train"], "B2 不应改变样本量"
    assert rC2["sw_mean"] is not None and rC3["c3_changed"] is True
    assert rC3["c3_path"].startswith("combined")
    print(f"[因子 B/C 路径] B2 样本量不变({rB2['n_train']})；"
          f"C2 权重均值={rC2['sw_mean']:.4f}；C3 特征被修改={rC3['c3_changed']}")
    # B3 应改变样本量
    rB3 = run_cell("german", "M1", "A1", "B3", "C1", 0, "LR", cfg)
    assert rB3["n_train"] > rB3["n_train_before_B"], "B3 SMOTE 应增加样本量"
    print(f"[因子 B] B3 SMOTE: {rB3['n_train_before_B']} -> {rB3['n_train']} 行")

    # ---- 7) NAT 稳健性路径：保留天然缺失且不注入 ----
    rnat = run_cell("adult", "NAT", "A1", "B1", "C1", 0, "LR", cfg,
                    natural_missing=True)
    assert rnat["inject_note"].startswith("未注入")
    assert np.isfinite(rnat["AUC"])
    print(f"[NAT 对照] adult 天然缺失/A1/B1/C1/LR -> AUC={rnat['AUC']:.4f} "
          f"DP={rnat['DP']:.4f} 列数={rnat['n_features_encoded']}")

    # ---- 8) 分组表结构 ----
    rg = run_cell("german", "M1", "A1", "B1", "C1", 0, "LR", cfg,
                  collect_groups=True)
    gs = rg["_groups"]
    assert len(gs) == 2 and {"group", "n", "sel", "tpr"} <= set(gs[0])
    print(f"[分组表] german 两组：{[{k: g[k] for k in ('group','n','sel','tpr')} for g in gs]}")

    # ---- 9) 错误路径：NAT 必须搭配 natural_missing=True ----
    try:
        run_cell("adult", "NAT", "A1", "B1", "C1", 0, "LR", cfg,
                 natural_missing=False)
        raise AssertionError("NAT 未配 natural_missing=True 应报错")
    except ValueError as e:
        assert "NAT" in str(e)
    print("[错误路径] NAT + natural_missing=False 已显式报错")
    print("run_cell 自检全部通过")


if __name__ == "__main__":
    _self_test()
