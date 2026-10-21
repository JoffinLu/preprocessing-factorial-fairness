"""Dataset construction for revision_v2.

Loaders preserve natural missingness. Semi-synthetic complete-data references
are constructed only AFTER the split, using training-derived fill values.
Diabetes patient identifiers are retained as grouping metadata, never features.
Bank call duration is excluded by default because it is unavailable before a call.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from pipeline.config import DATA_RAW  # noqa: E402

DATASETS: tuple[str, ...] = ("adult", "german", "bank", "diabetes", "compas")

# ---------------------------------------------------------------------------
# 列名表（与项目既有审计脚本 code/01_download_and_audit.py 保持一致）
# ---------------------------------------------------------------------------
ADULT_COLS = ["age", "workclass", "fnlwgt", "education", "education_num",
              "marital_status", "occupation", "relationship", "race", "sex",
              "capital_gain", "capital_loss", "hours_per_week", "native_country",
              "income"]

GERMAN_COLS = ["checking_status", "duration", "credit_history", "purpose",
               "credit_amount", "savings_status", "employment",
               "installment_commitment", "personal_status", "other_parties",
               "residence_since", "property_magnitude", "age",
               "other_payment_plans", "housing", "existing_credits", "job",
               "num_dependents", "own_telephone", "foreign_worker", "class"]

# COMPAS 非建模列（ID/日期/姓名/重复列等）：规则 4 要求丢弃
COMPAS_NON_MODELING = (
    "id", "name", "first", "last", "dob",
    "compas_screening_date", "screening_date", "type_of_assessment",
    "v_type_of_assessment", "v_decile_score", "v_score_text", "v_screening_date",
    "decile_score.1", "score_text", "priors_count.1",
    "days_b_screening_arrest", "c_days_from_compas", "r_days_from_arrest",
    "c_jail_in", "c_jail_out", "r_jail_in", "r_jail_out",
    "c_case_number", "r_case_number", "vr_case_number",
    "c_offense_date", "r_offense_date", "vr_offense_date", "c_arrest_date",
    "c_charge_desc", "r_charge_desc", "vr_charge_desc",
    "r_charge_degree", "vr_charge_degree",
    "violent_recid", "is_recid", "is_violent_recid",
    "in_custody", "out_custody", "start", "end", "event",
)
COMPAS_FEATURES = ["sex", "age", "age_cat", "juv_fel_count", "decile_score",
                   "juv_misd_count", "juv_other_count", "priors_count",
                   "c_charge_degree"]

# Gate 2b（D7）：极小敏感子群合并阈值
#   COMPAS 过滤后仅 6,172 行、race 有 6 类，其中 Asian 31 / Native American 11。
#   7:3 划分后这些子群在测试集只剩 3~10 人，DP/EO/DI 完全由它们决定
#   （实测 seed0：DP=0.5 由 Asian(sel=0) 与 Native American(sel=0.5) 两个极小组驱动）。
#   处置与 D5（Adult 的 race 二分化）同源：把占比 < 1% 的 race 取值并入 "Other"。
#   阈值稳健性：合并结果在阈值取 [32, 343]（即 0.52%~5.6%）内完全相同，
#   故结论不依赖阈值的具体取值。
RARE_GROUP_MIN_FRAC: float = 0.01

# Gate 2c（D11）：大样本数据集的行数上限（只用于运行成本控制）
#   KNNImputer 的代价约为 O(n^2 * p)。diabetes 清洗后 n=99,493、p=214，
#   实测单个 A3 单元格 >25 min、进程峰值内存 4.6 GB；54 个 A3 单元格需 >=22.5 CPU 小时。
#   处置：按 (标签 x 敏感属性) 分层抽取固定 30,000 行，抽样与随机种子无关，
#   全部 54 个单元格与 3 个种子共用同一批行，配对比较结构不变。
#   仅当 n > 上限时才抽样；其余数据集不受影响。
SUBSAMPLE_MAX_ROWS: dict[str, int] = {"diabetes": 30_000}
SUBSAMPLE_SEED: int = 2024

DIABETES_DROP_COLS = ["weight", "max_glu_serum", "A1Cresult",
                      "medical_specialty", "payer_code"]


@dataclass
class Dataset:
    """单个数据集的标准容器

    X_raw        : 仅特征（含分类列与缺失），pd.DataFrame
    y            : 目标（0/1），np.ndarray[int]
    sensitive    : 原始敏感属性取值，pd.Series（与 X_raw 行对齐）
    sensitive_name : 敏感属性名
    positive_label : 正类标签值（固定为 1）
    meta         : 记录 n / n_pos / imbalance_ratio 等，以及数据驱动选择
    """
    name: str
    X_raw: pd.DataFrame
    y: np.ndarray
    sensitive: pd.Series
    sensitive_name: str
    positive_label: int = 1
    meta: dict[str, Any] = field(default_factory=dict)
    groups: np.ndarray | None = None


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def _is_categorical(s: pd.Series) -> bool:
    """pandas 3.0 下字符串列 dtype 为 str，故用「非数值且非日期」判定分类列"""
    return (not pd.api.types.is_numeric_dtype(s)
            and not pd.api.types.is_datetime64_any_dtype(s))


def _merge_rare_groups(s: pd.Series, min_frac: float,
                       other_label: str = "Other") -> tuple[pd.Series, dict]:
    """把占比低于 min_frac 的敏感取值并入 other_label（Gate 2b / D7）

    理由：公平性指标（DP/EO/EOdd/TPR-gap/DI）都是「组间极差」，
    子群样本量过小时这些极差由个别样本决定，会把采样噪声误报成总体不公平。
    合并口径只依赖**全样本**分布，与随机种子无关，因此在 54 个单元格之间恒定。
    """
    n = len(s)
    thr = max(2, int(np.ceil(min_frac * n)))
    counts = s.value_counts()
    keep = set(counts[counts >= thr].index)
    merged_vals = [v for v in counts.index if v not in keep]
    out = s.where(s.isin(keep), other_label)
    meta = {"rare_merge_min_frac": min_frac, "rare_merge_threshold_n": thr,
            "rare_merge_kept_groups": sorted(map(str, keep)),
            "rare_merge_merged_groups": sorted(map(str, merged_vals)),
            "rare_merge_other_n": int((~s.isin(keep)).sum())}
    return out, meta


def _stratified_subsample(y: np.ndarray, sensitive: pd.Series, max_rows: int,
                          seed: int) -> tuple[np.ndarray, dict]:
    """按 (标签 × 敏感属性) 分层抽取固定行号（与实验种子无关）

    为什么分层：直接随机抽会让小敏感子群在小样本下进一步缩水，公平指标的
    组间极差会更不稳；按标签×敏感属性分层可保持两类边际分布与联合结构。

    为什么固定种子：抽样结果对全部 A/B/C × M × seed 单元格完全相同，
    因此单元格之间仍是配对比较（差异只来自因子本身）。
    """
    n = len(y)
    if n <= max_rows:
        return np.arange(n), {}
    key = pd.Series([f"{int(a)}|{b}" for a, b in zip(y, np.asarray(sensitive))])
    sizes = key.value_counts().sort_index()
    # 比例分配 + 最大余数法，保证总数恰为 max_rows
    quota = max_rows * sizes / n
    base = np.floor(quota).astype(int)
    rest = max_rows - int(base.sum())
    if rest > 0:
        order = (quota - base).sort_values(ascending=False).index[:rest]
        base.loc[order] += 1
    rng = np.random.default_rng(seed)
    idx_parts: list[np.ndarray] = []
    for stratum, k in base.items():
        pos = np.flatnonzero(key.to_numpy() == stratum)
        idx_parts.append(rng.choice(pos, size=int(k), replace=False))
    idx = np.sort(np.concatenate(idx_parts))
    meta = {"subsample_max_rows": int(max_rows), "subsample_seed": int(seed),
            "n_before_subsample": int(n), "subsample_method": "stratified(label x sensitive)",
            "subsample_strata": {str(s): int(k) for s, k in base.items()}}
    return idx, meta


def complete_from_training(train: pd.DataFrame, test: pd.DataFrame
                           ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build a semi-synthetic complete reference using training statistics only.

    This preliminary completion is part of the data-generating protocol, not
    factor A. Numerical gaps use the training median; categorical gaps use the
    training mode. An entirely unobserved training column uses 0 / __MISSING__.
    Values are fitted for every column, including test-only missing columns.
    """
    if list(train.columns) != list(test.columns):
        raise ValueError("Training and test columns must match")
    tr, te, fills = train.copy(), test.copy(), {}
    for c in train.columns:
        s = train[c]
        if _is_categorical(s):
            modes = s.mode(dropna=True)
            fill = modes.iloc[0] if len(modes) else "__MISSING__"
        else:
            med = s.median(skipna=True)
            fill = float(med) if pd.notna(med) else 0.0
        fills[c] = fill
        tr[c] = tr[c].fillna(fill)
        te[c] = te[c].fillna(fill)
    return tr, te, fills


# ---------------------------------------------------------------------------
# Adult
# ---------------------------------------------------------------------------
def _load_adult() -> Dataset:
    """规则 6：CSV 无表头；na_values="?"；income 去尾点；>50K->1；敏感属性 sex"""
    f = DATA_RAW / "adult" / "adult.data"
    df = pd.read_csv(f, header=None, names=ADULT_COLS, sep=r"\s*,\s*",
                     engine="python", skipinitialspace=True, na_values="?")
    df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")
    y = (df["income"] == ">50K").astype(int).to_numpy()
    sensitive = df["sex"].astype(str).reset_index(drop=True)
    # 敏感属性从特征中剔除（避免把受保护属性直接喂给模型）
    X = df.drop(columns=["income", "sex"]).reset_index(drop=True)
    return Dataset("adult", X, y, sensitive, "sex", 1,
                   {"n": len(df), "n_pos": int(y.sum()),
                    "imbalance_ratio": round((y == 0).sum() / max(int(y.sum()), 1), 4),
                    "target": "income(>50K=1)", "sens_dist":
                        sensitive.value_counts().to_dict()})


# ---------------------------------------------------------------------------
# German
# ---------------------------------------------------------------------------
def _load_german() -> Dataset:
    """Original Statlog table; bad credit is the positive class.

    Protected benchmark groups are age > 25 / age <= 25. The original sex
    variable cannot be reliably recovered from personal_status, so no sex
    fairness claim is made. Both age and personal_status are excluded as inputs.
    """
    f = DATA_RAW / "german" / "german.data"
    df = pd.read_csv(f, header=None, names=GERMAN_COLS, sep=r"\s+",
                     engine="python", na_values="?")
    df["class"] = df["class"].map({1: "good", 2: "bad"})
    # bad 为正类 1
    y = (df["class"] == "bad").astype(int).to_numpy()
    # UCI's South German Credit correction documents unreliable recovery of
    # sex from the original personal-status code. Use an explicit age benchmark
    # definition instead (AIF360 convention: privileged age > 25).
    sensitive = pd.Series((df["age"].to_numpy() > 25).astype(int), name="age_gt25")
    X = df.drop(columns=["class", "personal_status", "age"]).reset_index(drop=True)
    return Dataset("german", X, y, sensitive, "age_gt25", 1,
                   {"n": len(df), "n_pos": int(y.sum()),
                    "imbalance_ratio": round((y == 0).sum() / max(int(y.sum()), 1), 4),
                    "target": "class(bad=1)", "sens_dist":
                        sensitive.value_counts().to_dict(),
                    "note": "age > 25 versus age <= 25; original sex coding not used",
                    "protected_group_threshold": 25,
                    "sex_coding_warning_source": "https://archive.ics.uci.edu/dataset/573/south+german+credit"})


# ---------------------------------------------------------------------------
# Bank
# ---------------------------------------------------------------------------
def _load_bank() -> Dataset:
    """规则 2 + 规则 8：
      - job/marital/education/default/housing/loan 的 "unknown" 保留为普通类别（不转缺失）
      - pdays 的 999 表示「此前未联系过」，保留为数值不处理
      - 敏感属性 age 二分化：阈值取样本中位数（**数据驱动选择**，写入 meta）
    """
    f = DATA_RAW / "bank" / "bank-additional-full.csv"
    if not f.exists():
        f = DATA_RAW / "bank" / "bank-full.csv"
    df = pd.read_csv(f, sep=";")
    y = (df["y"].astype(str).str.strip() == "yes").astype(int).to_numpy()
    # 数据驱动阈值：全样本 age 中位数（确定性，与种子无关）
    age_median = float(df["age"].median())
    sensitive = pd.Series((df["age"].to_numpy() >= age_median).astype(int),
                          name="age_bin")
    # Gate 2b：`age` 必须从特征中剔除——它正是敏感属性 age_bin 的定义变量，
    # 留在特征里等于把受保护属性直接喂给模型，会使 C 因子（公平性干预）无意义。
    # 与 adult(sex)、german(personal_status)、diabetes(race)、compas(race) 的处理一致。
    X = df.drop(columns=["y", "age"]).reset_index(drop=True)
    return Dataset("bank", X, y, sensitive, "age_bin", 1,
                   {"n": len(df), "n_pos": int(y.sum()),
                    "imbalance_ratio": round((y == 0).sum() / max(int(y.sum()), 1), 4),
                    "target": "y(yes=1)",
                    "age_binarize_threshold": age_median,
                    "age_binarize_rule": "age >= median -> 1（数据驱动，非外生阈值）",
                    "dropped_protected_feature": "age",
                    "unknown_kept_cols": ["job", "marital", "education",
                                          "default", "housing", "loan"],
                    "pdays_999_note": "999=此前未联系过，保留为数值",
                       "sens_dist": sensitive.value_counts().to_dict()})


# ---------------------------------------------------------------------------
# Diabetes
# ---------------------------------------------------------------------------
def _load_diabetes() -> Dataset:
    """规则 3：丢弃缺失率 > 20% 的 5 列 → 剔除 race 缺失行 → readmitted 二值化
    （<30 -> 1，NO/>30 -> 0）。敏感属性 race。
    encounter_id / patient_nbr 为标识列（非建模列，与规则 4 对 ID 的处理一致），
    在此显式剔除并记入 meta（此为对规则 3 未列举项的工程判断，报告中已标注）。
    """
    f = DATA_RAW / "diabetes" / "diabetic_data.csv"
    df = pd.read_csv(f, na_values="?", low_memory=False)
    before = len(df)
    df = df.drop(columns=DIABETES_DROP_COLS)
    n_race_missing = int(df["race"].isna().sum())
    df = df[df["race"].notna()].reset_index(drop=True)
    y = (df["readmitted"] == "<30").astype(int).to_numpy()
    sensitive = df["race"].astype(str).reset_index(drop=True)
    X = df.drop(columns=["readmitted", "race", "encounter_id",
                         "patient_nbr"]).reset_index(drop=True)
    return Dataset("diabetes", X, y, sensitive, "race", 1,
                   {"n": len(df), "n_pos": int(y.sum()),
                    "imbalance_ratio": round((y == 0).sum() / max(int(y.sum()), 1), 4),
                    "target": "readmitted(<30=1)",
                    "dropped_cols_gt20pct_missing": DIABETES_DROP_COLS,
                    "n_rows_dropped_race_missing": n_race_missing,
                    "n_rows_before": before,
                    "dropped_id_cols": ["encounter_id", "patient_nbr"],
                    "sens_dist": sensitive.value_counts().to_dict()},
                   groups=df["patient_nbr"].to_numpy(copy=True))


# ---------------------------------------------------------------------------
# COMPAS
# ---------------------------------------------------------------------------
def _load_compas() -> Dataset:
    """规则 4：ProPublica 通行口径过滤 →
      days_b_screening_arrest ∈ [-30,30] & is_recid != -1 &
      c_charge_degree != "O" & score_text != "N/A"；
    丢弃全空 violent_recid 列与 ID/日期类非建模列；目标 two_year_recid；敏感属性 race。
    """
    f = DATA_RAW / "compas" / "compas-scores-two-years.csv"
    df = pd.read_csv(f)
    mask = (df["days_b_screening_arrest"].between(-30, 30)
            & (df["is_recid"] != -1)
            & (df["c_charge_degree"] != "O")
            & (df["score_text"] != "N/A"))
    raw_n = len(df)
    df = df[mask].reset_index(drop=True)
    # 全空的 violent_recid 列（本文件该列无值）
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=empty_cols)
    y = df["two_year_recid"].astype(int).to_numpy()
    sensitive_raw = df["race"].astype(str).reset_index(drop=True)
    # Gate 2b（D7）：合并极小 race 子群（Asian 31 / Native American 11）
    sensitive, merge_meta = _merge_rare_groups(sensitive_raw, RARE_GROUP_MIN_FRAC)
    sensitive.name = "race"
    X = df[COMPAS_FEATURES].reset_index(drop=True)
    leaked = [c for c in X.columns if c in ("is_recid", "is_violent_recid")]
    assert not leaked, f"目标泄漏列误入特征：{leaked}"
    return Dataset("compas", X, y, sensitive, "race", 1,
                   {"n": len(df), "n_pos": int(y.sum()),
                    "imbalance_ratio": round((y == 0).sum() / max(int(y.sum()), 1), 4),
                    "n_rows_before_filter": raw_n,
                    "target": "two_year_recid",
                    "dropped_empty_cols": empty_cols,
                    "kept_features": COMPAS_FEATURES,
                    "sens_dist_raw": sensitive_raw.value_counts().to_dict(),
                    "sens_dist": sensitive.value_counts().to_dict(),
                    **merge_meta})


_LOADERS: dict[str, Callable[[], Dataset]] = {
    "adult": _load_adult, "german": _load_german, "bank": _load_bank,
    "diabetes": _load_diabetes, "compas": _load_compas,
}


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------
def load_dataset(name: str, natural_missing: bool = False,
                 drop_bank_duration: bool = True) -> Dataset:
    """加载并清洗一个数据集（确定性）

    参数
        name            : 数据集名，取值见 DATASETS
        natural_missing : retained for API compatibility and protocol metadata;
                          both paths preserve gaps until after splitting.

    返回
        Dataset 数据类。meta 中记录 n / n_pos / imbalance_ratio 及数据驱动选择。
    """
    if name not in _LOADERS:
        raise ValueError(f"未知数据集：{name}，可选 {DATASETS}")
    ds = _LOADERS[name]()
    # Gate 2c（D11）：仅对运行成本不可接受的大样本数据集做分层行抽样
    cap = SUBSAMPLE_MAX_ROWS.get(name)
    if cap and len(ds.y) > cap:
        idx, sub_meta = _stratified_subsample(ds.y, ds.sensitive, cap, SUBSAMPLE_SEED)
        sens_sub = ds.sensitive.iloc[idx].reset_index(drop=True)
        # 抽样后重新按「分析样本」判定极小敏感子群（与 D7 同一规则）：
        #   diabetes 的 Asian 在全样本占 1.8% 得以保留，抽样到 3 万行后仅占 0.65%，
        #   测试集只剩约 58 人、正类约 6 人，EO/TPR-gap 会重新被这个小组主导。
        sens_sub, merge_meta = _merge_rare_groups(sens_sub, RARE_GROUP_MIN_FRAC)
        sens_sub.name = ds.sensitive_name
        ds = Dataset(name, ds.X_raw.iloc[idx].reset_index(drop=True),
                     ds.y[idx], sens_sub,
                     ds.sensitive_name, ds.positive_label,
                     {**ds.meta, **sub_meta, **merge_meta, "n": int(len(idx)),
                      "n_pos": int(ds.y[idx].sum()),
                      "imbalance_ratio": round(float((ds.y[idx] == 0).sum())
                                               / max(int(ds.y[idx].sum()), 1), 4),
                      "sens_dist_before_subsample": ds.meta.get("sens_dist"),
                       "sens_dist": sens_sub.value_counts().to_dict()},
                     groups=(None if ds.groups is None else ds.groups[idx].copy()))
    ds.meta["natural_missing_kept"] = bool(natural_missing)
    ds.meta["natural_missing_cells_zeroed"] = 0
    ds.meta["natural_missing_preserved_until_split"] = True
    if name == "bank" and drop_bank_duration:
        ds.X_raw = ds.X_raw.drop(columns=["duration"], errors="ignore")
        ds.meta["dropped_unavailable_at_prediction"] = ["duration"]
    ds.meta["n_features_raw"] = int(ds.X_raw.shape[1])
    # 两种不平衡比口径（论文 Table 1 必须写清用哪个）：
    #   imbalance_ratio         = n_neg / n_pos（>1 表示负类占多数；COMPAS 为 1.20）
    #   minority_majority_ratio = min / max（恒 <=1；COMPAS 为 0.84，与范围说明书一致）
    n_pos = int(ds.y.sum())
    n_neg = int(len(ds.y) - n_pos)
    ds.meta["imbalance_ratio"] = round(n_neg / max(n_pos, 1), 4)
    ds.meta["minority_majority_ratio"] = round(min(n_pos, n_neg) / max(n_pos, n_neg, 1), 4)
    return ds


# ---------------------------------------------------------------------------
# 自检：python code/pipeline/data.py
# ---------------------------------------------------------------------------
def _self_test() -> None:
    for name in DATASETS:
        ds = load_dataset(name)
        assert len(ds.X_raw) == len(ds.y) == len(ds.sensitive)
        assert ds.sensitive.notna().all()
        assert ds.meta["natural_missing_cells_zeroed"] == 0
        if name == "diabetes":
            assert ds.groups is not None and len(ds.groups) == len(ds.y)
            assert "patient_nbr" not in ds.X_raw.columns
        if name == "bank":
            assert "duration" not in ds.X_raw.columns
        tr, te, _ = complete_from_training(ds.X_raw.iloc[:100], ds.X_raw.iloc[100:120])
        assert not tr.isna().any().any() and not te.isna().any().any()
        print(f"{name}: rows={len(ds.y)}, raw_missing={int(ds.X_raw.isna().sum().sum())}")
    print("data protocol checks passed")


if __name__ == "__main__":
    _self_test()
