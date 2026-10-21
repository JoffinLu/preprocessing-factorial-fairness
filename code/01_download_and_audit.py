"""阶段 1：数据集下载与字段审计（T3 项目）

用途
  1. 从公开源下载 5 个数据集到 data/raw/
  2. 逐字段审计：真实缺失率与缺失模式、结构式缺失标记、类别分布、
     敏感属性分布与子群样本量
  3. 输出 results/data_audit.csv（字段级）与 results/data_audit_summary.csv（数据集级）

运行：python code/01_download_and_audit.py
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
RESULTS = BASE / "results"
RAW.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (research data audit)"}

# 每个数据集给出候选源列表，按顺序尝试，第一个成功的即采用
SOURCES: dict[str, list[str]] = {
    "adult": [
        "https://archive.ics.uci.edu/static/public/2/adult.zip",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult.adult.zip",
    ],
    "german": [
        "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip",
    ],
    "bank": [
        "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
    ],
    "diabetes": [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip",
    ],
    "compas": [
        "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
    ],
}


def fetch(name: str, urls: list[str]) -> Path:
    """下载并解压到 data/raw/<name>/，返回该目录"""
    dest = RAW / name
    if dest.exists() and any(dest.iterdir()):
        print(f"[skip] {name} 已存在")
        return dest
    dest.mkdir(parents=True, exist_ok=True)

    payload = None
    for u in urls:
        try:
            r = requests.get(u, headers=UA, timeout=120)
            r.raise_for_status()
            payload = r.content
            print(f"[get ] {name} <- {u} ({len(payload) / 1e6:.2f} MB)")
            break
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {name} <- {u} :: {type(e).__name__}")
    if payload is None:
        raise RuntimeError(f"{name}: 全部候选源均下载失败")

    head = payload[:2]
    if head == b"PK":  # zip
        extract_zip(payload, dest)
    else:
        (dest / urls[0].rsplit("/", 1)[-1]).write_bytes(payload)
    return dest


def extract_zip(payload: bytes, dest: Path, depth: int = 0) -> None:
    """解压 zip 到 dest；若成员本身是 zip（如 UCI bank+marketing.zip 的嵌套结构）则递归解压"""
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        for m in z.namelist():
            if m.endswith("/"):
                continue
            data = z.read(m)
            name = Path(m).name
            if name.lower().endswith(".zip"):
                (dest / name).write_bytes(data)
                print(f"{'   ' * (depth + 1)}extract {dest.name}/{name} (嵌套)")
                extract_zip(data, dest, depth + 1)
                continue
            (dest / name).write_bytes(data)
            print(f"{'   ' * (depth + 1)}extract {dest.name}/{name}")


def sentinel_scan(df: pd.DataFrame) -> dict[str, int]:
    """统计常见结构式缺失/哨兵值标记"""
    obj_markers = {"?", "unknown", "Unknown", "UNKNOWN", "none", "None", "N/A", "NA", "?"}
    num_markers = {-1, -9, -8, -7}
    out: dict[str, int] = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            n = int(s.isin(list(num_markers)).sum())
        else:
            n = int(s.astype(str).str.strip().isin(obj_markers).sum())
        if n:
            out[str(c)] = n
    return out


def audit(df: pd.DataFrame, dataset: str, target: str | None,
          sensitive: dict[str, list[str]] | None = None) -> tuple[list[dict], dict]:
    """字段级审计 + 数据集级汇总"""
    n = len(df)
    rows = []
    for c in df.columns:
        s = df[c]
        nmiss = int(s.isna().sum())
        rows.append({
            "dataset": dataset,
            "column": str(c),
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=True)),
            "n_missing": nmiss,
            "missing_pct": round(100 * nmiss / n, 3),
            "is_sentinel_col": "",
        })

    sents = sentinel_scan(df)
    sent_df = pd.DataFrame({"column": list(sents), "n_sentinel": list(sents.values())})
    if not sent_df.empty:
        rows = pd.DataFrame(rows).merge(
            sent_df, on="column", how="left"
        ).fillna({"n_sentinel": 0}).to_dict("records")
    for r in rows:
        r["is_sentinel_col"] = "YES" if r.get("n_sentinel", 0) else ""

    summary = {
        "dataset": dataset,
        "n_rows": n,
        "n_cols": df.shape[1],
        "cols_with_nan": int((df.isna().sum() > 0).sum()),
        "rows_with_any_nan": int(df.isna().any(axis=1).sum()),
        "pct_rows_with_any_nan": round(100 * df.isna().any(axis=1).mean(), 2),
        "total_nan": int(df.isna().sum().sum()),
        "cols_with_sentinel": len(sents),
        "total_sentinel": sum(sents.values()),
        "sentinel_cols": ";".join(f"{k}={v}" for k, v in sents.items()),
        "target": target or "",
    }

    if target and target in df.columns:
        vc = df[target].value_counts(dropna=False)
        summary["target_dist"] = ";".join(f"{k}={v}" for k, v in vc.items())
        if len(vc) == 2:
            lo, hi = vc.min(), vc.max()
            summary["imbalance_ratio"] = round(hi / lo, 2)
            summary["minority_pct"] = round(100 * lo / n, 2)

    if sensitive:
        for attr, groups in sensitive.items():
            if attr not in df.columns:
                summary[f"sens_{attr}"] = "列不存在"
                continue
            sub = df[attr].value_counts(dropna=False)
            summary[f"sens_{attr}"] = ";".join(f"{k}={v}" for k, v in sub.items())
    return rows, summary


# ---------------- 各数据集的读取与配置 ----------------

ADULT_COLS = ["age", "workclass", "fnlwgt", "education", "education_num",
              "marital_status", "occupation", "relationship", "race", "sex",
              "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]

GERMAN_COLS = ["checking_status", "duration", "credit_history", "purpose", "credit_amount",
               "savings_status", "employment", "installment_commitment", "personal_status",
               "other_parties", "residence_since", "property_magnitude", "age",
               "other_payment_plans", "housing", "existing_credits", "job",
               "num_dependents", "own_telephone", "foreign_worker", "class"]

BANK_COLS = ["age", "job", "marital", "education", "default", "housing", "loan", "contact",
             "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome",
             "emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed", "y"]


def load_all() -> list[tuple[str, pd.DataFrame, str | None, dict]]:
    out = []

    # Adult：训练集 adult.data，缺失以 " ?" 编码
    f = RAW / "adult" / "adult.data"
    df = pd.read_csv(f, header=None, names=ADULT_COLS, sep=r"\s*,\s*",
                     engine="python", skipinitialspace=True, na_values="?")
    df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")
    out.append(("adult", df, "income",
                {"sex": [], "race": []}))

    # German Credit：空格分隔，官方标注无缺失
    f = RAW / "german" / "german.data"
    dfg = pd.read_csv(f, header=None, names=GERMAN_COLS, sep=r"\s+", engine="python",
                      na_values="?")
    dfg["class"] = dfg["class"].map({1: "good", 2: "bad"})
    out.append(("german", dfg, "class", {"personal_status": [], "age": []}))

    # Bank Marketing：分号分隔；unknown 为结构式缺失
    f = RAW / "bank" / "bank-additional-full.csv"
    if not f.exists():
        f = RAW / "bank" / "bank-full.csv"
    dfb = pd.read_csv(f, sep=";")
    out.append(("bank", dfb, "y", {"age": [], "marital": []}))

    # Diabetes 130-US hospitals：'?' 为真实缺失
    f = RAW / "diabetes" / "diabetic_data.csv"
    dfd = pd.read_csv(f, na_values="?")
    out.append(("diabetes", dfd, "readmitted", {"race": [], "gender": []}))

    # COMPAS
    f = RAW / "compas" / "compas-scores-two-years.csv"
    dfc = pd.read_csv(f)
    out.append(("compas", dfc, "two_year_recid", {"race": [], "sex": []}))

    return out


def decision_checks(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """面向实验设计的核查：目标二值化口径、结构式缺失规模、标准预处理惯例的影响"""
    chk: list[dict] = []

    def add(dataset, item, value, note=""):
        chk.append({"dataset": dataset, "item": item, "value": value, "note": note})

    # --- bank：pdays 哨兵值 / default 的 unknown 规模 ---
    b = frames["bank"]
    add("bank", "pdays 取值数", b["pdays"].nunique())
    add("bank", "pdays==999 行数", int((b["pdays"] == 999).sum()), "表「未联系过」；-1 不存在")
    add("bank", "pdays==-1 行数", int((b["pdays"] == -1).sum()), "该文件中不存在")
    add("bank", "default=='unknown'", int((b["default"] == "unknown").sum()),
        f"{100 * (b['default'] == 'unknown').mean():.1f}%")
    add("bank", "6 个哨兵列合计占比", round(
        100 * sum((b[c].astype(str) == "unknown").mean() for c in
                  ["job", "marital", "education", "default", "housing", "loan"]), 2), "行占比之和")

    # --- adult：缺失列与缺失行 ---
    a = frames["adult"]
    miss_cols = [c for c in a.columns if a[c].isna().any()]
    add("adult", "含真实缺失的列", ";".join(miss_cols))
    add("adult", "任一缺失行数", int(a.isna().any(axis=1).sum()),
        f"{100 * a.isna().any(axis=1).mean():.2f}%")

    # --- german：敏感属性的可构造性 ---
    g = frames["german"]
    add("german", "personal_status 码表",
        "A91=男离异/分居; A92=女离异/分居/已婚; A93=男单身; A94=男已婚/丧偶")
    add("german", "由 A92 判定的 female 数", int((g["personal_status"] == "A92").sum()),
        f"{100 * (g['personal_status'] == 'A92').mean():.1f}%")
    add("german", "age 二分化(<=25)数", int((g["age"] <= 25).sum()),
        f"{100 * (g['age'] <= 25).mean():.1f}%")

    # --- diabetes：目标二值化 + race 缺失 ---
    d = frames["diabetes"]
    pos = (d["readmitted"] == "<30")
    add("diabetes", "目标二值化口径", "<30 记为正类，其余（NO/>30）记为负类")
    add("diabetes", "二值化后正类数", int(pos.sum()), f"{100 * pos.mean():.2f}%")
    add("diabetes", "二值化不平衡比", round((~pos).sum() / pos.sum(), 2))
    add("diabetes", "race 缺失(nan)数", int(d["race"].isna().sum()), "重要：敏感属性本身缺失，需决定是否剔除")
    add("diabetes", "weight 缺失率", f"{100 * d['weight'].isna().mean():.2f}%", "建议直接丢列")
    add("diabetes", "medical_specialty 缺失率", f"{100 * d['medical_specialty'].isna().mean():.2f}%", "建议直接丢列")
    add("diabetes", "payer_code 缺失率", f"{100 * d['payer_code'].isna().mean():.2f}%", "建议直接丢列")

    # --- compas：ProPublica 标准过滤的影响 ---
    c = frames["compas"]
    add("compas", "原始行数", len(c))
    f = c[(c["days_b_screening_arrest"].between(-30, 30)) &
          (c["is_recid"] != -1) &
          (c["c_charge_degree"] != "O") &
          (c["score_text"] != "N/A")]
    add("compas", "标准过滤后行数", len(f),
        "过滤条件：days_b_screening_arrest∈[-30,30] & is_recid≠-1 & c_charge_degree≠O & score_text≠N/A")
    add("compas", "过滤后不平衡比", round(f["two_year_recid"].mean() /
                                          (1 - f["two_year_recid"].mean()), 2))
    add("compas", "过滤后 race 分布",
        ";".join(f"{k}={v}" for k, v in f["race"].value_counts().items()))
    add("compas", "violent_recid 非空数", int(c["violent_recid"].notna().sum()),
        "本文件该列全空，需另取 compas-scores-two-years-violent.csv")

    return pd.DataFrame(chk)


def main() -> None:
    for name, urls in SOURCES.items():
        try:
            fetch(name, urls)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {name}: {e}")

    loaded = load_all()
    frames = {n: df for n, df, _, _ in loaded}

    all_rows: list[dict] = []
    summaries: list[dict] = []
    for name, df, target, sens in loaded:
        rows, summ = audit(df, name, target, sens)
        all_rows.extend(rows)
        summaries.append(summ)
        print(f"\n===== {name} =====")
        print(f"  shape={df.shape}  含 nan 列={summ['cols_with_nan']}  含 nan 行={summ['rows_with_any_nan']}"
              f" ({summ['pct_rows_with_any_nan']}%)  哨兵列={summ['cols_with_sentinel']}")
        if summ.get("target_dist"):
            print(f"  target[{target}]: {summ['target_dist']}")
            if "imbalance_ratio" in summ:
                print(f"  不平衡比={summ['imbalance_ratio']}  少数类占比={summ['minority_pct']}%")
        for k, v in summ.items():
            if k.startswith("sens_"):
                print(f"  {k}: {v}")
        if summ["sentinel_cols"]:
            print(f"  哨兵明细: {summ['sentinel_cols']}")

    pd.DataFrame(all_rows).to_csv(RESULTS / "data_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(RESULTS / "data_audit_summary.csv", index=False, encoding="utf-8-sig")

    print("\n\n########## 决策核查 ##########")
    chk = decision_checks(frames)
    chk.to_csv(RESULTS / "decision_checks.csv", index=False, encoding="utf-8-sig")
    for _, r in chk.iterrows():
        print(f"  [{r['dataset']:8s}] {r['item']:24s} = {r['value']}   {r['note']}")

    print(f"\n已写出 {RESULTS / 'data_audit.csv'}")
    print(f"已写出 {RESULTS / 'data_audit_summary.csv'}")
    print(f"已写出 {RESULTS / 'decision_checks.csv'}")


if __name__ == "__main__":
    main()
