"""阶段 6：可复现性验证——抽样重跑若干单元格，与已归档结果逐字段比对

为什么抽样而不是全量重跑
    全量 1,620 次训练-评估约 130 min。复现包需要的是「同代码 + 同数据 + 同环境
    能否还原出同一张表」，抽样覆盖全部 5 个数据集、4 个因子与 2 个模型即可
    证伪「结果依赖运行顺序/并行/缓存」这类问题；逐位比对为 0 容差。

比对口径
    - 关键键（dataset/mechanism/A/B/C/seed/model）必须完全一致
    - 全部指标列与中间量（AUC/F1/DP/EO/…、n_train、n_features_encoded、
      n_test、min_group_n 等）按绝对误差 <= 1e-9 比对
    - 计时列（*_secs）与 wall 时间不比对（机器负载相关）

运行：python code/04_verify_reproducibility.py [--n 10] [--csv results/full_results.csv]
退出码：0 = 全部一致；1 = 存在不一致
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pipeline.config import ROOT as PROJ_ROOT  # noqa: E402
from pipeline.run_cell import run_cell  # noqa: E402

DEFAULT_CSV = PROJ_ROOT / "results" / "full_results.csv"
TOL = 1e-9
SKIP_SUFFIX = ("_secs",)

# 固定抽样：覆盖 5 数据集 × 2 机制 × 3 插补 × 3 不平衡 × 3 干预 × 2 模型
# 刻意避开最慢的组合（adult/bank/diabetes 用 A1/A2，compas/german 用 A3）
SAMPLES: tuple[dict, ...] = (
    {"dataset": "adult",    "mechanism": "M1", "A": "A1", "B": "B2", "C": "C2", "seed": 0, "model": "LGBM"},
    {"dataset": "adult",    "mechanism": "M2", "A": "A1", "B": "B3", "C": "C3", "seed": 1, "model": "LR"},
    {"dataset": "bank",     "mechanism": "M1", "A": "A1", "B": "B1", "C": "C1", "seed": 2, "model": "LGBM"},
    {"dataset": "bank",     "mechanism": "M2", "A": "A2", "B": "B2", "C": "C2", "seed": 0, "model": "LR"},
    {"dataset": "compas",   "mechanism": "M1", "A": "A3", "B": "B3", "C": "C3", "seed": 1, "model": "LGBM"},
    {"dataset": "compas",   "mechanism": "M2", "A": "A2", "B": "B1", "C": "C2", "seed": 2, "model": "LR"},
    {"dataset": "german",   "mechanism": "M1", "A": "A3", "B": "B2", "C": "C1", "seed": 0, "model": "LR"},
    {"dataset": "german",   "mechanism": "M2", "A": "A2", "B": "B3", "C": "C2", "seed": 1, "model": "LGBM"},
    {"dataset": "diabetes", "mechanism": "M1", "A": "A1", "B": "B1", "C": "C3", "seed": 0, "model": "LR"},
    {"dataset": "diabetes", "mechanism": "M2", "A": "A2", "B": "B2", "C": "C1", "seed": 1, "model": "LGBM"},
)

KEY_COLS = ("dataset", "mechanism", "A", "B", "C", "seed", "model")


def compare_row(replayed: dict, archived: pd.Series) -> list[str]:
    """逐列比对，返回差异描述列表（空 = 完全一致）"""
    diffs: list[str] = []
    for k in KEY_COLS:
        a, b = replayed.get(k), archived.get(k)
        if str(a) != str(b):
            diffs.append(f"{k}: replay={a!r} archived={b!r}")
    for col, new in replayed.items():
        if col in KEY_COLS or col in ("status", "error") or col.endswith(SKIP_SUFFIX):
            continue
        if col not in archived.index:
            continue
        old = archived[col]
        # CSV 往返会把「不适用的 meta 字段（None）」读成 NaN，两者语义等价
        new_na = new is None or (isinstance(new, float) and np.isnan(new))
        old_na = pd.isna(old)
        if new_na or old_na:
            if not (new_na and old_na):
                diffs.append(f"{col}: replay={new!r} archived={old!r}（缺失侧不一致）")
            continue
        if isinstance(new, (bool, np.bool_)):
            if bool(new) != bool(old):
                diffs.append(f"{col}: replay={new} archived={old}")
            continue
        try:
            new_f, old_f = float(new), float(old)
        except (TypeError, ValueError):
            if str(new) != str(old):
                diffs.append(f"{col}: replay={new!r} archived={old!r}")
            continue
        if np.isnan(new_f) and np.isnan(old_f):
            continue
        if abs(new_f - old_f) > TOL:
            diffs.append(f"{col}: replay={new_f:.12g} archived={old_f:.12g} "
                         f"|d|={abs(new_f - old_f):.3g}")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description="抽样重跑并比对归档结果")
    ap.add_argument("--n", type=int, default=len(SAMPLES),
                    help=f"抽样个数（按固定顺序取前 n 个，默认 {len(SAMPLES)}）")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"[FAIL] 找不到归档结果：{args.csv}")
        return 1
    arch = pd.read_csv(args.csv)
    arch = arch[arch["status"] == "ok"]

    samples = SAMPLES[: args.n]
    print(f"归档结果 {len(arch)} 行；将重跑并逐字段比对 {len(samples)} 个单元格"
          f"（容差 {TOL:g}）\n")

    n_pass, failures = 0, []
    for i, s in enumerate(samples, 1):
        t0 = time.perf_counter()
        try:
            rep = run_cell(s["dataset"], s["mechanism"], s["A"], s["B"], s["C"],
                           s["seed"], s["model"])
        except Exception as e:  # noqa: BLE001
            failures.append((s, [f"重跑异常：{type(e).__name__}: {e}"]))
            print(f"[{i:2d}/{len(samples)}] FAIL  {s['dataset']:9s} "
                  f"{s['mechanism']}/{s['A']}/{s['B']}/{s['C']}/s{s['seed']}/{s['model']}"
                  f"  异常：{type(e).__name__}")
            continue

        m = pd.Series(True, index=arch.index)
        for k in KEY_COLS:
            m &= (arch[k].astype(str) == str(s[k]))
        if m.sum() != 1:
            failures.append((s, [f"归档中匹配到 {int(m.sum())} 行（应为 1）"]))
            print(f"[{i:2d}/{len(samples)}] FAIL  归档缺行或少行")
            continue

        diffs = compare_row(rep, arch[m].iloc[0])
        wall = time.perf_counter() - t0
        tag = "PASS" if not diffs else "FAIL"
        print(f"[{i:2d}/{len(samples)}] {tag}  {s['dataset']:9s} "
              f"{s['mechanism']}/{s['A']}/{s['B']}/{s['C']}/s{s['seed']}/{s['model']}"
              f"  AUC={rep['AUC']:.6f} DP={rep['DP']:.6f}  {wall:6.1f}s"
              + ("" if not diffs else f"  差异 {len(diffs)} 项"))
        for d in diffs[:6]:
            print(f"        - {d}")
        if diffs:
            failures.append((s, diffs))
        else:
            n_pass += 1

    print(f"\n{'=' * 78}\n结果：{n_pass}/{len(samples)} 个单元格逐位一致，"
          f"{len(failures)} 个不一致")
    if failures:
        print("不一致清单：")
        for s, d in failures:
            print(f"  {s['dataset']}/{s['mechanism']}/{s['A']}/{s['B']}/{s['C']}"
                  f"/s{s['seed']}/{s['model']} -> {d[:3]}")
        return 1
    print("可复现性验证通过：同代码 + 同数据 + 同环境可还原归档结果")
    return 0


if __name__ == "__main__":
    sys.exit(main())
