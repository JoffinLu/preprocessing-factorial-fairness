"""Comparison arm runner

用法：
    # 跑单个方法 / 数据集 / 种子
    python run_comparison.py --method APW --dataset adult --mechanism M1 --seed 0

    # dry-run：跑 Adult seed0 MCAR 上 APW 和 TabFairGDT
    python run_comparison.py --method APW TabFairGDT --dataset adult --mechanism M1 --seed 0 \
        --out results/revised/comparison/comparison_dryrun.csv

    # 全量 90 次（3 方法 × 3 数据集 × 2 机制 × 5 区组）
    python run_comparison.py --out results/revised/comparison/comparison.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

# 引入 adapter
from common import build_result_row, evaluate_method, prepare_comparison_data, ORDER_COLS_COMPARISON
# 延迟 import，避免所有 adapter 的依赖都加载
_APW_AVAILABLE = _TABFAIRGDT_AVAILABLE = _AIM_AVAILABLE = None


def _import_adapter(method: str):
    """按需导入 adapter，失败时返回 None + 错误信息"""
    try:
        if method == "APW":
            from apw_adapter import run_apw  # noqa: F401
            return run_apw, None
        elif method == "TabFairGDT":
            from tabfairgdt_adapter import run_tabfairgdt  # noqa: F401
            return run_tabfairgdt, None
        elif method == "AIM":
            from aim_adapter import run_aim  # noqa: F401
            return run_aim, None
        else:
            return None, f"Unknown method: {method}"
    except Exception as e:  # noqa: BLE001
        return None, f"Import failed: {type(e).__name__}: {e}"


def _run_one(method: str, dataset: str, mechanism: str, seed: int) -> dict:
    """运行单个单元，成功返回结果行，失败返回 error 行"""
    runner, err = _import_adapter(method)
    if err is not None:
        # import 失败，直接记录
        t0 = time.perf_counter()
        return build_result_row(
            method=method, dataset=dataset, mechanism=mechanism, seed=seed,
            prep_secs=time.perf_counter() - t0, train_secs=0.0,
            n_train=0, n_test=0, sensitive_name="", n_groups=0,
            eval_dict={"AUC": None, "F1": None, "accuracy": None,
                       "balanced_accuracy": None, "precision": None, "recall": None,
                       "DP": None, "EO": None, "EOdd": None, "TPR_gap": None, "DI": None},
            status="import_error", error=err, traceback=None,
        )

    try:
        return runner(dataset, mechanism, seed)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc(limit=10)
        t0 = time.perf_counter()
        return build_result_row(
            method=method, dataset=dataset, mechanism=mechanism, seed=seed,
            prep_secs=0.0, train_secs=0.0,
            n_train=0, n_test=0, sensitive_name="", n_groups=0,
            eval_dict={"AUC": None, "F1": None, "accuracy": None,
                       "balanced_accuracy": None, "precision": None, "recall": None,
                       "DP": None, "EO": None, "EOdd": None, "TPR_gap": None, "DI": None},
            status="error",
            error=f"{type(e).__name__}: {e}",
            traceback=tb,
        )


def _append(path: Path, rows: list[dict]) -> None:
    """增量追加（与主管线 02_run_experiment.py 风格一致）"""
    if not rows:
        return
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        old_cols = list(pd.read_csv(path, nrows=0).columns)
        all_cols = old_cols + [c for c in df.columns if c not in old_cols]
        df = df.reindex(columns=all_cols)
        df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        ordered = [c for c in ORDER_COLS_COMPARISON if c in df.columns]
        ordered += [c for c in df.columns if c not in ordered]
        df.reindex(columns=ordered).to_csv(path, index=False, encoding="utf-8-sig")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Comparison arm runner")
    p.add_argument("--method", nargs="+", default=["APW", "TabFairGDT", "AIM"],
                   help="对比臂方法列表（空格分隔）")
    p.add_argument("--dataset", nargs="+", default=["adult", "compas", "diabetes"],
                   help="数据集列表")
    p.add_argument("--mechanism", nargs="+", default=["M1", "M2"],
                   help="缺失机制：M1(MCAR) / M2(MAR)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                   help="区组种子列表")
    p.add_argument("--out", type=str, default="results/revised/comparison/comparison.csv",
                   help="输出 CSV 路径")
    p.add_argument("--limit", type=int, default=0,
                   help="只跑前 N 个单元（0=全部）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # 枚举所有单元（确定性顺序）
    units: list[dict] = []
    for method in args.method:
        for dataset in args.dataset:
            for mechanism in args.mechanism:
                for seed in args.seeds:
                    units.append({
                        "method": method, "dataset": dataset,
                        "mechanism": mechanism, "seed": seed,
                    })
    if args.limit:
        units = units[:args.limit]

    out_path = Path(args.out)
    print("=" * 78)
    print(f"方法    : {args.method}")
    print(f"数据集  : {args.dataset}")
    print(f"机制    : {args.mechanism}")
    print(f"种子    : {args.seeds}")
    print(f"待执行  : {len(units)} 次运行")
    print(f"输出    : {out_path}")
    print("=" * 78)

    t_all = time.perf_counter()
    rows: list[dict] = []
    n_ok = n_err = 0

    for i, u in enumerate(units, 1):
        t0 = time.perf_counter()
        row = _run_one(u["method"], u["dataset"], u["mechanism"], u["seed"])
        rows.append(row)
        elapsed = time.perf_counter() - t0

        ok = row["status"] == "ok"
        n_ok += ok
        n_err += not ok

        # 进度
        eta = (len(units) - i) * (time.perf_counter() - t_all) / i if i > 0 else 0
        status_str = "OK" if ok else f"[{row['status']}]"
        util_str = f"AUC={row.get('AUC', '?'):.3f}" if row.get("AUC") is not None else "—"
        print(f"  [{i:2d}/{len(units)}] {u['method']:12s} {u['dataset']:8s} "
              f"{u['mechanism']:3s} seed{u['seed']} {elapsed:5.1f}s → "
              f"{status_str:20s} {util_str}")

        # 定期追加
        if len(rows) >= 50 or i == len(units):
            _append(out_path, rows)
            rows = []

    el = time.perf_counter() - t_all
    print("=" * 78)
    print(f"完成：成功 {n_ok}，失败 {n_err}，总耗时 {el/60:.1f} min")
    print(f"结果：{out_path}")

    if n_err:
        err_df = pd.read_csv(out_path)
        err_path = out_path.with_name(out_path.stem + "_errors.csv")
        err_df[err_df["status"] != "ok"].to_csv(err_path, index=False, encoding="utf-8-sig")
        print(f"失败明细：{err_path}")

    # 写 manifest
    manifest = {
        "protocol": "comparison arm v1.1",
        "methods": args.method,
        "datasets": args.dataset,
        "mechanisms": args.mechanism,
        "seeds": args.seeds,
        "run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_min": round(el / 60.0, 2),
        "total_units": len(units),
        "n_ok": n_ok,
        "n_err": n_err,
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest：{manifest_path}")

    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
