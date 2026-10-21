"""全量实验入口（T3 项目）

用法
    # 冒烟测试（前 8 个单元格，单进程）
    python code/02_run_experiment.py --limit 8 --jobs 1 --out results/smoke.csv

    # 指定数据集与机制
    python code/02_run_experiment.py --datasets german compas --mechanisms M1 --jobs 4

    # 主设计（810 次预处理 / 1,620 次训练-评估），断点续跑
    python code/02_run_experiment.py --jobs 6 --resume

    # 附加稳健性（Adult 天然缺失对照，81 / 162）
    python code/02_run_experiment.py --datasets adult --mechanisms NAT \
        --natural-missing --out results/robustness_natural_missing.csv --jobs 4

设计要点
    - **一个预处理 → 多个模型**：`run_cell.py` 的预处理与模型已解耦，本入口以
      `(dataset, mechanism, A, B, C, seed)` 为执行单元，单元内依次训练全部模型。
      于是预处理次数 = 单元数（810），训练-评估次数 = 单元数 × 模型数（1,620），
      与范围说明书 §4.6 完全对应。
    - **断点续跑**：`--resume` 时读取已有 CSV，**只有该单元的全部模型都 `status == "ok"`
      才跳过**；部分完成的单元会补齐缺失的模型。
    - **增量落盘**：每完成 `batch` 个单元就追加写入，进程崩溃不丢已完成结果。
    - **失败可见**：异常写入 `status="error"` 与 `error` 列，不静默跳过。
    - **确定性**：所有随机性由 (dataset, seed) 派生（见 pipeline/run_cell.py 顶部说明），
      与并行度、执行顺序**无关**；串行与并行的结果可逐位比对。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from pipeline.config import (A_LEVELS, B_LEVELS, C_LEVELS, DATASETS,  # noqa: E402
                             DEFAULT_CONFIG, MECHANISMS, MODELS, ROOT, SEEDS,
                             Config)
from pipeline.run_cell import ORDER_COLS  # noqa: E402

KEY_COLS: tuple[str, ...] = ("dataset", "mechanism", "A", "B", "C", "seed", "model",
                             "natural_missing")


def _run_one_unit(unit: dict[str, Any]) -> list[dict]:
    """子进程入口：完成一个预处理单元，并训练该单元的全部模型

    预处理（A → B → C）只执行一次；单个模型失败不影响其余模型，
    失败行带 `status="error"` 与 traceback，保证可追溯。
    """
    from pipeline.run_cell import prepare_cell, run_cell_from_prep

    out: list[dict] = []
    base = {k: unit[k] for k in ("dataset", "mechanism", "A", "B", "C", "seed")}
    base["natural_missing"] = unit["natural_missing"]
    try:
        prep = prepare_cell(unit["dataset"], unit["mechanism"], unit["A"], unit["B"],
                            unit["C"], unit["seed"], unit["cfg"],
                            natural_missing=unit["natural_missing"])
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc(limit=6)
        err = f"{type(e).__name__}: {e}"
        for model in unit["models"]:
            out.append({**base, "model": model, "status": "error",
                        "error": err, "traceback": tb, "total_secs": float("nan")})
        return out

    for model in unit["models"]:
        try:
            out.append(run_cell_from_prep(prep, model))
        except Exception as e:  # noqa: BLE001
            out.append({**base, "model": model, "status": "error",
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(limit=6),
                        "total_secs": float("nan")})
    return out


def _existing_ok_keys(path: Path) -> set[tuple]:
    """读取已有结果，返回已成功完成的 (预处理, 模型) 组合键集合"""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        old = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return set()
    if "status" not in old.columns:
        return set()
    ok = old[old["status"] == "ok"]
    return {tuple(r) for r in ok[list(KEY_COLS)].itertuples(index=False, name=None)}


def _unit_key(unit: dict, model: str) -> tuple:
    return (unit["dataset"], unit["mechanism"], unit["A"], unit["B"], unit["C"],
            unit["seed"], model, unit["natural_missing"])


def _append(path: Path, rows: list[dict]) -> None:
    """增量追加（首写带表头；列集不一致时按并集补齐）"""
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
        ordered = [c for c in ORDER_COLS if c in df.columns]
        ordered += [c for c in df.columns if c not in ordered]
        df.reindex(columns=ordered).to_csv(path, index=False,
                                           encoding="utf-8-sig")


def build_units(cfg: Config, natural_missing: bool) -> list[dict]:
    """按配置枚举预处理单元（固定顺序，确定性）：每个单元挂载全部待跑模型"""
    grouped: dict[tuple, dict] = {}
    for combo in cfg.iter_combos():
        key = (combo["dataset"], combo["mechanism"], combo["A"], combo["B"],
               combo["C"], combo["seed"])
        if key not in grouped:
            grouped[key] = {**combo, "models": [], "cfg": cfg,
                            "natural_missing": natural_missing}
        grouped[key]["models"].append(combo["model"])
    for u in grouped.values():
        u["models"] = tuple(u["models"])
    return list(grouped.values())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T3 四因子全因子实验运行器")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--mechanisms", nargs="+", default=list(MECHANISMS),
                   help="M1(MCAR) / M2(MAR) / NAT(不注入，须配 --natural-missing)")
    p.add_argument("--a", nargs="+", default=list(A_LEVELS))
    p.add_argument("--b", nargs="+", default=list(B_LEVELS))
    p.add_argument("--c", nargs="+", default=list(C_LEVELS))
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 个单元格（0=全部）")
    p.add_argument("--jobs", type=int, default=1, help="并行进程数（1=串行）")
    p.add_argument("--out", type=str, default="results/full_results.csv")
    p.add_argument("--resume", action="store_true", help="跳过已成功的单元格")
    p.add_argument("--natural-missing", action="store_true",
                   help="保留天然缺失、不执行置零（NAT 稳健性对照必须开启）")
    p.add_argument("--max-cardinality", type=int, default=None,
                   help="覆盖配置中的分类列高基数截断上限")
    p.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = Config.from_yaml(args.config) if args.config.exists() else Config()
    overrides = {"datasets": tuple(args.datasets), "mechanisms": tuple(args.mechanisms),
                 "a_levels": tuple(args.a), "b_levels": tuple(args.b),
                 "c_levels": tuple(args.c), "models": tuple(args.models),
                 "seeds": tuple(args.seeds), "out_csv": args.out}
    if args.max_cardinality is not None:
        overrides["max_cardinality"] = args.max_cardinality
    cfg = Config(**{**cfg.to_dict(), **overrides})

    if "NAT" in cfg.mechanisms and not args.natural_missing:
        print("[错误] mechanism=NAT 必须同时指定 --natural-missing", file=sys.stderr)
        return 2

    units = build_units(cfg, args.natural_missing)
    out_path = cfg.out_path()
    skipped = 0
    if args.resume:
        done = _existing_ok_keys(out_path)
        kept: list[dict] = []
        for u in units:
            remain = tuple(m for m in u["models"] if _unit_key(u, m) not in done)
            if len(remain) < len(u["models"]):
                skipped += len(u["models"]) - len(remain)
            if remain:
                kept.append({**u, "models": remain})
        units = kept
    if args.limit:
        units = units[:args.limit]

    n_cells = sum(len(u["models"]) for u in units)
    print("=" * 78)
    print(f"配置      : {args.config if args.config.exists() else '内置默认'}")
    print(f"数据集    : {list(cfg.datasets)}")
    print(f"机制      : {list(cfg.mechanisms)}（natural_missing={args.natural_missing}）")
    print(f"因子水平  : A={list(cfg.a_levels)} B={list(cfg.b_levels)} C={list(cfg.c_levels)}")
    print(f"模型/种子 : {list(cfg.models)} / {list(cfg.seeds)}")
    print(f"缺失率    : {cfg.missing_rate:.0%}   划分 : {1-cfg.test_size:.0%}:{cfg.test_size:.0%}"
          f"   高基数截断 : {cfg.max_cardinality}")
    print(f"待执行    : 预处理 {len(units)} 次 → 训练-评估 {n_cells} 次"
          f"（已跳过 {skipped} 次训练-评估）")
    print(f"输出      : {out_path}")
    print(f"并行      : {args.jobs} 进程")
    print("=" * 78)
    if args.dry_run or not units:
        if not units:
            print("无可执行单元（可能已全部完成）")
        return 0

    t_all = time.perf_counter()
    rows: list[dict] = []
    n_ok = n_err = 0
    batch = max(1, min(20, len(units)))

    def _report(done: int) -> None:
        el = time.perf_counter() - t_all
        rate = done / el if el > 0 else 0.0
        eta = (len(units) - done) / rate if rate > 0 else float("nan")
        print(f"  进度 {done}/{len(units)} 预处理单元  成功 {n_ok} 失败 {n_err}  "
              f"已用 {el/60:.1f} min  预计剩余 {eta/60:.1f} min", flush=True)

    if args.jobs <= 1:
        for i, u in enumerate(units, 1):
            for r in _run_one_unit(u):
                rows.append(r)
                n_ok += r["status"] == "ok"
                n_err += r["status"] != "ok"
            if len(rows) >= batch or i == len(units):
                _append(out_path, rows)
                rows = []
                _report(i)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = {ex.submit(_run_one_unit, u): u for u in units}
            for i, fut in enumerate(as_completed(futures), 1):
                for r in fut.result():
                    rows.append(r)
                    n_ok += r["status"] == "ok"
                    n_err += r["status"] != "ok"
                if len(rows) >= batch or i == len(units):
                    _append(out_path, rows)
                    rows = []
                    _report(i)

    el = time.perf_counter() - t_all
    print("=" * 78)
    print(f"完成：成功 {n_ok}，失败 {n_err}，总耗时 {el/60:.1f} min")
    print(f"结果：{out_path}")
    if n_err:
        err_path = out_path.with_name(out_path.stem + "_errors.csv")
        df = pd.read_csv(out_path)
        df[df["status"] != "ok"].to_csv(err_path, index=False, encoding="utf-8-sig")
        print(f"失败明细：{err_path}（重跑时不要加 --resume 以覆盖重试）")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
