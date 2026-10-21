"""Provenance-checked factorial runner; one task shares A across B/C and models."""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
from pipeline.config import Config, DEFAULT_CONFIG, ROOT

KEY = ("dataset", "mechanism", "A", "B", "C", "seed", "model", "natural_missing")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_block(block):
    import pipeline.run_cell as runner
    rows = []
    try:
        for unit in block["units"]:
            base = {k: unit[k] for k in KEY[:6]}
            base["natural_missing"] = block["natural_missing"]
            try:
                prep = runner.prepare_cell(unit["dataset"], unit["mechanism"], unit["A"],
                    unit["B"], unit["C"], unit["seed"], block["cfg"],
                    natural_missing=block["natural_missing"])
            except Exception as exc:
                rows.extend({**base, "model": m, "status": "error", "error": str(exc),
                             "traceback": traceback.format_exc(limit=7)} for m in unit["models"])
                continue
            for model in unit["models"]:
                try:
                    rows.append(runner.run_cell_from_prep(prep, model))
                except Exception as exc:
                    rows.append({**base, "model": model, "status": "error", "error": str(exc),
                                 "traceback": traceback.format_exc(limit=7)})
    finally:
        if hasattr(runner, "clear_caches"):
            runner.clear_caches()
    return rows


def build_blocks(cfg, natural, done):
    blocks = OrderedDict()
    for combo in cfg.iter_combos():
        if tuple(combo[k] for k in KEY[:-1]) + (natural,) in done:
            continue
        bk = tuple(combo[k] for k in ("dataset", "mechanism", "A", "seed"))
        block = blocks.setdefault(bk, {"units": OrderedDict(), "cfg": cfg, "natural_missing": natural})
        uk = (combo["B"], combo["C"])
        unit = block["units"].setdefault(uk, {**combo, "models": []})
        unit["models"].append(combo["model"])
    for block in blocks.values():
        block["units"] = list(block["units"].values())
    return list(blocks.values())


def append_rows(path, rows):
    frame = pd.DataFrame(rows)
    if path.exists() and path.stat().st_size:
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        added = [c for c in frame if c not in columns]
        if added:
            pd.read_csv(path).reindex(columns=columns + added).to_csv(path, index=False, encoding="utf-8-sig")
        frame.reindex(columns=columns + added).to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        frame.reindex(columns=list(KEY) + [c for c in frame if c not in KEY]).to_csv(path, index=False, encoding="utf-8-sig")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    for flag in ("datasets", "mechanisms", "models", "a", "b", "c"):
        p.add_argument("--" + flag, nargs="+")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--missing-rate", type=float)
    p.add_argument("--resample-variant", choices=["within_group_smote", "random_over"])
    p.add_argument("--repair-variant", choices=["train_quantile", "train_quantile_reference", "transductive_dir"])
    p.add_argument("--out")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--natural-missing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    cfg = Config.from_yaml(args.config) if args.config.exists() else Config()
    changes = {}
    for flag, field in [("datasets", "datasets"), ("mechanisms", "mechanisms"), ("models", "models"),
                        ("a", "a_levels"), ("b", "b_levels"), ("c", "c_levels"), ("seeds", "seeds")]:
        if getattr(args, flag) is not None:
            changes[field] = tuple(getattr(args, flag))
    for flag, field in [("missing_rate", "missing_rate"), ("out", "out_csv"),
                        ("resample_variant", "resample_variant"), ("repair_variant", "repair_variant")]:
        if getattr(args, flag) is not None:
            changes[field] = getattr(args, flag)
    cfg = replace(cfg, **changes)
    if args.natural_missing != (cfg.mechanisms == ("NAT",)):
        raise ValueError("--natural-missing requires the sole mechanism NAT, and vice versa")
    path = cfg.out_path()
    mp = path.with_suffix(".manifest.json")
    config = json.loads(json.dumps(cfg.to_dict()))
    source_paths = [Path(__file__), CODE_DIR / "fairness_metrics.py", *sorted((CODE_DIR / "pipeline").glob("*.py"))]
    sources = {q.relative_to(ROOT).as_posix(): sha256(q) for q in source_paths}
    raw_paths = {"adult": "adult/adult.data", "german": "german/german.data",
                 "bank": "bank/bank-additional-full.csv", "diabetes": "diabetes/diabetic_data.csv",
                 "compas": "compas/compas-scores-two-years.csv"}
    data_sources = {raw_paths[d]: sha256(ROOT / "data/raw" / raw_paths[d]) for d in cfg.datasets}
    environment = {name: importlib.metadata.version(name) for name in
                   ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "aif360", "BlackBoxAuditing")}
    requirements_hash = sha256(ROOT / "requirements.txt")
    done, manifest, pending_cleanup = set(), None, None
    if path.exists() and path.stat().st_size:
        if not args.resume or not mp.exists():
            raise ValueError("Existing output requires --resume and its provenance manifest")
        manifest = json.loads(mp.read_text(encoding="utf-8"))
        if (manifest["config"] != config or manifest["source_sha256"] != sources
                or manifest.get("data_sha256") != data_sources or manifest.get("packages") != environment
                or manifest.get("requirements_sha256") != requirements_hash):
            raise ValueError("Resume protocol/code mismatch; choose a separate output")
        if manifest.get("status") in ("complete", "partial", "errors") and manifest.get("output_sha256") != sha256(path):
            raise ValueError("Archived output checksum changed")
        old = pd.read_csv(path)
        ok = old[old.status.eq("ok")]
        if ok.duplicated(list(KEY)).any():
            raise ValueError("Duplicated successful keys")
        done = set(ok[list(KEY)].itertuples(index=False, name=None))
        allowed = {tuple(c[k] for k in KEY[:-1]) + (args.natural_missing,) for c in cfg.iter_combos()}
        if not done.issubset(allowed):
            raise ValueError("Existing result keys are outside the configured factorial design")
        if len(ok) != len(old):
            pending_cleanup = (old[~old.status.eq("ok")], ok)
    blocks = build_blocks(cfg, args.natural_missing, done)
    if args.limit:
        blocks = blocks[:args.limit]
    print(json.dumps({"output": str(path), "config": config, "expected_rows": cfg.n_combos(),
        "already_ok": len(done), "worker_blocks": len(blocks), "jobs": args.jobs}, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    if pending_cleanup is not None:
        failed, successful = pending_cleanup
        failed.to_csv(path.with_suffix(".previous_errors.csv"), index=False)
        successful.to_csv(path, index=False, encoding="utf-8-sig")
    manifest = manifest or {"config": config, "source_sha256": sources, "data_sha256": data_sources,
        "packages": environment, "requirements_sha256": requirements_hash,
        "natural_missing": args.natural_missing,
        "started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version, "platform": platform.platform(),
        "logical_cpus": os.cpu_count(), "processor": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
        "jobs": args.jobs, "command": sys.argv, "expected_rows": cfg.n_combos(), "wall_time_seconds": 0.0}
    manifest["status"] = "running"
    write_json(mp, manifest)
    start = time.perf_counter()
    n_ok = n_error = 0

    def receive(rows, count):
        nonlocal n_ok, n_error
        append_rows(path, rows)
        errors = [r for r in rows if r["status"] != "ok"]
        n_ok += len(rows) - len(errors)
        n_error += len(errors)
        print(f"blocks={count}/{len(blocks)} new_ok={n_ok} errors={n_error} elapsed_min={(time.perf_counter()-start)/60:.2f}", flush=True)
        if errors:
            print(errors[0].get("traceback", errors[0]["error"]), flush=True)
    if args.jobs == 1:
        for i, block in enumerate(blocks, 1):
            receive(run_block(block), i)
    elif blocks:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_block, block) for block in blocks]
            for i, future in enumerate(as_completed(futures), 1):
                receive(future.result(), i)
    manifest["wall_time_seconds"] += time.perf_counter() - start
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    result = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["status"])
    total_ok = int(result.status.eq("ok").sum())
    manifest.update(rows=len(result), successful_rows=total_ok, failed_rows=int((~result.status.eq("ok")).sum()))
    manifest["status"] = "complete" if total_ok == cfg.n_combos() and not n_error else "errors" if n_error else "partial"
    if {q.relative_to(ROOT).as_posix(): sha256(q) for q in source_paths} != sources:
        manifest["status"] = "source_changed_during_run"
    if {raw_paths[d]: sha256(ROOT / "data/raw" / raw_paths[d]) for d in cfg.datasets} != data_sources:
        manifest["status"] = "data_changed_during_run"
    if path.exists():
        manifest["output_sha256"] = sha256(path)
    write_json(mp, manifest)
    print(f"status={manifest['status']} total_ok={total_ok}/{cfg.n_combos()}", flush=True)
    return int(n_error > 0 or manifest["status"] in ("source_changed_during_run", "data_changed_during_run"))


if __name__ == "__main__":
    raise SystemExit(main())
