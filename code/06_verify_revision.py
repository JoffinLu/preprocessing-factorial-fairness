"""Verify revision manifests and independently replay representative treatments.

This is a sampled replay, not a claim that every archived model was retrained.
Metrics and metadata use absolute tolerance 1e-12, zero relative tolerance;
timing fields are excluded. CSV floats use pandas' round_trip parser. Exact
numeric equality is reported separately and is never called bitwise identity.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
from numbers import Real
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from pipeline.config import Config
from pipeline.run_cell import clear_caches, run_cell

KEY = ("dataset", "mechanism", "A", "B", "C", "seed", "model", "natural_missing")
TOLERANCE = 1e-12
TUPLE_FIELDS = {"datasets", "mechanisms", "a_levels", "b_levels", "c_levels", "models", "seeds"}
HOUSEKEEPING_FIELDS = {"error", "traceback"}

# First ten span all five datasets, both models, all A/B/C/M levels and seeds
# 0/19. Slow Diabetes KNN is deliberately omitted from this sampled replay.
PREFERRED = (
    ("adult", "M1", "A1", "B2", "C2", 0, "LGBM"),
    ("adult", "M2", "A2", "B3", "C3", 19, "LR"),
    ("bank", "M1", "A1", "B1", "C1", 19, "LR"),
    ("bank", "M2", "A2", "B2", "C2", 0, "LGBM"),
    ("compas", "M1", "A3", "B3", "C3", 19, "LGBM"),
    ("compas", "M2", "A2", "B1", "C2", 0, "LR"),
    ("german", "M1", "A3", "B2", "C1", 0, "LR"),
    ("german", "M2", "A2", "B3", "C2", 19, "LGBM"),
    ("diabetes", "M1", "A1", "B1", "C3", 0, "LR"),
    ("diabetes", "M2", "A2", "B2", "C1", 19, "LGBM"),
    ("german", "M2", "A3", "B1", "C3", 19, "LR"),
    ("bank", "M1", "A1", "B3", "C3", 0, "LGBM"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def config_from_manifest(manifest: dict) -> Config:
    values = dict(manifest["config"])
    allowed = {item.name for item in fields(Config)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown manifest configuration fields: {sorted(unknown)}")
    for name in TUPLE_FIELDS:
        if name in values:
            values[name] = tuple(values[name])
    return Config(**values)


def canonical_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if str(value).lower() in ("true", "1"):
        return True
    if str(value).lower() in ("false", "0"):
        return False
    raise ValueError(f"Not a boolean value: {value!r}")


def key_of(record) -> tuple:
    return tuple(int(record[k]) if k == "seed" else canonical_bool(record[k])
                 if k == "natural_missing" else str(record[k]) for k in KEY)


def select_rows(archive: pd.DataFrame, cfg: Config, n: int) -> pd.DataFrame:
    """Use fixed preferred keys, then deterministic coverage for smaller arms."""
    available = {key_of(row): idx for idx, row in archive.iterrows()}
    natural = bool(canonical_bool(archive.iloc[0]["natural_missing"]))
    chosen, seen = [], set()
    for entry in PREFERRED:
        key = entry + (natural,)
        if key in available and len(chosen) < n:
            chosen.append(available[key])
    coverage_fields = KEY[:-1]
    for idx in chosen:
        row = archive.loc[idx]
        seen.update((name, str(row[name])) for name in coverage_fields)
    remaining = archive.drop(index=chosen).sort_values(list(KEY[:-1]), kind="stable")
    while len(chosen) < min(n, len(archive)):
        best_idx, best_score = None, None
        for idx, row in remaining.iterrows():
            gain = sum((name, str(row[name])) not in seen for name in coverage_fields)
            # After coverage, prefer the cheaper imputers for arbitrary extra rows.
            score = (gain, {"A1": 2, "A2": 1, "A3": 0}.get(row["A"], 0))
            if best_score is None or score > best_score:
                best_idx, best_score = idx, score
        if best_idx is None:
            break
        chosen.append(best_idx)
        row = remaining.loc[best_idx]
        seen.update((name, str(row[name])) for name in coverage_fields)
        remaining = remaining.drop(index=best_idx)
    return archive.loc[chosen]


def is_missing(value) -> bool:
    return value is None or (not isinstance(value, (dict, list, tuple)) and bool(pd.isna(value)))


def compare_record(replayed: dict, archived: pd.Series) -> dict:
    differences = []
    numeric_deltas = []
    compared_fields, skipped_fields = [], []

    def check(new, old, name):
        if is_missing(new) or is_missing(old):
            if not (is_missing(new) and is_missing(old)):
                differences.append({"field": name, "kind": "missing_value_mismatch",
                                    "replayed": repr(new), "archived": repr(old)})
        elif isinstance(new, dict) and isinstance(old, dict):
            if set(new) != set(old):
                differences.append({"field": name, "kind": "JSON_key_mismatch"})
            for key in sorted(set(new) & set(old)):
                check(new[key], old[key], f"{name}.{key}")
        elif isinstance(new, list) and isinstance(old, list):
            if len(new) != len(old):
                differences.append({"field": name, "kind": "JSON_length_mismatch"})
            for i, (a, b) in enumerate(zip(new, old)):
                check(a, b, f"{name}[{i}]")
        elif isinstance(new, (bool, np.bool_)):
            if bool(new) != canonical_bool(old):
                differences.append({"field": name, "kind": "boolean_mismatch"})
        elif isinstance(new, Real) and not isinstance(new, (bool, np.bool_)):
            try:
                a, b = float(new), float(old)
            except (ValueError, TypeError):
                differences.append({"field": name, "kind": "numeric_type_mismatch"})
                return
            if not math.isfinite(a) or not math.isfinite(b):
                if a != b:
                    differences.append({"field": name, "kind": "nonfinite_mismatch"})
                return
            delta = abs(a - b)
            numeric_deltas.append(delta)
            if delta > TOLERANCE:
                differences.append({"field": name, "kind": "numeric_difference",
                                    "replayed": a, "archived": b, "absolute_difference": delta})
        elif str(new) != str(old):
            differences.append({"field": name, "kind": "metadata_mismatch",
                                "replayed": repr(new), "archived": repr(old)})

    for name in sorted(set(replayed) | set(archived.index)):
        if name.endswith("_secs") or name in HOUSEKEEPING_FIELDS:
            skipped_fields.append(name)
            continue
        if name not in replayed or name not in archived.index:
            differences.append({"field": name, "kind": "schema_mismatch"})
            continue
        new, old = replayed[name], archived[name]
        if name.endswith("_json") and not is_missing(new) and not is_missing(old):
            try:
                new, old = json.loads(new), json.loads(old)
            except (TypeError, ValueError) as exc:
                differences.append({"field": name, "kind": "invalid_JSON", "error": str(exc)})
                continue
        compared_fields.append(name)
        check(new, old, name)
    return {"passed": not differences, "differences": differences,
            "compared_fields": compared_fields, "excluded_fields": skipped_fields,
            "numeric_values_compared": len(numeric_deltas),
            "numeric_values_exactly_equal": all(delta == 0 for delta in numeric_deltas),
            "max_absolute_numeric_difference": max(numeric_deltas, default=0.0)}


def verify_hashes(expected: dict, base: Path) -> list[dict]:
    results = []
    for relative, digest in expected.items():
        path = (base / relative).resolve()
        if not path.is_relative_to(base.resolve()):
            raise ValueError(f"Manifest path outside its source root: {relative}")
        actual = sha256(path) if path.is_file() else None
        results.append({"path": relative, "expected": digest, "actual": actual,
                        "passed": actual == digest})
    return results


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                    allow_nan=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "results/revised/main.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "revision/reproduction_check.json")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--list-only", action="store_true", help="Check integrity and show selection without replay")
    args = parser.parse_args(argv)
    archive_path, report_path = project_path(args.csv), project_path(args.report)
    manifest_path = archive_path.with_suffix(".manifest.json")
    report = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "csv": str(archive_path), "manifest": str(manifest_path),
              "verifier_sha256": sha256(Path(__file__)), "python_executable": sys.executable,
              "python": sys.version, "platform": platform.platform(),
              "virtual_environment": sys.prefix != sys.base_prefix,
              "absolute_tolerance": TOLERANCE, "relative_tolerance": 0,
              "csv_float_parser": "round_trip", "timing_fields_excluded": "*_secs",
              "replay_is_bitwise_identity_claim": False, "requested_replays": args.n,
              "allow_partial": args.allow_partial, "cells": []}
    try:
        if args.n < 1:
            raise ValueError("--n must be positive")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cfg = config_from_manifest(manifest)
        complete = manifest.get("status") == "complete"
        report["manifest_status"] = manifest.get("status")
        report["audit_scope"] = ("complete_archive_with_sampled_replay" if complete
                                  else "partial_archive_precheck")
        if not complete and not args.allow_partial:
            raise ValueError("Full archive audit requires manifest status=complete; --allow-partial is only a precheck")
        report["source_hashes"] = verify_hashes(manifest["source_sha256"], ROOT)
        report["raw_data_hashes"] = verify_hashes(manifest["data_sha256"], ROOT / "data/raw")
        package_checks = []
        for name, expected in manifest["packages"].items():
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                actual = None
            package_checks.append({"name": name, "expected": expected, "actual": actual,
                                   "passed": actual == expected})
        report["packages"] = package_checks
        report["requirements_hash_passed"] = sha256(ROOT / "requirements.txt") == manifest["requirements_sha256"]
        if not all(item["passed"] for group in (report["source_hashes"], report["raw_data_hashes"], package_checks)
                   for item in group) or not report["requirements_hash_passed"]:
            raise ValueError("Prediction source, raw data, requirements or package version mismatch")
        snapshot = archive_path.read_bytes()
        snapshot_hash = hashlib.sha256(snapshot).hexdigest()
        report["snapshot_sha256"] = snapshot_hash
        report["snapshot_bytes"] = len(snapshot)
        report["output_checksum_sealed"] = complete
        if complete and snapshot_hash != manifest.get("output_sha256"):
            raise ValueError("Completed CSV checksum does not match manifest")
        archive = pd.read_csv(io.BytesIO(snapshot), float_precision="round_trip")
        if not set(KEY + ("status",)).issubset(archive.columns):
            raise ValueError("Archive is missing treatment keys/status")
        ok = archive[archive.status.eq("ok")].copy()
        if ok.empty or ok.duplicated(list(KEY)).any():
            raise ValueError("No successful rows or duplicate successful treatment keys")
        allowed = {tuple(combo[k] for k in KEY[:-1]) + (bool(manifest["natural_missing"]),)
                   for combo in cfg.iter_combos()}
        actual_keys = {key_of(row) for _, row in ok.iterrows()}
        if not actual_keys.issubset(allowed):
            raise ValueError("Archive keys outside the manifest design")
        if complete and (len(archive) != cfg.n_combos() or len(ok) != len(archive) or actual_keys != allowed):
            raise ValueError("Completed manifest does not contain exactly the complete successful design")
        report.update(archive_rows=len(archive), successful_rows=len(ok), expected_rows=cfg.n_combos(),
                      patient_overlap_max=(int(ok.patient_overlap.max()) if "patient_overlap" in ok else None))
        if report["patient_overlap_max"] not in (0, None):
            raise ValueError("Nonzero patient overlap in archived protocol metadata")
        selected = select_rows(ok, cfg, args.n)
        report["selected_keys"] = [dict(zip(KEY, key_of(row))) for _, row in selected.iterrows()]
        report["coverage"] = {name: sorted({str(row[name]) for _, row in selected.iterrows()})
                              for name in KEY[:-1]}
        report["replays_selected"] = len(selected)
        if args.list_only:
            report["status"] = "selection_only_no_replay"
            print(json.dumps(report["selected_keys"], indent=2), flush=True)
        else:
            for i, (_, archived) in enumerate(selected.iterrows(), 1):
                key = key_of(archived)
                start = time.perf_counter()
                clear_caches()
                try:
                    new = run_cell(*key[:6], key[6], cfg=cfg, natural_missing=key[7])
                    cell = compare_record(new, archived)
                except Exception as exc:
                    cell = {"passed": False, "exception": f"{type(exc).__name__}: {exc}"}
                finally:
                    clear_caches()
                cell.update(key=dict(zip(KEY, key)), wall_time_seconds=time.perf_counter() - start)
                report["cells"].append(cell)
                print(f"{i}/{len(selected)} {'PASS' if cell['passed'] else 'FAIL'} {'/'.join(map(str, key[:-1]))}", flush=True)
            report["replays_passed"] = sum(cell["passed"] for cell in report["cells"])
            report["all_numeric_values_exactly_equal"] = all(
                cell.get("numeric_values_exactly_equal", False) for cell in report["cells"])
            report["max_absolute_numeric_difference"] = max(
                (cell.get("max_absolute_numeric_difference", 0.) for cell in report["cells"]), default=0.)
            all_passed = report["replays_passed"] == len(selected)
            report["status"] = ("passed_sampled_replay" if complete else "passed_partial_precheck") if all_passed else "failed"
            report["all_archived_models_replayed"] = len(selected) == cfg.n_combos()
        report["source_csv_changed_during_check"] = sha256(archive_path) != snapshot_hash
        if complete and report["source_csv_changed_during_check"]:
            raise ValueError("Completed archive changed during verification")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], flush=True)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_report(report_path, report)
    print(f"status={report['status']} report={report_path}", flush=True)
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
