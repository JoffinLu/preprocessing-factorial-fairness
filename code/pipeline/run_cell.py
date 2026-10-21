"""revision_v2 pipeline: grouped split -> train-only reference completion ->
shared-plan numeric missingness -> encode/scale/impute -> imbalance -> fairness.

A bounded cache shares the expensive imputation among B/C cells. Cached arrays
are read-only; each downstream transformation returns its own mutable output.
clear_caches() is provided for worker-unit memory control.
"""
from __future__ import annotations

import hashlib
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from pipeline.config import MECHANISM_TO_NAME, Config
from pipeline.data import Dataset, complete_from_training, load_dataset
from pipeline.features import (categorical_fill_values, encode, fit_category_spec,
                               split_columns, standardize)
from pipeline.imbalance import resample
from pipeline.impute import impute
from pipeline.intervention import apply_fair_intervention
from pipeline.metrics import evaluate, group_table
from pipeline.missing import fit_missingness, inject_missing
from pipeline.model import fit_predict

INJECT_SEED_OFFSET_TEST = 5000
_DATASET_CACHE: dict[tuple, Dataset] = {}
_SPLIT_CACHE: OrderedDict = OrderedDict()
_A_CACHE: OrderedDict = OrderedDict()
MAX_A_CACHE = 4
MAX_SPLIT_CACHE = 8

ORDER_COLS = (
    "dataset", "mechanism", "A", "B", "C", "seed", "model", "natural_missing",
    "protocol_version", "missing_rate", "resample_variant", "repair_variant",
    "repair_numeric_only", "split_kind", "patient_overlap", "n_train_patients",
    "n_test_patients", "missing_driver_cols", "missing_target_cols",
    "n_train", "n_test", "n_features_encoded", "imbalance_ratio",
    "minority_majority_ratio", "sensitive_name", "n_groups",
    "AUC", "F1", "accuracy", "balanced_accuracy", "precision", "recall",
    "specificity", "n_pos_true", "n_neg_true", "n_pred_pos",
    "DP", "EO", "EOdd", "TPR_gap", "DI", "undefined_tpr_groups",
    "undefined_sel_groups", "min_group_n", "min_group_n_pos",
    "impute_secs", "train_secs", "total_secs", "n_train_before_B", "smote_k",
    "resample_note", "c_note", "c3_path", "c3_changed", "sw_mean", "sw_min",
    "sw_max", "lr_converged", "inject_note", "status",
)


def clear_caches() -> None:
    _A_CACHE.clear()
    _SPLIT_CACHE.clear()
    _DATASET_CACHE.clear()


def get_dataset(name: str, natural_missing: bool = False,
                drop_bank_duration: bool = True) -> Dataset:
    key = (name, bool(natural_missing), bool(drop_bank_duration))
    if key not in _DATASET_CACHE:
        _DATASET_CACHE[key] = load_dataset(name, natural_missing, drop_bank_duration)
    return _DATASET_CACHE[key]


def _fingerprint(array) -> str:
    return hashlib.sha256(pd.util.hash_array(np.asarray(array)).tobytes()).hexdigest()[:20]


def split_indices(dataset: str, y: np.ndarray, seed: int, test_size: float,
                  groups: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Paired split; Diabetes patients must never cross train and test.

    For grouped data choose among 32 GroupShuffleSplit candidates by absolute
    test-size deviation plus absolute test/train label-prevalence deviations.
    This balances design quantities only and never uses a model or its metrics.
    Grouped test counts are approximately, rather than exactly, 30% of rows.
    """
    if dataset == "diabetes" and groups is None:
        raise ValueError("Diabetes split requires patient grouping identifiers")
    key = (dataset, int(seed), float(test_size), _fingerprint(y),
           None if groups is None else _fingerprint(groups))
    if key in _SPLIT_CACHE:
        _SPLIT_CACHE.move_to_end(key)
        return _SPLIT_CACHE[key]
    if groups is None:
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(np.arange(len(y)), test_size=test_size,
                                  random_state=int(seed), stratify=y)
    else:
        from sklearn.model_selection import GroupShuffleSplit
        groups = np.asarray(groups)
        if len(groups) != len(y):
            raise ValueError("Patient identifiers must align with rows")
        splitter = GroupShuffleSplit(n_splits=32, test_size=test_size,
                                      random_state=int(seed))
        candidates = []
        prevalence = float(np.mean(y))
        for tr_i, te_i in splitter.split(np.zeros(len(y)), y, groups):
            if len(np.unique(y[tr_i])) < 2 or len(np.unique(y[te_i])) < 2:
                continue
            score = (abs(len(te_i) / len(y) - test_size)
                     + abs(float(np.mean(y[te_i])) - prevalence)
                     + abs(float(np.mean(y[tr_i])) - prevalence))
            candidates.append((score, tr_i, te_i))
        if not candidates:
            raise ValueError("No grouped split with both labels in both partitions")
        _, tr, te = min(candidates, key=lambda item: item[0])
        if set(groups[tr]) & set(groups[te]):
            raise AssertionError("Patient overlap in grouped split")
    tr, te = np.asarray(tr), np.asarray(te)
    tr.setflags(write=False)
    te.setflags(write=False)
    _SPLIT_CACHE[key] = (tr, te)
    while len(_SPLIT_CACHE) > MAX_SPLIT_CACHE:
        _SPLIT_CACHE.popitem(last=False)
    return tr, te


def _a_cache_key(dataset, mechanism, a, seed, cfg, natural_missing):
    return (dataset, mechanism, a, int(seed), bool(natural_missing),
            cfg.missing_rate, cfg.test_size, cfg.random_state,
            cfg.categorical_missing, cfg.max_cardinality,
            getattr(cfg, "drop_bank_duration", True),
            getattr(cfg, "protocol_version", "revision_v2"))


def _prepare_through_a(dataset, mechanism, a, seed, cfg, natural_missing):
    key = _a_cache_key(dataset, mechanism, a, seed, cfg, natural_missing)
    if key in _A_CACHE:
        _A_CACHE.move_to_end(key)
        return _A_CACHE[key]
    ds = get_dataset(dataset, natural_missing, getattr(cfg, "drop_bank_duration", True))
    idx_tr, idx_te = split_indices(dataset, ds.y, seed, cfg.test_size, ds.groups)
    Xtr = ds.X_raw.iloc[idx_tr].reset_index(drop=True)
    Xte = ds.X_raw.iloc[idx_te].reset_index(drop=True)
    ytr, yte = ds.y[idx_tr], ds.y[idx_te]
    str_, ste = ds.sensitive.iloc[idx_tr].to_numpy(), ds.sensitive.iloc[idx_te].to_numpy()
    if mechanism == "NAT":
        Xtr_m, Xte_m, drivers, targets = Xtr, Xte, (), ()
        inject_note = "natural missingness; no preliminary completion or injection"
    else:
        Xtr, Xte, _ = complete_from_training(Xtr, Xte)
        mechanism_name = MECHANISM_TO_NAME[mechanism]
        plan = fit_missingness(Xtr, mechanism_name, cfg.missing_rate, seed=cfg.random_state)
        Xtr_m = inject_missing(Xtr, mechanism_name, cfg.missing_rate, plan=plan,
                               mask_seed=seed)
        Xte_m = inject_missing(Xte, mechanism_name, cfg.missing_rate, plan=plan,
                               mask_seed=seed + INJECT_SEED_OFFSET_TEST)
        drivers, targets = plan.driver_columns, plan.target_columns
        inject_note = (f"{mechanism_name}; rate={cfg.missing_rate}; numeric targets only; "
                       "training-fitted mechanism shared with test; observed drivers="
                       + ",".join(drivers))
    numeric_cols, _ = split_columns(Xtr_m)
    fv = categorical_fill_values(Xtr_m) if cfg.categorical_missing == "mode" else None
    spec = fit_category_spec(Xtr_m, cfg.categorical_missing, cfg.max_cardinality, fv)
    mat_tr, names = encode(Xtr_m, cfg.categorical_missing, spec, fv)
    mat_te, names_te = encode(Xte_m, cfg.categorical_missing, spec, fv)
    if names != names_te:
        raise AssertionError("Training and test encoded columns differ")
    tr_s, te_s = standardize(mat_tr, mat_te)
    tr_a, te_a, impute_secs = impute(tr_s, te_s, a, seed=seed)
    overlap = 0 if ds.groups is None else len(set(ds.groups[idx_tr]) & set(ds.groups[idx_te]))
    prepared = {
        "ds": ds, "X_train_a": tr_a, "X_test_a": te_a,
        "y_train": ytr, "y_test": yte,
        "sensitive_train": str_, "sensitive_test": ste,
        "numeric_indices": tuple(range(len(numeric_cols))),
        "n_features_encoded": len(names), "impute_secs": impute_secs,
        "missing_driver_cols": "|".join(drivers), "missing_target_cols": "|".join(targets),
        "inject_note": inject_note, "patient_overlap": overlap,
        "split_kind": "label_stratified_rows" if ds.groups is None else "balanced_patient_group_shuffle_32",
        "n_train_patients": np.nan if ds.groups is None else len(set(ds.groups[idx_tr])),
        "n_test_patients": np.nan if ds.groups is None else len(set(ds.groups[idx_te])),
    }
    for arr in (tr_a, te_a, ytr, yte, str_, ste):
        arr.setflags(write=False)
    _A_CACHE[key] = prepared
    while len(_A_CACHE) > MAX_A_CACHE:
        _A_CACHE.popitem(last=False)
    return prepared


def prepare_cell(dataset: str, mechanism: str, a: str, b: str, c: str, seed: int,
                 cfg: Config | None = None, natural_missing: bool = False) -> dict:
    cfg = cfg or Config()
    t0 = time.perf_counter()
    if (mechanism == "NAT") != bool(natural_missing):
        raise ValueError("NAT and natural_missing=True must be used together")
    before = _prepare_through_a(dataset, mechanism, a, seed, cfg, natural_missing)
    variant = getattr(cfg, "resample_variant", "within_group_smote")
    repair_variant = getattr(cfg, "repair_variant", "train_quantile")
    tr_b, ytr_b, str_b, b_meta = resample(before["X_train_a"], before["y_train"],
                                         before["sensitive_train"], b, seed=seed,
                                         variant=variant)
    feature_indices = (before["numeric_indices"]
                       if getattr(cfg, "repair_numeric_only", True) else None)
    iv = apply_fair_intervention(tr_b, ytr_b, str_b, before["X_test_a"],
                                 before["sensitive_test"], c, seed=seed,
                                 repair_variant=repair_variant,
                                 reference_sensitive=before["sensitive_train"],
                                 reference_X=before["X_train_a"],
                                 feature_indices=feature_indices)
    return {
        "ds": before["ds"], "cfg": cfg, "dataset": dataset, "mechanism": mechanism,
        "A": a, "B": b, "C": c, "seed": int(seed), "natural_missing": bool(natural_missing),
        "X_train": iv.X_train, "y_train": ytr_b, "X_test": iv.X_test,
        "sample_weight": iv.sample_weight, "class_weight": b_meta["class_weight"],
        "y_test": before["y_test"], "sensitive_test": before["sensitive_test"],
        "n_features_encoded": before["n_features_encoded"],
        "impute_secs": before["impute_secs"], "n_train_before_B": b_meta["n_before"],
        "smote_k": b_meta.get("smote_k"), "resample_note": b_meta.get("note"),
        "c_note": iv.meta.get("sensitive_binary_note"), "c3_path": iv.meta.get("c3_path"),
        "c3_changed": iv.meta.get("c3_features_changed"),
        "sw_mean": iv.meta.get("sample_weight_mean"), "sw_min": iv.meta.get("sample_weight_min"),
        "sw_max": iv.meta.get("sample_weight_max"), "inject_note": before["inject_note"],
        "prep_secs": time.perf_counter() - t0,
        **{k: before[k] for k in ("patient_overlap", "split_kind", "n_train_patients",
                                  "n_test_patients", "missing_driver_cols", "missing_target_cols")},
    }


def run_cell_from_prep(prep: dict, model_name: str, collect_groups: bool = False) -> dict:
    cfg, ds = prep["cfg"], prep["ds"]
    mr = fit_predict(model_name, prep["X_train"], prep["y_train"], prep["X_test"],
                      sample_weight=prep["sample_weight"], class_weight=prep["class_weight"],
                      seed=prep["seed"])
    metrics = evaluate(prep["y_test"], mr["y_pred"], mr["y_prob"], prep["sensitive_test"],
                       dataset=prep["dataset"], model=model_name)
    row = {k: prep[k] for k in ("dataset", "mechanism", "A", "B", "C", "seed", "natural_missing",
                                "patient_overlap", "split_kind", "n_train_patients", "n_test_patients",
                                "missing_driver_cols", "missing_target_cols", "n_features_encoded",
                                "impute_secs", "n_train_before_B", "smote_k", "resample_note", "c_note",
                                "c3_path", "c3_changed", "sw_mean", "sw_min", "sw_max", "inject_note")}
    row.update({"model": model_name, "protocol_version": getattr(cfg, "protocol_version", "revision_v2"),
                "missing_rate": cfg.missing_rate,
                "resample_variant": getattr(cfg, "resample_variant", "within_group_smote"),
                "repair_variant": getattr(cfg, "repair_variant", "train_quantile"),
                "repair_numeric_only": getattr(cfg, "repair_numeric_only", True),
                "n_train": len(prep["y_train"]), "n_test": len(prep["y_test"]),
                "imbalance_ratio": ds.meta["imbalance_ratio"],
                "minority_majority_ratio": ds.meta["minority_majority_ratio"],
                "sensitive_name": ds.sensitive_name, "train_secs": mr["train_time"],
                "total_secs": prep["prep_secs"] + mr["train_time"],
                "lr_converged": mr["meta"].get("lr_converged"), "status": "ok"})
    row.update(metrics)
    if collect_groups:
        row["_groups"] = group_table(prep["y_test"], mr["y_pred"], prep["sensitive_test"],
                                     dataset=prep["dataset"], mechanism=prep["mechanism"],
                                     A=prep["A"], B=prep["B"], C=prep["C"], seed=prep["seed"],
                                     model=model_name).to_dict(orient="records")
    return row


def run_cell(dataset: str, mechanism: str, a: str, b: str, c: str, seed: int,
             model_name: str, cfg: Config | None = None, natural_missing: bool = False,
             collect_groups: bool = False) -> dict:
    prep = prepare_cell(dataset, mechanism, a, b, c, seed, cfg, natural_missing)
    return run_cell_from_prep(prep, model_name, collect_groups)


def _self_test() -> None:
    for b, c in (("B1", "C1"), ("B2", "C2"), ("B3", "C3")):
        result = run_cell("german", "M2", "A1", b, c, 0, "LR")
        assert result["status"] == "ok" and result["patient_overlap"] == 0
        print(b, c, "AUC", round(result["AUC"], 4))
    clear_caches()
    print("pipeline protocol checks passed")


if __name__ == "__main__":
    _self_test()
