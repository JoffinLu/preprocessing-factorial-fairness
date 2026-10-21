"""Native-group reweighing and train-fitted numerical marginal repair.

C2 applies w(g,y)=n_g*n_y/(n*n_gy) to ALL original protected groups, without
majority-versus-rest collapsing. Positive support in every group-label cell is
required; otherwise the requested independent weighted distribution is not
identified and the cell fails explicitly.

C3 (train_quantile) maps each group empirical numerical marginal CDF to the
equal-group Wasserstein barycenter of training quantile functions. Only the
specified original numeric feature indices are repaired; dummy columns remain
unchanged. This is an inductive empirical marginal mapping, NOT the AIF360 DIR
algorithm and NOT a guarantee of demographic parity. Group membership is needed
at prediction time. train_quantile_reference holds the fitting reference at the
original imputed training rows before B, isolating the role of resampling in the
repair reference distribution. transductive_dir is a separate named ablation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

METHODS = ("C1", "C2", "C3")
REPAIR_VARIANTS = ("train_quantile", "train_quantile_reference", "transductive_dir")


def _groups(s: np.ndarray) -> list:
    return list(dict.fromkeys(np.asarray(s).tolist()))


@dataclass
class InterventionResult:
    X_train: np.ndarray
    X_test: np.ndarray
    sample_weight: np.ndarray | None
    meta: dict[str, Any] = field(default_factory=dict)


def reweighing_weights(sensitive: np.ndarray, labels: np.ndarray) -> np.ndarray:
    s, y = np.asarray(sensitive), np.asarray(labels)
    if len(s) != len(y) or not len(y):
        raise ValueError("Reweighing requires nonempty aligned arrays")
    w = np.empty(len(y), dtype=float)
    n = len(y)
    for g in _groups(s):
        group = s == g
        for label in np.unique(y):
            is_label = y == label
            joint = group & is_label
            if not joint.any():
                raise ValueError(f"Reweighing support missing: group={g!r}, label={label!r}")
            w[joint] = group.sum() * is_label.sum() / (n * joint.sum())
    if not np.isclose(w.sum(), n, rtol=1e-12):
        raise AssertionError("Reweighing failed to preserve total mass")
    return w


class TrainQuantileRepair:
    """Training empirical CDF -> equal-group quantile barycenter.

    With ties, use midpoint empirical ranks. Outside a training group's observed
    support, clamp ranks to 0/1. The 257-point quantile grid and linear
    interpolation are fixed protocol choices, shared by every experimental cell.
    """
    def __init__(self, feature_indices=None, n_quantiles: int = 257):
        self.feature_indices = feature_indices
        self.n_quantiles = int(n_quantiles)

    def fit(self, X: np.ndarray, sensitive: np.ndarray, *, group_order=None):
        X, s = np.asarray(X, dtype=float), np.asarray(sensitive)
        if len(X) != len(s) or not np.isfinite(X).all():
            raise ValueError("Repair requires aligned, finite training data")
        self.groups_ = tuple(_groups(s) if group_order is None else group_order)
        if set(self.groups_) != set(_groups(s)):
            raise ValueError("Repair group support differs from original training groups")
        self.n_features_in_ = X.shape[1]
        self.feature_indices_ = tuple(range(X.shape[1]) if self.feature_indices is None
                                      else self.feature_indices)
        if any(i < 0 or i >= X.shape[1] for i in self.feature_indices_):
            raise ValueError("Repair feature index outside matrix")
        self.quantile_grid_ = np.linspace(0, 1, self.n_quantiles)
        targets = np.zeros((len(self.feature_indices_), self.n_quantiles))
        self.maps_ = {}
        for g in self.groups_:
            local = X[s == g]
            maps = []
            for j, feature in enumerate(self.feature_indices_):
                values, counts = np.unique(local[:, feature], return_counts=True)
                midpoint_cdf = (np.cumsum(counts) - counts / 2) / len(local)
                maps.append((values, midpoint_cdf))
                targets[j] += np.quantile(local[:, feature], self.quantile_grid_,
                                           method="linear") / len(self.groups_)
            self.maps_[g] = tuple(maps)
        self.target_quantiles_ = targets
        return self

    def transform(self, X: np.ndarray, sensitive: np.ndarray) -> np.ndarray:
        X, s = np.asarray(X, dtype=float), np.asarray(sensitive)
        if len(X) != len(s) or X.shape[1] != self.n_features_in_:
            raise ValueError("Repair transform shape mismatch")
        unknown = set(_groups(s)) - set(self.groups_)
        if unknown:
            raise ValueError(f"Unseen protected groups at repair time: {unknown}")
        if not np.isfinite(X).all():
            raise ValueError("Repair requires finite test features")
        out = X.copy()
        for g in self.groups_:
            rows = np.flatnonzero(s == g)
            for j, feature in enumerate(self.feature_indices_):
                values, cdf = self.maps_[g][j]
                q = np.interp(X[rows, feature], values, cdf, left=0.0, right=1.0)
                out[rows, feature] = np.interp(q, self.quantile_grid_,
                                              self.target_quantiles_[j])
        return out


def _transductive_dir(X_train, y_train, s_train, X_test, s_test,
                      group_order, feature_indices):
    from aif360.algorithms.preprocessing import DisparateImpactRemover
    from aif360.datasets import BinaryLabelDataset
    import pandas as pd

    mapping = {g: i for i, g in enumerate(group_order)}
    n_train = len(X_train)
    features = np.vstack([X_train, X_test])[:, feature_indices]
    sensitive = np.concatenate([s_train, s_test])
    codes = np.array([mapping[g] for g in sensitive], dtype=float)
    columns = [f"f{i}" for i in range(features.shape[1])] + ["__sensitive__"]
    frame = pd.DataFrame(np.column_stack([features, codes]), columns=columns)
    # Test labels are deliberately unavailable to the transformation.
    frame["label"] = np.concatenate([y_train, np.zeros(len(X_test), dtype=int)])
    data = BinaryLabelDataset(df=frame, label_names=["label"],
                              protected_attribute_names=["__sensitive__"])
    repaired = DisparateImpactRemover(repair_level=1.0).fit_transform(data)
    transformed = np.asarray(repaired.features)[:, :features.shape[1]]
    tr, te = X_train.copy(), X_test.copy()
    tr[:, feature_indices], te[:, feature_indices] = transformed[:n_train], transformed[n_train:]
    return tr, te


def apply_fair_intervention(X_train: np.ndarray, y_train: np.ndarray,
                            sensitive_train: np.ndarray, X_test: np.ndarray,
                            sensitive_test: np.ndarray, method: str,
                            seed: int = 0, *, repair_variant: str = "train_quantile",
                            reference_sensitive: np.ndarray | None = None,
                            reference_X: np.ndarray | None = None,
                            feature_indices=None) -> InterventionResult:
    if method not in METHODS or repair_variant not in REPAIR_VARIANTS:
        raise ValueError("Unknown fairness method/repair variant")
    Xtr, Xte = np.asarray(X_train, dtype=float), np.asarray(X_test, dtype=float)
    y, str_, ste = np.asarray(y_train, dtype=int), np.asarray(sensitive_train), np.asarray(sensitive_test)
    original_groups = _groups(str_ if reference_sensitive is None else reference_sensitive)
    if set(_groups(str_)) != set(original_groups):
        raise ValueError("Resampling changed protected group support")
    if set(_groups(ste)) - set(original_groups):
        raise ValueError("Test group absent from original training sample")
    indices = tuple(range(Xtr.shape[1]) if feature_indices is None else feature_indices)
    meta = {"seed": int(seed), "method": method,
            "sensitive_binary_note": f"native {len(original_groups)} groups; no binary collapse",
            "n_sensitive_classes": len(original_groups),
            "group_definition": "original_training_groups_before_resampling",
            "repair_variant": repair_variant, "c3_path": None,
            "sample_weight_mean": None}
    if method == "C1":
        return InterventionResult(Xtr.copy(), Xte.copy(), None, meta)
    if method == "C2":
        w = reweighing_weights(str_, y)
        meta.update({"sample_weight_mean": float(w.mean()),
                     "sample_weight_min": float(w.min()),
                     "sample_weight_max": float(w.max()),
                     "reweighing_formula": "P(g)P(y)/P(g,y)"})
        return InterventionResult(Xtr.copy(), Xte.copy(), w, meta)
    if repair_variant in ("train_quantile", "train_quantile_reference"):
        if repair_variant == "train_quantile_reference":
            if reference_X is None or reference_sensitive is None:
                raise ValueError("Reference repair requires pre-resampling training X and groups")
            fitted_X, fitted_s = np.asarray(reference_X), np.asarray(reference_sensitive)
            reference_note = "original_imputed_training_rows_before_resampling"
        else:
            fitted_X, fitted_s = Xtr, str_
            reference_note = "post_resampling_training_rows"
        repair = TrainQuantileRepair(indices).fit(fitted_X, fitted_s, group_order=original_groups)
        tr, te = repair.transform(Xtr, str_), repair.transform(Xte, ste)
        path = ("train_fitted_numerical_quantile_barycenter"
                if repair_variant == "train_quantile" else "pre_resampling_reference_quantile_barycenter")
        meta["repair_fitting_reference"] = reference_note
    else:
        tr, te = _transductive_dir(Xtr, y, str_, Xte, ste, original_groups, indices)
        path = "transductive_aif360_dir_combined_train_test"
    meta.update({"c3_path": path,
                 "c3_features_changed": bool(not np.allclose(tr, Xtr)),
                 "repair_feature_count": len(indices),
                 "requires_sensitive_at_prediction": True})
    return InterventionResult(tr, te, None, meta)


def _self_test() -> None:
    rng = np.random.default_rng(2)
    s = np.repeat(["a", "b", "c"], 40)
    y = np.tile(np.repeat([0, 1], 20), 3)
    X = rng.normal(size=(120, 4)) + np.repeat([0, 3, 5], 40)[:, None]
    w = reweighing_weights(s, y)
    assert np.isclose(w.sum(), len(y))
    a = apply_fair_intervention(X, y, s, X[:30], s[:30], "C3", feature_indices=(0, 1))
    b = apply_fair_intervention(X, y, s, X[:30] + 100, s[:30], "C3", feature_indices=(0, 1))
    assert np.array_equal(a.X_train, b.X_train)
    assert np.array_equal(a.X_train[:, 2:], X[:, 2:])
    assert not np.array_equal(a.X_train[:, :2], X[:, :2])
    print("intervention protocol checks passed")


if __name__ == "__main__":
    _self_test()
