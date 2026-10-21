"""Controlled MCAR/MAR on numerical targets with always-observed drivers.

The first one or two numerical columns (in source-column order) are auxiliary
variables. They remain complete in BOTH mechanisms; only the remaining numeric
columns are masked. Categorical columns are untouched. In MAR, every missingness
probability depends only on these final observed auxiliaries. A plan is fitted
on the training reference and reused unchanged on test; only mask RNG differs.
The requested rate is the rate among eligible numeric targets, not all entries.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MECHANISMS = ("MCAR", "MAR")
MIN_OBSERVED = 3
_P_MIN, _P_MAX = 1e-4, 0.95


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(scores, -60.0, 60.0)))


def _calibrate_intercept(scores: np.ndarray, rate: float) -> float:
    lo, hi = -60.0, 60.0
    for _ in range(80):
        mid = (lo + hi) / 2
        mean_p = np.clip(_sigmoid(scores + mid), _P_MIN, _P_MAX).mean()
        if mean_p > rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


@dataclass(frozen=True)
class MissingnessPlan:
    mechanism: str
    rate: float
    driver_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    mechanism_seed: int

    def probabilities(self, X: pd.DataFrame) -> np.ndarray:
        if X[list(self.driver_columns)].isna().any().any():
            raise ValueError("Missingness drivers must remain fully observed")
        if self.mechanism == "MCAR":
            return np.full((len(X), len(self.target_columns)), self.rate)
        z = (X[list(self.driver_columns)].to_numpy(dtype=float) - self.means) / self.scales
        scores = z @ self.coefficients.T + self.intercepts
        return np.clip(_sigmoid(scores), _P_MIN, _P_MAX)

    def metadata(self) -> dict:
        return {"mechanism": self.mechanism, "rate": self.rate,
                "driver_columns": list(self.driver_columns),
                "target_columns": list(self.target_columns),
                "mechanism_seed": self.mechanism_seed,
                "rate_denominator": "eligible_numeric_target_entries"}


def fit_missingness(train: pd.DataFrame, mechanism: str, rate: float = 0.30,
                    seed: int = 0) -> MissingnessPlan:
    if mechanism not in MECHANISMS:
        raise ValueError(f"Unknown mechanism {mechanism!r}")
    if not _P_MIN < rate < _P_MAX:
        raise ValueError(f"rate must be between {_P_MIN} and {_P_MAX}")
    numeric = [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c])]
    if len(numeric) < 2:
        raise ValueError("Controlled missingness needs at least two numerical columns")
    n_drivers = min(2, len(numeric) - 1)
    drivers, targets = tuple(numeric[:n_drivers]), tuple(numeric[n_drivers:])
    if train[list(drivers + targets)].isna().any().any():
        raise ValueError("Fit missingness on the train-completed reference, not raw gaps")
    v = train[list(drivers)].to_numpy(dtype=float)
    means, scales = v.mean(axis=0), v.std(axis=0)
    scales = np.where(scales > 0, scales, 1.0)
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(size=(len(targets), len(drivers)))
    coefficients /= np.maximum(np.linalg.norm(coefficients, axis=1, keepdims=True), 1e-12)
    scores = ((v - means) / scales) @ coefficients.T
    intercepts = np.array([_calibrate_intercept(scores[:, j], rate)
                           for j in range(len(targets))])
    for arr in (means, scales, coefficients, intercepts):
        arr.setflags(write=False)
    return MissingnessPlan(mechanism, float(rate), drivers, targets, means, scales,
                           coefficients, intercepts, int(seed))


def inject_missing(X: pd.DataFrame, mechanism: str, rate: float = 0.30,
                    seed: int = 0, *, plan: MissingnessPlan | None = None,
                    mask_seed: int | None = None) -> pd.DataFrame:
    """Apply a fixed plan; callers MUST share a train-fitted plan with test.

    Without a plan this remains a convenient standalone train-only operation.
    Protecting at least MIN_OBSERVED entries depends only on the sampled mask,
    never on unobserved values, so it does not introduce an MNAR path.
    """
    if plan is None:
        plan = fit_missingness(X, mechanism, rate, seed)
    if plan.mechanism != mechanism or abs(plan.rate - rate) > 1e-12:
        raise ValueError("Supplied plan does not match mechanism/rate")
    out = X.copy().reset_index(drop=True)
    probabilities = plan.probabilities(out)
    rng = np.random.default_rng(seed if mask_seed is None else mask_seed)
    mask = rng.random(probabilities.shape) < probabilities
    for j, col in enumerate(plan.target_columns):
        missing_idx = np.flatnonzero(mask[:, j])
        n_obs = len(out) - len(missing_idx)
        required = min(MIN_OBSERVED, len(out))
        if n_obs < required:
            keep = rng.choice(missing_idx, size=required - n_obs, replace=False)
            mask[keep, j] = False
        out[col] = out[col].mask(mask[:, j])
    assert not out[list(plan.driver_columns)].isna().any().any()
    return out


def _self_test() -> None:
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(8000, 6)), columns=list("abcdef"))
    X["category"] = rng.choice(["a", "b"], len(X))
    for mechanism in MECHANISMS:
        plan = fit_missingness(X, mechanism, seed=2)
        out = inject_missing(X, mechanism, plan=plan, mask_seed=8)
        assert not out[list(plan.driver_columns)].isna().any().any()
        assert out["category"].equals(X["category"])
        observed = out[list(plan.target_columns)].isna().mean().mean()
        assert abs(observed - .3) < .02
        assert np.array_equal(plan.probabilities(X), plan.probabilities(X.copy()))
        print(mechanism, "eligible missing fraction", round(float(observed), 4))
    print("missingness protocol checks passed")


if __name__ == "__main__":
    _self_test()
