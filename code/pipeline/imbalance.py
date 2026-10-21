"""Class imbalance treatment with nominal-group-preserving resampling.

B3 balances the global minority class to the majority-class count. Additional
minority rows are allocated in proportion to ORIGINAL minority group counts by
largest remainders (ties follow first row occurrence, never label spelling).
SMOTE neighbours are restricted to the same protected group and class. A group
with only one minority row uses explicit duplication, recorded in metadata.
The random_over ablation uses the identical allocation but duplicates rows.
"""
from __future__ import annotations

import numpy as np

METHODS = ("B1", "B2", "B3")
VARIANTS = ("within_group_smote", "random_over")
SMOTE_K = 5


def _dist(y: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(y, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def nominal_groups(s: np.ndarray) -> list:
    # Stable under any one-to-one relabeling of a nominal protected attribute.
    return list(dict.fromkeys(np.asarray(s).tolist()))


def class_weight_balanced(y: np.ndarray) -> dict[int, float]:
    values, counts = np.unique(y, return_counts=True)
    return {int(v): len(y) / (len(values) * int(n)) for v, n in zip(values, counts)}


def _allocate(additional: int, counts: np.ndarray) -> np.ndarray:
    quota = additional * counts.astype(float) / counts.sum()
    allocation = np.floor(quota).astype(int)
    remainder = additional - allocation.sum()
    order = np.argsort(-(quota - allocation), kind="stable")
    allocation[order[:remainder]] += 1
    return allocation


def resample(X_train: np.ndarray, y_train: np.ndarray,
             sensitive_train: np.ndarray, method: str, seed: int = 0, *,
             variant: str = "within_group_smote"
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if method not in METHODS:
        raise ValueError(f"Unknown imbalance method {method!r}")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown resampling variant {variant!r}")
    X, y, s = (np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=int),
               np.asarray(sensitive_train))
    if len(X) != len(y) or len(s) != len(y):
        raise ValueError("Resampling arrays have unequal lengths")
    meta = {"before_dist": _dist(y), "after_dist": _dist(y),
            "n_before": len(y), "n_after": len(y), "class_weight": None,
            "smote_k": None, "resample_variant": variant}
    if method == "B1":
        return X, y, s, meta
    if method == "B2":
        meta["class_weight"] = class_weight_balanced(y)
        return X, y, s, meta
    values, counts = np.unique(y, return_counts=True)
    if len(values) != 2:
        raise ValueError("B3 expects a binary outcome with both classes observed")
    minority = values[int(np.argmin(counts))]
    additional = int(counts.max() - counts.min())
    if additional == 0:
        meta["note"] = "Classes already balanced; no synthetic rows"
        return X, y, s, meta
    groups = [g for g in nominal_groups(s) if np.any((s == g) & (y == minority))]
    group_counts = np.array([np.sum((s == g) & (y == minority)) for g in groups])
    allocations = _allocate(additional, group_counts)
    rng = np.random.default_rng(seed)
    blocks_x, blocks_y, blocks_s = [X], [y], [s]
    fallback_groups, used_k, allocation_meta = [], [], []
    for g, original_n, n_add in zip(groups, group_counts, allocations):
        n_add = int(n_add)
        allocation_meta.append({"group": str(g), "original_minority": int(original_n),
                                "added_minority": n_add})
        if not n_add:
            continue
        Xg = X[(s == g) & (y == minority)]
        if variant == "random_over" or len(Xg) == 1:
            generated = Xg[rng.integers(len(Xg), size=n_add)].copy()
            if len(Xg) == 1 and variant == "within_group_smote":
                fallback_groups.append(str(g))
        else:
            from sklearn.neighbors import NearestNeighbors
            k = min(SMOTE_K, len(Xg) - 1)
            nn = NearestNeighbors(n_neighbors=k + 1).fit(Xg)
            neighbours = nn.kneighbors(Xg, return_distance=False)
            # Remove each query's own row by index (duplicates need not sort
            # self first); choose the first k remaining nearest neighbours.
            neighbours = np.array([row[row != i][:k]
                                   for i, row in enumerate(neighbours)])
            base = rng.integers(len(Xg), size=n_add)
            other = neighbours[base, rng.integers(k, size=n_add)]
            fraction = rng.random(n_add).reshape(-1, 1)
            generated = Xg[base] + fraction * (Xg[other] - Xg[base])
            used_k.append(k)
        blocks_x.append(generated)
        blocks_y.append(np.full(n_add, minority, dtype=y.dtype))
        blocks_s.append(np.full(n_add, g, dtype=s.dtype))
    xo, yo, so = np.vstack(blocks_x), np.concatenate(blocks_y), np.concatenate(blocks_s)
    meta.update({"after_dist": _dist(yo), "n_after": len(yo),
                 "smote_k": min(used_k) if used_k else None,
                 "sensitive_alignment": "nominal group held fixed; same-group minority neighbours",
                 "minority_allocation": allocation_meta,
                 "singleton_duplication_groups": fallback_groups})
    if fallback_groups:
        meta["note"] = "Singleton minority groups duplicated: " + ",".join(fallback_groups)
    assert len(set(_dist(yo).values())) == 1
    return xo, yo, so, meta


def _self_test() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 4))
    y = np.array([1] * 20 + [0] * 80)
    s = np.array(["red", "blue", "green", "red"] * 25)
    rename = {"red": "z", "blue": "a", "green": "q"}
    sx = np.array([rename[g] for g in s])
    for variant in VARIANTS:
        a = resample(X, y, s, "B3", seed=4, variant=variant)
        b = resample(X, y, sx, "B3", seed=4, variant=variant)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        assert np.array_equal(np.array([rename[g] for g in a[2]]), b[2])
        assert _dist(a[1]) == {0: 80, 1: 80}
        print(variant, a[3]["after_dist"])
    print("resampling protocol checks passed")


if __name__ == "__main__":
    _self_test()
