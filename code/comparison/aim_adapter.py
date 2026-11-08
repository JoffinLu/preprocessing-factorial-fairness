"""AIM 对比臂适配器（KDD 2024, commit 0384547）

AIM = Attributing, Interpreting, Mitigating Data Unfairness
原始仓库：https://github.com/ZhiningLiu1998/AIM

方法概述
  归因-解释-缓解一体化框架。核心是 UnLoc 类：
  1. fit_transform(X, y, s, ...) → 返回 df_score（含 bias_weights）
  2. fair_removal(X_train, y_train, s_train, weights, edit_ratio)
     → 按 bias score 删除 top k% 最有偏见的样本
  3. 在编辑后的训练集上训练 LogisticRegression

⚠️ 重要可行性约束（协议 S3 已记录）：
  AIM 的邻近性计算是 O(n²) 内存（Gower 距离 + 全差值矩阵）。
  - Adult（32k 训练行）：完全不可行（≈20GB+）→ 采用分层降采样 5000 行
  - COMPAS（4.3k 训练行）：勉强可行（≈148MB）
  - Diabetes（21k 训练行）：同 Adult → 分层降采样 5000 行

适配要点（协议 S3 §2.1 & §4）：
  1. AIM fit_transform 要求 X[:, 0] == s（敏感属性必须在第一列，二值化）
  2. AIM 的 UnLoc 硬编码了 save_path='data_cache'，需要 os.chdir 绕过
  3. AIM 邻近性计算不能有缺失 → 先做 A1 均值/众数插补（原始 DataFrame）
  4. edit_ratio=0.1（官方 notebook 默认），fair_criterion 通过 filter 参数
  5. 训练模型用 LogisticRegression（官方 notebook 默认）
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 引入 AIM 核心代码
_AIM_DIR = Path(__file__).resolve().parent / "third_party" / "AIM"
if str(_AIM_DIR) not in sys.path:
    sys.path.insert(0, str(_AIM_DIR))

from common import build_result_row, evaluate_method, prepare_comparison_data  # noqa: E402


def _impute_raw_df(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """原始 DataFrame 上的 A1 均值/众数插补"""
    df_tr = df_train.copy()
    df_te = df_test.copy()
    for col in df_tr.columns:
        if pd.api.types.is_numeric_dtype(df_tr[col]):
            fill_val = df_tr[col].mean()
            df_tr[col] = df_tr[col].fillna(fill_val)
            df_te[col] = df_te[col].fillna(fill_val)
        else:
            modes = df_tr[col].mode()
            fill_val = modes.iloc[0] if len(modes) > 0 else "MISSING"
            df_tr[col] = df_tr[col].fillna(fill_val)
            df_te[col] = df_te[col].fillna(fill_val)
    return df_tr, df_te


def _downsample_stratified(X: pd.DataFrame, y: np.ndarray, s: np.ndarray,
                           n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """分层降采样：保持标签比例，降至 n 行"""
    n = min(n, len(y))
    if n >= len(y):
        return X, y, s
    X_sub, _, y_sub, _, s_sub, _ = train_test_split(
        X, y, s, train_size=n, random_state=seed, stratify=y
    )
    return X_sub.reset_index(drop=True), y_sub, s_sub


def run_aim(dataset: str, mechanism: str, seed: int) -> dict:
    """运行 AIM 一次，返回标准结果行"""
    t0 = time.perf_counter()

    # 1. 准备数据
    data = prepare_comparison_data(dataset, mechanism, seed)
    Xtr_m = data["X_train_missing"]
    ytr = np.asarray(data["y_train"]).astype(int)
    str_bin = data["sensitive_train_bin"]
    str_raw = data["sensitive_train_raw"]
    Xte_m = data["X_test_missing"]
    yte = np.asarray(data["y_test"]).astype(int)
    ste_bin = data["sensitive_test_bin"]
    ste_raw = data["sensitive_test_raw"]
    sensitive_name = data["ds"].sensitive_name

    sensitive_col = sensitive_name

    # 2. A1 插补（AIM 邻近性计算不能有缺失）
    tr_df = Xtr_m.copy()
    tr_df[sensitive_col] = str_raw.reset_index(drop=True).values
    te_df = Xte_m.copy()
    te_df[sensitive_col] = ste_raw.reset_index(drop=True).values
    tr_imp, te_imp = _impute_raw_df(tr_df, te_df)

    # 3. 分层降采样（O(n²) 约束：Adult/Diabetes 降至 5000 行）
    if dataset in ("adult", "diabetes") and len(ytr) > 5000:
        tr_imp_sub, ytr_sub, str_sub = _downsample_stratified(
            tr_imp, ytr, str_bin, n=5000, seed=seed
        )
    else:
        tr_imp_sub = tr_imp
        ytr_sub = ytr
        str_sub = str_bin

    prep_secs = time.perf_counter() - t0
    t1 = time.perf_counter()

    # 4. 构造 AIM 输入：先对原始 DataFrame 做 one-hot 编码（AIM 假设所有列都是数值）
    #    敏感属性列必须是 0/1 数值（二值化），放在第一列
    from sklearn.preprocessing import OneHotEncoder as _OHE

    # 先确保敏感属性列是 int（0/1）
    tr_aim = tr_imp_sub.copy()
    tr_aim[sensitive_col] = str_sub.astype(int)

    # One-hot 编码所有非数值列
    cat_cols = [c for c in tr_aim.columns
                if tr_aim[c].dtype == object
                or pd.api.types.is_string_dtype(tr_aim[c])
                or tr_aim[c].dtype == "category"]

    if cat_cols:
        ohe = _OHE(handle_unknown="ignore", sparse_output=False, dtype=float)
        encoded = ohe.fit_transform(tr_aim[cat_cols])
        new_col_names = []
        for i, col in enumerate(cat_cols):
            for j in range(ohe.categories_[i].size):
                new_col_names.append(f"{col}={ohe.categories_[i][j]}")
        tr_aim = tr_aim.drop(columns=cat_cols)
        tr_aim = pd.concat([tr_aim, pd.DataFrame(encoded, columns=new_col_names, index=tr_aim.index)], axis=1)

    # 同样处理测试集
    te_aim = te_imp.copy()
    te_aim[sensitive_col] = ste_bin.astype(int)
    if cat_cols:
        encoded_te = ohe.transform(te_aim[cat_cols])
        te_aim = te_aim.drop(columns=cat_cols)
        te_aim = pd.concat([te_aim, pd.DataFrame(encoded_te, columns=new_col_names, index=te_aim.index)], axis=1)

    # 确保敏感属性在第一列
    tr_aim = tr_aim[[sensitive_col] + [c for c in tr_aim.columns if c != sensitive_col]]
    te_aim = te_aim[[sensitive_col] + [c for c in te_aim.columns if c != sensitive_col]]

    # 5. 创建 AIM 缓存目录 + os.chdir 绕过硬编码的 './data_cache'
    # AIM 硬编码了 diff/, gower/, proximity/, score/ 四个子目录
    _cache_base = Path(tempfile.gettempdir()) / f"aim_cache_{dataset}_{mechanism}_{seed}_{os.getpid()}"
    _cache_base.mkdir(parents=True, exist_ok=True)
    _cache_data = _cache_base / "data_cache"
    for sub in ("diff", "gower", "proximity", "score"):
        (_cache_data / sub).mkdir(parents=True, exist_ok=True)

    _original_cwd = os.getcwd()

    try:
        os.chdir(_cache_base)

        # 6. 运行 AIM UnLoc
        warnings.filterwarnings("ignore")
        from unloc import UnLoc

        unloc = UnLoc()  # 空构造！所有配置通过方法参数传入

        # Monkey-patch: Windows 文件名不能含 {} 或引号，AIM 把 prox_kwargs dict repr 放进了文件名
        _original_save_df = unloc._save_df
        _original_load_df = unloc._load_df

        def _safe_filename(name: str) -> str:
            # 把 {'t_con': 0.1, ...} 替换成 TC0.1_TC0_RP0.5
            safe = (name.replace("'", "").replace('"', "").replace("{", "")
                        .replace("}", "").replace(": ", "_").replace(", ", "_")
                        .replace(" ", ""))
            return safe

        def _patched_save_df(df, name, path='./data_cache/score'):
            _original_save_df(df, _safe_filename(name), path)

        def _patched_load_df(name, path='./data_cache/score'):
            return _original_load_df(_safe_filename(name), path)

        unloc._save_df = _patched_save_df
        unloc._load_df = _patched_load_df

        data_split_info = f"{dataset}_{mechanism}_seed{seed}_train"
        all_features = list(tr_aim.columns)
        prox_kwargs = {"t_con": 0.1, "t_cat": 0, "restart_prob": 0.5}

        # fit_transform → 返回 df_score（含 bias_weights）
        Xtr_arr = tr_aim.to_numpy().astype(float)
        assert (Xtr_arr[:, 0] == str_sub).all(), "AIM 要求 X[:, 0] == s"

        df_score = unloc.fit_transform(
            X=Xtr_arr,
            y=ytr_sub,
            s=str_sub,
            data_split_info=data_split_info,
            features=all_features,
            s_col=sensitive_col,
            prox_kwargs=prox_kwargs,
        )

        # df_score 应该有 'bias_weights' 列；如果没有，找权重相关列
        if "bias_weights" in df_score.columns:
            weights = df_score["bias_weights"].to_numpy()
        else:
            weight_col = [c for c in df_score.columns if "weight" in c.lower() or "bias" in c.lower()]
            if weight_col:
                weights = df_score[weight_col[0]].to_numpy()
            else:
                weights = df_score.iloc[:, 0].to_numpy()

        # fair_removal → 删除 edit_ratio=10% 最不公平的样本
        # 返回 4-tuple: (X_edited, y_edited, s_edited, n_removed)
        result = unloc.fair_removal(
            X_train=Xtr_arr,
            y_train=ytr_sub,
            s_train=str_sub,
            weights=weights,
            edit_ratio=0.1,
            how="removal",
            filter=None,
            verbose=False,
            random_state=seed,
        )
        X_edited, y_edited, s_edited, n_removed = result
        print(f"[AIM] removed {n_removed}/{len(ytr_sub)} unfair samples ({n_removed/len(ytr_sub)*100:.1f}%)")

        # 7. AIM 输入已经是全数值（one-hot 后）→ 直接训练 LR
        #    X_edited 是 numpy array，列顺序与 tr_aim 一致
        te_arr = te_aim.to_numpy().astype(float)

        clf = LogisticRegression(max_iter=10000, random_state=seed, solver="lbfgs")
        clf.fit(X_edited, y_edited)

        train_secs = time.perf_counter() - t1

        # 8. 在测试集上预测
        yp_te = clf.predict(te_arr)
        yp_prob_te = clf.predict_proba(te_arr)[:, 1]

        eval_dict = evaluate_method(yte, yp_te, yp_prob_te, ste_raw, dataset,
                                    "AIM", sensitive_bin=ste_bin)

        return build_result_row(
            method="AIM",
            dataset=dataset,
            mechanism=mechanism,
            seed=seed,
            prep_secs=prep_secs,
            train_secs=train_secs,
            n_train=len(ytr_sub),
            n_test=len(yte),
            sensitive_name=sensitive_name,
            n_groups=int(ste_raw.astype(str).nunique()),
            eval_dict=eval_dict,
        )

    finally:
        os.chdir(_original_cwd)
        if _cache_base.exists():
            shutil.rmtree(_cache_base, ignore_errors=True)
