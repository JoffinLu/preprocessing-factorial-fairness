"""TabFairGDT 对比臂适配器（ICDM 2025, commit f93cefb）

TabFairGDT = Tabular Fair Synthetic Data Generator
原始仓库：https://github.com/Panagiotou/TABFAIRGDT
PyPI：pip install tabfairgdt（已验证可安装 0.1.1）

方法概述
  自回归 CART 决策树生成公平感知的合成表格数据。
  每个特征用一棵 CART 建模（条件于已生成的前序特征），
  目标列用公平 CART（调整叶节点概率分布减小人口均等差）。
  在合成训练数据上训练 LR 预测模型。

适配要点（协议 S3 §2.2 & §4）：
  1. TabFairGDT 输入要求 DataFrame，目标与敏感列必须是 Categorical dtype
  2. TabFairGDT 不处理缺失 → 先做 A1 均值/众数插补（对原始 DataFrame，未编码）
  3. 敏感属性列名需要传递给 TabFairGDT constructor
  4. 训练模型用 LogisticRegression（官方 example 默认）
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import build_result_row, evaluate_method, prepare_comparison_data  # noqa: E402


def _impute_raw_df(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对原始 DataFrame 做 A1 均值/众数插补（训练集 fit，测试集 transform）

    - 数值列：用训练集均值
    - 分类列（object/category/str）：用训练集众数（取第一个众数）
    """
    df_tr = df_train.copy()
    df_te = df_test.copy()

    for col in df_tr.columns:
        if pd.api.types.is_numeric_dtype(df_tr[col]):
            # 数值列：训练集均值
            fill_val = df_tr[col].mean()
            df_tr[col] = df_tr[col].fillna(fill_val)
            df_te[col] = df_te[col].fillna(fill_val)
        else:
            # 分类列：训练集众数
            modes = df_tr[col].mode()
            fill_val = modes.iloc[0] if len(modes) > 0 else "MISSING"
            df_tr[col] = df_tr[col].fillna(fill_val)
            df_te[col] = df_te[col].fillna(fill_val)

    return df_tr, df_te


def _infer_dtype_map(df: pd.DataFrame, target_col: str, sensitive_col: str) -> dict:
    """自动推导 TabFairGDT 的 dtype_map

    敏感属性和目标列必须是 'category'；其余列根据 dtype 自动映射。
    """
    dtype_map = {}
    for col in df.columns:
        if col == target_col or col == sensitive_col:
            dtype_map[col] = "category"
        elif pd.api.types.is_float_dtype(df[col]):
            dtype_map[col] = "float"
        elif pd.api.types.is_integer_dtype(df[col]):
            dtype_map[col] = "int"
        else:
            dtype_map[col] = "category"
    return dtype_map


def run_tabfairgdt(dataset: str, mechanism: str, seed: int) -> dict:
    """运行 TabFairGDT 一次，返回标准结果行"""
    t0 = time.perf_counter()

    # 1. 准备数据（主管线加载+划分+缺失注入）
    data = prepare_comparison_data(dataset, mechanism, seed)
    Xtr_m = data["X_train_missing"]   # 原始 DataFrame（已注入缺失）
    ytr = data["y_train"]
    str_raw = data["sensitive_train_raw"]
    Xte_raw = data["X_test_raw"]       # 测试集原始 DataFrame
    yte = data["y_test"]
    ste_raw = data["sensitive_test_raw"]
    ste_bin = data["sensitive_test_bin"]
    sensitive_name = data["ds"].sensitive_name

    # 2. 对原始 DataFrame 做 A1 插补（TabFairGDT 不处理缺失 → 适配代价）
    #    注意：保留原始列名和 dtype，不编码、不标准化
    sensitive_col = sensitive_name
    target_col = "Y"

    # 构造 TabFairGDT 输入：敏感属性列加回 Xtr_m
    tr_df = Xtr_m.copy()
    tr_df[sensitive_col] = str_raw.reset_index(drop=True).values
    tr_df[target_col] = np.asarray(ytr).astype(int)

    te_df = Xte_raw.copy()
    te_df[sensitive_col] = ste_raw.reset_index(drop=True).values
    te_df[target_col] = np.asarray(yte).astype(int)

    # A1 插补（对原始 DataFrame，未编码）
    tr_imp, te_imp = _impute_raw_df(tr_df, te_df)

    # 确保敏感属性和目标列是 Categorical dtype（TabFairGDT 要求）
    tr_imp = tr_imp.astype({sensitive_col: "category", target_col: "category"})
    te_imp = te_imp.astype({sensitive_col: "category", target_col: "category"})

    prep_secs = time.perf_counter() - t0
    t1 = time.perf_counter()

    # 3. 初始化 TabFairGDT（协议 S3 §2.2 官方默认配置）
    from tabfairgdt import TABFAIRGDT

    dtype_map = _infer_dtype_map(tr_imp, target_col, sensitive_col)

    generator = TABFAIRGDT(
        protected_attribute=sensitive_col,
        target=target_col,
        dtype_map=dtype_map,
        criterion="dp",
        acc_threshold=0.05,
        seed=seed,
        parallel=False,
        verbose=False,
    )

    # 4. 拟合 + 生成合成训练数据（官方推荐 lamda=0.5）
    generator.fit(tr_imp, lamda=0.5)
    synthetic_df = generator.generate(k=len(ytr))

    # 5. 从合成 DataFrame 提取特征/标签 → 编码 → 训练 LR 预测模型
    synth_X_raw = synthetic_df.drop(columns=[target_col, sensitive_col])
    synth_y = synthetic_df[target_col].astype(int).to_numpy()

    # TabFairGDT 保留了原始 dtype（分类列是 category/object，值为字符串）
    # 需要对训练/测试的特征做统一编码（分类 → one-hot，数值保持）
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    cat_cols = [c for c in synth_X_raw.columns
                if synth_X_raw[c].dtype == "category"
                or pd.api.types.is_string_dtype(synth_X_raw[c])
                or synth_X_raw[c].dtype == object]
    num_cols = [c for c in synth_X_raw.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    synth_X = preprocess.fit_transform(synth_X_raw)

    clf = LogisticRegression(max_iter=10000, random_state=seed, solver="lbfgs")
    clf.fit(synth_X, synth_y)

    train_secs = time.perf_counter() - t1

    # 6. 在测试集上预测（用同一个编码器 transform）
    te_X_raw = te_imp.drop(columns=[target_col, sensitive_col])
    te_X = preprocess.transform(te_X_raw)
    yp_te = clf.predict(te_X)
    yp_prob_te = clf.predict_proba(te_X)[:, 1]

    # 7. 用主管线评估：公平性用**原始多类别**敏感属性（与主表口径一致）
    yte_arr = np.asarray(yte).astype(int)
    eval_dict = evaluate_method(yte_arr, yp_te, yp_prob_te, ste_raw, dataset,
                                "TabFairGDT", sensitive_bin=ste_bin)

    return build_result_row(
        method="TabFairGDT",
        dataset=dataset,
        mechanism=mechanism,
        seed=seed,
        prep_secs=prep_secs,
        train_secs=train_secs,
        n_train=len(ytr),
        n_test=len(yte),
        sensitive_name=sensitive_name,
        n_groups=int(ste_raw.astype(str).nunique()),
        eval_dict=eval_dict,
    )
