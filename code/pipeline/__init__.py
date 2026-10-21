"""配置化实验管线（T3 项目）

模块职责
    config        配置与冻结参数
    data          5 个数据集的加载与清洗
    missing       MCAR / MAR 缺失注入
    features      类别编码（one-hot）与数值标准化
    impute        因子 A：A1 均值 / A2 MICE / A3 KNN
    imbalance     因子 B：B1 无处理 / B2 类别权重 / B3 SMOTE
    intervention  因子 C：C1 无干预 / C2 Reweighing / C3 DIR
    model         LR / LGBM 训练与预测
    metrics       性能 + 公平指标
    run_cell      单单元格执行

固定处理顺序：A（插补） → B（不平衡） → C（公平性干预）。
"""
from __future__ import annotations

__all__ = ["config", "data", "missing", "features", "impute",
           "imbalance", "intervention", "model", "metrics", "run_cell"]
