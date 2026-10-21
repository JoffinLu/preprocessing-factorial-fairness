"""实验配置与默认值（T3 项目：配置化实验管线）

设计说明
    - 全部冻结参数集中在 `Config` 中，`configs/mvp.yaml` 是唯一真源。
    - 代码内**不出现任何绝对路径**：ROOT 一律由 `Path(__file__).resolve()` 推导。
    - 因子水平（A/B/C/M）、数据集、种子、模型均为冻结设计，不得改动。

冻结的四因子全因子设计（3×3×3×2 = 54 组合）
    因子 A 缺失值处理   : A1 均值插补 / A2 MICE / A3 KNN
    因子 B 类别不平衡   : B1 无处理 / B2 类别权重 / B3 SMOTE
    因子 C 公平性干预   : C1 无干预 / C2 Reweighing / C3 DisparateImpactRemover
    因子 M 缺失机制     : M1 MCAR / M2 MAR（缺失率固定 30%）

数据清洗与处理的固定顺序：A（插补） → B（不平衡） → C（公平性干预）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterator

import yaml

# ---------------------------------------------------------------------------
# 路径（全部相对本文件推导，禁止绝对路径）
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parents[2]
CODE_DIR: Path = ROOT / "code"
CONFIG_DIR: Path = ROOT / "configs"
RESULTS_DIR: Path = ROOT / "results"
DATA_RAW: Path = ROOT / "data" / "raw"
DEFAULT_CONFIG: Path = CONFIG_DIR / "mvp.yaml"

# ---------------------------------------------------------------------------
# 冻结的因子水平与常量
# ---------------------------------------------------------------------------
DATASETS: tuple[str, ...] = ("adult", "german", "bank", "diabetes", "compas")
MECHANISMS: tuple[str, ...] = ("M1", "M2")            # M1=MCAR, M2=MAR
A_LEVELS: tuple[str, ...] = ("A1", "A2", "A3")          # 插补
B_LEVELS: tuple[str, ...] = ("B1", "B2", "B3")          # 不平衡
C_LEVELS: tuple[str, ...] = ("C1", "C2", "C3")          # 公平性干预
MODELS: tuple[str, ...] = ("LR", "LGBM")
SEEDS: tuple[int, ...] = (0, 1, 2)

MECHANISM_TO_NAME: dict[str, str] = {"M1": "MCAR", "M2": "MAR"}

FACTOR_DESC: dict[str, dict[str, str]] = {
    "A": {"A1": "均值插补", "A2": "MICE 多重插补", "A3": "KNN 插补"},
    "B": {"B1": "无处理", "B2": "类别权重", "B3": "SMOTE"},
    "C": {"C1": "无干预", "C2": "样本再加权(Reweighing)",
          "C3": "特征去相关(DisparateImpactRemover)"},
    "M": {"M1": "MCAR", "M2": "MAR"},
}


@dataclass(frozen=True)
class Config:
    """实验配置（不可变数据类，可直接传给子进程）"""

    datasets: tuple[str, ...] = DATASETS
    mechanisms: tuple[str, ...] = MECHANISMS
    a_levels: tuple[str, ...] = A_LEVELS
    b_levels: tuple[str, ...] = B_LEVELS
    c_levels: tuple[str, ...] = C_LEVELS
    models: tuple[str, ...] = MODELS
    seeds: tuple[int, ...] = SEEDS

    missing_rate: float = 0.30          # 缺失率固定 30%
    test_size: float = 0.30             # 分层 7:3 划分
    random_state: int = 42              # 仅用于与划分无关的兜底随机性
    categorical_missing: str = "as_category"   # 分类列缺失处理：as_category / mode
    max_cardinality: int = 20           # 分类列按训练集频次保留的最大取值数（Gate 2b）
    use_natural_missing: bool = False   # True 时跳过节 3 规则 7 的「置零」步骤
    out_csv: str = "results/full_results.csv"

    # ---------------- 校验与构造 ----------------
    def __post_init__(self) -> None:
        if self.categorical_missing not in ("as_category", "mode"):
            raise ValueError("categorical_missing 只能为 'as_category' 或 'mode'")
        if not 0.0 < self.missing_rate < 1.0:
            raise ValueError("missing_rate 须在 (0,1) 内")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size 须在 (0,1) 内")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """从 YAML 读取配置；只接受 Config 中已声明的字段"""
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        allowed = {f.name for f in fields(cls)}
        kw: dict[str, Any] = {}
        tuple_fields = {"datasets", "mechanisms", "a_levels", "b_levels",
                        "c_levels", "models", "seeds"}
        for k, v in raw.items():
            if k not in allowed or v is None:
                continue
            kw[k] = tuple(v) if k in tuple_fields else v
        return cls(**kw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ---------------- 组合枚举与路径 ----------------
    def out_path(self) -> Path:
        """把相对路径 out_csv 解析到项目根，规避绝对路径写死"""
        p = Path(self.out_csv)
        return p if p.is_absolute() else (ROOT / p)

    def iter_combos(self) -> Iterator[dict[str, Any]]:
        """按固定顺序枚举全部单元格（确定性）"""
        for dataset in self.datasets:
            for mechanism in self.mechanisms:
                for a in self.a_levels:
                    for b in self.b_levels:
                        for c in self.c_levels:
                            for seed in self.seeds:
                                for model in self.models:
                                    yield {"dataset": dataset, "mechanism": mechanism,
                                           "A": a, "B": b, "C": c, "seed": seed,
                                           "model": model}

    def n_combos(self) -> int:
        return (len(self.datasets) * len(self.mechanisms) * len(self.a_levels)
                * len(self.b_levels) * len(self.c_levels) * len(self.seeds)
                * len(self.models))


# ---------------------------------------------------------------------------
# 自检：python code/pipeline/config.py
# ---------------------------------------------------------------------------
def _self_test() -> None:
    cfg = Config()
    assert cfg.n_combos() == 1620, cfg.n_combos()
    n_pre = (len(cfg.datasets) * len(cfg.mechanisms) * len(cfg.a_levels)
             * len(cfg.b_levels) * len(cfg.c_levels) * len(cfg.seeds))
    assert n_pre == 810, n_pre
    combos = list(cfg.iter_combos())
    assert len(combos) == 1620
    assert len({tuple(sorted(d.items())) for d in combos}) == 1620, "组合存在重复"
    # 相对路径可被解析到 ROOT 且不含绝对前缀
    assert not Path(cfg.out_csv).is_absolute()
    assert cfg.out_path() == ROOT / "results" / "full_results.csv"
    assert str(ROOT).endswith("T3-预处理管线公平性")
    # YAML 往返一致性
    if DEFAULT_CONFIG.exists():
        cfg_yaml = Config.from_yaml(DEFAULT_CONFIG)
        assert cfg_yaml.n_combos() == 1620
        assert cfg_yaml.a_levels == A_LEVELS
    print(f"config 自检通过：{cfg.n_combos()} 个单元格组合，"
          f"其中预处理 {n_pre} 次、训练-评估 {cfg.n_combos()} 次")
    print(f"ROOT = {ROOT}")


if __name__ == "__main__":
    _self_test()
