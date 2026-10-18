"""
数据集加载与固定划分。

周 2 要求"固定划分、固定下游模型"，所以这里做三件事：

1. 从 OpenML 下载数据集，落盘缓存（一次下载，之后离线可用）；
2. 用固定随机种子生成 train / val / test 划分，并把**索引**存到磁盘，
   保证任何人任何时候跑出来的划分完全一样；
3. 把混合类型数据统一预处理成数值矩阵（类别列 one-hot、缺失值填补），
   让所有方法在同一份输入上比较。

划分比例默认 60% / 20% / 20%：
训练集用于特征选择与拟合，验证集留给选题 1、2、4 后面调参，
测试集只在最后评估时使用。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENML_DIR = PROJECT_ROOT / "data" / "openml"


# ============================================================
# 下载与缓存
# ============================================================

def _cache_paths(name: str):
    return (
        OPENML_DIR / f"{name}.csv",
        OPENML_DIR / f"{name}_splits.json",
    )


def _schema_path(name: str) -> Path:
    return OPENML_DIR / f"{name}_schema.json"


def load_dataset(name: str, openml_id: int) -> pd.DataFrame:
    """
    返回带 `__target__` 列的 DataFrame。
    第一次调用会联网下载，之后直接读缓存。
    """

    OPENML_DIR.mkdir(parents=True, exist_ok=True)
    data_path, _ = _cache_paths(name)

    if data_path.exists():
        # 缺 schema 说明是旧版本留下的缓存（那时还没记录列类型），
        # 重新取一次把类型元数据补上，保证后面不会被错误地 one-hot。
        if _schema_path(name).exists():
            return pd.read_csv(data_path)

    bunch = fetch_openml(
        data_id=openml_id,
        as_frame=True,
        parser="auto",
        # 默认会写到 C:\Users\<用户>\scikit_learn_data，
        # 这里统一收进仓库，保证换机器、换用户都能复用同一份缓存。
        data_home=str(OPENML_DIR / "_sklearn_cache"),
    )

    frame = bunch.data.copy()
    target = bunch.target

    # 目标列统一成字符串（分类任务），去掉多余的空格
    if isinstance(target, pd.DataFrame):
        target = target.iloc[:, 0]

    frame["__target__"] = target.astype(str).str.strip()
    frame.to_csv(data_path, index=False)

    # 记录每一列的真实类型。
    # sklearn 的 parser="auto" 会把取值很少的数值列推断成类别类型，
    # 直接用 pandas 的 dtype 判断会把这些整数特征错误地 one-hot 展开，
    # 所以这里以 OpenML 自己的元数据为准，并把结论落盘。
    schema = {}
    details = getattr(bunch, "details", None) or {}
    for item in details.get("features", []):
        column = item.get("name")
        if column not in frame.columns or column == "__target__":
            continue
        data_type = str(item.get("data_type", "")).lower()
        schema[column] = (
            "numeric" if data_type in ("numeric", "real", "integer") else "categorical"
        )

    _schema_path(name).write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return frame


def load_schema(name: str) -> dict:
    path = _schema_path(name)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ============================================================
# 固定划分
# ============================================================

def make_splits(
    name: str,
    n_samples: int,
    seed: int,
    val_size: float = 0.2,
    test_size: float = 0.2,
) -> dict:
    """
    生成（或读取）固定划分。划分一旦生成就固定下来，不随代码改动而变。
    """

    OPENML_DIR.mkdir(parents=True, exist_ok=True)
    _, split_path = _cache_paths(name)

    if split_path.exists():
        stored = json.loads(split_path.read_text(encoding="utf-8"))
        if stored.get("n_samples") == n_samples and stored.get("seed") == seed:
            return {k: np.array(v) for k, v in stored["splits"].items()}

    index = np.arange(n_samples)

    train_index, rest_index = train_test_split(
        index, test_size=val_size + test_size, random_state=seed, shuffle=True
    )

    relative_test = test_size / (val_size + test_size)

    val_index, test_index = train_test_split(
        rest_index, test_size=relative_test, random_state=seed, shuffle=True
    )

    splits = {
        "train": np.sort(train_index),
        "val": np.sort(val_index),
        "test": np.sort(test_index),
    }

    split_path.write_text(
        json.dumps(
            {
                "name": name,
                "n_samples": n_samples,
                "seed": seed,
                "val_size": val_size,
                "test_size": test_size,
                "splits": {k: v.tolist() for k, v in splits.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return splits


# ============================================================
# 预处理
# ============================================================

class Preprocessor:
    """
    把混合类型特征统一成数值矩阵。

    * 数值列：中位数填补缺失值；
    * 类别列：one-hot 编码（缺失单独算一类）。

    编码器只在训练集上拟合，再套用到 val / test，避免信息泄漏。
    """

    def __init__(self, schema: dict | None = None):
        self.schema = schema or {}
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.encoder: OneHotEncoder | None = None
        self.fill_values: dict[str, float] = {}
        self.feature_names: list[str] = []

    def fit(self, frame: pd.DataFrame):
        features = frame.drop(columns="__target__")

        for column in features.columns:
            declared = self.schema.get(column)
            if declared == "numeric":
                self.numeric_columns.append(column)
            elif declared == "categorical":
                self.categorical_columns.append(column)
            elif pd.api.types.is_numeric_dtype(features[column]):
                self.numeric_columns.append(column)
            else:
                self.categorical_columns.append(column)

        for column in self.numeric_columns:
            value = features[column].astype(float).median()
            self.fill_values[column] = 0.0 if pd.isna(value) else float(value)

        if self.categorical_columns:
            self.encoder = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            )
            self.encoder.fit(features[self.categorical_columns].astype(str))
            encoded = list(self.encoder.get_feature_names_out(self.categorical_columns))
        else:
            encoded = []

        self.feature_names = self.numeric_columns + encoded

        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        features = frame.drop(columns="__target__")

        blocks = []

        if self.numeric_columns:
            numeric = features[self.numeric_columns].astype(float).copy()
            for column in self.numeric_columns:
                numeric[column] = numeric[column].fillna(self.fill_values[column])
            blocks.append(numeric.to_numpy())

        if self.categorical_columns and self.encoder is not None:
            blocks.append(
                self.encoder.transform(
                    features[self.categorical_columns].astype(str)
                )
            )

        matrix = np.hstack(blocks) if blocks else np.zeros((len(frame), 0))

        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def prepare(name: str, openml_id: int, seed: int, val_size=0.2, test_size=0.2):
    """
    一步拿到：数值矩阵、标签、特征名、固定划分。
    """

    frame = load_dataset(name, openml_id)

    labels, y = np.unique(frame["__target__"], return_inverse=True)

    splits = make_splits(name, len(frame), seed, val_size, test_size)

    # 预处理只在训练集上拟合。
    #
    # 这里原先写的是 .fit(frame) —— 在**整份数据**上拟合，再套回 train/val/test。
    # 那等于让测试集的取值参与决定中位数与 one-hot 的类别集合，是实打实的泄漏，
    # 而且与本文件 Preprocessor 的说明正好相反（说明写的是"只在训练集上拟合"）。
    # 现在先切分、再只用训练行拟合，最后统一变换，说明与行为一致。
    training_frame = frame.iloc[splits["train"]]
    preprocessor = Preprocessor(load_schema(name)).fit(training_frame)
    matrix = preprocessor.transform(frame)

    return {
        "name": name,
        "matrix": matrix,
        "y": y,
        "classes": labels.tolist(),
        "feature_names": preprocessor.feature_names,
        "splits": splits,
        "n_samples": len(frame),
        "n_features": matrix.shape[1],
    }
