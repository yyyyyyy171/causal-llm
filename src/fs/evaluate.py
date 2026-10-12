"""
固定下游模型的评测。

周 2 要求"固定划分、固定下游模型"，所以这里把下游模型写死成
标准化 + 逻辑回归：对所有数据集、所有方法都完全一致。
换下游模型属于下一层实验（选题 1 要换成 TabPFN / XGBoost），
到时候只需要在配置里换名字，评测流程不动。
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DOWNSTREAM_MODELS = {
    "logistic_regression": lambda seed: Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=seed)),
        ]
    ),
}


def evaluate(
    matrix: np.ndarray,
    y: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    columns,
    seed: int,
    model_name: str = "logistic_regression",
):
    """
    在给定特征子集上训练并评估，返回准确率与耗时。
    columns 为空表示使用全部特征（基线）。
    """

    if model_name not in DOWNSTREAM_MODELS:
        raise ValueError(f"未知的下游模型: {model_name}")

    columns = np.asarray(columns, dtype=int)

    if columns.size == 0:
        train_x = matrix[train_index]
        test_x = matrix[test_index]
    else:
        train_x = matrix[np.ix_(train_index, columns)]
        test_x = matrix[np.ix_(test_index, columns)]

    model = DOWNSTREAM_MODELS[model_name](seed)

    started = time.time()
    model.fit(train_x, y[train_index])
    fit_seconds = time.time() - started

    accuracy = float(model.score(test_x, y[test_index]))

    return {
        "accuracy": accuracy,
        "n_selected": int(columns.size),
        "fit_seconds": fit_seconds,
    }
