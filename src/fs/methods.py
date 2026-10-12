"""
五种特征选择方法（周 2 的工具箱）。

统一接口：`rank_features(matrix, y, seed) -> 全部特征按重要性从高到低排序`
返回完整排序而不是只返回前 k 个，这样同一份结果可以被不同预算（10%、25%、50%）
重复使用——选题 1、2、4 后面都要按不同比例取特征。

五种方法：

    mutual_info        互信息（逐特征打分）
    mrmr               最大相关最小冗余（互信息贪心，减掉与已选特征的冗余）
    tree_importance    随机森林的重要性
    l1                 L1 正则逻辑回归的系数绝对值
    forward_selection  前向选择（用固定下游模型在训练集上做 3 折交叉验证）
"""

from __future__ import annotations

import inspect

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SequentialFeatureSelector,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


def _standardize(matrix: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(matrix)


def _order(scores: np.ndarray) -> np.ndarray:
    """分数越高越靠前；并列时按特征下标稳定排序。"""

    return np.argsort(-np.asarray(scores, dtype=float), kind="stable")


# ============================================================
# 1. 互信息
# ============================================================

def mutual_info(matrix, y, seed, k=None):
    scores = mutual_info_classif(
        matrix, y, random_state=seed, discrete_features=False
    )
    return _order(scores)


# ============================================================
# 2. mRMR
# ============================================================

def _mi_between_features(matrix, seed, subsample=1000):
    """
    特征之间的互信息矩阵。

    用 mutual_info_regression 一次算出一整行（特征 i 与所有特征的互信息），
    所以只需要 p 次调用，而不是 p² 次。
    """

    rng = np.random.default_rng(seed)
    n = matrix.shape[0]
    sample = matrix[rng.choice(n, subsample, replace=False)] if n > subsample else matrix

    p = matrix.shape[1]
    table = np.zeros((p, p))

    for i in range(p):
        target = sample[:, i]
        if np.std(target) == 0:
            continue
        table[i] = mutual_info_regression(
            sample, target, random_state=seed, discrete_features=False
        )

    return np.clip((table + table.T) / 2.0, 0.0, None)


def mrmr(matrix, y, seed, k=None):
    relevance = mutual_info_classif(
        matrix, y, random_state=seed, discrete_features=False
    )
    redundancy = _mi_between_features(matrix, seed)

    p = matrix.shape[1]

    remaining = list(range(p))
    selected = [int(np.argmax(relevance))]
    remaining.remove(selected[0])

    while remaining:
        best_feature = None
        best_score = -np.inf

        for feature in remaining:
            penalty = float(np.mean(redundancy[feature, selected]))
            score = relevance[feature] - penalty
            if score > best_score:
                best_score = score
                best_feature = feature

        selected.append(int(best_feature))
        remaining.remove(best_feature)

    return np.array(selected)


# ============================================================
# 3. 树模型重要性
# ============================================================

def tree_importance(matrix, y, seed, k=None):
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(matrix, y)
    return _order(model.feature_importances_)


# ============================================================
# 4. L1 逻辑回归
# ============================================================

def l1(matrix, y, seed, k=None):
    # sklearn 1.8 起 penalty='l1' 被弃用，改用 l1_ratio=1；
    # 老版本仍然只认 penalty，所以按签名自动选择。
    parameters = inspect.signature(LogisticRegression.__init__).parameters

    if "l1_ratio" in parameters:
        penalty_kwargs = {"l1_ratio": 1.0}
    else:
        penalty_kwargs = {"penalty": "l1"}

    base = LogisticRegression(
        solver="liblinear",
        C=1.0,
        max_iter=5000,
        random_state=seed,
        **penalty_kwargs,
    )

    standardized = _standardize(matrix)

    # liblinear 只做二分类；多分类时用 One-vs-Rest 包装，
    # 与它历史上自动采用的策略一致，然后把各类的系数绝对值加起来作为重要性。
    if len(np.unique(y)) > 2:
        model = OneVsRestClassifier(base)
        model.fit(standardized, y)
        scores = np.sum(
            [
                np.abs(np.atleast_2d(estimator.coef_)).sum(axis=0)
                for estimator in model.estimators_
            ],
            axis=0,
        )
    else:
        base.fit(standardized, y)
        scores = np.abs(np.atleast_2d(base.coef_)).sum(axis=0)

    return _order(scores)


# ============================================================
# 5. 前向选择
# ============================================================

def forward_selection(matrix, y, seed, k=10):
    """
    前向选择。它只产出被选中的那 k 个特征，
    剩余特征按原始下标接在后面，保证下游可以取任意前缀。
    """

    # sklearn 要求 n_features_to_select 严格小于特征总数，
    # 所以在特征数很少的数据集上要把 k 压到 p-1。
    k = int(k or 10)
    k = max(1, min(k, matrix.shape[1] - 1))

    estimator = LogisticRegression(max_iter=2000, random_state=seed)

    selector = SequentialFeatureSelector(
        estimator,
        n_features_to_select=k,
        direction="forward",
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
    )
    selector.fit(_standardize(matrix), y)

    chosen = np.flatnonzero(selector.get_support())
    rest = np.setdiff1d(np.arange(matrix.shape[1]), chosen)

    return np.concatenate([chosen, rest])


# ============================================================
# 注册表
# ============================================================

METHODS = {
    "mutual_info": mutual_info,
    "mrmr": mrmr,
    "tree_importance": tree_importance,
    "l1": l1,
    "forward_selection": forward_selection,
}

METHOD_LABELS = {
    "mutual_info": "互信息",
    "mrmr": "mRMR",
    "tree_importance": "树模型重要性",
    "l1": "L1",
    "forward_selection": "前向选择",
}
