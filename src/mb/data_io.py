"""
读取实验数据、真值马尔可夫毯、以及语言模型先验。
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"


def data_path(network: str, size: int) -> Path:
    return DATA_ROOT / network / f"{network}_n{size}.txt"


def load_data(network: str, size: int):
    """返回 (data, variables)。data 是 (n, p) 的字符串数组。"""

    path = data_path(network, size)
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().strip().split("\t")
        rows = [line.rstrip("\n").split("\t") for line in handle if line.strip()]

    return np.array(rows, dtype=object), header


def load_true_mb(network: str):
    """真值马尔可夫毯邻接矩阵（对称 0/1）。"""

    path = DATA_ROOT / network / f"{network}_true_MB_matrix.txt"

    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append([int(x) for x in line.split()])

    return np.array(rows, dtype=int)


def true_mb_sets(network: str) -> list[set[int]]:
    matrix = load_true_mb(network)
    return [set(np.flatnonzero(matrix[i])) for i in range(matrix.shape[0])]


def load_prior_scores(network: str, mode: str, min_consistency: float = 0.0):
    """
    读取语言模型先验，转成 {(i, j): score}，i < j。

    score 的构造：
        模型认为直接相连      -> +confidence
        模型认为没有直接边    -> -confidence
        uncertain            -> 0

    另外只保留 consistency >= min_consistency 的判断，
    用来测试"只要稳定先验"是否更好。
    """

    csv_path = DATA_ROOT / network / f"{network}_LLM_prior_{mode}_summary.csv"
    if not csv_path.exists():
        return {}

    _, variables = load_data(network, 100)
    index = {name: i for i, name in enumerate(variables)}

    scores: dict[tuple[int, int], float] = {}

    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:

            a = row["variable_a"]
            b = row["variable_b"]

            if a not in index or b not in index:
                continue

            consistency = float(row.get("consistency", 1.0))
            if consistency < min_consistency:
                continue

            confidence = float(row.get("final_confidence", 0.0) or 0.0)
            confidence = float(min(max(confidence, 0.0), 1.0))

            direct_edge = str(row.get("direct_edge", "")).strip().lower() == "true"
            relation = str(row.get("final_relation", "")).strip()

            if direct_edge or relation in ("A_to_B", "B_to_A"):
                score = confidence
            elif relation == "no_direct_edge":
                score = -confidence
            else:
                score = 0.0

            i, j = index[a], index[b]
            scores[(min(i, j), max(i, j))] = score

    return scores


def f1_of(predicted: set, truth: set):
    if not predicted and not truth:
        return 1.0, 1.0, 1.0

    tp = len(predicted & truth)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return precision, recall, f1
