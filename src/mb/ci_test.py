"""
条件独立检验（G 检验 / 似然比检验），支持"先验调节阈值"。

这是选题二融合方法的落点之一：文档要求
"把先验作为条件独立检验阈值的调节项"。

做法：对变量对 (x, y)，若语言模型认为它们**直接相连**，
就把该次检验的显著性水平放宽（更容易判定为依赖）；
若语言模型认为**没有直接边**，就收紧阈值（更难判定为依赖）。

    alpha_eff = alpha * exp(beta * s_xy)

其中 s_xy ∈ [-1, 1] 是先验得分（正=倾向相连，负=倾向不相连），
beta 控制先验强度。beta = 0 时退化为纯数据的经典检验。
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2


class CITest:

    def __init__(self, data, alpha=0.05, prior=None, beta=0.0):
        """
        参数
        ----
        data : (n_samples, n_vars) 的整数编码数据
        alpha : 基线显著性水平
        prior : {(i, j): score}，i < j，score ∈ [-1, 1]
        beta : 先验强度，0 表示不使用先验
        """

        self.data = self._encode(data)
        self.alpha = alpha
        self.prior = prior or {}
        self.beta = beta
        self.n_vars = self.data.shape[1]
        self._cache: dict = {}
        self.n_tests = 0

    # --------------------------------------------------------
    # 数据编码
    # --------------------------------------------------------

    @staticmethod
    def _encode(data):
        data = np.asarray(data)
        out = np.empty_like(data, dtype=np.int32)
        for col in range(data.shape[1]):
            _, codes = np.unique(data[:, col], return_inverse=True)
            out[:, col] = codes
        return out

    # --------------------------------------------------------
    # 先验调节
    # --------------------------------------------------------

    def score(self, x: int, y: int) -> float:
        key = (min(x, y), max(x, y))
        return float(self.prior.get(key, 0.0))

    def alpha_eff(self, x: int, y: int) -> float:
        if self.beta == 0.0:
            return self.alpha

        s = self.score(x, y)
        if s == 0.0:
            return self.alpha

        value = self.alpha * math.exp(self.beta * s)

        return float(min(max(value, self.alpha / 20.0), 0.5))

    # --------------------------------------------------------
    # 统计量
    # --------------------------------------------------------

    @staticmethod
    def _g_statistic(a, b):
        table = np.zeros((a.max() + 1, b.max() + 1), dtype=np.float64)
        np.add.at(table, (a, b), 1.0)

        table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]

        if table.shape[0] < 2 or table.shape[1] < 2:
            return 0.0, 1

        total = table.sum()
        expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / total

        mask = table > 0
        g = 2.0 * np.sum(table[mask] * np.log(table[mask] / expected[mask]))
        dof = (table.shape[0] - 1) * (table.shape[1] - 1)

        return float(g), int(dof)

    def raw_p_value(self, x: int, y: int, cond=()) -> float:
        """条件独立的 p 值；带缓存，同一 (x, y, S) 只算一次。"""

        cond = tuple(sorted(cond))
        key = (min(x, y), max(x, y), cond)

        if key in self._cache:
            return self._cache[key]

        self.n_tests += 1

        data = self.data

        if not cond:
            g, dof = self._g_statistic(data[:, x], data[:, y])
            p = float(chi2.sf(g, dof))
        else:
            keys = data[:, list(cond)]
            _, inverse = np.unique(keys, axis=0, return_inverse=True)

            g_total = 0.0
            dof_total = 0

            for stratum in np.unique(inverse):
                mask = inverse == stratum
                if mask.sum() < 5:
                    continue
                g_s, dof_s = self._g_statistic(
                    data[mask, x], data[mask, y]
                )
                g_total += g_s
                dof_total += dof_s

            p = float(chi2.sf(g_total, dof_total)) if dof_total > 0 else 1.0

        self._cache[key] = p

        return p

    # --------------------------------------------------------
    # 判定
    # --------------------------------------------------------

    def independent(self, x: int, y: int, cond=()) -> bool:
        """按（可能被先验调节过的）阈值判定是否条件独立。"""

        return self.raw_p_value(x, y, cond) >= self.alpha_eff(x, y)

    def association(self, x: int, y: int, cond=()) -> float:
        """
        关联强度：-log10(p)。

        加入先验后，语言模型认为直接相连的变量对会得到额外加成，
        这就是文档里"先验作为搜索的初始候选"的实现方式。
        """

        p = max(self.raw_p_value(x, y, cond), 1e-300)
        return -math.log10(p)
