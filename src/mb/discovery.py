"""
马尔可夫毯发现：IAMB 与 HITON-MB（Python 版）。

两个算法都支持注入语言模型先验：

* 先验作为**阈值调节项**（由 CITest 的 beta 实现）
* 先验作为**搜索初始候选/优先级**（由这里的 gamma 实现：
  选下一个变量时，给语言模型认为直接相连的变量加一个加成）

gamma = 0 时就是纯数据的经典算法，作为基线。
"""

from __future__ import annotations

from itertools import combinations


def _subset_independent(ci, target, y, candidates, max_k, cap=12):
    """
    是否存在大小不超过 max_k 的子集 S ⊆ candidates，使 y ⊥ target | S。

    candidates 过多时只取关联最强的 cap 个（标准实现里的常见剪枝），
    否则子集枚举会爆炸。
    """

    pool = list(candidates)

    if len(pool) > cap:
        pool = sorted(
            pool,
            key=lambda v: -ci.association(target, y, (v,)),
        )[:cap]

    for size in range(min(max_k, len(pool)) + 1):
        for subset in combinations(pool, size):
            if ci.independent(target, y, subset):
                return True

    return False


def _pick_next(ci, target, candidates, current, prior, gamma):
    """按关联强度（可加先验加成）挑下一个进入 MB 的变量。"""

    def score(y):
        association = ci.association(target, y, tuple(current))
        if gamma:
            association += gamma * ci.score(target, y)
        return association

    return max(candidates, key=score)


def iamb(ci, target, n_vars, max_k=3, prior=None, gamma=0.0):
    """
    IAMB：增量式关联马尔可夫毯。

    max_k 只是为了和其它算法保持同样的调用签名（IAMB 不使用它）。
    """

    mb = []
    remaining = set(range(n_vars)) - {target}

    while remaining:

        y = _pick_next(ci, target, remaining, mb, prior, gamma)

        if ci.independent(target, y, tuple(mb)):
            break

        mb.append(y)
        remaining.discard(y)

        # 后向剪枝：删掉冗余变量
        changed = True
        while changed:
            changed = False
            for x in list(mb):
                rest = tuple(v for v in mb if v != x)
                if ci.independent(target, x, rest):
                    mb.remove(x)
                    changed = True
                    break

    return set(mb)


def hiton_pc(ci, target, n_vars, max_k=3, prior=None, gamma=0.0):
    """HITON-PC：先求目标的父子集（parents and children）。"""

    pc = []
    remaining = set(range(n_vars)) - {target}

    while remaining:

        # 1) 若存在 S ⊆ PC 使 y ⊥ target | S，则 y 被剔除
        for y in list(remaining):
            if _subset_independent(ci, target, y, pc, max_k):
                remaining.discard(y)

        if not remaining:
            break

        # 2) 把与 target 关联最强（可加先验加成）的变量并入 PC
        y = _pick_next(ci, target, remaining, pc, prior, gamma)
        pc.append(y)
        remaining.discard(y)

    # 3) 后向剪枝
    for x in list(pc):
        rest = [v for v in pc if v != x]
        if _subset_independent(ci, target, x, rest, max_k):
            pc.remove(x)

    return pc


def hiton_mb(ci, target, n_vars, max_k=3, prior=None, gamma=0.0):
    """
    HITON-MB：父子集 ∪ 每个父子的父子集（补上配偶），再统一剪枝。
    """

    pc = hiton_pc(ci, target, n_vars, max_k, prior, gamma)

    mb = set(pc)

    for x in pc:
        mb.update(hiton_pc(ci, x, n_vars, max_k, prior, gamma))

    mb.discard(target)

    for y in list(mb):
        rest = [v for v in mb if v != y]
        if _subset_independent(ci, target, y, rest, max_k):
            mb.discard(y)

    return mb


METHODS = {
    "IAMB": iamb,
    "HITON-MB": hiton_mb,
}
