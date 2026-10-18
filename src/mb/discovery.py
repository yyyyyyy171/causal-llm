"""
马尔可夫毯发现：IAMB 与 HITON-MB（Python 版）。

两个算法都支持注入语言模型先验：

* 先验作为**阈值调节项**（由 CITest 的 beta 实现）
* 先验作为**搜索初始候选/优先级**（由这里的 gamma 实现：
  选下一个变量时，给语言模型认为直接相连的变量加一个加成）

gamma = 0 时就是纯数据的经典算法，作为基线。
"""

from __future__ import annotations

import statistics
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


def _pick_next(ci, target, candidates, current, prior, gamma, normalize=False):
    """
    按关联强度（可加先验加成）挑下一个进入 MB 的变量。

    normalize=True 时把先验分数换算到"当前候选集的关联强度尺度"上：

        S(y) = A(y) + gamma * P(y) * scale,   scale = 当前候选集 A 的中位数

    为什么需要它：A 是 -log10(p)，量级 0~10；先验分数在 [-1, 1]。
    不归一化时 gamma=1.5 对排序几乎没有影响，先验等于没接上。
    归一化后 gamma 才有"一个最大置信度的先验，相当于挪动一个中位关联强度"
    这样可解释的含义，并且不会因为不同网络、不同搜索阶段 A 的量级不同
    而让先验的实际权重剧烈变化。

    先验分数的正负号与相对大小保持不变——它本来就在 [-1, 1]，
    再做一次标准化只会破坏"正=倾向在毯内、负=倾向不在"的含义。
    """

    if not gamma:
        return max(candidates, key=lambda y: ci.association(target, y, tuple(current)))

    associations = {y: ci.association(target, y, tuple(current)) for y in candidates}

    if normalize:
        scale = statistics.median(associations.values())
        if not scale > 0:            # 全部 p≈1 时中位数为 0，先验无参照尺度
            scale = max(associations.values()) or 1.0
        return max(
            candidates,
            key=lambda y: associations[y] + gamma * ci.score(target, y) * scale,
        )

    return max(
        candidates,
        key=lambda y: associations[y] + gamma * ci.score(target, y),
    )


def iamb(ci, target, n_vars, max_k=3, prior=None, gamma=0.0,
         normalize=False, allowed=None, trace=None):
    """
    IAMB：增量式关联马尔可夫毯。

    max_k 只是为了和其它算法保持同样的调用签名（IAMB 不使用它）。
    """

    mb = []
    pool = range(n_vars) if allowed is None else allowed
    remaining = set(pool) - {target}

    while remaining:

        y = _pick_next(ci, target, remaining, mb, prior, gamma, normalize)

        if ci.independent(target, y, tuple(mb)):
            break

        if trace is not None:
            trace.append({
                "chosen": y,
                "candidate_count": len(remaining),
                "p_value": ci.raw_p_value(target, y, tuple(mb)),
                "mb_before": sorted(mb),
            })

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
