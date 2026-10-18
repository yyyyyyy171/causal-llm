"""
对照 bnlearn 的 IAMB 与仓库里 MATLAB（Causal-Learner）的 IAMB 基线。

为什么要单独做成脚本：周 4 产出文档里有一句定量结论
（"20 个网格里 14 个在小数点后四位一致，最大负差 −0.033"），
如果这个计算只存在于某次临时粘贴的代码里，那句话就没人能复核。
这里把口径固定下来：取 bnlearn 的 mi 检验，与 results/BN_baseline_summary.csv
的 IAMB_F1 逐格相减，一致的定义是 |Δ| < 1e-4（即四位小数一致）。

用法：
    python scripts/compare_iamb_with_matlab.py
"""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 判断"一致"的阈值：1e-4 表示小数点后四位相同。
# 不用严格相等，是因为 MATLAB 与 bnlearn 是两套独立实现，
# 浮点运算顺序不同会带来 1e-5 量级的噪声，这个量级不影响结论。
TOLERANCE = 1e-4


def load(path: Path, column: str, only=None) -> dict:
    rows = {}
    with open(path, "r", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if only and not only(record):
                continue
            rows[(record["network"], int(record["sample_size"]))] = float(record[column])
    return rows


def main() -> int:

    bnlearn = load(
        PROJECT_ROOT / "results" / "bnlearn_mb_summary.csv",
        "f1",
        only=lambda r: r["test"] == "mi",
    )
    matlab = load(PROJECT_ROOT / "results" / "BN_baseline_summary.csv", "IAMB_F1")

    keys = sorted(k for k in matlab if k in bnlearn)

    print(f"{'network':<12}{'n':>6}{'bnlearn(mi)':>13}{'MATLAB':>10}{'delta':>11}  一致")

    same, different = [], []

    for key in keys:
        delta = bnlearn[key] - matlab[key]
        is_same = abs(delta) < TOLERANCE
        (same if is_same else different).append((key, delta))
        print(
            f"{key[0]:<12}{key[1]:>6}{bnlearn[key]:>13.4f}{matlab[key]:>10.4f}"
            f"{delta:>+11.4f}  {'是' if is_same else '否'}"
        )

    print()
    print(f"可比网格 {len(keys)} 个：四位小数一致 {len(same)} 个，不一致 {len(different)} 个")

    if different:
        worst = min(different, key=lambda item: item[1])
        names = "、".join(f"{k[0]} n={k[1]}" for k, _ in different)
        print(f"不一致的网格：{names}")
        print(f"最大负差：{worst[1]:+.4f}（{worst[0][0]} n={worst[0][1]}）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
