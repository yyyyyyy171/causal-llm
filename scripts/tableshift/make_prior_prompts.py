"""
为 TableShift 任务生成"特征 → 标签"的角色提问（每个任务一次，量很小）。

为什么只问特征与标签、不枚举特征两两组合：
软约束在 IAMB 里只用 (候选变量, 目标) 的先验分数来调整候选排序，
特征之间的先验不进入排序分数。按需提问可以把 8B 的生成量压到
每个任务 30–50 条 × 5 次采样，T4 几分钟就能跑完。

用法：
    python scripts/tableshift/make_prior_prompts.py --data data/tableshift
产出：
    data/tableshift/<task>/prior_prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TEMPLATE = """You are helping with causal feature selection for a tabular prediction task.

Task: {task}
Target variable: {target} (binary)
Candidate feature: {feature} ({dtype}; {profile})

Question: In the real-world data-generating process for this task, what is the role of
"{feature}" with respect to the target "{target}"?

Choose exactly one:
- "parent": the feature causes the target
- "child": the target causes the feature
- "spouse": neither causes the other, but they share a common cause (they would still be
  dependent given every other variable)
- "none": none of the above; the feature is not in the target's Markov blanket
- "uncertain": the name and description do not let you decide

Answer with JSON only:
{{"feature": "{feature}", "target": "{target}", "role": "parent|child|spouse|none|uncertain",
  "confidence": 0.0-1.0, "reason": "one sentence"}}
"""


def profile(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        s = series.dropna()
        if s.empty:
            return "all missing"
        return f"range {s.min():.3g} to {s.max():.3g}"
    counts = series.astype("string").value_counts().head(5)
    return "values: " + ", ".join(f"{k} ({v})" for k, v in counts.items())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/tableshift")
    parser.add_argument("--target", default="target")
    args = parser.parse_args()

    root = Path(args.data)
    written = 0
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        train_path = task_dir / "train.csv"
        if not train_path.exists():
            continue
        train = pd.read_csv(train_path, nrows=5000)
        if args.target not in train.columns:
            print(f"[{task_dir.name}] 没有目标列 {args.target}，跳过")
            continue
        out_path = task_dir / "prior_prompts.jsonl"
        count = 0
        with open(out_path, "w", encoding="utf-8") as handle:
            for column in train.columns:
                if column == args.target:
                    continue
                record = {
                    "task": task_dir.name,
                    "feature": column,
                    "target": args.target,
                    "prompt": TEMPLATE.format(
                        task=task_dir.name.replace("_", " "),
                        target=args.target,
                        feature=column,
                        dtype="numeric" if pd.api.types.is_numeric_dtype(train[column]) else "categorical",
                        profile=profile(train[column]),
                    ),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        print(f"[{task_dir.name}] {count} 条 -> {out_path}")
        written += count
    if not written:
        raise SystemExit("没有生成任何提示，先检查 --data 目录")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
