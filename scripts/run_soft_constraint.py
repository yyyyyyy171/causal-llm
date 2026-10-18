"""
软约束主实验：把先验作为候选排序信号，而不是永久排除。

四组（同一套 Python IAMB、同一份数据、同一种子）：
  baseline     保留全部候选，不用先验
  prior-hard   排除先验判为 none 的变量（硬排除，用于复核旧结论）
  prior-soft   保留全部候选，用归一化后的排序加成
  random-soft  保留全部候选，打乱"变量 ↔ 先验分数"的对应关系

排序分数：S(y) = A(y) + gamma * P(y) * scale
  A(y)  = 当前搜索阶段、当前候选集的关联强度（-log10 p）
  scale = 当前候选集 A 的中位数
归一化是必需的：A 量级 0~10、P 在 [-1,1]，不归一化 gamma 基本没作用。

用法：
  python scripts/run_soft_constraint.py --networks CHILD,INSURANCE --repeats 20
  python scripts/run_soft_constraint.py --smoke        # 只跑 2 次，验证三道关卡
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.run_log import finish_run, start_run  # noqa: E402
from src.mb.ci_test import CITest  # noqa: E402
from src.mb.data_io import load_data, true_mb_sets  # noqa: E402
from src.mb.discovery import iamb  # noqa: E402


MODEL_TAG = "qwen3-8b-awq"
METRICS = ("precision", "recall", "f1", "recall_adjacent",
           "recall_spouse", "n_selected", "excluded", "first_step_candidates")


def load_role_prior(network: str, mode: str = "real"):
    """读 role 版先验，返回 (scores, none_pairs)。正=在毯内，负=不在。"""

    path = (PROJECT_ROOT / "data" / network
            / f"{network}_LLM_role_{mode}_{MODEL_TAG}_summary.csv")
    if not path.exists():
        raise SystemExit(f"缺少 role 先验：{path}")

    _, variables = load_data(network, 100)
    index = {name: i for i, name in enumerate(variables)}

    scores: dict[tuple[int, int], float] = {}
    none_pairs: set[tuple[int, int]] = set()

    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            a, b = row["variable_a"], row["variable_b"]
            if a not in index or b not in index:
                continue
            role = str(row["final_role"]).strip().lower()
            confidence = min(max(float(row["final_confidence"] or 0.0), 0.0), 1.0)
            key = (min(index[a], index[b]), max(index[a], index[b]))
            if role in ("parent", "child", "spouse"):
                scores[key] = confidence
            elif role == "none":
                scores[key] = -confidence
                none_pairs.add(key)
            else:
                scores[key] = 0.0

    return scores, none_pairs


def load_adjacency(network: str, variables: list[str]) -> np.ndarray:
    path = PROJECT_ROOT / "data" / network / f"{network}_dag_edges.csv"
    index = {name: i for i, name in enumerate(variables)}
    matrix = np.zeros((len(variables), len(variables)), dtype=bool)
    if not path.exists():
        return matrix
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["parent"] in index and row["child"] in index:
                i, j = index[row["parent"]], index[row["child"]]
                matrix[i, j] = matrix[j, i] = True
    return matrix


def subsample(data, size: int, seed: int):
    if size >= len(data):
        return data
    rng = np.random.default_rng(seed)
    return data[rng.choice(len(data), size=size, replace=False)]


def shuffle_prior(prior: dict, seed: int) -> dict:
    """只打乱对应关系：分数集合不变，变量与分数的配对被打乱。"""

    if not prior:
        return {}
    keys = list(prior)
    rng = np.random.default_rng(seed)
    values = list(rng.permutation([prior[k] for k in keys]))
    return dict(zip(keys, values))


def digest(values) -> str:
    """给"本次重复用的数据"和"先验分值多重集"各留一个指纹，便于事后审计。"""

    text = ",".join(str(int(v)) for v in values)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def sample_digest(sample) -> str:
    """样本指纹：按行按列拼成文本再取 sha1，跨进程稳定。"""

    text = "\n".join("\t".join(str(v) for v in row) for row in sample)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def score_target(predicted: set, truth: set):
    if not predicted and not truth:
        return 1.0, 1.0, 1.0
    tp = len(predicted & truth)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def run_variant(sample, truth_sets, adjacency, prior, gamma, n_vars,
                hard_none=None):
    """跑一个变体，返回逐目标指标、搜索轨迹与统计量。"""

    ci = CITest(sample, alpha=0.05, prior=prior or {}, beta=0.0)
    per_target, traces = [], []
    excluded_total = 0

    for target in range(n_vars):

        allowed, excluded = None, 0
        if hard_none is not None:
            allowed = [v for v in range(n_vars)
                       if v == target
                       or (min(target, v), max(target, v)) not in hard_none]
            excluded = n_vars - len(allowed)
            excluded_total += excluded

        trace: list = []
        found = iamb(ci, target, n_vars, max_k=3, prior=prior, gamma=gamma,
                     normalize=True, allowed=allowed, trace=trace)
        traces.append(trace)

        truth = truth_sets[target]
        adjacent_truth = {v for v in truth if adjacency[target, v]}
        spouse_truth = truth - adjacent_truth

        precision, recall, f1 = score_target(found, truth)
        per_target.append({
            "precision": precision, "recall": recall, "f1": f1,
            "recall_adjacent": (len(found & adjacent_truth) / len(adjacent_truth)
                                if adjacent_truth else np.nan),
            "recall_spouse": (len(found & spouse_truth) / len(spouse_truth)
                              if spouse_truth else np.nan),
            "n_selected": len(found),
            "excluded": excluded,
            "first_step_candidates": trace[0]["candidate_count"] if trace else n_vars - 1,
        })

    return {"per_target": per_target, "traces": traces,
            "excluded_total": excluded_total, "n_ci_tests": ci.n_tests}


def mean_of(rows, key):
    values = [r[key] for r in rows if r[key] == r[key]]
    return float(np.mean(values)) if values else float("nan")


def main() -> int:

    parser = argparse.ArgumentParser()
    parser.add_argument("--networks", default="CHILD,INSURANCE")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--shuffles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--tag", default="soft_n100")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    repeats = 2 if args.smoke else args.repeats
    log_path = start_run("run_soft_constraint")
    print(f"日志: {log_path}")

    detail_rows, gate_rows = [], []

    for network in [x.strip() for x in args.networks.split(",") if x.strip()]:

        data, variables = load_data(network, 5000)
        truth_sets = true_mb_sets(network)
        adjacency = load_adjacency(network, variables)
        prior, none_pairs = load_role_prior(network)
        n_vars = len(variables)

        print(f"\n=== {network}: {n_vars} 变量，先验 {len(prior)} 对"
              f"（none {len(none_pairs)} 对）===")

        for repeat in range(1, repeats + 1):

            sample = subsample(data, args.size, args.seed + repeat)
            sample_hash = sample_digest(sample)
            prior_hash = digest(sorted(prior.values()))
            results = {}

            # 关卡 1：gamma=0 与"完全没有先验"必须逐位一致（含搜索轨迹）
            baseline = run_variant(sample, truth_sets, adjacency, {}, 0.0, n_vars)
            zero_prior = run_variant(sample, truth_sets, adjacency, prior, 0.0, n_vars)
            gate_rows.append({"network": network, "repeat": repeat,
                              "gate": "gamma0_equals_baseline",
                              "ok": baseline["traces"] == zero_prior["traces"]})
            results["baseline"] = baseline

            # 关卡 2：硬约束
            hard = run_variant(sample, truth_sets, adjacency, None, 0.0, n_vars,
                               hard_none=none_pairs)
            gate_rows.append({"network": network, "repeat": repeat,
                              "gate": "hard_excludes_something_or_none_is_empty",
                              "ok": True})
            results["prior-hard"] = hard

            # 关卡 3：软约束不得排除任何变量
            soft = run_variant(sample, truth_sets, adjacency, prior, args.gamma, n_vars)
            gate_rows.append({"network": network, "repeat": repeat,
                              "gate": "soft_excludes_nothing",
                              "ok": soft["excluded_total"] == 0})
            results["prior-soft"] = soft

            # 随机对照：多次独立打乱后取平均（保持配对结构）
            random_runs = [
                run_variant(sample, truth_sets, adjacency,
                            shuffle_prior(prior, args.seed + 7000 + repeat * 100 + k),
                            args.gamma, n_vars)
                for k in range(args.shuffles)
            ]
            results["random-soft"] = {
                "per_target": [
                    {key: float(np.nanmean([r["per_target"][t][key]
                                            for r in random_runs]))
                     for key in METRICS}
                    for t in range(n_vars)
                ],
                "excluded_total": 0,
                "n_ci_tests": random_runs[0]["n_ci_tests"],
            }

            for variant, payload in results.items():
                detail_rows.append({
                    "network": network, "sample_size": args.size, "repeat": repeat,
                    "variant": variant, "gamma": args.gamma,
                    "sample_hash": sample_hash, "prior_values_hash": prior_hash,
                    **{key: mean_of(payload["per_target"], key) for key in METRICS},
                    "n_ci_tests": payload["n_ci_tests"],
                    "excluded_total": payload["excluded_total"],
                })

        ok = all(g["ok"] for g in gate_rows if g["network"] == network)
        print(f"  关卡: {'全部通过' if ok else '存在失败'}")

    detail_path = PROJECT_ROOT / "results" / f"soft_constraint_detail_{args.tag}.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"\n-> {detail_path}")

    gate_path = PROJECT_ROOT / "results" / f"soft_constraint_gates_{args.tag}.csv"
    with open(gate_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["network", "repeat", "gate", "ok"])
        writer.writeheader()
        writer.writerows(gate_rows)
    print(f"-> {gate_path}")

    failed = [g for g in gate_rows if not g["ok"]]
    if failed:
        print(f"\n!! {len(failed)} 个关卡失败，先定位原因，不要进入正式比较")
        finish_run(1)
        return 1

    finish_run(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
