"""
评估语言模型的因果判断：边判别与方向准确率（周5 的产出）。

指标：

* 边判别：把 direct_edge=true 当作"预测有边"，与真实 DAG 的边集比较，
  给出 precision / recall / F1。
* 方向准确率：在**真实存在的边**上，模型判出的方向是否正确；
  同时报告"覆盖率"（模型敢不敢在这些边上给方向）。
* 一致性：5 次采样中多数意见的占比。

真值来源：全部 5 个网络的 DAG 都由 `src/mb/rds_dag.py` 从
`data/<网络>/<网络>.rds` 里的 bn.fit 对象解析得到（ALARM 用仓库里的
`alarm_graph.txt`）。解析结果已做交叉验证：由 DAG 算出的道德图与
`<网络>_true_MB_matrix.txt` 逐位一致。可用 `scripts/export_dags.py` 复查。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.run_log import finish_run, start_run  # noqa: E402
from src.mb.data_io import load_data  # noqa: E402
from src.mb.rds_dag import load_dag as load_dag_truth  # noqa: E402


def load_dag(network: str, variables: list[str]):
    """返回有向边集合 {(parent, child)}。"""

    _, arcs = load_dag_truth(network, variables)
    return arcs


def read_prior(network: str, mode: str):
    path = (
        PROJECT_ROOT / "data" / network
        / f"{network}_LLM_prior_{mode}_summary.csv"
    )
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate(network: str, mode: str, dag_edges, verbose: bool = True) -> dict:

    rows = read_prior(network, mode)
    if rows is None:
        return {}

    # 真实无向边（忽略方向）
    undirected_true = {
        frozenset({a, b}) for (a, b) in dag_edges
    }

    tp = fp = fn = 0
    direction_correct = 0
    direction_attempted = 0
    direction_undecided = 0
    consistencies = []

    for row in rows:

        a, b = row["variable_a"], row["variable_b"]
        direct_edge = row["direct_edge"].strip().lower() == "true"
        relation = row["final_relation"].strip()

        consistencies.append(float(row["consistency"]))

        truth = frozenset({a, b}) in undirected_true

        if direct_edge and truth:
            tp += 1
        elif direct_edge and not truth:
            fp += 1
        elif (not direct_edge) and truth:
            fn += 1

        if truth:
            true_direction = (
                "A_to_B" if (a, b) in dag_edges else "B_to_A"
            )
            if relation in ("A_to_B", "B_to_A"):
                direction_attempted += 1
                if relation == true_direction:
                    direction_correct += 1
            else:
                direction_undecided += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    result = {
        "network": network,
        "mode": mode,
        "pairs": len(rows),
        "true_edges": len(dag_edges),
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "direction_correct": direction_correct,
        "direction_attempted": direction_attempted,
        "direction_undecided": direction_undecided,
        "direction_accuracy": (
            direction_correct / direction_attempted
            if direction_attempted else 0.0
        ),
        "mean_consistency": float(np.mean(consistencies)) if consistencies else 0.0,
    }

    if verbose:
        print(
            f"{network:10s} {mode:5s} pairs={result['pairs']:4d} "
            f"edgeP={precision:.3f} edgeR={recall:.3f} edgeF1={f1:.3f} "
            f"dirAcc={result['direction_accuracy']:.3f}"
            f"({direction_correct}/{direction_attempted},"
            f" 未决 {direction_undecided}) "
            f"cons={result['mean_consistency']:.3f}"
        )

    return result


def main() -> int:

    parser = argparse.ArgumentParser()
    parser.add_argument("--networks", default="ASIA,ALARM")
    parser.add_argument("--modes", default="real,anon")
    parser.add_argument("--out", default="")

    args = parser.parse_args()

    started_log = start_run("eval_llm_prior")
    print(f"日志文件: {started_log}")

    results = []

    for network in [x.strip() for x in args.networks.split(",") if x.strip()]:

        _, variables = load_data(network, 100)
        dag_edges = load_dag(network, variables)

        if dag_edges is None:
            print(f"{network}: 没有 DAG 真值，跳过")
            continue

        for mode in [x.strip() for x in args.modes.split(",") if x.strip()]:
            result = evaluate(network, mode, dag_edges)
            if result:
                results.append(result)
            else:
                print(
                    f"{network:10s} {mode:5s} 语言模型先验还没跑，"
                    f"先执行 run_llm_prior.py --network {network} --mode {mode}"
                )

    if args.out and results:
        path = PROJECT_ROOT / "results" / args.out
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"\n-> {path}")

    finish_run(0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
