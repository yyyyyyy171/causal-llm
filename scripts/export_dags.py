"""
导出 5 个 bnlearn 网络的 DAG 真值，并做一致性交叉验证。

背景：方向准确率必须对照真实 DAG。`data/<网络>/<网络>.rds` 里存的是 bn.fit 对象，
直接解析即可拿到 parents，所以 Python 管道不依赖 R 也能恢复整张有向图
（见 src/mb/rds_dag.py 的说明）。本机装了 R（`D:\R\R-4.6.1`，含 bnlearn），
因此还能用 bnlearn 独立复核：`scripts/verify_dag_bnlearn.R`。

交叉验证：由 DAG 算出的**道德图邻接矩阵**应当与仓库里已有的
`<网络>_true_MB_matrix.txt` 完全一致。两者来源完全不同
（一个是解析 rds 的 parents，一个是既有的真值矩阵），
对上了才说明 DAG 解析正确。

产出：`data/<网络>/<网络>_dag_edges.csv`（parent,child 两列）。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_config  # noqa: E402
from src.common.run_log import finish_run, start_run  # noqa: E402
from src.mb.data_io import load_data, load_true_mb  # noqa: E402
from src.mb.rds_dag import load_dag, moral_adjacency  # noqa: E402


NETWORKS = ["ASIA", "CHILD", "INSURANCE", "HAILFINDER", "ALARM"]


def main() -> int:

    cfg = load_config("mb_discovery")
    log_path = start_run("export_dags", config=cfg)
    print(f"日志文件: {log_path}")

    ok = True

    for network in NETWORKS:

        try:
            _, variables = load_data(network, 100)
            nodes, arcs = load_dag(network, variables)
        except Exception as error:  # noqa: BLE001
            print(f"{network:12s} 失败: {type(error).__name__}: {error}")
            ok = False
            continue

        if arcs is None:
            print(f"{network:12s} 没有可用的 DAG 真值")
            ok = False
            continue

        # 一致性交叉验证
        computed = moral_adjacency(arcs, variables)
        try:
            truth = load_true_mb(network)
            same_shape = computed.shape == truth.shape
            mismatch = int(np.abs(computed - truth).sum()) if same_shape else -1
        except FileNotFoundError:
            same_shape, mismatch = False, -1

        verdict = "一致" if (same_shape and mismatch == 0) else f"不一致（差异 {mismatch}）"

        print(
            f"{network:12s} {len(nodes):3d} 节点 {len(arcs):3d} 条边 | "
            f"道德图 vs 真值马尔可夫毯矩阵: {verdict}"
        )

        if not (same_shape and mismatch == 0):
            ok = False

        out_path = PROJECT_ROOT / "data" / network / f"{network}_dag_edges.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["parent", "child"])
            writer.writerows(sorted(arcs))

    print()
    print("全部一致" if ok else "有网络未通过，见上面的提示")

    finish_run(0 if ok else 1)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
