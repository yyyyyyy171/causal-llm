"""
周 2：共用的特征选择工具箱（5 个 OpenML 数据集 x 5 种方法）。

产出：
    results/fs_benchmark_long.csv        长表，含所有特征数预算的结果
    results/fs_method_by_dataset.csv     "方法 x 数据集"主结果表
    results/fs_selected_features.csv     每个方法选中的特征顺序（便于复查与复用）

用法：
    python scripts/run_fs_benchmark.py
    python scripts/run_fs_benchmark.py --datasets breast-w,diabetes --methods mutual_info,l1
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_config  # noqa: E402
from src.common.run_log import finish_run, start_run  # noqa: E402
from src.fs.data import prepare  # noqa: E402
from src.fs.evaluate import evaluate  # noqa: E402
from src.fs.methods import METHOD_LABELS, METHODS  # noqa: E402


RESULTS_DIR = PROJECT_ROOT / "results"

# 主结果表的表头（中文，方便直接放进周报）
COLUMN_LABELS = {
    "n_samples": "样本数",
    "n_features": "特征数",
    "k": "选定特征数",
    "all_features": "全部特征",
    "majority_accuracy": "多数类基线",
}


def main() -> int:

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="feature_selection")
    pre_args, _ = pre.parse_known_args()

    cfg = load_config(pre_args.config)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument(
        "--datasets",
        default=",".join(d["name"] for d in cfg["datasets"]),
    )
    parser.add_argument(
        "--methods", default=",".join(cfg["methods"])
    )
    args = parser.parse_args()

    log_path = start_run("run_fs_benchmark", config=cfg)
    print(f"日志文件: {log_path}")

    wanted_datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    wanted_methods = [x.strip() for x in args.methods.split(",") if x.strip()]

    seed = int(cfg["seed"])
    primary_k = int(cfg["primary_k"])
    ratios = list(cfg.get("k_ratios", []))
    model_name = cfg.get("downstream_model", "logistic_regression")

    rows = []
    feature_rows = []

    for spec in cfg["datasets"]:

        if spec["name"] not in wanted_datasets:
            continue

        started = time.time()
        bundle = prepare(
            spec["name"],
            spec["openml_id"],
            seed,
            val_size=float(cfg["val_size"]),
            test_size=float(cfg["test_size"]),
        )

        matrix = bundle["matrix"]
        y = bundle["y"]
        train_index = bundle["splits"]["train"]
        test_index = bundle["splits"]["test"]

        # 数据集体检：多数类基线。
        # 全特征准确率若与多数类基线相差无几，说明这个数据集没有可学信号；
        # 若全特征准确率恰好等于 1.0，则多半是标签泄漏或退化版本。
        counts = np.bincount(y[test_index])
        majority = float(counts.max() / counts.sum())

        print(
            f"\n=== {spec['name']} : {bundle['n_samples']} 样本 x "
            f"{bundle['n_features']} 特征，"
            f"{len(bundle['classes'])} 类，多数类基线 {majority:.4f}，"
            f"加载 {time.time()-started:.1f}s ===",
            flush=True,
        )

        # 基线：全部特征
        baseline = evaluate(
            matrix, y, train_index, test_index, [], seed, model_name
        )
        rows.append(
            {
                "dataset": spec["name"],
                "n_samples": bundle["n_samples"],
                "n_features": bundle["n_features"],
                "method": "all_features",
                "k": bundle["n_features"],
                "budget": "all",
                "majority_accuracy": f"{majority:.4f}",
                "accuracy": f"{baseline['accuracy']:.4f}",
                "select_seconds": "0.0000",
                "fit_seconds": f"{baseline['fit_seconds']:.4f}",
            }
        )
        print(f"  全部特征 准确率={baseline['accuracy']:.4f}")

        if baseline["accuracy"] >= 0.9999:
            print(
                "  【警告】全部特征即可完全分开，疑似标签泄漏或退化数据版本，"
                "这个数据集的结果不要用。",
                flush=True,
            )

        # 各种特征数预算
        # 特征数不足 primary_k 的数据集（例如只有 8 列的 diabetes）自动降级
        effective_primary = min(primary_k, bundle["n_features"])

        budgets = {effective_primary}
        for ratio in ratios:
            budgets.add(max(1, int(round(ratio * bundle["n_features"]))))
        budgets = sorted(b for b in budgets if b <= bundle["n_features"])

        for method in wanted_methods:

            selector = METHODS[method]
            label = METHOD_LABELS.get(method, method)

            started = time.time()
            if method == "forward_selection":
                ranking = selector(
                    matrix[train_index], y[train_index], seed,
                    k=effective_primary,
                )
            else:
                ranking = selector(matrix[train_index], y[train_index], seed)
            select_seconds = time.time() - started

            for rank, feature in enumerate(ranking, start=1):
                feature_rows.append(
                    {
                        "dataset": spec["name"],
                        "method": method,
                        "rank": rank,
                        "feature": bundle["feature_names"][feature],
                    }
                )

            for k in budgets:
                result = evaluate(
                    matrix, y, train_index, test_index,
                    ranking[:k], seed, model_name,
                )
                rows.append(
                    {
                        "dataset": spec["name"],
                        "n_samples": bundle["n_samples"],
                        "n_features": bundle["n_features"],
                        "method": method,
                        "k": k,
                        "budget": "primary" if k == effective_primary else "extra",
                        "majority_accuracy": f"{majority:.4f}",
                        "accuracy": f"{result['accuracy']:.4f}",
                        "select_seconds": f"{select_seconds:.4f}",
                        "fit_seconds": f"{result['fit_seconds']:.4f}",
                    }
                )

            primary_accuracy = next(
                r["accuracy"] for r in rows
                if r["dataset"] == spec["name"]
                and r["method"] == method
                and r["budget"] == "primary"
            )
            print(
                f"  {label:<12} k={effective_primary} 准确率={primary_accuracy} "
                f"（选择耗时 {select_seconds:.1f}s）",
                flush=True,
            )

    # --------------------------------------------------------
    # 落盘
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    long_path = RESULTS_DIR / "fs_benchmark_long.csv"
    with open(long_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # "方法 x 数据集"主结果表
    table = {}
    for row in rows:
        if row["budget"] == "extra":
            continue
        table.setdefault(row["dataset"], {})["n_samples"] = row["n_samples"]
        table[row["dataset"]]["n_features"] = row["n_features"]
        table[row["dataset"]]["majority_accuracy"] = row["majority_accuracy"]
        if row["method"] == "all_features":
            table[row["dataset"]]["all_features"] = row["accuracy"]
        else:
            table[row["dataset"]][row["method"]] = row["accuracy"]
            # 所有方法的 primary 预算相同，直接覆盖（不能用 setdefault，
            # 否则会被 all_features 那一行先写进去的特征总数顶掉）
            table[row["dataset"]]["k"] = row["k"]

    columns = [
        "n_samples", "n_features", "k",
        "majority_accuracy", "all_features",
    ] + wanted_methods

    pivot_path = RESULTS_DIR / "fs_method_by_dataset.csv"
    with open(pivot_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["数据集"] + [
            COLUMN_LABELS.get(c, METHOD_LABELS.get(c, c)) for c in columns
        ]
        writer.writerow(header)
        for dataset, values in table.items():
            writer.writerow([dataset] + [values.get(c, "") for c in columns])

    features_path = RESULTS_DIR / "fs_selected_features.csv"
    with open(features_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)

    print(f"\n长表   -> {long_path}")
    print(f"主结果 -> {pivot_path}")
    print(f"特征序 -> {features_path}")

    finish_run(0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
