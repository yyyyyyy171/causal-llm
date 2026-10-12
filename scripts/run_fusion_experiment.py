"""
选题二 周6-7 验证性实验（第一部分）。

问题：小样本时，加入语言模型先验能否提高马尔可夫毯的恢复质量？
      把变量名匿名后，这个提升还在不在？

做法：
    在 5 个 bnlearn 网络上，每个样本量重复 R 次（从 n5000 里无放回抽样），
    对每个目标变量运行 IAMB / HITON-MB 的若干变体，计算 Precision / Recall / F1，
    再对目标与重复取平均。

变体：
    baseline              纯数据
    prior-thresh          先验调节条件独立检验阈值
    prior-search          先验作为搜索优先级
    prior-thresh+search   两者同时使用
    random-thresh         随机先验（对照：证明提升不是随便扰动就能拿到）

注意：匿名模式（--mode anon）使用匿名变量名跑出来的先验，
在结构上与真实名模式完全一致，只是语义被剥掉。
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mb.ci_test import CITest  # noqa: E402
from src.mb.data_io import (  # noqa: E402
    f1_of,
    load_data,
    load_prior_scores,
    load_true_mb,
)
from src.mb.discovery import METHODS  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.common.run_log import finish_run, start_run  # noqa: E402


RESULTS_DIR = PROJECT_ROOT / "results"


# 内置默认值；正式实验以 configs/mb_discovery.json 为准
VARIANTS = (
    # name,                beta,  gamma, prior kind
    ("baseline", 0.0, 0.0, "none"),
    ("prior-thresh", 1.0, 0.0, "real"),
    ("prior-search", 0.0, 1.5, "real"),
    ("prior-thresh+search", 1.0, 1.5, "real"),
    ("random-thresh", 1.0, 0.0, "random"),
)


def subsample(data, size, seed):
    """从大样本里无放回抽样，得到指定样本量的子集。"""

    n = data.shape[0]
    if size >= n:
        return data
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=size, replace=False)
    return data[idx]


def randomize_prior(prior, seed):
    """对照实验：保留先验强度的分布，但把"哪个变量对有先验"打乱。"""

    if not prior:
        return {}
    keys = list(prior)
    values = [prior[k] for k in keys]
    rng = random.Random(seed)
    rng.shuffle(values)
    return dict(zip(keys, values))


def evaluate(network, size, repeat, methods, targets, source_size, prior_real,
             beta, gamma, prior_kind, alpha, max_k, seed_base):

    data, variables = load_data(network, source_size)
    sample = subsample(data, size, seed=seed_base + repeat)
    truth = load_true_mb(network)

    if prior_kind == "real":
        prior = prior_real
    elif prior_kind == "random":
        prior = randomize_prior(prior_real, seed=seed_base + 999 + repeat)
    else:
        prior = {}

    ci = CITest(sample, alpha=alpha, prior=prior, beta=beta)

    rows = []

    for target in targets:
        true_set = set(np.flatnonzero(truth[target]))

        for method in methods:
            predicted = METHODS[method](
                ci, target, len(variables), max_k=max_k,
                prior=prior, gamma=gamma,
            )
            precision, recall, f1 = f1_of(predicted, true_set)
            rows.append(
                {
                    "network": network,
                    "sample_size": size,
                    "repeat": repeat,
                    "method": method,
                    "target": variables[target],
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "cits": ci.n_tests,
                }
            )

    return rows


def main() -> int:

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="mb_discovery")
    pre_args, _ = pre.parse_known_args()

    try:
        cfg = load_config(pre_args.config)
    except FileNotFoundError:
        cfg = {}

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument("--networks", default="ALARM,ASIA,CHILD,INSURANCE")
    parser.add_argument(
        "--sizes",
        default=",".join(str(s) for s in cfg.get("sizes", [100, 500])),
    )
    parser.add_argument(
        "--repeats", type=int, default=int(cfg.get("repeats", 5))
    )
    parser.add_argument(
        "--methods",
        default=",".join(cfg.get("methods", ["IAMB", "HITON-MB"])),
    )
    parser.add_argument("--variants", default="all")
    parser.add_argument("--mode", default="real", choices=("real", "anon"))
    parser.add_argument("--targets", type=int, default=0,
                        help="只跑前 N 个目标变量（0 = 全部）")
    parser.add_argument(
        "--alpha", type=float, default=float(cfg.get("alpha", 0.05))
    )
    parser.add_argument(
        "--max-k", type=int, default=int(cfg.get("max_k", 3))
    )
    parser.add_argument(
        "--source-size", type=int, default=int(cfg.get("source_size", 5000))
    )
    parser.add_argument(
        "--seed", type=int, default=int(cfg.get("seed", 20261012))
    )
    parser.add_argument("--tag", default="")

    args = parser.parse_args()

    started_log = start_run("run_fusion_experiment", config=cfg)
    print(f"日志文件: {started_log}")

    networks = [x.strip() for x in args.networks.split(",") if x.strip()]
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]

    # 配置里的变体优先于脚本内置的默认变体
    config_variants = tuple(
        (v["name"], float(v["beta"]), float(v["gamma"]), v["prior"])
        for v in cfg.get("variants", [])
    )
    known_variants = config_variants or VARIANTS

    if args.variants == "all":
        variants = known_variants
    else:
        wanted = {x.strip() for x in args.variants.split(",")}
        variants = tuple(v for v in known_variants if v[0] in wanted)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    started = time.time()

    for network in networks:

        prior_real = load_prior_scores(network, args.mode)

        print(
            f"\n=== {network} | mode={args.mode} | "
            f"先验覆盖 {len(prior_real)} 对 ===",
            flush=True,
        )

        _, variables = load_data(network, 100)
        targets = list(range(len(variables)))
        if args.targets:
            targets = targets[: args.targets]

        for size in sizes:
            for variant_name, beta, gamma, prior_kind in variants:
                for repeat in range(args.repeats):
                    rows = evaluate(
                        network, size, repeat, methods, targets,
                        args.source_size, prior_real, beta, gamma,
                        prior_kind, args.alpha, args.max_k, args.seed,
                    )
                    for row in rows:
                        row["variant"] = variant_name
                    detail_rows.extend(rows)

                parts = []
                for method in methods:
                    f1s = [
                        r["f1"] for r in detail_rows
                        if r["network"] == network
                        and r["sample_size"] == size
                        and r["variant"] == variant_name
                        and r["method"] == method
                    ]
                    parts.append(f"{method}={np.mean(f1s):.4f}")
                print(
                    f"  n={size:<5} {variant_name:<22} "
                    + "  ".join(parts)
                    + f"  ({time.time()-started:.0f}s)",
                    flush=True,
                )

    # --------------------------------------------------------
    # 落盘
    # --------------------------------------------------------

    tag = args.tag or args.mode

    detail_path = RESULTS_DIR / f"fusion_detail_{tag}.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    # 汇总：对 target 和 repeat 取平均
    summary = {}
    for row in detail_rows:
        key = (row["network"], row["sample_size"], row["method"], row["variant"])
        summary.setdefault(key, []).append(row)

    summary_path = RESULTS_DIR / f"fusion_summary_{tag}.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["network", "sample_size", "method", "variant",
             "precision", "recall", "f1", "f1_std", "mean_cits"]
        )
        for key in sorted(summary):
            rows = summary[key]
            f1s = [r["f1"] for r in rows]
            writer.writerow([
                *key,
                f"{np.mean([r['precision'] for r in rows]):.4f}",
                f"{np.mean([r['recall'] for r in rows]):.4f}",
                f"{np.mean(f1s):.4f}",
                f"{np.std(f1s):.4f}",
                f"{np.mean([r['cits'] for r in rows]):.0f}",
            ])

    print(f"\n明细 -> {detail_path}")
    print(f"汇总 -> {summary_path}")
    print(f"总耗时 {time.time()-started:.0f}s")

    finish_run(0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
