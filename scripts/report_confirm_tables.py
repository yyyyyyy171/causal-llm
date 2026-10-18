"""
第 6–7 周报告用表：全部从 results/ 下的原始结果文件现算。

为什么单独写一个脚本：周报里的每个数字都必须能追到某个 CSV 的某一行。
报告规格文件（scripts/week_reports/week06_07.json）只引用这里生成的
scripts/week_reports/tables/*.json，不手抄任何结论性数字。

用法：
    python scripts/report_confirm_tables.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path

from scipy.stats import ttest_1samp, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = ROOT / "scripts" / "week_reports" / "tables"

NETWORKS = ("CHILD", "INSURANCE")

# 由 main() 解析后赋值：结果 tag 与确认类表格的输出目录。
TAG = "confirm50_g2"
CONFIRM_TABLES = TABLES


def load(path: Path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def fmt_p(p: float) -> str:
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def paired_stats(diffs):
    n = len(diffs)
    mean = sum(diffs) / n
    sd = (sum((d - mean) ** 2 for d in diffs) / (n - 1)) ** 0.5
    se = sd / sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "lo": mean - 1.96 * se,
        "hi": mean + 1.96 * se,
        "positive": sum(1 for d in diffs if d > 0),
        "zero": sum(1 for d in diffs if d == 0),
        "negative": sum(1 for d in diffs if d < 0),
    }


def one_sided_p(diffs):
    t_p = ttest_1samp(diffs, 0, alternative="greater").pvalue
    w_p = wilcoxon(diffs, alternative="greater").pvalue
    return float(t_p), float(w_p)


def two_sided_p(diffs):
    t_p = ttest_1samp(diffs, 0, alternative="two-sided").pvalue
    w_p = wilcoxon(diffs, alternative="two-sided").pvalue
    return float(t_p), float(w_p)


def paired_table(path: Path):
    table = defaultdict(dict)
    for row in load(path):
        table[(row["network"], int(row["repeat"]))][row["variant"]] = row
    return table


def holm(pvalues):
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [0.0] * n
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (n - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def write(name, header, rows, outdir=None):
    target = outdir or TABLES
    target.mkdir(parents=True, exist_ok=True)
    path = target / name
    path.write_text(
        json.dumps({"header": header, "rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"-> {path.relative_to(ROOT)}（{len(rows)} 行）")


def wconf(name, header, rows):
    """确认类表格统一写到 CONFIRM_TABLES，避免 n=500 覆盖 n=100 的表。"""

    write(name, header, rows, outdir=CONFIRM_TABLES)


def main() -> int:

    global TAG, CONFIRM_TABLES

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="confirm50_g2",
                        help="结果文件后缀，如 confirm50_g2 或 confirm_n500_g2")
    parser.add_argument("--confirm-outdir", default="",
                        help="确认类表格输出目录；留空则写入 tables/")
    parser.add_argument("--subset-start", type=int, default=21,
                        help="独立子集的起始重复号；<=1 表示不生成该表（n=500 时全部重复都未参与 γ 选择）")
    args = parser.parse_args()

    TAG = args.tag
    CONFIRM_TABLES = (
        (ROOT / args.confirm_outdir).resolve() if args.confirm_outdir else TABLES
    )

    detail_path = RESULTS / f"soft_constraint_detail_{TAG}.csv"
    table = paired_table(detail_path)

    # ---------- 1. 主终点：soft − random ----------
    main_rows = []
    for network in NETWORKS:
        repeats = sorted(k[1] for k in table if k[0] == network)
        diffs = [
            float(table[(network, r)]["prior-soft"]["f1"])
            - float(table[(network, r)]["random-soft"]["f1"])
            for r in repeats
        ]
        stats = paired_stats(diffs)
        t_p, w_p = one_sided_p(diffs)
        main_rows.append([
            network,
            str(stats["n"]),
            fmt(stats["mean"]),
            f"{stats['sd']:.4f}",
            f"[{fmt(stats['lo'])}, {fmt(stats['hi'])}]",
            f"{stats['positive']}/{stats['zero']}/{stats['negative']}",
            fmt_p(t_p),
            fmt_p(w_p),
        ])
    wconf(
        "confirm_main.json",
        ["网络", "配对重复", "平均差值 F1（soft−random）", "SD",
         "95% 置信区间", "正/零/负", "单侧 t 检验 p", "单侧 Wilcoxon p"],
        main_rows,
    )

    # ---------- 2. 与基线的关系 ----------
    baseline_rows = []
    for network in NETWORKS:
        repeats = sorted(k[1] for k in table if k[0] == network)
        for variant, label, sided in (
            ("prior-soft", "soft − baseline", "greater"),
            ("prior-hard", "hard − baseline", "two-sided"),
            ("random-soft", "random − baseline", "two-sided"),
        ):
            diffs = [
                float(table[(network, r)][variant]["f1"])
                - float(table[(network, r)]["baseline"]["f1"])
                for r in repeats
            ]
            stats = paired_stats(diffs)
            t_p, w_p = (one_sided_p if sided == "greater" else two_sided_p)(diffs)
            baseline_rows.append([
                network, label, fmt(stats["mean"]),
                f"[{fmt(stats['lo'])}, {fmt(stats['hi'])}]",
                "单侧" if sided == "greater" else "双侧",
                fmt_p(t_p),
                fmt_p(w_p),
            ])
    wconf(
        "confirm_baseline.json",
        ["网络", "配对比较", "平均差值 F1", "95% 置信区间",
         "检验方向", "t 检验 p", "Wilcoxon p"],
        baseline_rows,
    )

    # ---------- 2b. 四组的绝对水平 ----------
    level_rows = []
    for network in NETWORKS:
        repeats = sorted(k[1] for k in table if k[0] == network)
        row = [network]
        for variant in ("baseline", "prior-hard", "prior-soft", "random-soft"):
            values = [float(table[(network, r)][variant]["f1"]) for r in repeats]
            row.append(f"{sum(values) / len(values):.4f}")
        level_rows.append(row)
    wconf(
        "confirm_levels.json",
        ["网络", "baseline", "prior-hard", "prior-soft", "random-soft"],
        level_rows,
    )

    # ---------- 3. 独立子集（第 21–50 次，未参与 γ 选择） ----------
    subset_rows = []
    for network in NETWORKS:
        repeats = [r for r in sorted(k[1] for k in table if k[0] == network)
                   if r >= args.subset_start]
        diffs = [
            float(table[(network, r)]["prior-soft"]["f1"])
            - float(table[(network, r)]["random-soft"]["f1"])
            for r in repeats
        ]
        stats = paired_stats(diffs)
        t_p, _ = one_sided_p(diffs)
        subset_rows.append([
            network, str(stats["n"]), fmt(stats["mean"]),
            f"[{fmt(stats['lo'])}, {fmt(stats['hi'])}]",
            f"{stats['positive']}/{stats['zero']}/{stats['negative']}",
            fmt_p(t_p),
        ])
    if args.subset_start > 1:
        wconf(
            "confirm_subset.json",
            ["网络", "配对重复", "平均差值 F1（soft−random）",
             "95% 置信区间", "正/零/负", "单侧 t 检验 p"],
            subset_rows,
        )

    # ---------- 4. 三道关卡 ----------
    gates = load(RESULTS / f"soft_constraint_gates_{TAG}.csv")
    gate_rows = []
    total_all = total_bad = 0
    for network in NETWORKS:
        rows = [g for g in gates if g["network"] == network]
        bad = [g for g in rows if g["ok"] != "True"]
        total_all += len(rows)
        total_bad += len(bad)
        gate_rows.append([network, str(len(rows) // 3), str(len(rows)), str(len(bad))])
    gate_rows.append(["合计", "—", str(total_all), str(total_bad)])
    wconf(
        "soft_gates.json",
        ["网络", "重复次数", "关卡检查数", "失败数"],
        gate_rows,
    )

    # ---------- 5. role 版先验的评测 ----------
    role = load(RESULTS / "llm_role_eval_qwen3-8b-awq.csv")
    role_rows = []
    for network in ("ASIA", "CHILD", "INSURANCE", "ALARM", "HAILFINDER"):
        real = next(r for r in role if r["network"] == network and r["mode"] == "real")
        anon = next(r for r in role if r["network"] == network and r["mode"] == "anon")
        role_rows.append([
            network,
            f"{float(real['edge_precision']):.3f}",
            f"{float(real['edge_recall']):.3f}",
            f"{float(real['edge_f1']):.3f}",
            f"{float(real['direction_coverage']):.3f}",
            f"{float(real['direction_accuracy']):.3f}",
            f"{anon['direction_undecided']}/{anon['true_edges']}",
        ])
    write(
        "role_prior_eval.json",
        ["网络", "边精确率", "边召回率", "边 F1", "方向覆盖率",
         "方向准确率（有条件）", "匿名：判不确定/真实边"],
        role_rows,
    )

    # ---------- 6. 配偶诊断 ----------
    spouse = load(RESULTS / "prior_vs_spouses_real_qwen3-8b-awq_role.csv")
    spouse_rows = []
    for row in spouse:
        hits = int(row["said_inblanket_for_spouse"])
        total = int(row["spouse"])
        spouse_rows.append([
            row["network"], row["blanket_pairs"], row["child_or_parent"], row["spouse"],
            row["said_inblanket_for_child_parent"],
            f"{hits}/{total}",
            f"{hits / total:.2f}" if total else "—",
        ])
    write(
        "spouse_diagnosis.json",
        ["网络", "毯内变量对", "其中父子", "其中配偶",
         "父子判在毯内", "配偶判在毯内", "配偶命中率"],
        spouse_rows,
    )

    # ---------- 7. 阈值路线（72 次检验，Holm 校正） ----------
    fusion = load(RESULTS / "fusion_both_paired_role8b_real.csv")
    pvalues = [float(r["wilcox_p"]) for r in fusion]
    adjusted = holm(pvalues)
    for row, adj in zip(fusion, adjusted):
        row["holm_p"] = adj

    route_rows = []
    for variant, label in (
        ("prior-thresh", "先验调阈值"),
        ("random-thresh", "随机先验调阈值"),
        ("alpha-only", "只调阈值（无先验）"),
    ):
        rows = [r for r in fusion if r["variant"] == variant]
        deltas = [abs(float(r["mean_delta"])) for r in rows]
        route_rows.append([
            label,
            str(len(rows)),
            str(sum(1 for r in rows if float(r["wilcox_p"]) < 0.05)),
            str(sum(1 for r in rows if r["holm_p"] < 0.05)),
            f"{sum(deltas) / len(deltas):.4f}",
            f"{max(deltas):.4f}",
        ])
    write(
        "threshold_route.json",
        ["变体", "检验数", "未校正 p<0.05", "Holm 后 p<0.05",
         "平均 |ΔF1|", "最大 |ΔF1|"],
        route_rows,
    )

    # ---------- 8. 硬约束路线（alpha0.05 预算） ----------
    constraint = load(RESULTS / "bnlearn_constraint_paired_role8b_n100.csv")
    pvalues = [float(r["wilcox_p"]) for r in constraint]
    adjusted = holm(pvalues)
    for row, adj in zip(constraint, adjusted):
        row["holm_p"] = adj
    constraint_rows = []
    for network in ("ASIA", "CHILD", "INSURANCE"):
        for row in constraint:
            if row["network"] != network or row["budget"] != "alpha0.05":
                continue
            if row["variant"] not in ("prior-blacklist", "random-blacklist", "prior-thresh"):
                continue
            constraint_rows.append([
                network,
                row["variant"],
                f"{float(row['baseline_f1']):.4f}",
                f"{float(row['variant_f1']):.4f}",
                fmt(float(row["mean_delta"])),
                row["wins"],
                fmt_p(float(row["wilcox_p"])),
                fmt_p(float(row["holm_p"])),
            ])
    write(
        "constraint_route.json",
        ["网络", "变体", "基线 F1", "变体 F1", "ΔF1", "变体胜/总",
         "Wilcoxon p", "Holm p"],
        constraint_rows,
    )

    # ---------- 9. γ 扫描（n=100，20 次重复） ----------
    scan_rows = []
    for gamma, tag in ((0.5, "soft_n100_g05"), (1.0, "soft_n100_g1"), (2.0, "soft_n100_g20")):
        scan = paired_table(RESULTS / f"soft_constraint_detail_{tag}.csv")
        row = [f"{gamma:.1f}"]
        for network in NETWORKS:
            repeats = sorted(k[1] for k in scan if k[0] == network)
            diffs = [
                float(scan[(network, r)]["prior-soft"]["f1"])
                - float(scan[(network, r)]["random-soft"]["f1"])
                for r in repeats
            ]
            stats = paired_stats(diffs)
            t_p, _ = one_sided_p(diffs)
            row.append(f"{fmt(stats['mean'])}（p={fmt_p(t_p)}）")
        scan_rows.append(row)
    write(
        "gamma_scan.json",
        ["γ", "CHILD：soft−random", "INSURANCE：soft−random"],
        scan_rows,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
