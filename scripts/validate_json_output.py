"""
结构化输出验收：检查语言模型返回的 JSON 是否可解析、字段是否完整。

对应第一周验收项"一段结构化输出的结果"：除了跑通一次，还要拿出校验结果。
扫描 data/ 下所有先验结果文件，逐条检查文件是否合法 JSON、记录与每次采样
的必需字段是否齐全、解析失败的次数。

输出 results/json_validation.csv（逐文件一行）与控制台摘要。
解析失败不被隐藏：缓存里保留原始输出与 parse_status。

用法：
    python scripts/validate_json_output.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.run_log import finish_run, start_run  # noqa: E402


DATA_ROOT = PROJECT_ROOT / "data"

RECORD_FIELDS = {
    "direction": [
        "variable_a", "variable_b", "runs",
        "final_relation", "final_confidence", "consistency",
    ],
    "mb": [
        "variable_a", "variable_b", "runs",
        "in_markov_blanket", "final_confidence", "consistency",
    ],
}

RUN_FIELDS = ["run_id", "raw_response", "parsed", "parse_status"]


def task_of(path: Path) -> str:
    return "mb" if "_LLM_mb_" in path.name else "direction"


def check_file(path: Path) -> dict:

    task = task_of(path)
    required = RECORD_FIELDS[task]

    row = {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "task": task,
        "file_parses": False,
        "records": 0,
        "runs": 0,
        "parse_failures": 0,
        "records_missing_fields": 0,
        "runs_missing_fields": 0,
        "answers_missing_fields": 0,
        "note": "",
    }

    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as error:
        row["note"] = f"{type(error).__name__}: {error}"
        return row

    row["file_parses"] = True
    row["records"] = len(data)

    for record in data:

        if any(field not in record for field in required):
            row["records_missing_fields"] += 1

        for run in record.get("runs", []):

            row["runs"] += 1

            if any(field not in run for field in RUN_FIELDS):
                row["runs_missing_fields"] += 1

            if run.get("parse_status") != "success":
                row["parse_failures"] += 1
                continue

            parsed = run.get("parsed") or {}

            if "confidence" not in parsed:
                row["answers_missing_fields"] += 1
            if task == "direction" and not any(
                key in parsed for key in ("relation", "direct_edge")
            ):
                row["answers_missing_fields"] += 1
            if task == "mb" and "in_markov_blanket" not in parsed:
                row["answers_missing_fields"] += 1

    return row


def main() -> int:

    log_path = start_run("validate_json_output")
    print(f"日志文件: {log_path}")

    found = sorted(
        list(DATA_ROOT.glob("*/*_LLM_prior_*.json"))
        + list(DATA_ROOT.glob("*/*_LLM_mb_*.json"))
    )

    # 早期实验留下的是旧格式（比如 *_progress.json），字段与新管道不同，
    # 把它们排除在校验之外并单独列出，避免把历史文件误判成结构缺陷。
    files = [path for path in found if "_progress" not in path.name]
    skipped = [path for path in found if path not in files]

    if skipped:
        print("跳过旧格式文件：")
        for path in skipped:
            print(f"  {path.relative_to(PROJECT_ROOT)}")

    if not files:
        print("没有找到任何先验结果文件")
        finish_run(1)
        return 1

    rows = [check_file(path) for path in files]

    out_path = PROJECT_ROOT / "results" / "json_validation.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"{'file':<46s} {'记录':>5s} {'采样':>6s} {'失败':>5s} {'缺字段':>6s}")
    for row in rows:
        missing = (
            row["records_missing_fields"]
            + row["runs_missing_fields"]
            + row["answers_missing_fields"]
        )
        print(
            f"{row['file']:<46s} {row['records']:>5d} {row['runs']:>6d} "
            f"{row['parse_failures']:>5d} {missing:>6d}"
        )

    total_runs = sum(row["runs"] for row in rows)
    total_failures = sum(row["parse_failures"] for row in rows)
    total_missing = sum(
        row["records_missing_fields"]
        + row["runs_missing_fields"]
        + row["answers_missing_fields"]
        for row in rows
    )

    print()
    print(
        f"文件 {len(rows)} 个，采样 {total_runs} 条，解析失败 {total_failures} 条"
        f"（{total_failures / max(1, total_runs) * 100:.2f}%），字段缺失 {total_missing} 处"
    )

    all_parse = all(row["file_parses"] for row in rows)
    no_structural = not any(
        row["records_missing_fields"] or row["runs_missing_fields"] for row in rows
    )
    print("结论：" + (
        "结构化输出校验通过（解析失败的记录均已在缓存中保留原始输出）"
        if all_parse and no_structural
        else "存在结构性问题，见上表"
    ))
    print(f"产出：{out_path.relative_to(PROJECT_ROOT)}")

    finish_run(0 if (all_parse and no_structural) else 1)
    return 0 if (all_parse and no_structural) else 1


if __name__ == "__main__":
    raise SystemExit(main())
