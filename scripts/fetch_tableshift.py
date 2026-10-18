"""
把 TableShift 数据集的固定划分导出成本地 CSV（周 11-12 / 验证性实验第二部分用）。

为什么做成"只导出"：
TableShift 的依赖非常重（numpy 1.23.5、ray、torch、folktables、rtdl…），
官方自己就建议用 Docker 或独立 conda 环境，官方文档里也写着"在 Docker 之外
配置训练依赖很困难"。把它装进我们已有的环境会污染一套本来干净的工具链。
所以这里只在**独立的 tableshift 环境**里跑一次，把数据落成 CSV；
后面的特征选择与下游评测全部用我们自己的代码（src/fs、src/mb），不再依赖它。

这样还有一个好处：数据一旦落盘，实验就是完全可复现的，
别人不需要装 TableShift 也能重跑我们的表。

用法：
    # 先跑一遍体检，看任务名对不对、每个划分有多少行
    python scripts/fetch_tableshift.py --dry-run

    # 确认无误再真正导出（默认 3 个任务，按文档验证性实验要求）
    python scripts/fetch_tableshift.py

产出：
    data/tableshift/<任务>/<划分>.csv
    data/tableshift/<任务>/schema.json      （列名、目标列、行数、划分来源）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "data" / "tableshift"

# 文档要求"3 个 TableShift 任务"，且验证性实验要比较分布内到分布外的性能下降。
# 这里选的是公开可直接获取、且各自来自不同领域的三个：
#   diabetes_readmission  医院再入院（医疗，表格特征混合）
#   acsfoodstamps         食品券领取（社会经济，ACS 调查）
#   acsincome             收入（社会经济，ACS 调查）
# 真正开跑前请先 --dry-run 核对：某些任务需要签署数据使用协议、要手工放数据。
DEFAULT_TASKS = ["diabetes_readmission", "acsfoodstamps", "acsincome"]

SPLITS = ["train", "validation", "id_test", "ood_test"]


def load_dataset(name: str):
    """导入并加载一个 TableShift 数据集。导入失败时给出可操作的提示。"""

    try:
        from tableshift import get_dataset
    except ImportError as error:
        raise SystemExit(
            "没有安装 tableshift。请在**独立的**环境里装它，不要装进本项目现有环境：\n"
            "  1) git clone https://github.com/mlfoundations/tableshift\n"
            "  2) cd tableshift && conda env create -f environment.yml\n"
            "  3) conda activate tableshift\n"
            "  4) 再运行本脚本\n"
            f"（原始报错：{error}）"
        )

    return get_dataset(name)


def describe(name: str) -> dict:
    """体检：每个划分有多少行、多少列、目标列是否单列。"""

    dataset = load_dataset(name)
    report = {"task": name, "splits": {}}

    for split in SPLITS:
        try:
            X, y, _, _ = dataset.get_pandas(split)
        except Exception as error:                      # noqa: BLE001
            report["splits"][split] = f"取不到（{type(error).__name__}: {error}）"
            continue
        report["splits"][split] = {
            "rows": int(len(X)),
            "features": int(X.shape[1]),
            "target": str(y.name) if hasattr(y, "name") else "y",
        }

    return report


def export(name: str) -> dict:
    """把各划分落成 CSV，并写一份 schema.json。"""

    dataset = load_dataset(name)
    target_dir = OUT_ROOT / name
    target_dir.mkdir(parents=True, exist_ok=True)

    schema = {"task": name, "splits": {}, "source": "TableShift (NeurIPS 2023 D&B)"}

    for split in SPLITS:
        try:
            X, y, _, _ = dataset.get_pandas(split)
        except Exception as error:                      # noqa: BLE001
            print(f"  {split}: 跳过（{type(error).__name__}: {error}）")
            continue

        # 目标列单独放一列，名字固定成 target；CSV 只存一份，避免 X/y 走散
        table = X.copy()
        table["target"] = y.values if hasattr(y, "values") else y

        path = target_dir / f"{split}.csv"
        table.to_csv(path, index=False)

        schema["splits"][split] = {
            "rows": int(len(table)),
            "features": int(X.shape[1]),
            "target_column": "target",
            "file": path.name,
        }
        print(f"  {split}: {len(table)} 行 x {X.shape[1]} 列 -> {path.name}")

    (target_dir / "schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return schema


def main() -> int:

    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--dry-run", action="store_true",
                        help="只体检，不写文件")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    failed = []

    for name in tasks:
        print(f"===== {name} =====")
        try:
            if args.dry_run:
                report = describe(name)
                for split, info in report["splits"].items():
                    print(f"  {split}: {info}")
            else:
                export(name)
        except SystemExit:
            raise
        except Exception as error:                      # noqa: BLE001
            print(f"  失败：{type(error).__name__}: {error}")
            failed.append(f"{name}（{type(error).__name__}）")

    print()
    if failed:
        print("以下任务没成功，先看报错再决定是否换任务：")
        for item in failed:
            print("  -", item)
        return 1

    print("全部完成。" if not args.dry_run
          else "体检完成；确认任务名与行数后再去掉 --dry-run 正式导出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
