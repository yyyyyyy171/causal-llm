import json
import sys
from pathlib import Path

import torch


# ============================================================
# 项目路径
# ============================================================

PROJECT_ROOT = Path(r"E:\Causal-LLM")
SOURCE_ROOT = Path(r"E:\Causal-LLM")

sys.path.insert(0, str(SOURCE_ROOT))


# ============================================================
# 导入你已经验证过的本地 LLM 代码
# ============================================================

from src.llm_prior.model import load_model
from src.llm_prior.prompt import build_causal_prompt
from src.llm_prior.inference import generate_prior


# ============================================================
# 文件
# ============================================================

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "ALARM"
    / "ALARM_LLM_pairs.json"
)

PROGRESS_FILE = (
    PROJECT_ROOT
    / "data"
    / "ALARM"
    / "ALARM_LLM_prior_progress.json"
)

FINAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "ALARM"
    / "ALARM_LLM_prior.json"
)


# ============================================================
# 参数
# ============================================================

N_RUNS = 5

TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 256

BASE_SEED = 20261003


# ============================================================
# JSON 解析
# ============================================================

def extract_json(text: str) -> dict:

    text = text.strip()

    # 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 提取第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:

        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"无法解析 JSON:\n{text}"
    )


# ============================================================
# 保存
# ============================================================

def save_json(path: Path, obj):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2
        )

    temp_path.replace(path)


# ============================================================
# 生成最终 relation
# ============================================================

def aggregate_relation(runs):

    valid_runs = [
        x
        for x in runs
        if x.get("parsed") is not None
    ]

    if not valid_runs:
        return {
            "final_relation": "uncertain",
            "final_confidence": 0.0
        }

    # 统计 relation
    relation_count = {}

    for run in valid_runs:

        relation = run["parsed"].get(
            "relation",
            "uncertain"
        )

        relation_count[relation] = (
            relation_count.get(relation, 0) + 1
        )

    # 多数投票
    final_relation = max(
        relation_count,
        key=relation_count.get
    )

    # 对最终关系对应的 confidence 求平均
    confidence_values = []

    for run in valid_runs:

        parsed = run["parsed"]

        if (
            parsed.get("relation")
            == final_relation
        ):
            try:

                confidence = float(
                    parsed.get(
                        "confidence",
                        0.0
                    )
                )

                confidence_values.append(
                    confidence
                )

            except (
                TypeError,
                ValueError
            ):
                pass

    if confidence_values:

        final_confidence = (
            sum(confidence_values)
            / len(confidence_values)
        )

    else:

        final_confidence = 0.0

    return {
        "final_relation": final_relation,
        "final_confidence": final_confidence,
        "relation_count": relation_count,
    }


# ============================================================
# 主程序
# ============================================================

def main():

    print("=" * 70)
    print("ALARM LLM Prior 正式批量实验")
    print("=" * 70)

    # --------------------------------------------------------
    # 读取变量对
    # --------------------------------------------------------

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        pairs = json.load(f)

    print(
        f"变量对总数: {len(pairs)}"
    )

    print(
        f"每对重复次数: {N_RUNS}"
    )

    print(
        f"理论总推理次数: {len(pairs) * N_RUNS}"
    )

    # --------------------------------------------------------
    # 读取已有进度
    # --------------------------------------------------------

    if PROGRESS_FILE.exists():

        with open(
            PROGRESS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            results = json.load(f)

        print(
            f"检测到已有进度: {len(results)} 对"
        )

    else:

        results = []

        print(
            "没有检测到已有进度，从第 1 对开始。"
        )

    # --------------------------------------------------------
    # 建立已完成索引
    # --------------------------------------------------------

    completed_keys = {
        (
            x["variable_a"],
            x["variable_b"]
        )
        for x in results
    }

    # --------------------------------------------------------
    # 加载模型
    # --------------------------------------------------------

    print()
    print("正在加载本地 Qwen 模型...")

    tokenizer, model = load_model()

    print("模型加载完成。")

    # --------------------------------------------------------
    # 变量对循环
    # --------------------------------------------------------

    total_pairs = len(pairs)

    for pair_index, pair in enumerate(
        pairs,
        start=1
    ):

        variable_a = pair["variable_a"]
        variable_b = pair["variable_b"]

        pair_key = (
            variable_a,
            variable_b
        )

        # ----------------------------------------------------
        # 断点续跑
        # ----------------------------------------------------

        if pair_key in completed_keys:

            print(
                f"\n[{pair_index}/{total_pairs}] "
                f"{variable_a} -> {variable_b} "
                f"已完成，跳过。"
            )

            continue

        print()
        print("=" * 70)
        print(
            f"处理第 {pair_index}/{total_pairs} 个变量对"
        )
        print(
            f"A = {variable_a}"
        )
        print(
            f"B = {variable_b}"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        prompt = build_causal_prompt(
            variable_a,
            variable_b
        )

        runs = []

        # ----------------------------------------------------
        # 5 次随机采样
        # ----------------------------------------------------

        for run_id in range(
            1,
            N_RUNS + 1
        ):

            print(
                f"\n第 {run_id}/{N_RUNS} 次"
            )

            # ----------------------------------------------
            # 为每一次实验设置固定但不同的随机种子
            # ----------------------------------------------

            seed = (
                BASE_SEED
                + pair_index * 100
                + run_id
            )

            torch.manual_seed(seed)

            if torch.cuda.is_available():

                torch.cuda.manual_seed_all(
                    seed
                )

            # ----------------------------------------------
            # 调用模型
            # ----------------------------------------------

            response = generate_prior(
                tokenizer,
                model,
                prompt,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            print(
                "原始输出:"
            )

            print(response)

            # ----------------------------------------------
            # 解析 JSON
            # ----------------------------------------------

            try:

                parsed = extract_json(
                    response
                )

                parse_status = "success"

                print(
                    "JSON解析成功"
                )

            except Exception as e:

                parsed = None

                parse_status = "failed"

                print(
                    "JSON解析失败:",
                    e
                )

            runs.append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "raw_response": response,
                    "parsed": parsed,
                    "parse_status": parse_status
                }
            )

        # ----------------------------------------------------
        # 5 次结果聚合
        # ----------------------------------------------------

        aggregated = aggregate_relation(
            runs
        )

        result = {

            "variable_a": variable_a,

            "variable_b": variable_b,

            "runs": runs,

            "final_relation":
                aggregated[
                    "final_relation"
                ],

            "final_confidence":
                aggregated[
                    "final_confidence"
                ],

            "relation_count":
                aggregated[
                    "relation_count"
                ],
        }

        results.append(
            result
        )

        # ----------------------------------------------------
        # 每完成一对立即保存
        # ----------------------------------------------------

        save_json(
            PROGRESS_FILE,
            results
        )

        print()
        print(
            "最终关系:",
            result["final_relation"]
        )

        print(
            "最终置信度:",
            result["final_confidence"]
        )

        print(
            "当前已完成:",
            len(results),
            "/",
            total_pairs
        )

    # ========================================================
    # 全部完成
    # ========================================================

    save_json(
        FINAL_FILE,
        results
    )

    print()
    print("=" * 70)
    print("全部变量对处理完成")
    print("=" * 70)

    print(
        "完成数量:",
        len(results)
    )

    print(
        "最终结果:",
        FINAL_FILE
    )

    print(
        "进度文件:",
        PROGRESS_FILE
    )


if __name__ == "__main__":
    main()