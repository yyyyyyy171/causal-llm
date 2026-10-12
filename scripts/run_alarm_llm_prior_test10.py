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
# 导入已经验证过的本地 LLM 代码
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
    / "ALARM_LLM_test10.json"
)

PROGRESS_FILE = (
    PROJECT_ROOT
    / "data"
    / "ALARM"
    / "ALARM_LLM_test10_progress.json"
)

FINAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "ALARM"
    / "ALARM_LLM_test10_result.json"
)


# ============================================================
# 参数
# ============================================================

# 每个变量对重复采样次数
N_RUNS = 5

# LLM采样参数
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 256

# 基础随机种子
BASE_SEED = 20261003


# ============================================================
# JSON解析
# ============================================================

def extract_json(text: str) -> dict:

    text = text.strip()

    # --------------------------------------------------------
    # 方法1：直接解析
    # --------------------------------------------------------

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # 方法2：提取第一个 { 到最后一个 }
    # --------------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:

        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)

        except json.JSONDecodeError:
            pass

    # --------------------------------------------------------
    # 如果仍然失败
    # --------------------------------------------------------

    raise ValueError(
        f"无法解析 JSON:\n{text}"
    )


# ============================================================
# JSON保存
# ============================================================

def save_json(path: Path, obj):

    # --------------------------------------------------------
    # 确保目录存在
    # --------------------------------------------------------

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 临时文件
    # --------------------------------------------------------

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    # --------------------------------------------------------
    # 写入临时文件
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 写入成功后再替换正式文件
    # --------------------------------------------------------

    temp_path.replace(path)


# ============================================================
# 生成最终relation
# ============================================================

def aggregate_relation(runs):

    # --------------------------------------------------------
    # 只保留成功解析的结果
    # --------------------------------------------------------

    valid_runs = [
        x
        for x in runs
        if x.get("parsed") is not None
    ]

    # --------------------------------------------------------
    # 如果全部解析失败
    # --------------------------------------------------------

    if not valid_runs:

        return {
            "final_relation": "uncertain",
            "final_confidence": 0.0,
            "relation_count": {}
        }

    # --------------------------------------------------------
    # 统计relation
    # --------------------------------------------------------

    relation_count = {}

    for run in valid_runs:

        relation = run["parsed"].get(
            "relation",
            "uncertain"
        )

        relation_count[relation] = (
            relation_count.get(relation, 0) + 1
        )

    # --------------------------------------------------------
    # 多数投票
    # --------------------------------------------------------

    final_relation = max(
        relation_count,
        key=relation_count.get
    )

    # --------------------------------------------------------
    # 对最终relation对应的confidence求平均
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 计算平均confidence
    # --------------------------------------------------------

    if confidence_values:

        final_confidence = (
            sum(confidence_values)
            / len(confidence_values)
        )

    else:

        final_confidence = 0.0

    # --------------------------------------------------------
    # 返回聚合结果
    # --------------------------------------------------------

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

    # ========================================================
    # 1. 读取变量对
    # ========================================================

    print()
    print("正在读取变量对文件...")

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8-sig"
    ) as f:

        pairs = json.load(f)

    # --------------------------------------------------------
    # 基本检查
    # --------------------------------------------------------

    if not isinstance(pairs, list):

        raise ValueError(
            "INPUT_FILE 的JSON顶层结构不是list，"
            f"实际类型为: {type(pairs)}"
        )

    print(
        f"变量对总数: {len(pairs)}"
    )

    print(
        f"每对重复次数: {N_RUNS}"
    )

    print(
        f"理论总推理次数: "
        f"{len(pairs) * N_RUNS}"
    )

    # ========================================================
    # 2. 读取已有进度
    # ========================================================

    if PROGRESS_FILE.exists():

        print()
        print("检测到已有进度文件。")

        with open(
            PROGRESS_FILE,
            "r",
            encoding="utf-8-sig"
        ) as f:

            results = json.load(f)

        # ----------------------------------------------------
        # 检查进度文件结构
        # ----------------------------------------------------

        if not isinstance(results, list):

            raise ValueError(
                "PROGRESS_FILE 的JSON顶层结构不是list，"
                f"实际类型为: {type(results)}"
            )

        print(
            f"检测到已有进度: "
            f"{len(results)} 对"
        )

    else:

        results = []

        print()
        print(
            "没有检测到已有进度，"
            "从第 1 对开始。"
        )

    # ========================================================
    # 3. 建立已完成索引
    # ========================================================

    completed_keys = {
        (
            x["variable_a"],
            x["variable_b"]
        )
        for x in results
    }

    print(
        f"已完成变量对数量: "
        f"{len(completed_keys)}"
    )

    print(
        f"剩余变量对数量: "
        f"{len(pairs) - len(completed_keys)}"
    )

    # ========================================================
    # 4. 加载模型
    # ========================================================

    print()
    print("正在加载本地 Qwen 模型...")

    tokenizer, model = load_model()

    print("模型加载完成。")

    # ========================================================
    # 5. 变量对循环
    # ========================================================

    total_pairs = len(pairs)

    for pair_index, pair in enumerate(
        pairs,
        start=1
    ):

        # ----------------------------------------------------
        # 获取变量名
        # ----------------------------------------------------

        variable_a = pair["variable_a"]
        variable_b = pair["variable_b"]

        pair_key = (
            variable_a,
            variable_b
        )

        # ====================================================
        # 5.1 断点续跑
        # ====================================================

        if pair_key in completed_keys:

            print()
            print(
                f"[{pair_index}/{total_pairs}] "
                f"{variable_a} -> {variable_b} "
                f"已完成，跳过。"
            )

            continue

        # ====================================================
        # 5.2 当前变量对
        # ====================================================

        print()
        print("=" * 70)

        print(
            f"处理第 "
            f"{pair_index}/{total_pairs} 个变量对"
        )

        print(
            f"A = {variable_a}"
        )

        print(
            f"B = {variable_b}"
        )

        print("=" * 70)

        # ====================================================
        # 5.3 构建Prompt
        # ====================================================

        prompt = build_causal_prompt(
            variable_a,
            variable_b
        )

        # ----------------------------------------------------
        # 当前变量对的5次运行结果
        # ----------------------------------------------------

        runs = []

        # ====================================================
        # 5.4 5次随机采样
        # ====================================================

        for run_id in range(
            1,
            N_RUNS + 1
        ):

            print()
            print(
                f"第 {run_id}/{N_RUNS} 次"
            )

            # ------------------------------------------------
            # 固定但不同的随机种子
            # ------------------------------------------------

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

            print(
                f"随机种子: {seed}"
            )

            # ------------------------------------------------
            # 调用模型
            # ------------------------------------------------

            try:

                response = generate_prior(
                    tokenizer,
                    model,
                    prompt,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_new_tokens=MAX_NEW_TOKENS,
                )

            except Exception as e:

                print()
                print(
                    "模型推理失败:"
                )

                print(
                    repr(e)
                )

                # --------------------------------------------
                # 保存失败信息
                # --------------------------------------------

                runs.append(
                    {
                        "run_id": run_id,
                        "seed": seed,
                        "raw_response": None,
                        "parsed": None,
                        "parse_status": "generation_failed",
                        "error": repr(e)
                    }
                )

                # --------------------------------------------
                # 当前run失败后继续下一次
                # --------------------------------------------

                continue

            # ------------------------------------------------
            # 输出原始结果
            # ------------------------------------------------

            print()
            print(
                "原始输出:"
            )

            print(
                response
            )

            # =================================================
            # 解析JSON
            # =================================================

            try:

                parsed = extract_json(
                    response
                )

                parse_status = "success"

                print()
                print(
                    "JSON解析成功:"
                )

                print(
                    json.dumps(
                        parsed,
                        ensure_ascii=False,
                        indent=2
                    )
                )

            except Exception as e:

                parsed = None

                parse_status = "failed"

                print()
                print(
                    "JSON解析失败:"
                )

                print(
                    repr(e)
                )

            # ------------------------------------------------
            # 保存当前run
            # ------------------------------------------------

            runs.append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "raw_response": response,
                    "parsed": parsed,
                    "parse_status": parse_status
                }
            )

        # ====================================================
        # 5.5 5次结果聚合
        # ====================================================

        aggregated = aggregate_relation(
            runs
        )

        # ====================================================
        # 5.6 当前变量对最终结果
        # ====================================================

        result = {

            "variable_a":
                variable_a,

            "variable_b":
                variable_b,

            "runs":
                runs,

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

        # ====================================================
        # 5.7 添加到总结果
        # ====================================================

        results.append(
            result
        )

        completed_keys.add(
            pair_key
        )

        # ====================================================
        # 5.8 每完成一对立即保存
        # ====================================================

        save_json(
            PROGRESS_FILE,
            results
        )

        # ====================================================
        # 5.9 输出当前进度
        # ====================================================

        print()
        print(
            "-" * 70
        )

        print(
            "当前变量对完成"
        )

        print(
            f"A = {variable_a}"
        )

        print(
            f"B = {variable_b}"
        )

        print(
            "最终关系:",
            result["final_relation"]
        )

        print(
            "最终置信度:",
            result["final_confidence"]
        )

        print(
            "关系统计:",
            result["relation_count"]
        )

        print(
            f"当前已完成: "
            f"{len(results)} / {total_pairs}"
        )

        print(
            f"当前进度: "
            f"{len(results) / total_pairs * 100:.2f}%"
        )

        print(
            "进度文件已保存:"
        )

        print(
            PROGRESS_FILE
        )

        print(
            "-" * 70
        )

    # ========================================================
    # 6. 全部完成
    # ========================================================

    save_json(
        FINAL_FILE,
        results
    )

    # ========================================================
    # 7. 最终统计
    # ========================================================

    success_pair_count = 0

    for result in results:

        runs = result.get(
            "runs",
            []
        )

        successful_runs = sum(
            1
            for run in runs
            if run.get(
                "parse_status"
            ) == "success"
        )

        if successful_runs == N_RUNS:

            success_pair_count += 1

    # ========================================================
    # 8. 输出最终结果
    # ========================================================

    print()
    print("=" * 70)
    print("全部变量对处理完成")
    print("=" * 70)

    print(
        "完成数量:",
        len(results),
        "/",
        total_pairs
    )

    print(
        "5次JSON全部解析成功的变量对:",
        success_pair_count,
        "/",
        total_pairs
    )

    print()
    print(
        "最终结果:"
    )

    print(
        FINAL_FILE
    )

    print()
    print(
        "进度文件:"
    )

    print(
        PROGRESS_FILE
    )

    print("=" * 70)


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":

    main()
