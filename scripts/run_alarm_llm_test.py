import json
import sys
from pathlib import Path

# ============================================================
# 路径
# ============================================================

PROJECT_ROOT = Path(r"E:\Causal-LLM")
SOURCE_ROOT = Path(r"E:\Causal-LLM")

# 让 Python 找到你原来的模型代码
sys.path.insert(0, str(SOURCE_ROOT))

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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "ALARM"
    / "ALARM_LLM_test_5runs.json"
)


# ============================================================
# 从模型输出中提取 JSON
# ============================================================

def extract_json(text: str) -> dict:

    text = text.strip()

    # 情况1：完整输出就是 JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 情况2：JSON 外面有其他文字
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:

        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"无法从模型输出中解析 JSON:\n{text}"
    )


# ============================================================
# 主程序
# ============================================================

def main():

    print("=" * 60)
    print("ALARM LLM Prior 测试")
    print("=" * 60)

    # --------------------------------------------------------
    # 读取变量对
    # --------------------------------------------------------

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        pairs = json.load(f)

    print("变量对总数:", len(pairs))

    # 只测试第一对
    pair = pairs[0]

    variable_a = pair["variable_a"]
    variable_b = pair["variable_b"]

    print()
    print("测试变量对:")
    print("A =", variable_a)
    print("B =", variable_b)

    # --------------------------------------------------------
    # 加载模型
    # --------------------------------------------------------

    print()
    print("正在加载模型...")

    tokenizer, model = load_model()

    print("模型加载完成")

    # --------------------------------------------------------
    # 运行 5 次
    # --------------------------------------------------------

    runs = []

    for run_id in range(1, 6):

        print()
        print("=" * 60)
        print(f"第 {run_id}/5 次")
        print("=" * 60)

        prompt = build_causal_prompt(
            variable_a,
            variable_b
        )

        response = generate_prior(
    tokenizer,
    model,
    prompt,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=256,
)

        print()
        print("原始输出:")
        print(response)

        # ----------------------------------------------------
        # 解析 JSON
        # ----------------------------------------------------

        try:

            parsed = extract_json(response)

            print()
            print("JSON解析成功:")
            print(
                json.dumps(
                    parsed,
                    ensure_ascii=False,
                    indent=2
                )
            )

            parse_status = "success"

        except Exception as e:

            print()
            print("JSON解析失败:")
            print(e)

            parsed = None
            parse_status = "failed"

        runs.append(
            {
                "run_id": run_id,
                "raw_response": response,
                "parsed": parsed,
                "parse_status": parse_status
            }
        )

    # --------------------------------------------------------
    # 保存测试结果
    # --------------------------------------------------------

    result = {
        "variable_a": variable_a,
        "variable_b": variable_b,
        "runs": runs
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # 汇总
    # --------------------------------------------------------

    success_count = sum(
        1
        for run in runs
        if run["parse_status"] == "success"
    )

    print()
    print("=" * 60)
    print("测试完成")
    print("=" * 60)

    print(
        f"JSON解析成功: {success_count}/5"
    )

    print()
    print("结果保存:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()