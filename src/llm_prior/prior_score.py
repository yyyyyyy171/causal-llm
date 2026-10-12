import json


def parse_llm_response(raw_response):
    """
    将LLM返回的JSON字符串解析成结构化结果。
    """

    try:
        result = json.loads(raw_response)

        relation = result.get("relation", "uncertain")
        confidence = result.get("confidence", 0.0)
        reason = result.get("reason", "")

        return {
            "relation": relation,
            "confidence": float(confidence),
            "reason": reason
        }

    except Exception as e:
        print("JSON解析失败：")
        print(raw_response)
        print("错误信息：", e)

        return {
            "relation": "uncertain",
            "confidence": 0.0,
            "reason": "JSON解析失败"
        }


def parse_results(input_file, output_file):
    """
    读取LLM原始结果，并转换成结构化结果。
    """

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:
        results = json.load(f)

    parsed_results = []

    for item in results:

        variable_a = item["variable_a"]
        variable_b = item["variable_b"]

        raw_response = item["raw_response"]

        parsed = parse_llm_response(
            raw_response
        )

        result = {
            "variable_a": variable_a,
            "variable_b": variable_b,
            "relation": parsed["relation"],
            "confidence": parsed["confidence"],
            "reason": parsed["reason"]
        }

        parsed_results.append(result)

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            parsed_results,
            f,
            ensure_ascii=False,
            indent=4
        )

    print()
    print("=" * 60)
    print("LLM先验解析完成！")
    print("结果保存到：")
    print(output_file)
    print("=" * 60)


if __name__ == "__main__":

    input_file = (
        r"E:\LLM-Causal-FS"
        r"\results\llm_prior"
        r"\llm_prior_results.json"
    )

    output_file = (
        r"E:\LLM-Causal-FS"
        r"\results\llm_prior"
        r"\llm_prior_structured.json"
    )

    parse_results(
        input_file,
        output_file
    )