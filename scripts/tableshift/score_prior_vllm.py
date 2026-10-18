"""
用 vLLM 给 TableShift 的 (特征, 标签) 对打角色分（Colab / Kaggle GPU）。

输入 data/tableshift/<task>/prior_prompts.jsonl
输出 data/tableshift/<task>/prior_role.csv    feature, final_role, confidence, score, consistency
     data/tableshift/<task>/prior_raw.jsonl   每次采样的原始输出（审计用）

score 换算与 bnlearn 主实验一致：
    parent / child / spouse -> +confidence
    none                    -> -confidence
    uncertain               -> 0

用法（在装好 vLLM 的环境里）：
    python scripts/tableshift/score_prior_vllm.py --model Qwen/Qwen3-8B-AWQ --samples 5
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

POSITIVE = ("parent", "child", "spouse")
FENCE = re.compile("^\x60{3}(?:json)?|" + "\x60{3}$")


def parse_json(text: str):
    cleaned = FENCE.sub("", text.strip()).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    return None


def make_sampling_params(SamplingParams, max_tokens: int, seed: int):
    params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=max_tokens, n=1, seed=seed)
    actual = getattr(params, "max_tokens", None)
    if actual != max_tokens:
        raise SystemExit(f"SamplingParams.max_tokens 未生效（读回 {actual!r}），先修参数名再跑")
    return params


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/tableshift")
    parser.add_argument("--model", default="Qwen/Qwen3-8B-AWQ")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    for task_dir in sorted(p for p in Path(args.data).iterdir() if p.is_dir()):
        prompts_path = task_dir / "prior_prompts.jsonl"
        if not prompts_path.exists():
            continue
        records = [json.loads(line) for line in prompts_path.read_text(encoding="utf-8").splitlines() if line]

        chats = []
        for _ in range(args.samples):
            for record in records:
                message = [{"role": "user", "content": record["prompt"]}]
                try:
                    text = tokenizer.apply_chat_template(
                        message, tokenize=False, add_generation_prompt=True, enable_thinking=False
                    )
                except TypeError:
                    text = tokenizer.apply_chat_template(
                        message, tokenize=False, add_generation_prompt=True
                    )
                chats.append(text)

        outputs = llm.generate(chats, make_sampling_params(SamplingParams, args.max_tokens, 20261018))

        by_feature = {}
        raw_rows = []
        for index, output in enumerate(outputs):
            record = records[index % len(records)]
            text = output.outputs[0].text
            parsed = parse_json(text)
            raw_rows.append({"task": task_dir.name, "feature": record["feature"],
                             "sample": index // len(records), "raw": text, "parsed": parsed})
            by_feature.setdefault(record["feature"], []).append(parsed)

        rows = []
        for feature, answers in by_feature.items():
            roles = [(a or {}).get("role", "uncertain") for a in answers]
            roles = [r if r in POSITIVE + ("none", "uncertain") else "uncertain" for r in roles]
            final_role, count = Counter(roles).most_common(1)[0]
            confidences = [float((a or {}).get("confidence", 0) or 0)
                           for a, r in zip(answers, roles) if r == final_role]
            confidence = sum(confidences) / len(confidences) if confidences else 0.0
            score = 0.0 if final_role == "uncertain" else (
                confidence if final_role in POSITIVE else -confidence)
            rows.append({"feature": feature, "final_role": final_role,
                         "confidence": round(confidence, 4), "score": round(score, 4),
                         "consistency": round(count / len(roles), 3)})

        with open(task_dir / "prior_role.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["feature", "final_role", "confidence", "score", "consistency"])
            writer.writeheader()
            writer.writerows(rows)
        with open(task_dir / "prior_raw.jsonl", "w", encoding="utf-8") as handle:
            for row in raw_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{task_dir.name}] {len(rows)} 个特征 -> prior_role.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
