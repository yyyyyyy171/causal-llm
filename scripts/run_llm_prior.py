"""
选题二 周5 实验：让本地语言模型判断变量对的因果方向。

相对旧脚本 run_alarm_llm_prior.py 的改进：

1. 支持全部 5 个网络（ALARM/ASIA/CHILD/INSURANCE/HAILFINDER）。
2. 支持真实变量名与匿名变量名两种模式（匿名消融）。
3. 每一次生成都落盘缓存（JSONL），同一输入永不重复计算
   ——对应文档里"所有语言模型的输入输出都缓存到磁盘"的约定。
4. 断点续跑：中断后重新运行会跳过已完成的部分。
5. 批量生成，充分利用显卡。
6. 聚合时额外记录 consistency（5 次采样的一致比例），
   用于判断先验是否"稳定"。

用法示例：

    python scripts/run_llm_prior.py --network ALARM --mode real --limit 20
    python scripts/run_llm_prior.py --network ASIA --mode both
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_prior.inference import generate_batch  # noqa: E402
from src.llm_prior.model import MODEL_NAME_4B, load_model  # noqa: E402
from src.llm_prior.prompt import (  # noqa: E402
    MB_PROMPT_VERSION,
    PROMPT_VERSION,
    build_causal_prompt,
    build_markov_blanket_prompt,
)
from src.llm_prior.var_semantics import load_variables  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.common.run_log import finish_run, start_run  # noqa: E402


DATA_ROOT = PROJECT_ROOT / "data"


def model_suffix(model_name: str, default_model: str) -> str:
    """
    缓存与结果文件名里的模型标识。

    同一个网络用不同规模的模型跑出来的先验必须分开存放，否则 8B 会直接
    命中 4B 留下的缓存。默认模型不加后缀，保证已有结果不被改名。
    """

    if model_name == default_model:
        return ""
    import re

    tag = re.sub(r"[^a-z0-9]+", "-", model_name.split("/")[-1].lower()).strip("-")
    return f"_{tag}"


# ============================================================
# 工具函数
# ============================================================

def extract_json(text: str):
    """从模型输出里抠出第一个合法 JSON 对象。"""

    text = (text or "").strip()

    # 去掉可能的 markdown 代码围栏
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None

    return None


def normalize_relation(parsed: dict | None) -> str:
    """把解析结果压成 4 种关系之一。"""

    if not parsed:
        return "uncertain"

    relation = str(parsed.get("relation", "uncertain")).strip()

    allowed = {"A_to_B", "B_to_A", "no_direct_edge", "uncertain"}

    return relation if relation in allowed else "uncertain"


def run_positive(parsed: dict | None) -> bool:
    """
    单次生成里的"肯定"判断。

    task=direction 时是"是否直接相连"；task=mb 时是"是否在马尔可夫毯里"。
    优先用模型自己填的布尔字段（in_markov_blanket / direct_edge），
    没有时用 relation 推导，保证不同 Prompt 版本的结果都能聚合。
    """

    if not parsed:
        return False

    for field in ("in_markov_blanket", "direct_edge"):
        if field in parsed:
            value = parsed[field]
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ("true", "yes", "1")

    return normalize_relation(parsed) in ("A_to_B", "B_to_A")


def aggregate(runs: list[dict], task: str = "direction") -> dict:
    """
    多数投票 + 一致性。

    consistency = 得票最多的关系占比；
    只有 5 次全部一致（consistency=1.0）才算"稳定先验"。
    """

    relations = [normalize_relation(r.get("parsed")) for r in runs]
    counts: dict[str, int] = {}

    for relation in relations:
        counts[relation] = counts.get(relation, 0) + 1

    final_relation = max(counts, key=counts.get)

    # "肯定/否定"单独投一次票：这个信号给融合方法当搜索约束用
    edge_votes = [run_positive(r.get("parsed")) for r in runs]
    n_edge = sum(1 for v in edge_votes if v)
    positive = n_edge * 2 > len(edge_votes)

    confidences = [
        float(r["parsed"].get("confidence", 0.0))
        for r in runs
        if r.get("parsed") and normalize_relation(r["parsed"]) == final_relation
    ]

    final_confidence = (
        sum(confidences) / len(confidences) if confidences else 0.0
    )

    result = {
        "final_relation": final_relation,
        "final_confidence": final_confidence,
        "consistency": counts[final_relation] / max(1, len(relations)),
        "relation_count": counts,
    }

    if task == "mb":
        result["in_markov_blanket"] = positive
        result["positive_ratio"] = n_edge / max(1, len(edge_votes))
        # relation_count 对毯成员任务没有意义，改成记录赞成/反对票数更清楚
        result["relation_count"] = {
            "in_blanket": n_edge,
            "not_in_blanket": len(edge_votes) - n_edge,
        }
    else:
        result["direct_edge"] = positive
        result["edge_ratio"] = n_edge / max(1, len(edge_votes))

    return result


def pair_list(network: str) -> list[tuple[str, str]]:
    """网络内全部无序变量对，顺序由数据表头决定（与旧脚本一致）。"""

    variables = load_variables(network)
    return list(combinations(variables, 2))


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


# ============================================================
# 缓存
# ============================================================

def cache_path(network: str, mode: str, task: str = "direction",
               suffix: str = "") -> Path:
    # 方向任务沿用原来的文件名，避免让已有缓存失效；
    # 马尔可夫毯任务用 mb_ 前缀单独存。
    name = f"{mode}.jsonl" if task == "direction" else f"{task}_{mode}.jsonl"
    if suffix:
        name = f"{name[:-6]}{suffix}.jsonl"
    return DATA_ROOT / network / "llm_cache" / name


def load_cache(network: str, mode: str, task: str, expected_version: str,
               suffix: str = ""):
    path = cache_path(network, mode, task, suffix)
    done: dict[tuple[str, str, int], dict] = {}

    if not path.exists():
        return done

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            # 只认当前 prompt 版本的缓存，旧版本的记录直接忽略
            if record.get("prompt_version") != expected_version:
                continue
            key = (record["variable_a"], record["variable_b"], record["run_id"])
            done[key] = record

    return done


def append_cache(network: str, mode: str, task: str, record: dict,
                 suffix: str = "") -> None:
    path = cache_path(network, mode, task, suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# 主流程
# ============================================================

def run_mode(
    args,
    tokenizer,
    model,
    network: str,
    mode: str,
    pairs: list[tuple[str, str]],
) -> None:

    anonymous = mode == "anon"
    task = args.task
    suffix = model_suffix(args.model, MODEL_NAME_4B)

    prompt_version = MB_PROMPT_VERSION if task == "mb" else PROMPT_VERSION

    def build_prompt(a: str, b: str) -> str:
        if task == "mb":
            return build_markov_blanket_prompt(
                a, b, network=network, anonymous=anonymous
            )
        return build_causal_prompt(a, b, network=network, anonymous=anonymous)

    cache = load_cache(network, mode, task, prompt_version, suffix)
    print(f"[{network}/{mode}] 缓存中已有 {len(cache)} 次生成记录")

    results: dict[tuple[str, str], dict] = {}

    # 把缓存里已有的结果重建出来
    for (a, b, _run), record in cache.items():
        entry = results.setdefault(
            (a, b),
            {"variable_a": a, "variable_b": b, "runs": []},
        )
        entry["runs"].append(record)

    for entry in results.values():
        entry["runs"].sort(key=lambda r: r["run_id"])

    todo: list[tuple[str, str, int]] = [
        (a, b, run_id)
        for (a, b) in pairs
        for run_id in range(1, args.runs + 1)
        if (a, b, run_id) not in cache
    ]

    total = len(todo)
    done_count = 0
    started = time.time()

    print(f"[{network}/{mode}] 待生成 {total} 次（batch={args.batch_size}）")

    for start in range(0, total, args.batch_size):

        batch = todo[start:start + args.batch_size]

        prompts = [build_prompt(a, b) for (a, b, _run) in batch]

        seed = args.base_seed + start
        set_seed(seed)

        outputs = generate_batch(
            tokenizer,
            model,
            prompts,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )

        for (a, b, run_id), raw in zip(batch, outputs):

            parsed = extract_json(raw)

            record = {
                "variable_a": a,
                "variable_b": b,
                "run_id": run_id,
                "seed": seed,
                "prompt_version": prompt_version,
                "raw_response": raw,
                "parsed": parsed,
                "parse_status": "success" if parsed else "failed",
            }

            append_cache(network, mode, task, record, suffix)

            entry = results.setdefault(
                (a, b), {"variable_a": a, "variable_b": b, "runs": []}
            )
            entry["runs"].append(record)

            done_count += 1

        elapsed = time.time() - started
        rate = done_count / elapsed if elapsed > 0 else 0.0
        remaining = (total - done_count) / rate if rate > 0 else 0.0

        print(
            f"  {done_count}/{total} 生成完成 "
            f"({rate:.2f} 次/秒, 预计剩余 {remaining/60:.1f} 分钟)",
            flush=True,
        )

    # --------------------------------------------------------
    # 聚合与落盘
    # --------------------------------------------------------

    final = []

    for (a, b), entry in sorted(results.items()):
        runs = sorted(entry["runs"], key=lambda r: r["run_id"])
        aggregated = aggregate(runs, task=task)
        final.append(
            {
                "variable_a": a,
                "variable_b": b,
                "runs": runs,
                **aggregated,
            }
        )

    prefix = "LLM_prior" if task == "direction" else "LLM_mb"
    out_json = DATA_ROOT / network / f"{network}_{prefix}_{mode}{suffix}.json"
    save_json(out_json, final)

    out_csv = DATA_ROOT / network / f"{network}_{prefix}_{mode}{suffix}_summary.csv"

    if task == "mb":
        header = (
            "variable_a,variable_b,in_markov_blanket,positive_ratio,"
            "consistency,final_confidence,relation_count\n"
        )
        def fields(item):
            return (
                f"{item['variable_a']},{item['variable_b']},"
                f"{item['in_markov_blanket']},{item['positive_ratio']:.3f},"
                f"{item['consistency']:.3f},{item['final_confidence']:.3f},"
                f"\"{json.dumps(item['relation_count'], ensure_ascii=False)}\"\n"
            )
    else:
        header = (
            "variable_a,variable_b,direct_edge,edge_ratio,"
            "final_relation,consistency,final_confidence,relation_count\n"
        )
        def fields(item):
            return (
                f"{item['variable_a']},{item['variable_b']},"
                f"{item['direct_edge']},{item['edge_ratio']:.3f},"
                f"{item['final_relation']},{item['consistency']:.3f},"
                f"{item['final_confidence']:.3f},"
                f"\"{json.dumps(item['relation_count'], ensure_ascii=False)}\"\n"
            )

    with open(out_csv, "w", encoding="utf-8") as handle:
        handle.write(header)
        for item in final:
            handle.write(fields(item))

    print(f"[{network}/{mode}] 完成 {len(final)} 对 -> {out_json.name}")
    print(f"[{network}/{mode}] 汇总表 -> {out_csv.name}")


def set_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:

    # 先读 --config，再用配置里的值作为各参数的默认值，
    # 这样"配置文件 + 命令行覆盖"两件事同时成立。
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="llm_prior")
    pre_args, _ = pre.parse_known_args()

    try:
        cfg = load_config(pre_args.config)
    except FileNotFoundError:
        cfg = {}

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument(
        "--task", default="direction", choices=("direction", "mb"),
        help="direction=问因果方向；mb=问是否同属马尔可夫毯",
    )
    parser.add_argument(
        "--network", default=(cfg.get("networks") or ["ALARM"])[0]
    )
    parser.add_argument(
        "--mode",
        default="real",
        choices=("real", "anon", "both"),
    )
    parser.add_argument("--runs", type=int, default=int(cfg.get("runs", 5)))
    parser.add_argument("--limit", type=int, default=0,
                        help="只跑前 N 对（0 表示全部）")
    parser.add_argument("--sample", type=int, default=0,
                        help="随机抽 N 对（0 表示不抽样）")
    parser.add_argument(
        "--sample-seed", type=int,
        default=int(cfg.get("base_seed", 20261012)),
    )
    parser.add_argument(
        "--batch-size", type=int, default=int(cfg.get("batch_size", 8))
    )
    parser.add_argument("--model", default=cfg.get("model", MODEL_NAME_4B))
    parser.add_argument("--quant", default=cfg.get("quant", "4bit"))
    parser.add_argument(
        "--device-map", default="auto", choices=("auto", "gpu0"),
        help="auto=按显存自动分层；gpu0=全部放 0 号卡（4bit 不支持 CPU 卸载时用）",
    )
    parser.add_argument(
        "--temperature", type=float, default=float(cfg.get("temperature", 0.7))
    )
    parser.add_argument(
        "--top-p", type=float, default=float(cfg.get("top_p", 0.9))
    )
    parser.add_argument(
        "--max-new-tokens", type=int,
        default=int(cfg.get("max_new_tokens", 192)),
    )
    parser.add_argument(
        "--base-seed", type=int, default=int(cfg.get("base_seed", 20261010))
    )

    args = parser.parse_args()

    started_log = start_run("run_llm_prior", config=cfg)
    print(f"日志文件: {started_log}")

    pairs = pair_list(args.network)

    if args.sample:
        rng = random.Random(args.sample_seed)
        pairs = rng.sample(pairs, min(args.sample, len(pairs)))
        pairs = sorted(pairs)

    if args.limit:
        pairs = pairs[: args.limit]

    print(f"网络 {args.network}: {len(pairs)} 对 x {args.runs} 次")

    modes = ("real", "anon") if args.mode == "both" else (args.mode,)

    tokenizer, model = load_model(
        model_name=args.model,
        quant=args.quant,
        local_files_only=True,
        device_map={"": 0} if args.device_map == "gpu0" else "auto",
    )

    for mode in modes:
        run_mode(args, tokenizer, model, args.network, mode, pairs)

    finish_run(0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
