"""
用 vLLM 跑语言模型先验（供 Google Colab / T4 或 Linux 4090 使用）。

和 run_llm_prior.py 的关系：缓存格式、断点续跑、聚合方式、输出文件名
完全一致，只是把"生成"这一层从 transformers 换成 vLLM 的批量推理。
因此本机（transformers）与 Colab（vLLM）产出的结果可以混用、可以互相续跑。

为什么需要它：本机是 Windows + 6GB 显卡，vLLM 没有 Windows 发行版，
且 8B 在 6GB 上会因显存分页严重退化。T4（16GB）官方支持，
连续批处理能把几千条短提示的吞吐提高一个数量级。

用法（在 Colab / Linux 上）：
    python scripts/run_llm_prior_vllm.py --network ASIA --mode real --task direction
    python scripts/run_llm_prior_vllm.py --network CHILD --mode both --model Qwen/Qwen3-8B-AWQ
"""

from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.common.config import load_config  # noqa: E402
from src.common.run_log import finish_run, start_run  # noqa: E402

# 复用 transformers 版里的公共部件，保证两条路径行为一致
from run_llm_prior import (  # noqa: E402
    DATA_ROOT,
    aggregate,
    append_cache,
    build_causal_prompt,
    build_markov_blanket_prompt,
    extract_json,
    load_cache,
    model_suffix,
    pair_list,
    save_json,
)
from src.llm_prior.prompt import MB_PROMPT_VERSION, PROMPT_VERSION  # noqa: E402


def build_sampling_params(SamplingParams, **wanted):
    """
    只传当前版本 vLLM 真正支持的字段。

    vLLM 的参数名在版本之间变动过（例如 seed、max_tokens 与 max_new_tokens），
    这里按签名过滤，避免换一个 Colab 镜像就报 TypeError。
    """

    supported = inspect.signature(SamplingParams.__init__).parameters

    # 新旧命名兼容：优先 max_tokens，其次 max_new_tokens
    if "max_new_tokens" in wanted and "max_tokens" in supported:
        wanted["max_tokens"] = wanted.pop("max_new_tokens")

    accepted = {key: value for key, value in wanted.items() if key in supported}
    dropped = sorted(set(wanted) - set(accepted))

    if dropped:
        print(f"  注意：当前 vLLM 不支持这些采样参数，已忽略 {dropped}")

    return SamplingParams(**accepted)


def main() -> int:

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="llm_prior")
    parser.add_argument("--network", default="ASIA")
    parser.add_argument("--mode", default="real", choices=("real", "anon", "both"))
    parser.add_argument("--task", default="direction", choices=("direction", "mb"))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3-8B-AWQ")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--base-seed", type=int, default=20261010)
    parser.add_argument("--requests-per-call", type=int, default=256,
                        help="一次 llm.chat 提交多少条请求（由调度器批处理）")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--enforce-eager", action="store_true",
        help="关闭 CUDA 图捕获。T4（Turing）上引擎初始化失败时先试这个",
    )
    parser.add_argument(
        "--quantization", default="",
        help="量化方式，一般留空让 vLLM 自动识别；AWQ 认不出来时填 awq",
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        cfg = {}

    started_log = start_run("run_llm_prior_vllm", config=cfg)
    print(f"日志文件: {started_log}")

    # vLLM 只在真正要生成时才导入，便于在没装的机器上先跑 --help
    from vllm import LLM, SamplingParams

    pairs = pair_list(args.network)
    if args.sample:
        import random
        pairs = sorted(random.Random(args.base_seed).sample(pairs, min(args.sample, len(pairs))))
    if args.limit:
        pairs = pairs[: args.limit]

    print(f"网络 {args.network}: {len(pairs)} 对 x {args.runs} 次；模型 {args.model}")

    print("正在加载 vLLM 引擎 ...")
    load_started = time.time()
    engine_kwargs = {
        "model": args.model,
        "dtype": "float16",           # T4 不支持 bfloat16
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,
    }
    if args.enforce_eager:
        engine_kwargs["enforce_eager"] = True
    if args.quantization:
        engine_kwargs["quantization"] = args.quantization

    llm = LLM(**engine_kwargs)
    print(f"引擎就绪，耗时 {time.time() - load_started:.1f}s")

    tokenizer = llm.get_tokenizer()

    modes = ("real", "anon") if args.mode == "both" else (args.mode,)
    suffix = model_suffix(args.model, cfg.get("model", "Qwen/Qwen3-8B"))
    prompt_version = MB_PROMPT_VERSION if args.task == "mb" else PROMPT_VERSION

    for mode in modes:

        anonymous = mode == "anon"

        def build_prompt(a: str, b: str) -> str:
            if args.task == "mb":
                return build_markov_blanket_prompt(a, b, network=args.network, anonymous=anonymous)
            return build_causal_prompt(a, b, network=args.network, anonymous=anonymous)

        cache = load_cache(args.network, mode, args.task, prompt_version, suffix)
        print(f"[{args.network}/{mode}] 缓存中已有 {len(cache)} 次生成")

        results: dict = {}
        for (a, b, _run), record in cache.items():
            entry = results.setdefault((a, b), {"variable_a": a, "variable_b": b, "runs": []})
            entry["runs"].append(record)

        todo = [
            (a, b, run_id)
            for (a, b) in pairs
            for run_id in range(1, args.runs + 1)
            if (a, b, run_id) not in cache
        ]

        total = len(todo)
        print(f"[{args.network}/{mode}] 待生成 {total} 次（每 {args.requests_per_call} 条提交一次）")

        done = 0
        started = time.time()

        for start in range(0, total, args.requests_per_call):

            chunk = todo[start:start + args.requests_per_call]

            seed = args.base_seed + start
            sampling = build_sampling_params(
                SamplingParams,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                n=1,
                seed=seed,
            )

            conversations = [
                [{"role": "user", "content": build_prompt(a, b)}]
                for (a, b, _run) in chunk
            ]

            outputs = llm.chat(conversations, sampling, use_tqdm=False)

            for (a, b, run_id), request_output in zip(chunk, outputs):

                raw = request_output.outputs[0].text if request_output.outputs else ""
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

                append_cache(args.network, mode, args.task, record, suffix)

                entry = results.setdefault((a, b), {"variable_a": a, "variable_b": b, "runs": []})
                entry["runs"].append(record)
                done += 1

            elapsed = time.time() - started
            rate = done / elapsed if elapsed > 0 else 0.0
            remaining = (total - done) / rate if rate > 0 else 0.0
            print(f"  {done}/{total} 完成（{rate:.2f} 次/秒，预计剩余 {remaining/60:.1f} 分钟）",
                  flush=True)

        final = []
        for (a, b), entry in sorted(results.items()):
            runs = sorted(entry["runs"], key=lambda r: r["run_id"])
            final.append({"variable_a": a, "variable_b": b, "runs": runs,
                          **aggregate(runs, task=args.task)})

        prefix = "LLM_prior" if args.task == "direction" else "LLM_mb"
        out_json = DATA_ROOT / args.network / f"{args.network}_{prefix}_{mode}{suffix}.json"
        save_json(out_json, final)

        if args.task == "mb":
            header = ("variable_a,variable_b,in_markov_blanket,positive_ratio,"
                      "consistency,final_confidence,relation_count\n")
            def line(item):
                return (f"{item['variable_a']},{item['variable_b']},"
                        f"{item['in_markov_blanket']},{item['positive_ratio']:.3f},"
                        f"{item['consistency']:.3f},{item['final_confidence']:.3f},"
                        f"\"{item['relation_count']}\"\n")
        else:
            header = ("variable_a,variable_b,direct_edge,edge_ratio,"
                      "final_relation,consistency,final_confidence,relation_count\n")
            def line(item):
                return (f"{item['variable_a']},{item['variable_b']},"
                        f"{item['direct_edge']},{item['edge_ratio']:.3f},"
                        f"{item['final_relation']},{item['consistency']:.3f},"
                        f"{item['final_confidence']:.3f},"
                        f"\"{item['relation_count']}\"\n")

        out_csv = DATA_ROOT / args.network / f"{args.network}_{prefix}_{mode}{suffix}_summary.csv"
        with open(out_csv, "w", encoding="utf-8") as handle:
            handle.write(header)
            for item in final:
                handle.write(line(item))

        print(f"[{args.network}/{mode}] 完成 {len(final)} 对 -> {out_json.name}、{out_csv.name}")

    finish_run(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
