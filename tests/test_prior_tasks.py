"""
把三种任务（direction / mb / role）的聚合结果各写一次汇总表。

这个测试是为了防住一次真实事故：加 role 任务时只改了 transformers 版的
CSV 写出逻辑，vLLM 版还留着旧的"mb 或 direction"二选一，于是 role 的
结果走到 else 分支去取 item['direct_edge'] —— 那个字段在 role 聚合里
根本不存在，直接 KeyError。生成跑了 140 次是对的，却卡在最后写表一步。

这里直接调用两个脚本共用的 summary_csv / write_summary_csv，
三种任务都过一遍，字段对不上就会当场炸。

运行：
    D:\Anaconda\envs\llm_causal_fs\python.exe tests\test_prior_tasks.py
（用哪个解释器都行，它不加载模型，只跑聚合与写表。）
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(r"E:\Causal-LLM")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_llm_prior as m                      # noqa: E402


def fake_runs(task: str):
    """造 5 次采样，形状与真实缓存记录一致（script 只取 parsed 与 run_id）。"""

    if task == "role":
        answers = [{"role_of_a": r, "confidence": 0.8}
                   for r in ("parent", "parent", "spouse", "none", "uncertain")]
    elif task == "mb":
        answers = [{"in_markov_blanket": b, "confidence": 0.8}
                   for b in (True, True, False, False, True)]
    else:
        answers = ([{"direct_edge": True, "relation": "A_to_B", "confidence": 0.9}] * 3
                   + [{"direct_edge": False, "relation": "no_direct_edge", "confidence": 0.6}] * 2)

    return [{"parsed": answer, "run_id": index}
            for index, answer in enumerate(answers, start=1)]


failures = []

for task in ("direction", "mb", "role"):
    runs = fake_runs(task)
    try:
        aggregated = m.aggregate(runs, task=task)
        item = {"variable_a": "a", "variable_b": "b", "runs": runs, **aggregated}

        header, line = m.summary_csv(task)
        row = line(item)

        out = Path(tempfile.gettempdir()) / f"test_summary_{task}.csv"
        m.write_summary_csv(out, task, [item])

        print(f"== {task} ==")
        print("  表头:", header.strip())
        print("  一行:", row.strip())
        print("  聚合键:", sorted(aggregated.keys()))
    except Exception as error:                  # noqa: BLE001
        failures.append(f"{task}: {type(error).__name__}: {error}")
        print(f"== {task} == 失败: {type(error).__name__}: {error}")

print()
if failures:
    print("有任务没通过：")
    for item in failures:
        print("  -", item)
    raise SystemExit(1)

print("三种任务的汇总表写出全部通过。")
