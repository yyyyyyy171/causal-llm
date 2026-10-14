"""
检查候选模型在 Hugging Face 上是否存在、有多大、是否需要授权。

为什么需要：Colab 上要跑 8B 的 4bit 量化版（T4 16GB 放不下纯 fp16 的 8B），
而量化仓库的命名不完全统一（AWQ / GPTQ-Int4 / 社区版），
先确认哪个能下到，再把它填进 notebook 的候选列表。

用法（本机或 Colab 都行）：
    python scripts/check_hf_models.py
    python scripts/check_hf_models.py --token hf_xxxxxxxx      # 需要授权时

也可以先设置环境变量 HF_TOKEN，脚本会自动带上。
"""

from __future__ import annotations

import argparse
import os
import sys


CANDIDATES = [
    # 8B 的 4bit 量化版：T4 上首选（2026-10-14 实测存在，下载量 200 万+）
    "Qwen/Qwen3-8B-AWQ",
    # 4B：兜底，纯 fp16 在 T4 上约 8GB
    "Qwen/Qwen3-4B-Instruct-2507",
    # 8B 全精度：本机已有缓存；T4 16GB 放不下（约 16GB 权重），仅作对照
    "Qwen/Qwen3-8B",
]


def main() -> int:

    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()

    try:
        from huggingface_hub import model_info
    except ImportError:
        print("缺少 huggingface_hub，先安装：pip install -U huggingface_hub")
        return 1

    print(f"{'模型':<42s} {'状态':<10s} {'下载量':>8s} {'文件数':>6s}  说明")
    print("-" * 92)

    available = []

    for repo in CANDIDATES:

        try:
            # files_metadata=True 才会带回每个文件的大小，用来算模型体积
            info = model_info(
                repo, token=args.token or None, files_metadata=True
            )
        except Exception as error:  # noqa: BLE001
            name = type(error).__name__
            # 区分两类失败：RepositoryNotFound = 仓库不存在（名字写错或确实没有）；
            # GatedRepoError / 401 = 需要授权，这时才该加 --token 重试。
            if "Gated" in name or "401" in str(error):
                hint = "需要授权，加 --token 重试"
            elif "RepositoryNotFound" in name or "404" in str(error):
                hint = "仓库不存在（换一个候选）"
            else:
                hint = f"网络或其它错误（{name}）"
            print(f"{repo:<42s} {'不可用':<10s} {'-':>8s} {'-':>6s}  {hint}")
            continue

        files = [item for item in info.siblings if item.rfilename.endswith((".safetensors", ".bin"))]

        total = 0
        for item in info.siblings:
            size = getattr(item, "size", None)
            if size and item.rfilename.endswith((".safetensors", ".bin")):
                total += size

        size_note = f"约 {total / 1e9:.1f} GB" if total else "大小未知"
        gated = "gated" if getattr(info, "gated", False) else ""

        print(
            f"{repo:<42s} {'可用':<10s} {getattr(info, 'downloads', 0) or 0:>8d} "
            f"{len(files):>6d}  {size_note} {gated}"
        )
        available.append(repo)

    print()
    if available:
        print("可以直接填进 notebook 的候选（按推荐顺序）：")
        for repo in available:
            print("   ", repo)
        print()
        print("若上面有 8B 的 AWQ/GPTQ 版本，就用它；T4 16GB 放得下。")
    else:
        print("一个都不可用：先确认网络能连上 huggingface.co，需要授权的用 --token 重试。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
