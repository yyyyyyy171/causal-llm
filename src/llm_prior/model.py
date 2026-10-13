"""
本地语言模型加载。

默认沿用原来的 Qwen3-8B + 8bit 配置（4090 上使用）。
本机是 6GB 显存的 RTX 3060，因此实测走
Qwen3-4B-Instruct-2507 + 4bit（见 model_4b_backup.py 的思路），
通过 load_model(quant="4bit") 指定。
"""

from __future__ import annotations

import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_NAME_4B = "Qwen/Qwen3-4B-Instruct-2507"


def _quant_config(quant: str, max_memory: dict | None):
    quant = quant.lower()

    if quant == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    if quant == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )

    if quant in ("none", "fp16"):
        return None

    raise ValueError(f"未知量化方式: {quant}")


def load_model(
    model_name: str = MODEL_NAME,
    quant: str = "8bit",
    max_memory: dict | None = None,
    local_files_only: bool = True,
    device_map="auto",
):
    """
    加载本地语言模型。

    local_files_only=True 时只用本地缓存（本机 HF_HOME=D:\\hf-cache），
    避免在没有网络的环境里卡在下载上。
    """

    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    print(f"正在加载模型 {model_name} (quant={quant}) ...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    # 批量生成需要左侧 padding
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "quantization_config": _quant_config(quant, max_memory),
        "device_map": device_map,
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }

    if max_memory is not None:
        kwargs["max_memory"] = max_memory

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    print("模型加载完成。")

    return tokenizer, model
