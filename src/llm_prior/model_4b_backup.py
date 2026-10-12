import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)


MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


def load_model():
    """
    加载Qwen模型。
    使用4-bit量化，适合RTX 3060 6GB。
    """

    print("正在加载Qwen模型...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Qwen模型加载完成！")

    return tokenizer, model