"""
批量推理。

与原版的差别：支持一次前向生成多条 prompt（--batch-size），
在 6GB 显卡上能显著摊薄耗时；同时兼容不支持 enable_thinking
参数的模型（如 Qwen3-4B-Instruct-2507）。
"""

from __future__ import annotations

import torch


def _apply_chat_template(tokenizer, messages_list):
    """
    逐步降级地套用 chat template，兼容不同版本的 transformers 与不同模型。
    """

    attempts = (
        dict(tokenize=False, add_generation_prompt=True, enable_thinking=False),
        dict(tokenize=False, add_generation_prompt=True),
    )

    last_error = None

    for kwargs in attempts:
        try:
            return [
                tokenizer.apply_chat_template(messages, **kwargs)
                for messages in messages_list
            ]
        except TypeError as error:
            last_error = error
            continue

    raise RuntimeError(f"无法套用 chat template: {last_error}")


@torch.inference_mode()
def generate_batch(
    tokenizer,
    model,
    prompts,
    do_sample=False,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=192,
):
    """
    对一批 prompt 生成回复，返回与 prompts 等长的字符串列表。
    """

    messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]

    texts = _apply_chat_template(tokenizer, messages_list)

    model_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }

    if do_sample:
        generation_kwargs.update(temperature=temperature, top_p=top_p)

    generated_ids = model.generate(**model_inputs, **generation_kwargs)

    new_tokens = generated_ids[:, model_inputs.input_ids.shape[1]:]

    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def generate_prior(
    tokenizer,
    model,
    prompt,
    do_sample=False,
    temperature=0.7,
    top_p=0.8,
    max_new_tokens=256,
):
    """单条生成的兼容接口，保留旧调用方式。"""

    return generate_batch(
        tokenizer,
        model,
        [prompt],
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )[0]
