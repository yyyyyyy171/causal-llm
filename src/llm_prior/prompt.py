"""
因果方向判断的 Prompt 构造。

与旧版本的差别：

* 旧版只写死了 ALARM 的 11 个变量含义，其余变量退化成
  "exact meaning is not provided"，模型只能靠缩写瞎猜。
* 新版从 var_semantics.py 取全网络（ALARM/ASIA/CHILD/INSURANCE/HAILFINDER）
  的变量含义，并支持**匿名模式**：变量名换成 V01, V02 ...，
  同时不提供任何含义与领域信息。

匿名模式是一次严格的消融：如果匿名后先验的收益消失，
说明起作用的是语义知识，而不是模型对经典因果图的记忆。
"""

from __future__ import annotations

from .var_semantics import (
    NETWORK_DOMAIN,
    anonymous_names,
    variable_meaning,
)


ALLOWED_RELATIONS = (
    "A_to_B",
    "B_to_A",
    "no_direct_edge",
    "uncertain",
)


# Prompt 每次改动都要升版本号：
# 推理缓存按版本号区分，旧版本的生成结果会自动失效而不必手工删除文件。
PROMPT_VERSION = "v3-no-invented-mediators"


FORMAT_EXAMPLE = """{
    "variable_a": "NAME_A",
    "variable_b": "NAME_B",
    "direct_edge": true,
    "relation": "A_to_B",
    "confidence": 0.75,
    "reason": "one sentence"
}"""


def build_causal_prompt(
    variable_a: str,
    variable_b: str,
    network: str = "ALARM",
    anonymous: bool = False,
) -> str:
    """
    构造一对变量的因果先验 Prompt。

    参数
    ----
    variable_a, variable_b
        变量名。匿名模式下会先被替换成匿名编号。
    network
        网络名，决定使用哪一份语义字典。
    anonymous
        是否匿名。True 时不提供变量名、含义和领域信息。
    """

    if anonymous:

        mapping = anonymous_names(network)

        name_a = mapping.get(variable_a, "V??")
        name_b = mapping.get(variable_b, "V??")

        setting = (
            "You are analyzing a probabilistic graphical model.\n"
            "The variables have been anonymized: their real names and\n"
            "meanings are hidden from you. You only know the two labels."
        )

        meaning_a = "hidden"
        meaning_b = "hidden"

    else:

        name_a = variable_a
        name_b = variable_b

        setting = (
            "You are analyzing the following Bayesian network:\n"
            f"{NETWORK_DOMAIN.get(network, network)}."
        )

        meaning_a = variable_meaning(network, variable_a)
        meaning_b = variable_meaning(network, variable_b)

    prompt = f"""
You are a causal analysis assistant.

{setting}

Your task is to provide a semantic causal prior for two variables.
This is NOT statistical causal discovery: you must answer from the
meaning of the variables alone, never from data.

Answer two separate questions.

Question 1: are A and B DIRECTLY connected by an edge?
A direct edge means that the model represents the influence of one
variable on the other without any other variable of the model in
between. Variables that are merely correlated, or related only through
a common cause or a common effect, are NOT directly connected.
Answer with the boolean field "direct_edge".

Question 2: if they are directly connected, which direction is it?
Choose "A_to_B" or "B_to_A". If you judge that A and B are not
directly connected, use "no_direct_edge". Use "uncertain" only when
you genuinely cannot answer either question.

Rules:

1. Answer only from the meaning of the variables. Never use a dataset,
   a published graph, or remembered experimental results.
2. Correlation does not imply a direct edge.
3. Do not force a direction, but do not hide behind "uncertain" either:
   whenever the semantics make one answer clearly more plausible,
   commit to it and express your doubt through a lower confidence.
4. "confidence" must be a number between 0 and 1 that reflects how
   confident you are in the relation you chose. Do not output 0 unless
   you truly have no information at all.
5. Do not invent intermediate variables. If A plausibly influences the
   state of B but the mechanism has no variable of its own in the model,
   that is still a direct edge. Concretely, the answer is "direct edge"
   when one variable is the cause of the other's state - for example a
   risk exposure causing a disease, a disease causing its symptom, a
   physical quantity driving the quantity derived from it, or a setting
   controlling the device it configures.
6. "direct_edge" and "relation" must be consistent: direct_edge is true
   exactly when the relation is "A_to_B" or "B_to_A".
7. Your answer is only a causal prior. The final structure must be
   validated with statistical causal discovery, so an honest
   "uncertain" is more useful than a guess.
8. Output ONLY valid JSON, with no markdown and no extra text.

Allowed relations:

- A_to_B
- B_to_A
- no_direct_edge
- uncertain

Variable A:
Name: {name_a}
Meaning: {meaning_a}

Variable B:
Name: {name_b}
Meaning: {meaning_b}

Return exactly this JSON structure. The example values below are only a
format illustration: replace every field with your own answer, and use
the real variable names {name_a} and {name_b}.

{FORMAT_EXAMPLE}
"""

    return prompt
