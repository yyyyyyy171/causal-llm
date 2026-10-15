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


# 第二次提问方式：直接问马尔可夫毯成员资格。
#
# 为什么需要它：第一种 Prompt 问的是"有没有直接边"，而马尔可夫毯 =
# 父子 + 配偶，配偶并不是邻居。诊断结果显示模型对配偶的判断在两个网络上
# 完全相反（ASIA 0/2 判成有边、CHILD 5/5 判成有边），而且它声称的"边"里
# 真正属于毯的只有 40% / 22%。问题问错了，再精巧的融合方式也救不回来。
#
# 这一版把问题改成"给定模型里其余所有变量之后，A 与 B 是否仍然相互依赖"，
# 也就是"B 是否在 A 的马尔可夫毯里"——直接对准任务的对象。
MB_PROMPT_VERSION = "mb-v1-blanket-membership"

# 第三次提问方式：直接问角色。
#
# 为什么还要再换一次：前两版都只在问"有没有直接边"或"在不在毯内"，
# 而融合算法把"没有直接边"当成负分处理（见 docs/先验进入算法的路径.md）。
# 配偶在 DAG 里本来就不相邻，对"有没有直接边"这个问题标准答案就是"没有"，
# 于是模型答得越对（就邻接而言），配偶被推出毯子的力量越大。
#
# 这一版把配偶变成**正面答案**：父 / 子 / 配偶三者都算"在毯内"，
# 只有"都不算"才是负分。并且明确允许"不确定"，不强迫模型在名字无信息时瞎猜。
ROLE_PROMPT_VERSION = "role-v1-parent-child-spouse"


ALLOWED_ROLES = ("parent", "child", "spouse", "none", "uncertain")


FORMAT_EXAMPLE_ROLE = """{
    "variable_a": "NAME_A",
    "variable_b": "NAME_B",
    "in_markov_blanket": true,
    "role_of_a": "parent",
    "confidence": 0.75,
    "reason": "one sentence"
}"""


FORMAT_EXAMPLE_MB = """{
    "variable_a": "NAME_A",
    "variable_b": "NAME_B",
    "in_markov_blanket": true,
    "confidence": 0.75,
    "reason": "one sentence"
}"""


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


def build_markov_blanket_prompt(
    variable_a: str,
    variable_b: str,
    network: str = "ALARM",
    anonymous: bool = False,
) -> str:
    """
    直接询问马尔可夫毯成员资格的 Prompt。

    与 build_causal_prompt 的区别：不再问"有没有直接边"，而是问
    "给定模型里其余所有变量之后，A 与 B 是否仍然相互依赖"，
    这正是马尔可夫毯的定义（父子 + 配偶），也就是下游算法真正需要的东西。
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

Answer one question about two of its variables.

Background. In a causal model, the Markov blanket of a variable is the
smallest set of variables that makes it independent of everything else.
It consists of three kinds of variables:

  * its parents, which cause it directly;
  * its children, which it causes directly;
  * the other parents of its children, called its spouses, which share a
    common effect with it.

Two variables belong to each other's Markov blanket exactly when they
remain dependent after you condition on every other variable in the
model - that is, when knowing one still tells you something about the
other even after everything else is known.

Question. Is variable B in the Markov blanket of variable A?
(The relation is symmetric, so this is the same as asking whether A is in
the Markov blanket of B.)

Rules:

1. Answer from the meaning of the variables and the structure this implies.
   Never use a dataset, a published graph, or remembered experimental
   results.
2. Answer YES when the two variables influence each other directly, or when
   they share a common effect (one is a cause of the other's effect).
   Answer NO when they are connected only through a chain, or only through
   a common cause, with nothing left over once everything else is known.
3. Do not invent intermediate variables.
4. "confidence" must be a number between 0 and 1. Do not output 0 unless you
   truly have no information at all.
5. Output ONLY valid JSON, with no markdown and no extra text.

Variable A:
Name: {name_a}
Meaning: {meaning_a}

Variable B:
Name: {name_b}
Meaning: {meaning_b}

Return exactly this JSON structure. The example values are only a format
illustration: replace every field with your own answer, and use the real
variable names {name_a} and {name_b}.

{FORMAT_EXAMPLE_MB}
"""

    return prompt


def build_role_prompt(
    variable_a: str,
    variable_b: str,
    network: str = "ALARM",
    anonymous: bool = False,
) -> str:
    """
    直接询问"A 相对于 B 扮演什么角色"的 Prompt。

    与 build_causal_prompt 的关键区别：那里"有没有直接边"只有"是/否"两个出口，
    配偶只能落在"否"上，而融合算法把"否"当成负分。这里把配偶变成**正面答案**，
    父 / 子 / 配偶三者都算"在毯内"，只有"都不是"才是负分。

    同时明确允许"不确定"：变量名确实没有信息时（例如匿名编号），
    模型应当照实回答，而不是为了给个答案而猜——猜出来的先验比没有先验更糟。
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

Background. The Markov blanket of a variable is the smallest set of
variables that makes it independent of everything else. It contains
exactly three kinds of variables:

  * its parents, which cause it directly;
  * its children, which it causes directly;
  * its spouses: the other parents of its children, which share a common
    effect with it and are NOT connected to it by an edge.

So a pair can belong to each other's Markov blanket while having NO
direct edge between them. Keep that in mind: the two questions below are
not the same question.

Answer these two questions.

Question 1: are A and B in each other's Markov blanket at all?
Answer with the boolean field "in_markov_blanket".

Question 2: which role does A play relative to B?
Choose exactly one value for "role_of_a":

  "parent"     A causes B directly (A -> B)
  "child"      B causes A directly (B -> A)
  "spouse"     neither causes the other, but they share a common child
  "none"       A is neither a parent, a child, nor a spouse of B, so they
               are not in each other's Markov blanket
  "uncertain"  the names and meanings given above genuinely do not let
               you decide

Rules:

1. Answer only from the meaning of the variables. Never use a dataset,
   a published graph, or remembered experimental results.
2. "spouse" is a POSITIVE answer, not a rejection. Two variables that
   both cause the same effect are spouses. Do not answer "none" merely
   because there is no direct edge between them.
3. Do not invent intermediate variables. If A plausibly influences the
   state of B and the mechanism has no variable of its own in the model,
   that is still "parent" (or "child" in the other direction).
4. "confidence" must be a number between 0 and 1, expressing how sure
   you are about the role you chose. Do not output 0 unless you truly
   have no information at all.
5. "in_markov_blanket" must be true exactly when the role is "parent",
   "child" or "spouse", and false when the role is "none" or "uncertain".
6. Use "uncertain" honestly when the names carry no usable information —
   for example when both variables are anonymised labels. Do NOT guess
   just to avoid answering "uncertain"; a guessed prior is worse than an
   honest abstention.
7. Output ONLY valid JSON, with no markdown and no extra text.

Variable A:
Name: {name_a}
Meaning: {meaning_a}

Variable B:
Name: {name_b}
Meaning: {meaning_b}

Return exactly this JSON structure. The example values below are only a
format illustration: replace every field with your own answer, and use
the real variable names {name_a} and {name_b}.

{FORMAT_EXAMPLE_ROLE}
"""

    return prompt
