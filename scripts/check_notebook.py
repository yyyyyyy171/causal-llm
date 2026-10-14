r"""
校验 Colab notebook 的基本正确性。

做三件事：

1. notebook 本身是合法 JSON；
2. **每个代码格都能编译**（这是关键——过去两次事故都是生成的单元格里有
   语法错误，而我在本机没检查过，等到 Colab 报错才发现：
   `\x` 非转义、f-string 里的 `\n` 被提前解释成换行）；
3. 没有"把凭据写进命令/URL"的写法（凭据泄露过两次，都是这个原因）。

用法：
    python scripts/check_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "colab_prior.ipynb"


def python_source(cell_source: str) -> str:
    """
    把单元格里的 shell 魔法（!cmd）与行魔法（%pip）换成占位语句，
    这样剩下的部分才能当纯 Python 编译。缩进保持不变。
    """

    lines = []
    for line in cell_source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            indent = line[: len(line) - len(stripped)]
            lines.append(f"{indent}pass  # shell magic")
        else:
            lines.append(line)
    return "\n".join(lines)


def main() -> int:

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    print(f"notebook: {NOTEBOOK.name}，单元格 {len(cells)} 个")

    problems = []
    code_cells = 0

    for index, cell in enumerate(cells, start=1):
        if cell["cell_type"] != "code":
            continue
        code_cells += 1
        source = python_source("".join(cell.get("source", [])))
        try:
            compile(source, f"<cell {index}>", "exec")
        except SyntaxError as error:
            problems.append(
                f"第 {index} 格语法错误：{error.msg}（第 {error.lineno} 行）"
            )

    print(f"代码格 {code_cells} 个，编译失败 {len(problems)} 个")

    source_all = "\n".join("".join(c.get("source", [])) for c in cells)

    forbidden = {
        "把 token 拼进 URL": "{token}@{rest}" in source_all,
        "推送时把带 token 的 URL 当参数": "'push', 'origin', BRANCH, url" in source_all,
        "直接打印 token": "print(token)" in source_all,
    }
    for name, bad in forbidden.items():
        if bad:
            problems.append(f"存在风险写法：{name}")

    required = {
        "凭据走 credential.helper store": "credential.helper" in source_all,
        "token 去掉所有空白字符": "''.join(token.split())" in source_all,
    }
    for name, ok in required.items():
        if not ok:
            problems.append(f"缺少必要写法：{name}")

    print()
    if problems:
        print("发现问题：")
        for item in problems:
            print("  -", item)
        return 1

    print("检查通过：notebook 合法、每个代码格可编译、无凭据写入命令的写法")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
