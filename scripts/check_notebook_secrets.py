"""
检查 Colab notebook 里是否残留"把凭据写进命令/URL"的写法。

为什么需要这个检查：我们曾经把 GitHub token 拼进 clone 的 URL，
推送/克隆失败时 git 把整条带 token 的命令打进了错误信息
（`https://ghp_xxx@github.com/...`），token 就这样泄露了。

正确的做法是：凭据写进 `~/.git-credentials`，由 `credential.helper store` 读取，
命令与 URL 里只出现仓库地址。

用法：
    python scripts/check_notebook_secrets.py
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "colab_prior.ipynb"


def main() -> int:

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    print(f"notebook 单元格数: {len(notebook['cells'])}")

    required = {
        "凭据走 credential.helper store": "credential.helper" in source,
        "token 去掉所有空白字符": "''.join(token.split())" in source,
        "克隆使用不带 token 的地址": "BRANCH, REPO_URL, REPO" in source,
        "推送命令不携带 token": "'push', 'origin', BRANCH]" in source,
    }

    forbidden = {
        "把 token 拼进 URL": "{token}@{rest}" in source,
        "推送时把带 token 的 URL 当参数": "'push', 'origin', BRANCH, url" in source,
        "直接打印 token": "print(token)" in source,
    }

    print("\n应该具备：")
    for name, ok in required.items():
        print(f"  {str(ok):<5} {name}")

    print("\n不应出现：")
    for name, bad in forbidden.items():
        print(f"  {str(bad):<5} {name}")

    ok = all(required.values()) and not any(forbidden.values())
    print("\n结论：" + ("通过" if ok else "存在风险写法，必须修"))

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
