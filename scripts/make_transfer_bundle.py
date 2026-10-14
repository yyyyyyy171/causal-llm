"""
打包成上传到 Google Drive 的 zip（供 Colab 解压使用）。

为什么不用 PowerShell 的 Compress-Archive：它在 Windows PowerShell 5.1 下
会把条目名写成反斜杠（`data\\ALARM\\x.csv`）。zip 规范要求用正斜杠，
Linux 的解压工具会把整串当成**一个文件名**，于是目录结构全部丢失、
文件扁平地堆在一层里（实测就是这么踩到的）。
这里用 Python 的 zipfile 显式写入 `/`，并且顶层包一层 `Causal-LLM/`，
这样在 Colab 里解压到 /content 后就得到 /content/Causal-LLM。

用法：
    python scripts/make_transfer_bundle.py
    python scripts/make_transfer_bundle.py --out D:\\causal-llm-bundle.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_DIRS = {".git", "__pycache__", "logs", "MathWorks"}
EXCLUDE_SUFFIXES = {".pyc"}
EXCLUDE_NAMES = {"causal-llm-bundle.zip"}

# 与 .gitignore 保持一致：OpenML 原始 CSV 与 sklearn 下载缓存都能重新获取，
# 不必打进包里（既省体积，也避免它们的目录条目给解压带来麻烦）。
EXCLUDE_DIR_NAMES = {"_sklearn_cache"}
EXCLUDE_DIR_PATHS = {("data", "openml")}


def main() -> int:

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(PROJECT_ROOT / "causal-llm-bundle.zip"))
    parser.add_argument("--root-name", default="Causal-LLM",
                        help="zip 顶层目录名，解压后即为 <root-name>/")
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()

    added = 0
    total_bytes = 0

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:

        for path in sorted(PROJECT_ROOT.rglob("*")):

            relative = path.relative_to(PROJECT_ROOT)

            if any(part in EXCLUDE_DIRS for part in relative.parts):
                continue
            if any(part in EXCLUDE_DIR_NAMES for part in relative.parts):
                continue
            if relative.suffix == ".csv" and ("data", "openml") in [
                relative.parts[:2]
            ]:
                continue
            if path.name in EXCLUDE_NAMES:
                continue
            if path.suffix in EXCLUDE_SUFFIXES:
                continue
            if not path.is_file():
                continue

            # zipfile 内部统一用正斜杠；这里显式转换，避免任何平台差异
            arcname = "/".join((args.root_name, *relative.parts))
            archive.write(path, arcname)

            added += 1
            total_bytes += path.stat().st_size

    size_mb = out_path.stat().st_size / 1e6

    print(f"打包 {added} 个文件（原始 {total_bytes / 1e6:.1f} MB）")
    print(f"已生成: {out_path}（{size_mb:.1f} MB）")
    print(f"zip 顶层: {args.root_name}/ ，解压到 /content 后即为 /content/{args.root_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
