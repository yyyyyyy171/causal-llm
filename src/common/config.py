"""
统一配置加载。

文档的周 1 要求"配置文件"作为实验规范的一部分：随机种子、样本量、
显著性水平、先验强度这些**会变的参数**都放在 configs/ 下，
脚本里只保留不会变的实现细节。这样"一条命令重生成结果表"才成立。
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def config_path(name: str) -> Path:
    if not name.endswith(".json"):
        name = f"{name}.json"
    return CONFIG_DIR / name


def load_config(name: str = "") -> dict:
    """
    读取配置。name 为空时返回空配置，调用方使用脚本内置默认值。
    """

    if not name:
        return {}

    path = config_path(name)

    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件 {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def dump_config(config: dict) -> str:
    """把配置渲染成便于写进日志的多行文本。"""

    return json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True)
