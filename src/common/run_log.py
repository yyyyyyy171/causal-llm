"""
运行日志落盘。

文档的周 1 要求"日志落盘"。这里统一处理三件事：

1. 本次运行的 stdout / stderr 同时写到 logs/<脚本>_<时间戳>.log；
2. 日志开头写入可复现信息：时间、命令行、配置文件内容、git 提交、环境；
3. 运行结束时记录退出码与耗时。

用法（脚本 main 的最开头）：

    from src.common.run_log import start_run, finish_run
    start_run("run_llm_prior", config=cfg)
    ...
    finish_run(0)
"""

from __future__ import annotations

import atexit
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"

_state: dict = {"log_path": None, "started": None, "finished": False}


class _Tee:
    """
    把写操作同时转发给多个流。

    这个类会替换 sys.stdout / sys.stderr，所以必须把底层真实流的常见接口
    一并代理过去，尤其是 fileno()：

    vLLM 启动引擎子进程时要取 stdout 的文件描述符，如果我们只实现
    write / flush，它会抛 `AttributeError: '_Tee' object has no attribute
    'fileno'`，而外层只报出 "Engine core initialization failed"，
    排查时极易误判成 CUDA、驱动或 CUDA 图捕获的问题（我们就这样绕了半天）。

    因此这里代理 fileno / isatty / encoding / errors / buffer 等接口，
    让日志包装对调用方透明。
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:  # noqa: BLE001
                pass
        return len(data)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------
    # 以下接口一律代理到第一个能响应的底层流
    # ------------------------------------------------------------

    def fileno(self) -> int:
        for stream in self.streams:
            try:
                return stream.fileno()
            except Exception:  # noqa: BLE001
                continue
        raise OSError("_Tee: 底层流没有可用的文件描述符")

    def isatty(self) -> bool:
        for stream in self.streams:
            try:
                if stream.isatty():
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    @property
    def encoding(self):
        for stream in self.streams:
            value = getattr(stream, "encoding", None)
            if value:
                return value
        return "utf-8"

    @property
    def errors(self):
        for stream in self.streams:
            value = getattr(stream, "errors", None)
            if value:
                return value
        return "replace"

    @property
    def buffer(self):
        for stream in self.streams:
            value = getattr(stream, "buffer", None)
            if value is not None:
                return value
        raise AttributeError("_Tee: 底层流没有 buffer")

    @property
    def closed(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def close(self) -> None:
        # 故意什么都不做：某些库（vLLM / multiprocessing）会在子进程里关闭 stdout，
        # 如果跟着关掉日志文件句柄，后续的日志写入就静默丢失了。
        pass


def _git_revision() -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "-c", f"safe.directory={PROJECT_ROOT.as_posix()}",
                "-C", str(PROJECT_ROOT),
                "rev-parse", "--short", "HEAD",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        revision = result.stdout.strip()
        return revision or "（尚无提交）"
    except Exception:  # noqa: BLE001
        return "（无法获取）"


def _git_status_short() -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "-c", f"safe.directory={PROJECT_ROOT.as_posix()}",
                "-C", str(PROJECT_ROOT),
                "status", "--porcelain",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = [x for x in result.stdout.splitlines() if x.strip()]
        return f"{len(lines)} 个未提交改动" if lines else "工作区干净"
    except Exception:  # noqa: BLE001
        return "（无法获取）"


def _gpu_info() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.stdout.strip() or "（未检测到）"
    except Exception:  # noqa: BLE001
        return "（未检测到）"


def start_run(script_name: str, config: dict | None = None) -> Path:
    """开始记录一次运行。返回日志文件路径。"""

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{script_name}_{stamp}.log"

    handle = open(log_path, "w", encoding="utf-8")

    sys.stdout = _Tee(sys.__stdout__, handle)
    sys.stderr = _Tee(sys.__stderr__, handle)

    _state["log_path"] = log_path
    _state["started"] = time.time()
    _state["finished"] = False

    print("=" * 72)
    print(f"脚本      : {script_name}")
    print(f"开始时间  : {datetime.now().isoformat(timespec='seconds')}")
    print(f"命令行    : {' '.join(sys.argv)}")
    print(f"工作目录  : {os.getcwd()}")
    print(f"git 提交  : {_git_revision()}（{_git_status_short()}）")
    print(f"Python    : {platform.python_version()} / {platform.platform()}")
    print(f"显卡      : {_gpu_info()}")
    if config:
        print("配置      :")
        for line in json.dumps(
            config, ensure_ascii=False, indent=2, sort_keys=True
        ).splitlines():
            print(f"    {line}")
    print("=" * 72)
    print()

    atexit.register(finish_run)

    # 未捕获异常时，日志结尾要写 1 而不是 0
    previous_hook = sys.excepthook

    def _hook(exc_type, exc_value, exc_traceback):
        finish_run(1)
        previous_hook(exc_type, exc_value, exc_traceback)

    sys.excepthook = _hook

    return log_path


def finish_run(code: int = 0) -> None:
    """结束记录。重复调用是安全的。"""

    if _state["finished"]:
        return
    _state["finished"] = True

    started = _state.get("started")
    elapsed = time.time() - started if started else 0.0

    print()
    print("=" * 72)
    print(f"结束时间  : {datetime.now().isoformat(timespec='seconds')}")
    print(f"耗时      : {elapsed:.1f} 秒")
    print(f"退出码    : {code}")
    print(f"日志文件  : {_state['log_path']}")
    print("=" * 72)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001
            pass
