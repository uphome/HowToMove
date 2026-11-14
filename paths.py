"""项目路径与运行环境入口。

任何需要引擎的模块先 `import paths`，之后 `from qwen3 import ...`、
`import server`、`from engine_rl import ...` 都能直接工作 —— 引擎内部
一行 import 都不用改（vendored fork 保持原包结构）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ENGINE_ROOT = Path(os.environ.get("AGENTICRL_ENGINE", PROJECT_ROOT / "engine"))
MODEL_PATH = Path(os.environ.get("AGENTICRL_MODEL", r"E:\codeall\Qwen3-0.6B"))
OUT_DIR = PROJECT_ROOT / "out"
REPORTS_DIR = PROJECT_ROOT / "reports"


def bootstrap() -> Path:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return ENGINE_ROOT


bootstrap()

__all__ = ["PROJECT_ROOT", "ENGINE_ROOT", "MODEL_PATH", "OUT_DIR", "REPORTS_DIR"]
