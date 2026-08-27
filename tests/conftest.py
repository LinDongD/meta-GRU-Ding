from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    """在项目 tmp/ 下创建本次测试独占目录，绕开 Windows 系统临时目录 ACL 冲突。"""
    path = Path("tmp") / "test-runs" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        # 测试生成的 CSV/PNG 均已关闭；清理失败也不应掩盖真正的测试结果。
        shutil.rmtree(path, ignore_errors=True)

