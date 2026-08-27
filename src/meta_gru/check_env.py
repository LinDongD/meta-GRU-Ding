from __future__ import annotations

import platform
import sys

import torch


def main() -> None:
    """打印当前解释器、PyTorch 与 CUDA 状态，便于确认 VS Code 环境。"""
    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {platform.python_version()}")
    print(f"PyTorch version   : {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"CUDA available    : {torch.cuda.is_available()}")
    print(f"CUDA device count : {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        memory_gib = properties.total_memory / 1024**3
        print(
            f"  [{index}] {properties.name}; compute capability "
            f"{properties.major}.{properties.minor}; {memory_gib:.1f} GiB"
        )


if __name__ == "__main__":
    main()
