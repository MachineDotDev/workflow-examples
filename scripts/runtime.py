"""Shared runtime checks and result output for the GPU examples."""

import json
import math
from pathlib import Path


def require_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This example requires an NVIDIA GPU. Check the Machine installation and runner labels."
        )
    torch.manual_seed(42)
    return torch.device("cuda")


def write_result(output, metrics):
    import torch

    def check(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite result")
        if isinstance(value, dict):
            for child in value.values():
                check(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check(child)

    check(metrics)
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    metrics["gpu"] = torch.cuda.get_device_name()
    metrics["torch"] = torch.__version__
    (path / "results.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, indent=2))
