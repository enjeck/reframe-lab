from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

import cv2 as cv


def opencv_runtime_info() -> dict[str, Any]:
    module_path = str(Path(getattr(cv, "__file__", "")).resolve())
    hint = os.environ.get("REFRAME_OPENCV_DISTRIBUTION", "").strip().lower()
    path_lower = module_path.lower()
    is_cool = hint == "cool" or "/opt/cool/" in path_lower or "cloud-optimized" in path_lower
    distribution = "cool" if is_cool else (hint or "standard")
    return {
        "opencv_version": cv.__version__,
        "opencv_module": module_path,
        "opencv_distribution": distribution,
        "cool_runtime": bool(is_cool),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }
