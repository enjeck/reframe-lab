from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class DNNEngineResult:
    net: Any
    engine: str
    benchmark_ms: dict[str, float]
    load_seconds: float


def _engine_constant(name: str) -> int | None:
    return getattr(cv.dnn, name, None)


def available_engine_names() -> list[str]:
    names: list[str] = []
    for name in ("ENGINE_NEW", "ENGINE_AUTO", "ENGINE_CLASSIC", "ENGINE_ORT"):
        if _engine_constant(name) is not None:
            names.append(name)
    return names


def _load_with_engine(path: Path, engine_name: str | None):
    if engine_name is not None:
        value = _engine_constant(engine_name)
        if value is not None:
            try:
                return cv.dnn.readNetFromONNX(str(path), engine=value)
            except (TypeError, cv.error):
                pass
    try:
        return cv.dnn.readNetFromONNX(str(path))
    except (AttributeError, cv.error):
        return cv.dnn.readNet(str(path))


def _forward_once(net: Any, sample_blob: np.ndarray) -> None:
    net.setInput(sample_blob)
    names = net.getUnconnectedOutLayersNames()
    if names:
        net.forward(names)
    else:
        net.forward()


def load_onnx_net(
    path: str | Path,
    *,
    model_name: str,
    sample_blob: np.ndarray | None = None,
    mode: str = "auto",
    warmup_runs: int = 1,
    benchmark_runs: int = 2,
) -> DNNEngineResult:
    """Load an ONNX model and optionally choose an OpenCV 5 DNN engine.

    ``mode`` may be ``default``, ``auto``, ``new``, ``classic``, ``ort``, or
    ``autotune``. Autotuning is deliberately short because it runs once when a
    persistent worker first loads a model. If an engine cannot load or execute the
    graph, it is skipped and the normal OpenCV loader remains the final fallback.
    """

    del model_name  # Kept in the signature for logs and future per-model policies.
    source = Path(path)
    started = time.perf_counter()
    normalized = mode.strip().lower().replace("engine_", "")
    explicit = {
        "new": "ENGINE_NEW",
        "auto": "ENGINE_AUTO",
        "classic": "ENGINE_CLASSIC",
        "ort": "ENGINE_ORT",
    }

    if normalized not in {"autotune", "tune"}:
        engine_name = explicit.get(normalized)
        net = _load_with_engine(source, engine_name)
        if sample_blob is not None:
            for _ in range(max(0, warmup_runs)):
                try:
                    _forward_once(net, sample_blob)
                except cv.error:
                    break
        label = engine_name or "DEFAULT"
        return DNNEngineResult(
            net=net,
            engine=label,
            benchmark_ms={},
            load_seconds=time.perf_counter() - started,
        )

    candidates = available_engine_names()
    # ENGINE_ORT is useful for some transformer graphs but can have a high startup
    # cost. Try it last so simpler OpenCV engines win ties.
    ordered = [
        name
        for name in ("ENGINE_NEW", "ENGINE_AUTO", "ENGINE_CLASSIC", "ENGINE_ORT")
        if name in candidates
    ]
    if not ordered:
        ordered = ["DEFAULT"]

    timings: dict[str, float] = {}
    valid: dict[str, Any] = {}
    for engine_name in ordered:
        try:
            net = _load_with_engine(source, None if engine_name == "DEFAULT" else engine_name)
            if sample_blob is None:
                timings[engine_name] = 0.0
                valid[engine_name] = net
                continue
            for _ in range(max(0, warmup_runs)):
                _forward_once(net, sample_blob)
            samples: list[float] = []
            for _ in range(max(1, benchmark_runs)):
                tick = time.perf_counter()
                _forward_once(net, sample_blob)
                samples.append((time.perf_counter() - tick) * 1000.0)
            timings[engine_name] = float(statistics.median(samples))
            valid[engine_name] = net
        except (cv.error, RuntimeError, ValueError, TypeError):
            continue

    if valid:
        chosen = min(valid, key=lambda name: timings.get(name, float("inf")))
        return DNNEngineResult(
            net=valid[chosen],
            engine=chosen,
            benchmark_ms=timings,
            load_seconds=time.perf_counter() - started,
        )

    net = _load_with_engine(source, None)
    return DNNEngineResult(
        net=net,
        engine="DEFAULT_FALLBACK",
        benchmark_ms=timings,
        load_seconds=time.perf_counter() - started,
    )
