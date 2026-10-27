from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .semantic import PersonDetector
from .text import TextDetector

ProgressCallback = Callable[[str], None]


@dataclass
class DetectorBundle:
    person: PersonDetector | None
    text: TextDetector | None
    metadata: dict[str, object]


_lock = threading.RLock()
_cache: dict[tuple[object, ...], DetectorBundle] = {}
_stats = {"hits": 0, "misses": 0, "clears": 0}


def clear_detector_cache() -> None:
    with _lock:
        _cache.clear()
        _stats["clears"] += 1


def detector_cache_stats() -> dict[str, int]:
    with _lock:
        return {**_stats, "entries": len(_cache)}


def get_detector_bundle(
    *,
    semantic_people: bool,
    semantic_text: bool,
    model_dir: str | Path | None,
    allow_download: bool,
    persistent: bool,
    dnn_optimization: bool,
    dnn_batch_size: int,
    progress: ProgressCallback | None,
) -> DetectorBundle:
    """Return detectors that stay warm across jobs on the same process.

    OpenCV ``Net`` objects and high-level detector wrappers are mutable. The web
    application currently runs one processing job per worker, so a process-level
    bundle is safe and avoids repeated graph loading and warm-up. The cache key keeps
    differently configured model directories and engine policies separate.
    """

    key = (
        threading.get_ident(),
        bool(semantic_people),
        bool(semantic_text),
        str(Path(model_dir).expanduser().resolve()) if model_dir else None,
        bool(allow_download),
        bool(dnn_optimization),
        int(max(1, dnn_batch_size)),
    )
    if persistent:
        with _lock:
            existing = _cache.get(key)
            if existing is not None:
                _stats["hits"] += 1
                metadata = dict(existing.metadata)
                metadata["cache_hit"] = True
                metadata["cache_stats"] = detector_cache_stats()
                return DetectorBundle(existing.person, existing.text, metadata)

    started = time.perf_counter()
    engine_mode = "autotune" if dnn_optimization else "auto"
    person = (
        PersonDetector(
            model_dir=model_dir,
            allow_download=allow_download,
            progress=progress,
            dnn_engine=engine_mode,
            batch_size=max(1, dnn_batch_size if dnn_optimization else 1),
        )
        if semantic_people
        else None
    )
    text = (
        TextDetector(
            model_dir=model_dir,
            allow_download=allow_download,
            progress=progress,
            dnn_engine=engine_mode,
            batch_size=max(1, dnn_batch_size if dnn_optimization else 1),
        )
        if semantic_text
        else None
    )
    metadata: dict[str, object] = {
        "cache_hit": False,
        "load_seconds": time.perf_counter() - started,
        "persistent": bool(persistent),
        "dnn_optimization": bool(dnn_optimization),
        "requested_batch_size": int(max(1, dnn_batch_size)),
        "person": person.runtime_metadata() if person is not None else None,
        "text": text.runtime_metadata() if text is not None else None,
    }
    bundle = DetectorBundle(person, text, metadata)
    with _lock:
        _stats["misses"] += 1
        if persistent:
            _cache[key] = bundle
        metadata["cache_stats"] = detector_cache_stats()
    return bundle
