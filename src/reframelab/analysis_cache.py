from __future__ import annotations

import gzip
import pickle
from pathlib import Path

from .models import AnalysisBundle

_CACHE_VERSION = 1


def save_analysis_bundle(path: str | Path, bundle: AnalysisBundle) -> Path:
    """Persist trusted internal analysis so framing corrections can skip re-analysis."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": _CACHE_VERSION, "bundle": bundle}
    temporary = target.with_suffix(target.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=4) as output:
        pickle.dump(payload, output, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(target)
    return target


def load_analysis_bundle(path: str | Path) -> AnalysisBundle:
    source = Path(path)
    with gzip.open(source, "rb") as input_file:
        payload = pickle.load(input_file)  # noqa: S301 - internal cache written by this application.
    if not isinstance(payload, dict) or payload.get("version") != _CACHE_VERSION:
        raise ValueError("Unsupported analysis cache version")
    bundle = payload.get("bundle")
    if not isinstance(bundle, AnalysisBundle):
        raise ValueError("Analysis cache does not contain a valid bundle")
    return bundle
