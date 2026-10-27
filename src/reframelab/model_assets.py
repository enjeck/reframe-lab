from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ProgressCallback = Callable[[str], None]
_FAILED_DOWNLOADS: set[str] = set()


@dataclass(frozen=True)
class ModelAsset:
    name: str
    filename: str
    url: str
    sha256: str
    license: str
    purpose: str
    fallback_urls: tuple[str, ...] = ()
    expected_size: int | None = None

    @property
    def download_urls(self) -> tuple[str, ...]:
        return (self.url, *self.fallback_urls)


def _opencv_zoo_fallbacks(folder: str, filename: str) -> tuple[str, ...]:
    path = f"models/{folder}/{filename}"
    return (
        f"https://github.com/opencv/opencv_zoo/raw/refs/heads/main/{path}",
        f"https://media.githubusercontent.com/media/opencv/opencv_zoo/main/{path}",
    )


MODEL_ASSETS: dict[str, ModelAsset] = {
    "efficient_sam": ModelAsset(
        name="efficient_sam",
        filename="image_segmentation_efficientsam_ti_2025april.onnx",
        url=(
            "https://huggingface.co/opencv/image_segmentation_efficientsam/resolve/main/"
            "image_segmentation_efficientsam_ti_2025april.onnx"
        ),
        sha256="4eb496e0a7259d435b49b66faf1754aa45a5c382a34558ddda9a8c6fe5915d77",
        license="Apache-2.0",
        purpose="OpenCV DNN promptable object masks from a point or box",
        fallback_urls=_opencv_zoo_fallbacks(
            "image_segmentation_efficientsam",
            "image_segmentation_efficientsam_ti_2025april.onnx",
        ),
    ),
    "raft": ModelAsset(
        name="raft",
        filename="optical_flow_estimation_raft_2023aug.onnx",
        url=(
            "https://huggingface.co/opencv/optical_flow_estimation_raft/resolve/main/"
            "optical_flow_estimation_raft_2023aug.onnx"
        ),
        sha256="3b479e0447894245b822bf0b7392155d31ccd22bb9b0ee55503066bf6f6f869a",
        license="BSD-3-Clause and MIT conversion",
        purpose="OpenCV DNN learned dense optical flow",
        fallback_urls=_opencv_zoo_fallbacks(
            "optical_flow_estimation_raft",
            "optical_flow_estimation_raft_2023aug.onnx",
        ),
        expected_size=64_119_337,
    ),
    "depth_anything_v2_small": ModelAsset(
        name="depth_anything_v2_small",
        filename="depth_anything_v2_vits.onnx",
        url=(
            "https://huggingface.co/AXERA-TECH/Depth-Anything-V2/resolve/main/"
            "depth_anything_v2_vits.onnx"
        ),
        sha256="443e95f17819f347f5f987384b8cb7d7d18ed6af6ac46dec9b0152748ba7dfd0",
        license="Apache-2.0 base model; third-party ONNX export",
        purpose="OpenCV DNN relative monocular depth for foreground and occlusion evidence",
    ),
    "vittrack": ModelAsset(
        name="vittrack",
        filename="object_tracking_vittrack_2023sep.onnx",
        url=(
            "https://huggingface.co/opencv/object_tracking_vittrack/resolve/main/"
            "object_tracking_vittrack_2023sep.onnx"
        ),
        sha256="2990f0b7cd44d92afa48cd97db6de7be113fc1d9594fddb74e2725c10478e91d",
        license="Apache-2.0",
        purpose="OpenCV DNN vision-transformer target tracking",
        fallback_urls=_opencv_zoo_fallbacks(
            "object_tracking_vittrack",
            "object_tracking_vittrack_2023sep.onnx",
        ),
    ),
    "person_mediapipe": ModelAsset(
        name="person_mediapipe",
        filename="person_detection_mediapipe_2023mar_int8bq.onnx",
        url=(
            "https://huggingface.co/opencv/person_detection_mediapipe/resolve/main/"
            "person_detection_mediapipe_2023mar_int8bq.onnx"
        ),
        sha256="c5ed8c00c028b98e5d2c55b920a6e975af6c4cd538cfeea7c054f4fbbd8b9075",
        license="Apache-2.0",
        purpose="OpenCV DNN full- and upper-body person detection",
        fallback_urls=_opencv_zoo_fallbacks(
            "person_detection_mediapipe",
            "person_detection_mediapipe_2023mar_int8bq.onnx",
        ),
    ),
    "text_ppocr": ModelAsset(
        name="text_ppocr",
        filename="text_detection_en_ppocrv3_2023may.onnx",
        url=(
            "https://huggingface.co/opencv/text_detection_ppocr/resolve/main/"
            "text_detection_en_ppocrv3_2023may.onnx"
        ),
        sha256="03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587",
        license="Apache-2.0",
        purpose="OpenCV DNN PP-OCRv3 text-region detection",
        fallback_urls=_opencv_zoo_fallbacks(
            "text_detection_ppocr",
            "text_detection_en_ppocrv3_2023may.onnx",
        ),
    ),
    "yunet": ModelAsset(
        name="yunet",
        filename="face_detection_yunet_2023mar.onnx",
        url=(
            "https://huggingface.co/opencv/face_detection_yunet/resolve/main/"
            "face_detection_yunet_2023mar.onnx"
        ),
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        license="MIT",
        purpose="OpenCV DNN multi-face detection",
        fallback_urls=_opencv_zoo_fallbacks(
            "face_detection_yunet",
            "face_detection_yunet_2023mar.onnx",
        ),
    ),
}

# These three models support the production path. The remaining models are retained
# for explicit experiments and never download during normal web use.
CORE_MODEL_NAMES: tuple[str, ...] = (
    "person_mediapipe",
    "text_ppocr",
    "yunet",
)

EXPERIMENTAL_MODEL_NAMES: tuple[str, ...] = (
    "efficient_sam",
    "raft",
    "depth_anything_v2_small",
    "vittrack",
)


def default_model_dir() -> Path:
    configured = os.environ.get("REFRAME_LAB_MODEL_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    cache_root = os.environ.get("XDG_CACHE_HOME")
    if cache_root:
        return (Path(cache_root).expanduser() / "reframe-lab" / "models").resolve()
    return (Path.home() / ".cache" / "reframe-lab" / "models").resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_timeout() -> float:
    raw = os.environ.get("REFRAME_LAB_MODEL_DOWNLOAD_TIMEOUT", "120")
    try:
        return max(10.0, float(raw))
    except ValueError:
        return 120.0


def _source_label(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    return parsed.netloc or parsed.path


def _payload_description(path: Path, expected_size: int | None) -> str:
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(512).lstrip()
    lowered = prefix.lower()
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return f"a {size}-byte Git LFS pointer instead of the model binary"
    if lowered.startswith((b"<!doctype html", b"<html")):
        return f"an HTML response ({size} bytes) instead of the model binary"
    if lowered.startswith((b"{", b"[")) and size < 1024 * 1024:
        return f"a small JSON response ({size} bytes) instead of the model binary"
    if expected_size is not None and size != expected_size:
        return f"{size} bytes; the official file is {expected_size} bytes"
    return f"{size} bytes"


def model_path(name: str, model_dir: str | Path | None = None) -> Path:
    try:
        asset = MODEL_ASSETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_ASSETS))
        raise ValueError(f"Unknown model {name!r}. Available models: {choices}") from exc
    root = Path(model_dir).expanduser().resolve() if model_dir else default_model_dir()
    return root / asset.filename


def verify_model(name: str, path: str | Path) -> bool:
    asset = MODEL_ASSETS[name]
    candidate = Path(path)
    return candidate.is_file() and _sha256(candidate) == asset.sha256


def ensure_model(
    name: str,
    *,
    model_dir: str | Path | None = None,
    download: bool = True,
    progress: ProgressCallback | None = None,
    strict: bool = False,
) -> Path | None:
    """Return a verified model, downloading it atomically when allowed.

    Every downloaded byte stream is checked against the model publisher's SHA-256.
    Large model hosts sometimes return a Git LFS pointer, proxy error page, or access
    response with HTTP 200. Those responses are rejected and the next official mirror
    is tried. A bad payload is never installed under the model filename.
    """

    try:
        asset = MODEL_ASSETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_ASSETS))
        raise ValueError(f"Unknown model {name!r}. Available models: {choices}") from exc

    if os.environ.get("REFRAME_LAB_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        download = False
    destination = model_path(name, model_dir)
    if verify_model(name, destination):
        return destination
    if destination.exists():
        destination.unlink(missing_ok=True)
    if name in _FAILED_DOWNLOADS and not strict:
        return None
    if not download:
        if strict:
            raise FileNotFoundError(
                f"Model {name!r} is not installed. Run: reframe-lab models --download {name}"
            )
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(f"Downloading OpenCV model: {name}")
    temporary_dir = Path(tempfile.mkdtemp(prefix="reframe-model-", dir=destination.parent))
    temporary = temporary_dir / asset.filename
    failures: list[str] = []
    try:
        for attempt, url in enumerate(asset.download_urls, start=1):
            temporary.unlink(missing_ok=True)
            source = _source_label(url)
            if progress and attempt > 1:
                progress(f"Retrying {name} from official mirror: {source}")
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "reframe-lab/0.14.0 (+https://opencv.org/)",
                    "Accept": "application/octet-stream,*/*;q=0.8",
                    "Accept-Encoding": "identity",
                    "Cache-Control": "no-cache",
                },
            )
            try:
                with (
                    urllib.request.urlopen(request, timeout=_download_timeout()) as response,
                    temporary.open("wb") as out,
                ):
                    shutil.copyfileobj(response, out, length=1024 * 1024)
            except (OSError, urllib.error.URLError) as exc:
                failures.append(f"{source}: download failed ({exc})")
                continue

            actual = _sha256(temporary)
            if actual != asset.sha256:
                payload = _payload_description(temporary, asset.expected_size)
                failures.append(
                    f"{source}: received {payload}; checksum was {actual}, "
                    f"expected {asset.sha256}"
                )
                continue

            temporary.replace(destination)
            _FAILED_DOWNLOADS.discard(name)
            if progress:
                progress(f"Installed OpenCV model: {destination}")
            return destination

        _FAILED_DOWNLOADS.add(name)
        details = "; ".join(failures) or "no download source was attempted"
        message = f"Could not install model {name!r} from any official source. {details}"
        if strict:
            raise RuntimeError(message)
        if progress:
            progress(f"{message}. Using the classical fallback")
        return None
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


def model_status(model_dir: str | Path | None = None) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for name, asset in MODEL_ASSETS.items():
        path = model_path(name, model_dir)
        result.append(
            {
                "name": name,
                "purpose": asset.purpose,
                "license": asset.license,
                "path": str(path),
                "installed": verify_model(name, path),
                "sha256": asset.sha256,
                "group": "core" if name in CORE_MODEL_NAMES else "experimental",
            }
        )
    return result
