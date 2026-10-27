from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path

import pytest

from reframelab import model_assets
from reframelab.model_assets import ModelAsset, ensure_model


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _asset(payload: bytes) -> ModelAsset:
    return ModelAsset(
        name="raft",
        filename="raft.onnx",
        url="https://primary.example/raft.onnx",
        fallback_urls=("https://mirror.example/raft.onnx",),
        sha256=hashlib.sha256(payload).hexdigest(),
        license="test",
        purpose="test",
        expected_size=len(payload),
    )


def test_model_download_retries_after_git_lfs_pointer(monkeypatch, tmp_path: Path) -> None:
    payload = b"valid onnx payload for the test"
    pointer = (
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        b"size 1234\n"
    )
    monkeypatch.delenv("REFRAME_LAB_OFFLINE", raising=False)
    monkeypatch.setitem(model_assets.MODEL_ASSETS, "raft", _asset(payload))
    model_assets._FAILED_DOWNLOADS.clear()
    calls: list[str] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float):
        del timeout
        calls.append(request.full_url)
        return _Response(pointer if len(calls) == 1 else payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    messages: list[str] = []
    result = ensure_model(
        "raft",
        model_dir=tmp_path,
        strict=True,
        progress=messages.append,
    )

    assert result == tmp_path / "raft.onnx"
    assert result.read_bytes() == payload
    assert calls == [
        "https://primary.example/raft.onnx",
        "https://mirror.example/raft.onnx",
    ]
    assert any("official mirror" in message for message in messages)


def test_model_download_explains_non_model_payloads(monkeypatch, tmp_path: Path) -> None:
    payload = b"valid model bytes"
    asset = _asset(payload)
    monkeypatch.delenv("REFRAME_LAB_OFFLINE", raising=False)
    monkeypatch.setitem(model_assets.MODEL_ASSETS, "raft", asset)
    model_assets._FAILED_DOWNLOADS.clear()
    responses = iter(
        [
            b"<!doctype html><html><body>proxy error</body></html>",
            (
                b"version https://git-lfs.github.com/spec/v1\n"
                b"oid sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff\n"
                b"size 1234\n"
            ),
        ]
    )

    def fake_urlopen(request: urllib.request.Request, timeout: float):
        del request, timeout
        return _Response(next(responses))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError) as error:
        ensure_model("raft", model_dir=tmp_path, strict=True)

    message = str(error.value)
    assert "HTML response" in message
    assert "Git LFS pointer" in message
    assert not (tmp_path / "raft.onnx").exists()


def test_model_download_non_strict_returns_fallback_after_all_sources(
    monkeypatch,
    tmp_path: Path,
) -> None:
    payload = b"valid model bytes"
    monkeypatch.delenv("REFRAME_LAB_OFFLINE", raising=False)
    monkeypatch.setitem(model_assets.MODEL_ASSETS, "raft", _asset(payload))
    model_assets._FAILED_DOWNLOADS.clear()

    def fake_urlopen(request: urllib.request.Request, timeout: float):
        del request, timeout
        return _Response(b"<!doctype html><html><body>blocked</body></html>")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    messages: list[str] = []

    result = ensure_model(
        "raft",
        model_dir=tmp_path,
        strict=False,
        progress=messages.append,
    )

    assert result is None
    assert "raft" in model_assets._FAILED_DOWNLOADS
    assert any("Using the classical fallback" in message for message in messages)
