# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepGEMM is faked: a "compile" is a new entry in its cache directory."""

import sys
import threading

import pytest

from vllm.platforms import current_platform
from vllm.utils import deep_gemm, jit_monitor

pytestmark = [
    pytest.mark.cpu_test,
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="inotify"),
]


@pytest.fixture
def fake_deep_gemm(tmp_path, monkeypatch):
    monkeypatch.setenv("DG_JIT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(deep_gemm, "has_deep_gemm", lambda: True)
    monkeypatch.setattr(deep_gemm, "_lazy_init", lambda: None)
    cache = tmp_path / "cache"

    def gemm(entry: str | None):
        if entry is not None:
            (cache / entry).mkdir()
        return "out"

    monkeypatch.setattr(deep_gemm, "_fake_gemm_impl", gemm, raising=False)
    return cache


def test_error_mode_raises_on_the_compiling_call(fake_deep_gemm):
    jit_monitor._mode = "error"
    jit_monitor._setup_deep_gemm_jit_hook()
    gemm = deep_gemm._fake_gemm_impl

    # A cached kernel loads without a new entry.
    assert gemm(None) == "out"
    with pytest.raises(RuntimeError, match="DeepGEMM JIT compilation during"):
        gemm("sm90_paged_mqa_logits_metadata.0123abcd")


def test_warn_mode_reports_from_the_watcher(fake_deep_gemm, monkeypatch):
    seen: list[str] = []
    reported = threading.Event()

    def record(*, backend, event, fn_name, detail=None):
        seen.append(f"{backend}:{fn_name}:{detail}")
        reported.set()

    monkeypatch.setattr(jit_monitor, "_handle_jit_event", record)
    jit_monitor._setup_deep_gemm_jit_hook()
    (fake_deep_gemm / "sm100_fp8_gemm_1d1d.feedface").mkdir()

    assert reported.wait(timeout=5)
    assert seen == ["DeepGEMM:sm100_fp8_gemm_1d1d:feedface"]


def test_unset_cache_dir_follows_vllm_cache_root(tmp_path, monkeypatch):
    monkeypatch.delenv("DG_JIT_CACHE_DIR", raising=False)
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    assert jit_monitor._deep_gemm_cache_dir() == str(tmp_path / "deep_gemm" / "cache")
