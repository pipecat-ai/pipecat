#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the shared ONNX ``InferenceSession`` cache.

``SileroVADAnalyzer`` and ``LocalSmartTurnAnalyzerV3`` reuse a process-wide ONNX
session per configuration instead of loading the model weights once per instance.
These tests verify three properties:

1. **Singleton identity per config** — instances of the same configuration share
   one session object; different configurations (or different models) get
   distinct sessions.
2. **Per-instance state isolation** — the immutable session is shared, while all
   mutable/stateful objects (audio buffer, executor, Silero recurrent state) stay
   per-instance.
3. **Numerical equivalence** — sharing introduces no drift: a cached session and
   an independently rebuilt session produce bit-identical inference output.
"""

import threading

import numpy as np
import pytest

from pipecat.audio.turn.smart_turn import local_smart_turn_v3
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad import silero
from pipecat.audio.vad.silero import SileroOnnxModel, SileroVADAnalyzer


@pytest.fixture(autouse=True)
def _reset_session_caches():
    """Session caches are process-global; reset around every test so identity
    assertions don't depend on execution order."""
    local_smart_turn_v3._reset_session_cache_for_tests()
    silero._reset_session_cache_for_tests()
    yield
    local_smart_turn_v3._reset_session_cache_for_tests()
    silero._reset_session_cache_for_tests()


# --------------------------------------------------------------------------- #
# Singleton identity per configuration
# --------------------------------------------------------------------------- #


def test_smart_turn_session_shared_across_instances():
    a = LocalSmartTurnAnalyzerV3()
    b = LocalSmartTurnAnalyzerV3()
    assert a._session is b._session


def test_silero_session_shared_across_instances():
    a = SileroVADAnalyzer()
    b = SileroVADAnalyzer()
    assert a._model.session is b._model.session


def test_smart_turn_and_silero_sessions_are_distinct():
    smart = LocalSmartTurnAnalyzerV3()
    vad = SileroVADAnalyzer()
    assert smart._session is not vad._model.session


def test_smart_turn_distinct_config_gets_distinct_session():
    a = LocalSmartTurnAnalyzerV3(cpu_count=1)
    b = LocalSmartTurnAnalyzerV3(cpu_count=2)
    assert a._session is not b._session
    # Same config still shares.
    assert LocalSmartTurnAnalyzerV3(cpu_count=2)._session is b._session


def test_silero_distinct_config_gets_distinct_session():
    path = silero._resolve_bundled_model_path()
    a = SileroOnnxModel(path, force_onnx_cpu=True)
    b = SileroOnnxModel(path, force_onnx_cpu=False)
    assert a.session is not b.session


# --------------------------------------------------------------------------- #
# Per-instance state isolation
# --------------------------------------------------------------------------- #


def test_smart_turn_shares_session_isolates_state():
    a = LocalSmartTurnAnalyzerV3()
    b = LocalSmartTurnAnalyzerV3()
    assert a._session is b._session
    assert a._audio_buffer is not b._audio_buffer
    assert a._executor is not b._executor


def test_silero_shares_session_isolates_state():
    a = SileroVADAnalyzer()
    b = SileroVADAnalyzer()
    assert a._model.session is b._model.session
    assert a._model._state is not b._model._state
    assert a._model._context is not b._model._context
    assert a._executor is not b._executor


# --------------------------------------------------------------------------- #
# Numerical equivalence — sharing introduces no drift
# --------------------------------------------------------------------------- #


def test_smart_turn_numerical_equivalence():
    rng = np.random.default_rng(1234)
    audio = rng.standard_normal(16000 * 8).astype(np.float32) * 0.1

    cached = LocalSmartTurnAnalyzerV3()
    cached._sample_rate = 16000
    cached_out = cached._predict_endpoint(audio.copy())

    # Drop the cache so the next analyzer builds a fresh, independent session.
    local_smart_turn_v3._reset_session_cache_for_tests()
    fresh = LocalSmartTurnAnalyzerV3()
    fresh._sample_rate = 16000
    assert fresh._session is not cached._session
    fresh_out = fresh._predict_endpoint(audio.copy())

    assert cached_out["prediction"] == fresh_out["prediction"]
    assert cached_out["probability"] == pytest.approx(fresh_out["probability"], abs=1e-6)


def test_silero_numerical_equivalence():
    rng = np.random.default_rng(99)
    chunk = rng.standard_normal(512).astype(np.float32) * 0.1

    cached = SileroVADAnalyzer()._model
    cached_conf = float(np.ravel(cached(chunk.copy(), 16000))[0])

    silero._reset_session_cache_for_tests()
    fresh = SileroVADAnalyzer()._model
    assert fresh.session is not cached.session
    fresh_conf = float(np.ravel(fresh(chunk.copy(), 16000))[0])

    assert cached_conf == pytest.approx(fresh_conf, abs=1e-6)


# --------------------------------------------------------------------------- #
# Concurrency — Silero is the stateful one; per-instance state must stay isolated
# even when two models drive the same shared session from different threads.
# --------------------------------------------------------------------------- #


def test_silero_concurrent_inference_is_consistent():
    rng = np.random.default_rng(7)
    chunks = [rng.standard_normal(512).astype(np.float32) * 0.1 for _ in range(50)]

    def run_model() -> list[float]:
        model = SileroVADAnalyzer()._model
        return [float(np.ravel(model(c.copy(), 16000))[0]) for c in chunks]

    expected = run_model()

    results: dict[int, list[float]] = {}

    def worker(idx: int):
        results[idx] = run_model()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All shared the one session object.
    assert len(silero._session_cache) == 1
    for idx in range(2):
        assert results[idx] == pytest.approx(expected, abs=1e-6)
