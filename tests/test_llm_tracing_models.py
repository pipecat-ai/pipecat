#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for LLM tracing model attributes."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from pipecat.utils.tracing.service_attributes import add_llm_span_attributes
from pipecat.utils.tracing.service_decorators import _get_model_name


class _FakeSpan:
    def __init__(self, recording: bool = True):
        self.attributes = {}
        self._recording = recording

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def is_recording(self):
        return self._recording


class _ServiceWithModels:
    def __init__(self):
        self._settings = SimpleNamespace(model="requested-model")
        self._full_model_name = "response-model"


def _load_base_llm_module():
    module_path = Path(__file__).resolve().parents[1] / "src/pipecat/services/openai/base_llm.py"
    spec = importlib.util.spec_from_file_location("test_base_llm_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_get_model_name_prefers_requested_model_over_response_model():
    """Request model should not drift to a previously observed response model."""
    service = _ServiceWithModels()

    model_name = _get_model_name(service)

    assert model_name == "requested-model"


def test_add_llm_span_attributes_sets_response_model_when_provided():
    """LLM span attributes should include request and response model names."""
    span = _FakeSpan()

    add_llm_span_attributes(
        span=span,
        service_name="OpenAILLMService",
        model="requested-model",
        response_model="resolved-model-2026-01-01",
    )

    assert span.attributes["gen_ai.request.model"] == "requested-model"
    assert span.attributes["gen_ai.response.model"] == "resolved-model-2026-01-01"


def test_set_full_model_name_updates_active_span_response_model():
    """Response model should be written when response data is received."""
    base_llm = _load_base_llm_module()
    fake_span = _FakeSpan(recording=True)
    fake_trace = SimpleNamespace(get_current_span=lambda: fake_span)
    fake_service = SimpleNamespace(_full_model_name="")

    original_is_tracing_available = base_llm.is_tracing_available
    original_trace = getattr(base_llm, "trace", None)
    try:
        setattr(base_llm, "is_tracing_available", lambda: True)
        setattr(base_llm, "trace", fake_trace)
        base_llm.BaseOpenAILLMService.set_full_model_name(fake_service, "gpt-4o-2026-02-14")
    finally:
        setattr(base_llm, "is_tracing_available", original_is_tracing_available)
        if original_trace is None:
            delattr(base_llm, "trace")
        else:
            setattr(base_llm, "trace", cast(object, original_trace))

    assert fake_service._full_model_name == "gpt-4o-2026-02-14"
    assert fake_span.attributes["gen_ai.response.model"] == "gpt-4o-2026-02-14"
