#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for system-instruction resolution in tracing.

System instructions can live on the LLM service (``settings.system_instruction``,
the norm) or arrive as an initial system message in the context. The
``_get_system_instruction`` helper resolves the effective value with the
service setting taking priority, matching service behavior.
"""

from types import SimpleNamespace

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.utils.tracing.service_attributes import add_gemini_live_span_attributes
from pipecat.utils.tracing.service_decorators import _get_system_instruction

SYSTEM_CONTEXT = LLMContext(messages=[{"role": "system", "content": "from context"}])


def _service(system_instruction=None):
    return SimpleNamespace(_settings=SimpleNamespace(system_instruction=system_instruction))


def test_settings_system_instruction_takes_priority():
    service = _service("from settings")
    assert _get_system_instruction(service, SYSTEM_CONTEXT) == "from settings"


def test_falls_back_to_context_system_message():
    assert _get_system_instruction(_service(), SYSTEM_CONTEXT) == "from context"


def test_falls_back_when_service_has_no_settings():
    assert _get_system_instruction(SimpleNamespace(), SYSTEM_CONTEXT) == "from context"


def test_context_multipart_system_message():
    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "part one."},
                    {"type": "text", "text": "part two."},
                ],
            }
        ]
    )
    assert _get_system_instruction(_service(), context) == "part one. part two."


def test_none_when_no_instruction_anywhere():
    assert _get_system_instruction(_service(), LLMContext()) is None
    assert _get_system_instruction(_service(), None) is None
    assert (
        _get_system_instruction(_service(), LLMContext(messages=[{"role": "user", "content": "x"}]))
        is None
    )


def test_gemini_span_attributes_accept_dotted_kwarg():
    """The Gemini Live decorator passes ``gen_ai.system_instructions`` through
    ``add_gemini_live_span_attributes(**operation_attrs)`` — dotted keys must
    survive the **kwargs round trip and land on the span."""

    class _FakeSpan:
        def __init__(self):
            self.attributes = {}

        def set_attribute(self, key, value):
            self.attributes[key] = value

    span = _FakeSpan()
    add_gemini_live_span_attributes(
        span=span,
        service_name="GeminiLiveLLMService",
        model="gemini-live",
        operation_name="llm_setup",
        **{"gen_ai.system_instructions": "be helpful"},
    )
    assert span.attributes["gen_ai.system_instructions"] == "be helpful"
