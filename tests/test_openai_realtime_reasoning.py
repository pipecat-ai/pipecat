#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for OpenAI Realtime reasoning support (gpt-realtime-2).

Covers:
- ``SessionProperties.reasoning`` round-trips through Pydantic.
- Compatibility heuristic warns when reasoning is configured on a model
  that isn't known to support it, and stays quiet otherwise.
- Runtime ``LLMUpdateSettingsFrame`` carrying reasoning triggers a
  ``session.update`` and the outgoing event includes the new reasoning.
"""

import io

import pytest
from loguru import logger

from pipecat.frames.frames import LLMUpdateSettingsFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

# ---------------------------------------------------------------------------
# Pure data: SessionProperties round-trip
# ---------------------------------------------------------------------------


def test_session_properties_accepts_reasoning_object():
    sp = events.SessionProperties(reasoning=events.Reasoning(effort="high"))
    assert sp.reasoning is not None
    assert sp.reasoning.effort == "high"


def test_session_properties_coerces_reasoning_dict():
    """Pydantic coerces nested dicts into Reasoning automatically."""
    sp = events.SessionProperties.model_validate({"reasoning": {"effort": "low"}})
    assert isinstance(sp.reasoning, events.Reasoning)
    assert sp.reasoning.effort == "low"


def test_reasoning_accepts_future_effort_strings():
    """Forward compat: unknown effort strings pass through (the field accepts ``| str``)."""
    r = events.Reasoning(effort="ultra")  # not in the today's Literal set
    assert r.effort == "ultra"


def test_reasoning_serializes_into_session_update():
    """Confirm the wire shape sent to OpenAI matches the documented schema."""
    sp = events.SessionProperties(reasoning=events.Reasoning(effort="medium"))
    dumped = events.SessionUpdateEvent(session=sp).model_dump(exclude_none=True)
    assert dumped["session"]["reasoning"] == {"effort": "medium"}


# ---------------------------------------------------------------------------
# Strip-on-the-client compatibility behavior
# ---------------------------------------------------------------------------


def _capture_warnings():
    """Attach a fresh loguru sink that captures WARNING-and-above messages."""
    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    return sink, handler_id


class _EventRecorder:
    def __init__(self):
        self.events: list[events.ClientEvent] = []

    async def __call__(self, event: events.ClientEvent):
        self.events.append(event)


async def _send_and_capture(service: OpenAIRealtimeLLMService) -> events.SessionProperties:
    """Run ``_send_session_update`` and return the outgoing session payload."""
    sent = _EventRecorder()
    service.send_client_event = sent
    await service._send_session_update()
    session_updates = [e for e in sent.events if isinstance(e, events.SessionUpdateEvent)]
    assert len(session_updates) == 1
    return session_updates[0].session


@pytest.mark.asyncio
async def test_outgoing_session_update_strips_reasoning_on_unsupported_model():
    service = OpenAIRealtimeLLMService(
        api_key="test-key",
        settings=OpenAIRealtimeLLMService.Settings(
            model="gpt-realtime-1.5",
            system_instruction="be helpful",
            session_properties=events.SessionProperties(
                reasoning=events.Reasoning(effort="high"),
            ),
        ),
    )

    sink, handler_id = _capture_warnings()
    try:
        outgoing = await _send_and_capture(service)
    finally:
        logger.remove(handler_id)

    # Stripped on the wire.
    assert outgoing.reasoning is None
    # Warning surfaced for visibility.
    text = sink.getvalue()
    assert "stripping `reasoning`" in text
    assert "gpt-realtime-1.5" in text
    # Stored config preserved — strip happens on a copy.
    assert service._settings.session_properties.reasoning is not None
    assert service._settings.session_properties.reasoning.effort == "high"


@pytest.mark.asyncio
async def test_outgoing_session_update_keeps_reasoning_on_supported_model():
    service = OpenAIRealtimeLLMService(
        api_key="test-key",
        settings=OpenAIRealtimeLLMService.Settings(
            model="gpt-realtime-2",
            system_instruction="be helpful",
            session_properties=events.SessionProperties(
                reasoning=events.Reasoning(effort="high"),
            ),
        ),
    )

    sink, handler_id = _capture_warnings()
    try:
        outgoing = await _send_and_capture(service)
    finally:
        logger.remove(handler_id)

    assert outgoing.reasoning is not None
    assert outgoing.reasoning.effort == "high"
    assert sink.getvalue() == ""


@pytest.mark.asyncio
async def test_supported_model_variant_keeps_reasoning():
    """Substring match covers variants of a supported base model (e.g. date suffixes)."""
    service = OpenAIRealtimeLLMService(
        api_key="test-key",
        settings=OpenAIRealtimeLLMService.Settings(
            model="gpt-realtime-2-some-variant",
            system_instruction="be helpful",
            session_properties=events.SessionProperties(
                reasoning=events.Reasoning(effort="high"),
            ),
        ),
    )

    sink, handler_id = _capture_warnings()
    try:
        outgoing = await _send_and_capture(service)
    finally:
        logger.remove(handler_id)

    assert outgoing.reasoning is not None
    assert outgoing.reasoning.effort == "high"
    assert sink.getvalue() == ""


@pytest.mark.asyncio
async def test_no_warning_when_reasoning_is_unset_on_unsupported_model():
    service = OpenAIRealtimeLLMService(
        api_key="test-key",
        settings=OpenAIRealtimeLLMService.Settings(
            model="gpt-realtime-1.5",
            system_instruction="be helpful",
        ),
    )

    sink, handler_id = _capture_warnings()
    try:
        outgoing = await _send_and_capture(service)
    finally:
        logger.remove(handler_id)

    assert outgoing.reasoning is None
    assert sink.getvalue() == ""


# ---------------------------------------------------------------------------
# Runtime updates via LLMUpdateSettingsFrame
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_settings_update_with_reasoning_triggers_session_update():
    service = OpenAIRealtimeLLMService(
        api_key="test-key",
        settings=OpenAIRealtimeLLMService.Settings(
            model="gpt-realtime-2",
            system_instruction="be helpful",
            session_properties=events.SessionProperties(
                reasoning=events.Reasoning(effort="low"),
            ),
        ),
    )

    sent = _EventRecorder()
    service.send_client_event = sent

    # Send a runtime update that changes reasoning effort.
    new_sp = events.SessionProperties(reasoning=events.Reasoning(effort="high"))
    delta = OpenAIRealtimeLLMService.Settings(session_properties=new_sp)
    await service.process_frame(
        LLMUpdateSettingsFrame(delta=delta),
        FrameDirection.DOWNSTREAM,
    )

    session_updates = [e for e in sent.events if isinstance(e, events.SessionUpdateEvent)]
    assert len(session_updates) == 1
    sent_session = session_updates[0].session
    assert sent_session.reasoning is not None
    assert sent_session.reasoning.effort == "high"


@pytest.mark.asyncio
async def test_runtime_settings_update_strips_reasoning_on_unsupported_model():
    """Runtime updates honor the same strip-on-the-client rule as init-time."""
    service = OpenAIRealtimeLLMService(
        api_key="test-key",
        settings=OpenAIRealtimeLLMService.Settings(
            model="gpt-realtime-1.5",
            system_instruction="be helpful",
        ),
    )

    sent = _EventRecorder()
    service.send_client_event = sent

    new_sp = events.SessionProperties(reasoning=events.Reasoning(effort="high"))
    delta = OpenAIRealtimeLLMService.Settings(session_properties=new_sp)

    sink, handler_id = _capture_warnings()
    try:
        await service.process_frame(
            LLMUpdateSettingsFrame(delta=delta),
            FrameDirection.DOWNSTREAM,
        )
    finally:
        logger.remove(handler_id)

    session_updates = [e for e in sent.events if isinstance(e, events.SessionUpdateEvent)]
    assert len(session_updates) == 1
    assert session_updates[0].session.reasoning is None
    assert "stripping `reasoning`" in sink.getvalue()
