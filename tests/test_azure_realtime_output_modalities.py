#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests verifying that AzureRealtimeLLMService omits output_modalities.

Azure's Realtime API rejects the ``output_modalities`` parameter in both
``session.update`` and ``response.create`` events. These tests confirm it is
stripped before transmission, even when a caller explicitly configures it.
"""

import pytest

from pipecat.services.azure.realtime.llm import AzureRealtimeLLMService
from pipecat.services.openai.realtime import events


def _make_service(**kwargs) -> AzureRealtimeLLMService:
    defaults = dict(
        api_key="test-key",
        base_url="wss://example.openai.azure.com/openai/realtime?api-version=2025-04-01-preview&deployment=gpt-4o-realtime",
    )
    defaults.update(kwargs)
    return AzureRealtimeLLMService(**defaults)


# ---------------------------------------------------------------------------
# send_client_event: session.update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_update_omits_output_modalities_by_default():
    """output_modalities must not appear in session.update when not configured."""
    service = _make_service(
        settings=AzureRealtimeLLMService.Settings(
            model="gpt-4o-realtime-preview",
            system_instruction="be helpful",
            session_properties=events.SessionProperties(),
        )
    )

    sent_events: list[events.ClientEvent] = []

    # Capture what the base class would transmit by patching _ws_send.
    async def _capture(payload):
        sent_events.append(payload)

    service._ws_send = _capture
    await service.send_client_event(events.SessionUpdateEvent(session=events.SessionProperties()))

    assert len(sent_events) == 1
    assert "output_modalities" not in sent_events[0].get("session", {})


@pytest.mark.asyncio
async def test_session_update_strips_explicit_output_modalities():
    """output_modalities must be stripped from session.update even when explicitly set."""
    service = _make_service(
        settings=AzureRealtimeLLMService.Settings(
            model="gpt-4o-realtime-preview",
            system_instruction="be helpful",
            session_properties=events.SessionProperties(
                output_modalities=["audio", "text"],
            ),
        )
    )

    sent_events: list[dict] = []

    async def _capture(payload):
        sent_events.append(payload)

    service._ws_send = _capture
    await service.send_client_event(
        events.SessionUpdateEvent(
            session=events.SessionProperties(output_modalities=["audio", "text"])
        )
    )

    assert len(sent_events) == 1
    assert "output_modalities" not in sent_events[0].get("session", {})

    # The stored config is preserved — the strip only affects the wire payload.
    assert service._settings.session_properties.output_modalities == ["audio", "text"]


# ---------------------------------------------------------------------------
# send_client_event: response.create
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_create_omits_output_modalities():
    """output_modalities must not appear in response.create payloads."""
    service = _make_service()

    sent_events: list[dict] = []

    async def _capture(payload):
        sent_events.append(payload)

    service._ws_send = _capture
    await service.send_client_event(
        events.ResponseCreateEvent(response=events.ResponseProperties(output_modalities=["audio"]))
    )

    assert len(sent_events) == 1
    response_payload = sent_events[0].get("response") or {}
    assert "output_modalities" not in response_payload
