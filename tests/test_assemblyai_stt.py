#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the AssemblyAI streaming STT service connection parameters."""

import asyncio
from urllib.parse import parse_qs, urlparse

import pytest

from pipecat.frames.frames import BotStoppedSpeakingFrame, TTSTextFrame
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.assemblyai.observer import AssemblyAIContextObserver
from pipecat.services.assemblyai.stt import AssemblyAISTTService, is_u3_pro_model


def _query(service: AssemblyAISTTService) -> dict[str, list[str]]:
    """Build the WebSocket URL and return its parsed query parameters."""
    return parse_qs(urlparse(service._build_ws_url()).query)


def test_continuous_partials_defaults_to_true_for_u3_rt_pro():
    # u3-rt-pro is the default model; continuous_partials should be on by default.
    service = AssemblyAISTTService(api_key="test-key")
    assert _query(service)["continuous_partials"] == ["true"]


def test_continuous_partials_can_be_disabled():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(continuous_partials=False),
    )
    assert _query(service)["continuous_partials"] == ["false"]


def test_continuous_partials_omitted_for_universal_streaming():
    # continuous_partials is a U3Pro-only parameter and must not be sent otherwise.
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(model="universal-streaming-english"),
    )
    assert "continuous_partials" not in _query(service)


def test_interruption_delay_omitted_by_default():
    # Unset means "use the server default" — the param should not be sent.
    service = AssemblyAISTTService(api_key="test-key")
    assert "interruption_delay" not in _query(service)


def test_interruption_delay_sent_for_u3_rt_pro():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(interruption_delay=300),
    )
    assert _query(service)["interruption_delay"] == ["300"]


def test_interruption_delay_omitted_for_universal_streaming():
    # interruption_delay is a U3Pro-only parameter.
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(
            model="universal-streaming-english", interruption_delay=300
        ),
    )
    assert "interruption_delay" not in _query(service)


@pytest.mark.parametrize("value", [0, 1000])
def test_interruption_delay_boundaries_allowed(value):
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(interruption_delay=value),
    )
    assert _query(service)["interruption_delay"] == [str(value)]


@pytest.mark.parametrize("value", [-1, 1001])
def test_interruption_delay_out_of_range_raises(value):
    with pytest.raises(ValueError):
        AssemblyAISTTService(
            api_key="test-key",
            settings=AssemblyAISTTService.Settings(interruption_delay=value),
        )


# --- u3-rt-pro family detection ---


@pytest.mark.parametrize(
    "model, expected",
    [
        ("u3-rt-pro", True),
        ("u3-rt-pro-beta-1", True),
        ("universal-streaming-english", False),
        ("universal-streaming-multilingual", False),
        (None, False),
    ],
)
def test_is_u3_pro_model(model, expected):
    assert is_u3_pro_model(model) is expected


def test_u3_pro_features_sent_for_beta_variant():
    # The u3-rt-pro-beta-1 variant gets the same U3 Pro-only params as u3-rt-pro.
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(
            model="u3-rt-pro-beta-1",
            agent_context="May I take your order?",
            previous_context_n_turns=5,
            interruption_delay=300,
        ),
    )
    q = _query(service)
    assert q["speech_model"] == ["u3-rt-pro-beta-1"]
    assert q["agent_context"] == ["May I take your order?"]
    assert q["previous_context_n_turns"] == ["5"]
    assert q["interruption_delay"] == ["300"]
    assert q["continuous_partials"] == ["true"]


def test_beta_variant_allows_assemblyai_turn_detection_mode():
    # vad_force_turn_endpoint=False requires a u3-rt-pro family model; beta-1 qualifies.
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(model="u3-rt-pro-beta-1"),
        vad_force_turn_endpoint=False,
    )
    assert is_u3_pro_model(service._settings.model)


def test_update_agent_context_works_for_beta_variant():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(model="u3-rt-pro-beta-1"),
    )
    sent = []

    async def fake_send(**fields):
        sent.append(fields)

    service._send_update_configuration = fake_send
    asyncio.run(service.update_agent_context("hello"))

    assert sent == [{"agent_context": "hello"}]


# --- agent_context (context carryover) connection parameter ---


def test_agent_context_omitted_by_default():
    service = AssemblyAISTTService(api_key="test-key")
    assert "agent_context" not in _query(service)


def test_agent_context_sent_for_u3_rt_pro():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(agent_context="May I take your order?"),
    )
    assert _query(service)["agent_context"] == ["May I take your order?"]


def test_agent_context_omitted_for_universal_streaming():
    # agent_context (context carryover) is a u3-rt-pro-only parameter.
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(
            model="universal-streaming-english", agent_context="May I take your order?"
        ),
    )
    assert "agent_context" not in _query(service)


def test_agent_context_clipped_in_url():
    # Values longer than the limit are clipped to the last 1500 characters.
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(agent_context="a" * 2000),
    )
    assert _query(service)["agent_context"] == ["a" * 1500]


# --- previous_context_n_turns (context carryover window) ---


def test_previous_context_n_turns_omitted_by_default():
    # Unset means "use the server default" — the param should not be sent.
    service = AssemblyAISTTService(api_key="test-key")
    assert "previous_context_n_turns" not in _query(service)


def test_previous_context_n_turns_sent_for_u3_rt_pro():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(previous_context_n_turns=5),
    )
    assert _query(service)["previous_context_n_turns"] == ["5"]


def test_previous_context_n_turns_zero_disables_carryover():
    # 0 disables carryover entirely and must be sent (not treated as "unset").
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(previous_context_n_turns=0),
    )
    assert _query(service)["previous_context_n_turns"] == ["0"]


def test_previous_context_n_turns_omitted_for_universal_streaming():
    # Context carryover is a u3-rt-pro-only feature.
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(
            model="universal-streaming-english", previous_context_n_turns=5
        ),
    )
    assert "previous_context_n_turns" not in _query(service)


@pytest.mark.parametrize("value", [0, 100])
def test_previous_context_n_turns_boundaries_allowed(value):
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(previous_context_n_turns=value),
    )
    assert _query(service)["previous_context_n_turns"] == [str(value)]


@pytest.mark.parametrize("value", [-1, 101])
def test_previous_context_n_turns_out_of_range_raises(value):
    with pytest.raises(ValueError):
        AssemblyAISTTService(
            api_key="test-key",
            settings=AssemblyAISTTService.Settings(previous_context_n_turns=value),
        )


# --- update_agent_context() ---


def test_update_agent_context_clips_and_sends():
    service = AssemblyAISTTService(api_key="test-key")
    sent = []

    async def fake_send(**fields):
        sent.append(fields)

    service._send_update_configuration = fake_send
    asyncio.run(service.update_agent_context("a" * 2000))

    # Stored (so a reconnect re-seeds it) and sent via UpdateConfiguration, clipped.
    assert service._settings.agent_context == "a" * 1500
    assert sent == [{"agent_context": "a" * 1500}]


def test_update_agent_context_noop_for_non_u3():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(model="universal-streaming-english"),
    )
    sent = []

    async def fake_send(**fields):
        sent.append(fields)

    service._send_update_configuration = fake_send
    asyncio.run(service.update_agent_context("hello"))

    assert sent == []
    assert service._settings.agent_context is None


def test_update_agent_context_ignores_empty_text():
    service = AssemblyAISTTService(api_key="test-key")
    sent = []

    async def fake_send(**fields):
        sent.append(fields)

    service._send_update_configuration = fake_send
    asyncio.run(service.update_agent_context(""))

    assert sent == []


# --- _update_settings routing for agent_context ---


def _stub_connection(service: AssemblyAISTTService) -> tuple[list, list]:
    """Stub out network methods; return (sent UpdateConfigurations, reconnects)."""
    sent, reconnects = [], []

    async def fake_send(**fields):
        sent.append(fields)

    async def fake_disconnect():
        reconnects.append("disconnect")

    async def fake_connect():
        reconnects.append("connect")

    service._send_update_configuration = fake_send
    service._disconnect = fake_disconnect
    service._connect = fake_connect
    return sent, reconnects


def test_update_settings_agent_context_only_sends_without_reconnect():
    service = AssemblyAISTTService(api_key="test-key")
    sent, reconnects = _stub_connection(service)

    delta = AssemblyAISTTService.Settings(agent_context="a" * 2000)
    asyncio.run(service._update_settings(delta))

    # Hot update: clipped UpdateConfiguration, no reconnect.
    assert sent == [{"agent_context": "a" * 1500}]
    assert reconnects == []


def test_update_settings_agent_context_not_sent_for_non_u3():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(model="universal-streaming-english"),
    )
    sent, reconnects = _stub_connection(service)

    delta = AssemblyAISTTService.Settings(agent_context="hello")
    asyncio.run(service._update_settings(delta))

    # agent_context is u3-rt-pro-only; nothing goes to the server.
    assert sent == []
    assert reconnects == []


def test_update_settings_mixed_delta_reconnects_without_update_configuration():
    service = AssemblyAISTTService(api_key="test-key")
    sent, reconnects = _stub_connection(service)

    delta = AssemblyAISTTService.Settings(agent_context="hello", vad_threshold=0.5)
    asyncio.run(service._update_settings(delta))

    # Connect-time field changed → reconnect; the new connection's URL
    # re-seeds agent_context, so no separate UpdateConfiguration is sent.
    assert sent == []
    assert reconnects == ["disconnect", "connect"]
    assert "agent_context" in service._build_ws_url()


# --- AssemblyAIContextObserver ---


class _StubSTT:
    """Records agent_context values fed by the observer."""

    def __init__(self):
        self.contexts = []

    async def update_agent_context(self, text):
        self.contexts.append(text)


def _pushed(frame):
    return FramePushed(
        source=None,
        destination=None,
        frame=frame,
        direction=FrameDirection.DOWNSTREAM,
        timestamp=0,
    )


def test_observer_flushes_concatenated_reply_on_bot_stopped():
    stt = _StubSTT()
    observer = AssemblyAIContextObserver(stt)

    async def run():
        await observer.on_push_frame(_pushed(TTSTextFrame(text="Welcome", aggregated_by="word")))
        await observer.on_push_frame(_pushed(TTSTextFrame(text="back.", aggregated_by="word")))
        await observer.on_push_frame(_pushed(BotStoppedSpeakingFrame()))

    asyncio.run(run())
    assert stt.contexts == ["Welcome back."]


def test_observer_fires_on_agent_context_event():
    stt = _StubSTT()
    observer = AssemblyAIContextObserver(stt)
    received = []

    @observer.event_handler("on_agent_context")
    async def _on_agent_context(obs, text):
        received.append(text)

    async def run():
        await observer.on_push_frame(_pushed(TTSTextFrame(text="Hello", aggregated_by="word")))
        await observer.on_push_frame(_pushed(BotStoppedSpeakingFrame()))
        # Event handlers run as background tasks; let them complete.
        await asyncio.sleep(0.05)

    asyncio.run(run())
    assert received == ["Hello"]


def test_observer_dedupes_repeated_frame_and_ignores_empty_flush():
    stt = _StubSTT()
    observer = AssemblyAIContextObserver(stt)
    # The same frame object is seen on multiple hops (same id) and must count once.
    tts = TTSTextFrame(text="Hi", aggregated_by="word")

    async def run():
        await observer.on_push_frame(_pushed(tts))
        await observer.on_push_frame(_pushed(tts))
        await observer.on_push_frame(_pushed(BotStoppedSpeakingFrame()))
        # A second bot-stopped sibling with an empty buffer is a no-op.
        await observer.on_push_frame(_pushed(BotStoppedSpeakingFrame()))

    asyncio.run(run())
    assert stt.contexts == ["Hi"]
