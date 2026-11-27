#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the AssemblyAI streaming STT service connection parameters."""

import asyncio
import io
import json
import unittest
from urllib.parse import parse_qs, urlparse

import pytest
from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    TranscriptionFinalizedFrame,
    TranscriptionFrame,
    TranscriptionSpeaker,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.assemblyai.models import (
    AssemblyAIConnectionParams,
    TerminationMessage,
    TurnMessage,
    Word,
)
from pipecat.services.assemblyai.stt import AssemblyAISTTService, is_u3_pro_model
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


def _query(service: AssemblyAISTTService) -> dict[str, list[str]]:
    """Build the WebSocket URL and return its parsed query parameters."""
    return parse_qs(urlparse(service._build_ws_url()).query)


def _setup_service(service: AssemblyAISTTService, monkeypatch, sample_rate: int) -> None:
    """Set the service up with the given input sample rate, without connecting."""

    async def fake_connect():
        pass

    monkeypatch.setattr(service, "_connect", fake_connect)

    async def run():
        await service.setup(frame_processor_setup(TaskManager(), audio_in_sample_rate=sample_rate))

    asyncio.run(run())


def test_sample_rate_inherits_setup_when_omitted(monkeypatch):
    service = AssemblyAISTTService(api_key="test-key")

    _setup_service(service, monkeypatch, 8000)

    assert service.sample_rate == 8000
    assert _query(service)["sample_rate"] == ["8000"]


def test_explicit_sample_rate_overrides_setup(monkeypatch):
    service = AssemblyAISTTService(api_key="test-key", sample_rate=16000)

    _setup_service(service, monkeypatch, 8000)

    assert service.sample_rate == 16000
    assert _query(service)["sample_rate"] == ["16000"]


def test_default_model_is_assemblyai_stt_3_7_realtime_agent():
    # assemblyai-stt-3-7-realtime-agent is the default model sent to AssemblyAI.
    service = AssemblyAISTTService(api_key="test-key")
    assert _query(service)["speech_model"] == ["assemblyai-stt-3-7-realtime-agent"]


def test_deprecated_connection_params_default_to_assemblyai_stt_3_7_realtime_agent():
    with pytest.warns(DeprecationWarning):
        service = AssemblyAISTTService(
            api_key="test-key", connection_params=AssemblyAIConnectionParams()
        )
    assert _query(service)["speech_model"] == ["assemblyai-stt-3-7-realtime-agent"]


def test_continuous_partials_defaults_to_true_for_u3_pro():
    # assemblyai-stt-3-7-realtime-agent is the default U3 Pro model; continuous_partials
    # should be on by default.
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
        ("universal-3-5-pro", True),
        ("universal-3-6-pro", True),
        ("assemblyai-stt-3-7-realtime-agent", True),
        # Not intended for voice-agent use cases.
        ("assemblyai-stt-3-7-realtime", False),
        # The Sync/async name is not a streaming model.
        ("assemblyai-stt-3-7", False),
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


# --- universal-3-5-pro (U3 Pro family) ---


def test_u3_pro_features_sent_for_universal_3_5_pro():
    # universal-3-5-pro supports every u3-rt-pro param.
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(
            model="universal-3-5-pro",
            agent_context="May I take your order?",
            previous_context_n_turns=5,
            interruption_delay=300,
        ),
    )
    q = _query(service)
    assert q["speech_model"] == ["universal-3-5-pro"]
    assert q["agent_context"] == ["May I take your order?"]
    assert q["previous_context_n_turns"] == ["5"]
    assert q["interruption_delay"] == ["300"]
    assert q["continuous_partials"] == ["true"]


def test_universal_3_5_pro_allows_assemblyai_turn_detection_mode():
    # vad_force_turn_endpoint=False requires a U3 Pro family model; u3.5 qualifies.
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(model="universal-3-5-pro"),
        vad_force_turn_endpoint=False,
    )
    assert is_u3_pro_model(service._settings.model)


def test_update_agent_context_works_for_universal_3_5_pro():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(model="universal-3-5-pro"),
    )
    sent = []

    async def fake_send(**fields):
        sent.append(fields)

    service._send_update_configuration = fake_send
    asyncio.run(service.update_agent_context("hello"))

    assert sent == [{"agent_context": "hello"}]


# --- universal-3-6-pro (U3 Pro family; universal-3-5-pro upgraded) ---


def test_u3_pro_features_sent_for_universal_3_6_pro():
    # universal-3-6-pro is universal-3-5-pro upgraded and supports every u3-rt-pro param.
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(
            model="universal-3-6-pro",
            agent_context="May I take your order?",
            previous_context_n_turns=5,
            interruption_delay=300,
        ),
    )
    q = _query(service)
    assert q["speech_model"] == ["universal-3-6-pro"]
    assert q["agent_context"] == ["May I take your order?"]
    assert q["previous_context_n_turns"] == ["5"]
    assert q["interruption_delay"] == ["300"]
    assert q["continuous_partials"] == ["true"]


def test_universal_3_6_pro_allows_assemblyai_turn_detection_mode():
    # vad_force_turn_endpoint=False requires a U3 Pro family model; u3.6 qualifies.
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(model="universal-3-6-pro"),
        vad_force_turn_endpoint=False,
    )
    assert is_u3_pro_model(service._settings.model)


def test_update_agent_context_works_for_universal_3_6_pro():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(model="universal-3-6-pro"),
    )
    sent = []

    async def fake_send(**fields):
        sent.append(fields)

    service._send_update_configuration = fake_send
    asyncio.run(service.update_agent_context("hello"))

    assert sent == [{"agent_context": "hello"}]


# --- assemblyai-stt-3-7-realtime-agent (U3 Pro family; Universal-3.7 voice agents) ---


def test_u3_pro_features_sent_for_assemblyai_stt_3_7_realtime_agent():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(
            model="assemblyai-stt-3-7-realtime-agent",
            prompt="Outbound call to a health insurance payer.",
            agent_context="May I take your order?",
            previous_context_n_turns=5,
            interruption_delay=300,
            language_codes=[Language.EN],
        ),
    )
    q = _query(service)
    assert q["speech_model"] == ["assemblyai-stt-3-7-realtime-agent"]
    assert q["prompt"] == ["Outbound call to a health insurance payer."]
    assert q["agent_context"] == ["May I take your order?"]
    assert q["previous_context_n_turns"] == ["5"]
    assert q["interruption_delay"] == ["300"]
    assert q["continuous_partials"] == ["true"]
    assert q["language_codes"] == [json.dumps(["en"])]


def test_assemblyai_stt_3_7_realtime_agent_allows_assemblyai_turn_detection_mode():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(model="assemblyai-stt-3-7-realtime-agent"),
        vad_force_turn_endpoint=False,
    )
    assert is_u3_pro_model(service._settings.model)


def test_update_agent_context_works_for_assemblyai_stt_3_7_realtime_agent():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(model="assemblyai-stt-3-7-realtime-agent"),
    )
    sent = []

    async def fake_send(**fields):
        sent.append(fields)

    service._send_update_configuration = fake_send
    asyncio.run(service.update_agent_context("hello"))

    assert sent == [{"agent_context": "hello"}]


def test_deprecated_connection_params_accept_assemblyai_stt_3_7_realtime_agent():
    with pytest.warns(DeprecationWarning):
        service = AssemblyAISTTService(
            api_key="test-key",
            connection_params=AssemblyAIConnectionParams(
                speech_model="assemblyai-stt-3-7-realtime-agent"
            ),
        )
    assert _query(service)["speech_model"] == ["assemblyai-stt-3-7-realtime-agent"]
    assert is_u3_pro_model(service._settings.model)


def test_assemblyai_stt_3_7_realtime_is_not_a_voice_agent_model():
    # Not intended for voice-agent use cases, so AssemblyAI's built-in turn
    # detection is rejected for it.
    with pytest.raises(ValueError, match="requires a U3 Pro model"):
        AssemblyAISTTService(
            api_key="test-key",
            settings=AssemblyAISTTService.Settings(model="assemblyai-stt-3-7-realtime"),
            vad_force_turn_endpoint=False,
        )


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


# --- voice_focus / voice_focus_threshold ---


def test_voice_focus_omitted_by_default():
    service = AssemblyAISTTService(api_key="test-key")
    q = _query(service)
    assert "voice_focus" not in q
    assert "voice_focus_threshold" not in q


def test_voice_focus_sent_for_u3_rt_pro():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(voice_focus="near-field", voice_focus_threshold=0.5),
    )
    q = _query(service)
    assert q["voice_focus"] == ["near-field"]
    assert q["voice_focus_threshold"] == ["0.5"]


def test_voice_focus_sent_for_universal_3_5_pro():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(model="universal-3-5-pro", voice_focus="far-field"),
    )
    assert _query(service)["voice_focus"] == ["far-field"]


def test_voice_focus_omitted_for_universal_streaming():
    # voice_focus is a U3 Pro-only parameter.
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(
            model="universal-streaming-english",
            voice_focus="far-field",
            voice_focus_threshold=0.5,
        ),
    )
    q = _query(service)
    assert "voice_focus" not in q
    assert "voice_focus_threshold" not in q


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_voice_focus_threshold_boundaries_allowed(value):
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(
            voice_focus="near-field", voice_focus_threshold=value
        ),
    )
    assert _query(service)["voice_focus_threshold"] == [str(value)]


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_voice_focus_threshold_out_of_range_raises(value):
    with pytest.raises(ValueError):
        AssemblyAISTTService(
            api_key="test-key",
            settings=AssemblyAISTTService.Settings(
                voice_focus="near-field", voice_focus_threshold=value
            ),
        )


# --- mode (latency/accuracy preset) ---


def test_mode_omitted_by_default():
    service = AssemblyAISTTService(api_key="test-key")
    assert "mode" not in _query(service)


def test_mode_sent_for_u3_rt_pro():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(mode="max_accuracy"),
    )
    assert _query(service)["mode"] == ["max_accuracy"]


def test_mode_sent_for_universal_3_5_pro():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(model="universal-3-5-pro", mode="min_latency"),
    )
    assert _query(service)["mode"] == ["min_latency"]


def test_mode_omitted_for_universal_streaming():
    # mode is a U3 Pro-only parameter.
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(
            model="universal-streaming-english",
            mode="max_accuracy",
        ),
    )
    assert "mode" not in _query(service)


@pytest.mark.parametrize("value", ["min_latency", "balanced", "max_accuracy"])
def test_mode_values_accepted(value):
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(mode=value),
    )
    assert _query(service)["mode"] == [value]


# --- language_code ---


def test_language_code_omitted_by_default():
    # Unset means "not sent" — no steering, current behavior preserved.
    service = AssemblyAISTTService(api_key="test-key")
    assert "language_code" not in _query(service)


def test_language_code_sent_when_set():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(language_code="es"),
    )
    assert _query(service)["language_code"] == ["es"]


def test_language_code_sent_for_universal_streaming():
    # language_code is not U3 Pro-only; it is forwarded for any model.
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(
            model="universal-streaming-english",
            language_code="en",
        ),
    )
    assert _query(service)["language_code"] == ["en"]


def test_language_code_with_language_detection_warns():
    # Declaring a language and detecting one are independent server-side; setting
    # both warns about which is in effect and forwards both.
    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        service = AssemblyAISTTService(
            api_key="test-key",
            settings=AssemblyAISTTService.Settings(
                language_code="es",
                language_detection=True,
            ),
        )
    finally:
        logger.remove(handler_id)
    assert _query(service)["language_code"] == ["es"]
    assert _query(service)["language_detection"] == ["true"]
    assert "independent" in sink.getvalue()


# --- language_codes ---

# One past AssemblyAI's 10-language cap, all resolving to distinct base codes.
OVER_LIMIT_LANGUAGES = [
    Language.EN,
    Language.ES,
    Language.DE,
    Language.FR,
    Language.IT,
    Language.PT,
    Language.TR,
    Language.NL,
    Language.SV,
    Language.NO,
    Language.DA,
]


def test_language_codes_omitted_by_default():
    # Unset means "not sent" — no steering.
    service = AssemblyAISTTService(api_key="test-key")
    assert "language_codes" not in _query(service)


@pytest.mark.parametrize(
    "languages, expected",
    [
        ([Language.ES], ["es"]),
        ([Language.EN, Language.ES], ["en", "es"]),
        ([Language.EN, Language.ES, Language.FR], ["en", "es", "fr"]),
    ],
)
def test_language_codes_sent_json_encoded(languages, expected):
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(language_codes=languages),
    )
    assert _query(service)["language_codes"] == [json.dumps(expected)]


@pytest.mark.parametrize(
    "language, expected",
    [
        (Language.UR, "ur"),
        (Language.RU, "ru"),
        (Language.KO, "ko"),
        (Language.CA, "ca"),
        (Language.GL, "gl"),
        (Language.RO, "ro"),
        (Language.ET, "et"),
        (Language.FA, "fa"),
        (Language.YUE, "yue"),
        (Language.AF, "af"),
        (Language.MR, "mr"),
        (Language.ZU, "zu"),
        (Language.XH, "xh"),
        (Language.NN, "nn"),
    ],
)
def test_language_codes_covers_universal_3_6_pro_additions(language, expected):
    # universal-3-6-pro inherits universal-3-5-pro's full T1+T2+T3 language set;
    # these are verified-map entries, not the unlisted-language fallback.
    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        service = AssemblyAISTTService(
            api_key="test-key",
            settings=AssemblyAISTTService.Settings(language_codes=[language]),
        )
        query = _query(service)
    finally:
        logger.remove(handler_id)

    assert query["language_codes"] == [json.dumps([expected])]
    assert "not verified" not in sink.getvalue()


@pytest.mark.parametrize("model", ["universal-streaming-multilingual", "u3-rt-pro-beta-1"])
def test_language_codes_sent_for_u3_pro_models_only(model):
    # Steering is prompt-based, so only the U3 Pro family accepts it.
    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        service = AssemblyAISTTService(
            api_key="test-key",
            settings=AssemblyAISTTService.Settings(
                model=model,
                language_codes=[Language.EN, Language.ES],
            ),
        )
    finally:
        logger.remove(handler_id)

    if is_u3_pro_model(model):
        assert _query(service)["language_codes"] == [json.dumps(["en", "es"])]
        assert "language_codes is only supported" not in sink.getvalue()
    else:
        assert "language_codes" not in _query(service)
        assert "language_codes is only supported by U3 Pro models" in sink.getvalue()


def test_language_codes_with_language_detection_warns():
    # Declaring languages and detecting one are independent server-side; setting
    # both warns about which is in effect and forwards both.
    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        service = AssemblyAISTTService(
            api_key="test-key",
            settings=AssemblyAISTTService.Settings(
                language_codes=[Language.ES],
                language_detection=True,
            ),
        )
    finally:
        logger.remove(handler_id)
    assert _query(service)["language_codes"] == [json.dumps(["es"])]
    assert _query(service)["language_detection"] == ["true"]
    assert "independent" in sink.getvalue()


def test_language_code_and_language_codes_together_warns():
    # The two names are aliases for one server parameter, which binds
    # language_codes and ignores language_code. Both are still forwarded.
    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        service = AssemblyAISTTService(
            api_key="test-key",
            settings=AssemblyAISTTService.Settings(
                language_code="es",
                language_codes=[Language.EN, Language.ES],
            ),
        )
    finally:
        logger.remove(handler_id)
    query = _query(service)
    assert query["language_code"] == ["es"]
    assert query["language_codes"] == [json.dumps(["en", "es"])]
    assert "ignoring language_code" in sink.getvalue()


def test_language_codes_over_limit_raises():
    # AssemblyAI rejects an over-long list at connect time; fail where it was set.
    with pytest.raises(ValueError, match="at most 10 languages"):
        AssemblyAISTTService(
            api_key="test-key",
            settings=AssemblyAISTTService.Settings(language_codes=OVER_LIMIT_LANGUAGES),
        )


def test_language_codes_regional_variants_resolve_to_base_codes():
    # AssemblyAI declares base ISO codes, so variants of one language collapse
    # to a single code while declaration order is preserved.
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(
            language_codes=[Language.ES_MX, Language.EN_US, Language.EN_GB],
        ),
    )
    assert _query(service)["language_codes"] == [json.dumps(["es", "en"])]


def test_language_codes_unlisted_language_forwarded():
    # Which languages are supported is the server's call, so one outside the
    # verified map is forwarded as its base code rather than rejected here.
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(language_codes=[Language.PL]),
    )
    assert _query(service)["language_codes"] == [json.dumps(["pl"])]


def test_language_codes_at_limit_allowed():
    languages = [
        Language.EN,
        Language.ES,
        Language.DE,
        Language.FR,
        Language.IT,
        Language.PT,
        Language.TR,
        Language.NL,
        Language.SV,
        Language.NO,
    ]
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(language_codes=languages),
    )
    expected = ["en", "es", "de", "fr", "it", "pt", "tr", "nl", "sv", "no"]
    assert _query(service)["language_codes"] == [json.dumps(expected)]


# --- prompt + keyterms_prompt ---


def test_prompt_and_keyterms_sent_together_for_universal_3_5_pro():
    # U3 Pro models accept both parameters in the same session; the server
    # is the authority on their compatibility.
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(
            model="universal-3-5-pro",
            prompt="Some context for the session.",
            keyterms_prompt=["alpha", "beta"],
        ),
    )
    query = _query(service)
    assert query["prompt"] == ["Some context for the session."]
    assert query["keyterms_prompt"] == [json.dumps(["alpha", "beta"])]


def test_prompt_and_keyterms_sent_together_for_u3_rt_pro():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(
            model="u3-rt-pro",
            prompt="Some context for the session.",
            keyterms_prompt=["alpha", "beta"],
        ),
    )
    query = _query(service)
    assert query["prompt"] == ["Some context for the session."]
    assert query["keyterms_prompt"] == [json.dumps(["alpha", "beta"])]


def test_prompt_and_keyterms_raise_for_universal_streaming():
    with pytest.raises(ValueError, match="only supported by U3 Pro models"):
        AssemblyAISTTService(
            api_key="test-key",
            settings=AssemblyAISTTService.Settings(
                model="universal-streaming-english",
                prompt="Some context for the session.",
                keyterms_prompt=["alpha", "beta"],
            ),
        )


def test_prompt_alone_raises_for_universal_streaming():
    with pytest.raises(ValueError, match="only supported by U3 Pro models"):
        AssemblyAISTTService(
            api_key="test-key",
            settings=AssemblyAISTTService.Settings(
                model="universal-streaming-english",
                prompt="Some context for the session.",
            ),
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


# --- _update_settings routing for hot-updatable fields ---


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


def test_update_settings_language_codes_only_sends_without_reconnect():
    service = AssemblyAISTTService(api_key="test-key")
    sent, reconnects = _stub_connection(service)

    delta = AssemblyAISTTService.Settings(language_codes=[Language.EN, Language.ES])
    asyncio.run(service._update_settings(delta))

    # Hot update: UpdateConfiguration carries the resolved codes, no reconnect.
    assert sent == [{"language_codes": ["en", "es"]}]
    assert reconnects == []


def test_update_settings_language_codes_empty_list_clears_steering():
    service = AssemblyAISTTService(api_key="test-key")
    sent, reconnects = _stub_connection(service)

    delta = AssemblyAISTTService.Settings(language_codes=[])
    asyncio.run(service._update_settings(delta))

    # An empty list is the server's "clear steering back to the model default".
    assert sent == [{"language_codes": []}]
    assert reconnects == []


@pytest.mark.parametrize(
    "model", ["universal-streaming-english", "universal-streaming-multilingual"]
)
def test_update_settings_language_codes_not_sent_for_non_u3(model):
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(model=model),
    )
    sent, reconnects = _stub_connection(service)

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        delta = AssemblyAISTTService.Settings(language_codes=[Language.ES])
        asyncio.run(service._update_settings(delta))
    finally:
        logger.remove(handler_id)

    # Steering is U3 Pro-only; the server would discard this, so don't send it —
    # and don't reconnect either, since those models aren't steered at all.
    assert sent == []
    assert reconnects == []
    assert "only supported by U3 Pro models" in sink.getvalue()


def test_update_settings_language_codes_non_u3_warns_once():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(model="universal-streaming-english"),
    )
    _stub_connection(service)

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        asyncio.run(
            service._update_settings(AssemblyAISTTService.Settings(language_codes=[Language.ES]))
        )
        asyncio.run(
            service._update_settings(AssemblyAISTTService.Settings(language_codes=[Language.FR]))
        )
    finally:
        logger.remove(handler_id)

    # A language-switching bot would otherwise warn on every attempt.
    assert sink.getvalue().count("only supported by U3 Pro models") == 1


def test_update_settings_language_codes_with_agent_context_sends_both():
    service = AssemblyAISTTService(api_key="test-key")
    sent, reconnects = _stub_connection(service)

    delta = AssemblyAISTTService.Settings(agent_context="Hello.", language_codes=[Language.ES])
    asyncio.run(service._update_settings(delta))

    # Both fields are hot-updatable, so neither triggers a reconnect.
    assert sent == [{"agent_context": "Hello."}, {"language_codes": ["es"]}]
    assert reconnects == []


def test_update_settings_language_codes_over_limit_ignored():
    service = AssemblyAISTTService(api_key="test-key")
    sent, reconnects = _stub_connection(service)

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        delta = AssemblyAISTTService.Settings(language_codes=OVER_LIMIT_LANGUAGES)
        asyncio.run(service._update_settings(delta))
    finally:
        logger.remove(handler_id)

    # AssemblyAI closes the session on an over-long list, so it is never sent.
    assert sent == []
    assert reconnects == []
    assert "at most 10 languages" in sink.getvalue()


def test_update_settings_language_codes_over_limit_leaves_settings_intact():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(language_codes=[Language.ES]),
    )
    _stub_connection(service)

    asyncio.run(
        service._update_settings(AssemblyAISTTService.Settings(language_codes=OVER_LIMIT_LANGUAGES))
    )

    # A rejected list must not reach _settings, which a reconnect rebuilds from.
    assert _query(service)["language_codes"] == [json.dumps(["es"])]


def test_update_settings_over_limit_language_codes_still_reconnects_for_other_fields():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(language_codes=[Language.ES]),
    )
    sent, reconnects = _stub_connection(service)

    delta = AssemblyAISTTService.Settings(
        model="u3-rt-pro",
        language_codes=OVER_LIMIT_LANGUAGES,
    )
    asyncio.run(service._update_settings(delta))

    # Dropping language_codes leaves the connect-time model change to reconnect,
    # and the rebuilt URL carries the previous steering.
    assert reconnects == ["disconnect", "connect"]
    assert sent == []
    assert _query(service)["language_codes"] == [json.dumps(["es"])]


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


# --- _process_assistant_turn ---


def test__process_assistant_turn_delegates_to_update_agent_context():
    service = AssemblyAISTTService(api_key="test-key")
    sent = []

    async def fake_send(**fields):
        sent.append(fields)

    service._send_update_configuration = fake_send
    asyncio.run(service._process_assistant_turn("Hello there."))

    assert sent == [{"agent_context": "Hello there."}]
    assert service._settings.agent_context == "Hello there."


def test__process_assistant_turn_noop_for_non_u3():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(model="universal-streaming-english"),
    )
    sent = []

    async def fake_send(**fields):
        sent.append(fields)

    service._send_update_configuration = fake_send
    asyncio.run(service._process_assistant_turn("Hello."))

    assert sent == []


def test__process_assistant_turn_noop_when_carryover_disabled():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(previous_context_n_turns=0),
    )
    sent = []

    async def fake_send(**fields):
        sent.append(fields)

    service._send_update_configuration = fake_send
    asyncio.run(service._process_assistant_turn("Hello."))

    assert sent == []


def test_speech_started_proposes_turn_without_interrupting():
    # The service proposes the turn; the user turn strategies decide it and own
    # the interruption, so nothing is interrupted from here.
    service = AssemblyAISTTService(api_key="test-key", vad_force_turn_endpoint=False)
    events = []

    async def fake_broadcast_frame(frame_cls, **kwargs):
        events.append(("broadcast", frame_cls))

    async def fake_broadcast_interruption():
        events.append(("interruption", None))

    service.broadcast_frame = fake_broadcast_frame
    service.broadcast_interruption = fake_broadcast_interruption

    asyncio.run(service._handle_speech_started(None))

    assert events == [("broadcast", ProposedUserStartedSpeakingFrame)]


def test_termination_acknowledges_without_ending_pipeline():
    # The server sends Termination in reply to our Terminate, including on a
    # settings-triggered reconnect, so it must not end the pipeline.
    service = AssemblyAISTTService(api_key="test-key")
    pushed = []

    async def fake_push_frame(frame, direction=None):
        pushed.append(frame)

    service.push_frame = fake_push_frame

    message = TerminationMessage(
        type="Termination", audio_duration_seconds=1, session_duration_seconds=2
    )
    asyncio.run(service._handle_termination(message))

    assert service._termination_event.is_set()
    assert pushed == []


def test_should_interrupt_rides_on_recommended_strategies():
    # should_interrupt no longer gates a local broadcast; it configures the
    # strategies the service recommends via its metadata frame.
    for should_interrupt in (True, False):
        service = AssemblyAISTTService(
            api_key="test-key",
            vad_force_turn_endpoint=False,
            should_interrupt=should_interrupt,
        )
        strategies = service.service_metadata_frame().user_turn_strategies
        assert isinstance(strategies, ExternalUserTurnStrategies)
        assert strategies.enable_interruptions is should_interrupt


# --- Speaker diarization ---


def _turn(
    transcript: str,
    *,
    end_of_turn: bool,
    speaker: str | None = None,
    words: list[tuple[str, str | None]] | None = None,
) -> TurnMessage:
    return TurnMessage(
        turn_order=0,
        turn_is_formatted=end_of_turn,
        end_of_turn=end_of_turn,
        transcript=transcript,
        end_of_turn_confidence=1.0,
        words=[
            Word(start=0, end=1, text=text, confidence=1.0, word_is_final=True, speaker=label)
            for text, label in words or []
        ],
        speaker=speaker,
    )


def _transcription_frames(service: AssemblyAISTTService, message: TurnMessage) -> list:
    """Run one Turn message through the service and return the pushed frames."""
    pushed = []

    async def fake_push_frame(frame, direction=None):
        pushed.append(frame)

    service.push_frame = fake_push_frame
    asyncio.run(service._handle_transcription(message))
    return [f for f in pushed if isinstance(f, (TranscriptionFrame, InterimTranscriptionFrame))]


@pytest.mark.parametrize("vad_force_turn_endpoint", [True, False])
def test_final_turn_carries_speaker_and_keeps_user_id(vad_force_turn_endpoint):
    service = AssemblyAISTTService(
        api_key="test-key",
        vad_force_turn_endpoint=vad_force_turn_endpoint,
        settings=AssemblyAISTTService.Settings(speaker_labels=True),
    )
    service._user_id = "session-user"

    frames = _transcription_frames(service, _turn("Hello", end_of_turn=True, speaker="A"))

    assert len(frames) == 1
    assert isinstance(frames[0], TranscriptionFrame)
    assert frames[0].text == "Hello"
    assert frames[0].speaker.label == "A"
    assert frames[0].user_id == "session-user"


def test_interim_turn_without_speaker_has_no_label():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(
            speaker_labels=True, speaker_format="{speaker}: {text}"
        ),
    )

    frames = _transcription_frames(service, _turn("Hel", end_of_turn=False))

    assert isinstance(frames[0], InterimTranscriptionFrame)
    assert frames[0].text == "Hel"
    assert frames[0].speaker is None


def test_speaker_format_setting_labels_text():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(
            speaker_labels=True, speaker_format="{speaker}: {text}"
        ),
    )

    frames = _transcription_frames(service, _turn("Hello", end_of_turn=True, speaker="B"))

    assert frames[0].text == "B: Hello"
    assert frames[0].speaker.label == "B"


def test_deprecated_speaker_format_arg_moves_to_settings():
    with pytest.warns(DeprecationWarning, match="speaker_format"):
        service = AssemblyAISTTService(api_key="test-key", speaker_format="{speaker}: {text}")

    assert service._settings.speaker_format == "{speaker}: {text}"


def test_speaker_format_not_sent_to_assemblyai():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(speaker_format="{speaker}: {text}"),
    )

    assert "speaker_format" not in _query(service)


def test_update_settings_speaker_format_does_not_reconnect():
    service = AssemblyAISTTService(api_key="test-key")
    sent, reconnects = _stub_connection(service)

    delta = AssemblyAISTTService.Settings(speaker_format="{speaker}: {text}")
    changed = asyncio.run(service._update_settings(delta))

    assert "speaker_format" in changed
    assert sent == []
    assert reconnects == []


@pytest.mark.parametrize("placeholder", ["PENDING", "UNKNOWN"])
def test_unassigned_speaker_is_not_a_label(placeholder):
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(
            speaker_labels=True, speaker_format="{speaker}: {text}"
        ),
    )

    frames = _transcription_frames(service, _turn("Hello", end_of_turn=True, speaker=placeholder))

    assert frames[0].speaker is None
    assert frames[0].text == "Hello"


def test_speaker_revision_is_not_an_error():
    service = AssemblyAISTTService(
        api_key="test-key", settings=AssemblyAISTTService.Settings(speaker_labels=True)
    )
    errors = []

    async def fake_push_error(*args, **kwargs):
        errors.append((args, kwargs))

    service.push_error = fake_push_error
    message = {
        "type": "SpeakerRevision",
        "revisions": [{"turn_order": 0, "speaker_label": "B", "words": []}],
    }

    asyncio.run(service._handle_message(message))

    assert errors == []


def test_final_turn_is_split_by_word_speakers():
    """A turn holding two speakers becomes one frame per speaker run."""
    service = AssemblyAISTTService(
        api_key="test-key",
        vad_force_turn_endpoint=False,
        settings=AssemblyAISTTService.Settings(
            speaker_labels=True, speaker_format="{speaker}: {text}"
        ),
    )
    message = _turn(
        "Mine is the banana. I take the train.",
        end_of_turn=True,
        speaker="B",
        words=[
            ("Mine", "B"),
            ("is", "B"),
            ("the", "B"),
            ("banana.", "B"),
            ("I", "A"),
            ("take", "A"),
            ("the", "A"),
            ("train.", "A"),
        ],
    )

    frames = _transcription_frames(service, message)

    assert [(f.speaker, f.text, f.finalized) for f in frames] == [
        (TranscriptionSpeaker("B", 1), "B: Mine is the banana.", False),
        (TranscriptionSpeaker("A", 2), "A: I take the train.", True),
    ]


def test_unassigned_word_takes_the_turn_speaker():
    service = AssemblyAISTTService(
        api_key="test-key", settings=AssemblyAISTTService.Settings(speaker_labels=True)
    )
    message = _turn(
        "Yeah. Different.",
        end_of_turn=True,
        speaker="A",
        words=[("Yeah.", "PENDING"), ("Different.", "A")],
    )

    frames = _transcription_frames(service, message)

    assert [(f.speaker, f.text) for f in frames] == [
        (TranscriptionSpeaker("A", 1), "Yeah. Different."),
    ]


def test_turn_speaker_is_used_when_words_do_not_match_the_transcript():
    service = AssemblyAISTTService(
        api_key="test-key", settings=AssemblyAISTTService.Settings(speaker_labels=True)
    )
    message = _turn(
        "It takes 40 minutes.",
        end_of_turn=True,
        speaker="A",
        words=[("It", "A"), ("takes", "A"), ("forty", "B"), ("minutes.", "B")],
    )

    frames = _transcription_frames(service, message)

    assert [(f.speaker, f.text) for f in frames] == [
        (TranscriptionSpeaker("A", 1), "It takes 40 minutes."),
    ]


if __name__ == "__main__":
    unittest.main()


def test_disconnect_terminates_an_open_session_before_closing():
    service = AssemblyAISTTService(api_key="test-key")
    service._audio_buffer.extend(b"\1\2")
    events = []

    class FakeWebsocket:
        state = State.OPEN

        async def send(self, message):
            events.append(("send", message))
            if message == json.dumps({"type": "Terminate"}):
                # The receive loop sets this on the server's Termination reply.
                service._termination_event.set()

        async def close(self):
            events.append("close")

    async def fake_cancel_task(task, timeout=None):
        events.append(("cancel", task))

    service._websocket = FakeWebsocket()
    service._receive_task = "receive-task"
    service.cancel_task = fake_cancel_task

    asyncio.run(service._disconnect())

    assert events == [
        ("send", b"\1\2"),
        ("send", json.dumps({"type": "Terminate"})),
        ("cancel", "receive-task"),
        "close",
    ]
    assert service._receive_task is None
    assert service._websocket is None


# --- mode="manual" ---
#
# Manual mode turns AssemblyAI's turn detection off so Pipecat ends every turn:
# each VAD stop sends the buffered audio and a ForceEndpoint, which AssemblyAI
# answers with exactly one final (empty if nothing was said). Only
# assemblyai-stt-3-7-realtime-agent supports it; the server rejects other
# models with error 3006.

_MANUAL_MODEL = "assemblyai-stt-3-7-realtime-agent"
_FORCE_ENDPOINT = json.dumps({"type": "ForceEndpoint"})
_TERMINATE = json.dumps({"type": "Terminate"})


def _manual_service(**settings) -> AssemblyAISTTService:
    return AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(model=_MANUAL_MODEL, mode="manual", **settings),
    )


def test_mode_manual_sent_with_turn_silence_left_to_the_server():
    """No 100 ms defaults: max_turn_silence is manual mode's fallback turn end
    (10000 ms by default), and 100 ms would end turns on every short pause."""
    query = _query(_manual_service())

    assert query["mode"] == ["manual"]
    assert "min_turn_silence" not in query
    assert "max_turn_silence" not in query


def test_mode_manual_does_not_copy_min_turn_silence_into_max():
    query = _query(_manual_service(min_turn_silence=400))

    assert query["min_turn_silence"] == ["400"]
    assert "max_turn_silence" not in query


def test_mode_manual_sends_explicit_max_turn_silence():
    query = _query(_manual_service(max_turn_silence=5000))

    assert query["max_turn_silence"] == ["5000"]


@pytest.mark.parametrize(
    "model",
    [
        "u3-rt-pro",
        "universal-3-5-pro",
        "universal-3-6-pro",
        "universal-streaming-english",
        "universal-streaming-multilingual",
    ],
)
def test_mode_manual_rejected_on_other_models(model):
    with pytest.raises(ValueError, match="manual"):
        AssemblyAISTTService(
            api_key="test-key",
            settings=AssemblyAISTTService.Settings(model=model, mode="manual"),
        )


def test_mode_manual_requires_vad_force_turn_endpoint():
    """With AssemblyAI's turn detection mode nothing sends ForceEndpoint, so turns
    would only end after max_turn_silence."""
    with pytest.raises(ValueError, match="vad_force_turn_endpoint"):
        AssemblyAISTTService(
            api_key="test-key",
            vad_force_turn_endpoint=False,
            settings=AssemblyAISTTService.Settings(model=_MANUAL_MODEL, mode="manual"),
        )


def test_mode_manual_rejects_speaker_labels():
    """With speaker labels on, the server uses its speaker-label turn defaults
    instead of manual mode and would end turns Pipecat hasn't ended."""
    with pytest.raises(ValueError, match="speaker_labels"):
        _manual_service(speaker_labels=True)


def test_speaker_labels_still_allowed_outside_manual_mode():
    service = AssemblyAISTTService(
        api_key="test-key",
        settings=AssemblyAISTTService.Settings(mode="balanced", speaker_labels=True),
    )

    assert _query(service)["speaker_labels"] == ["true"]


@pytest.mark.parametrize("mode", [None, "min_latency", "balanced", "max_accuracy"])
def test_other_modes_keep_pipecat_turn_silence(mode):
    service = AssemblyAISTTService(
        api_key="test-key", settings=AssemblyAISTTService.Settings(mode=mode)
    )

    query = _query(service)
    assert query["min_turn_silence"] == ["100"]
    assert query["max_turn_silence"] == ["100"]


def test_update_settings_ignores_switching_into_manual_mode():
    """Pipecat mode sent 100 ms turn silences at construction; manual mode with
    those would end turns on every short pause."""
    service = AssemblyAISTTService(api_key="test-key")
    _, reconnects = _stub_connection(service)

    changed = asyncio.run(service._update_settings(AssemblyAISTTService.Settings(mode="manual")))

    assert changed == {}
    assert service._settings.mode is None
    assert reconnects == []


def test_update_settings_ignores_switching_out_of_manual_mode():
    service = _manual_service()
    _, reconnects = _stub_connection(service)

    changed = asyncio.run(service._update_settings(AssemblyAISTTService.Settings(mode="balanced")))

    assert changed == {}
    assert service._settings.mode == "manual"
    assert reconnects == []


def test_update_settings_still_switches_between_other_modes():
    service = AssemblyAISTTService(
        api_key="test-key", settings=AssemblyAISTTService.Settings(mode="balanced")
    )
    _, reconnects = _stub_connection(service)

    asyncio.run(service._update_settings(AssemblyAISTTService.Settings(mode="max_accuracy")))

    assert service._settings.mode == "max_accuracy"
    assert reconnects == ["disconnect", "connect"]


def test_update_settings_ignores_speaker_labels_in_manual_mode():
    service = _manual_service()
    _, reconnects = _stub_connection(service)

    changed = asyncio.run(
        service._update_settings(AssemblyAISTTService.Settings(speaker_labels=True))
    )

    assert changed == {}
    assert service._settings.speaker_labels is None
    assert reconnects == []


class _RecordingWebsocket:
    state = State.OPEN

    def __init__(self):
        self.sent: list = []

    async def send(self, message):
        self.sent.append(message)


def _pcm(ms: int, sample: bytes = b"\1\0") -> bytes:
    """16-bit mono audio at 16 kHz."""
    return sample * (16 * ms)


def _live_service(monkeypatch, *, manual: bool = True) -> tuple[AssemblyAISTTService, list]:
    """A service set up at 16 kHz with a recording websocket, and what it sends."""
    service = _manual_service() if manual else AssemblyAISTTService(api_key="test-key")
    _setup_service(service, monkeypatch, 16000)
    websocket = _RecordingWebsocket()
    service._websocket = websocket

    async def fake_push_frame(frame, direction=None):
        pass

    service.push_frame = fake_push_frame
    return service, websocket.sent


async def _send_audio(service: AssemblyAISTTService, audio: bytes):
    async for _ in service.run_stt(audio):
        pass


async def _vad_stop(service: AssemblyAISTTService):
    # stop_secs=0 skips the TTFB measurement, which would start a task.
    await service.process_frame(
        VADUserStoppedSpeakingFrame(stop_secs=0.0), FrameDirection.DOWNSTREAM
    )


@pytest.mark.parametrize("manual", [True, False])
def test_vad_stop_sends_force_endpoint_after_the_buffered_audio(monkeypatch, manual):
    """The tail of the turn reaches AssemblyAI before the ForceEndpoint, so the
    final covers all of it."""
    service, sent = _live_service(monkeypatch, manual=manual)

    async def run():
        await _send_audio(service, _pcm(130))  # two full 50 ms chunks and a 30 ms tail
        await _vad_stop(service)

    asyncio.run(run())

    assert [len(m) // 32 for m in sent[:3]] == [50, 50, 30]
    assert sent[3] == _FORCE_ENDPOINT
    assert service._finalize_requested


def test_vad_stop_sends_force_endpoint_when_no_audio_is_left(monkeypatch):
    service, sent = _live_service(monkeypatch)

    async def run():
        await _send_audio(service, _pcm(100))
        await _vad_stop(service)

    asyncio.run(run())

    assert [len(m) // 32 for m in sent[:2]] == [50, 50]
    assert sent[2:] == [_FORCE_ENDPOINT]


def test_vad_stop_holds_a_tail_under_20ms_for_the_next_chunk(monkeypatch):
    """AssemblyAI rejects audio chunks under 20 ms (error 3007), and a VAD stop can
    leave a shorter tail. It goes out at the start of the next chunk: padding it
    would insert audio and shift every later word timestamp."""
    service, sent = _live_service(monkeypatch)

    async def run():
        await _send_audio(service, _pcm(10, b"\1\0"))
        await _vad_stop(service)
        await _send_audio(service, _pcm(40, b"\2\0"))

    asyncio.run(run())

    # One full chunk: the held 10 ms, then the next 40 ms, with nothing inserted.
    assert sent == [_FORCE_ENDPOINT, _pcm(10, b"\1\0") + _pcm(40, b"\2\0")]


def test_disconnect_pads_a_tail_under_20ms(monkeypatch):
    """No audio follows the last tail, so it's padded with silence instead."""
    service, sent = _live_service(monkeypatch)
    service._websocket.close = lambda: asyncio.sleep(0)

    async def fake_send(message):
        sent.append(message)
        if message == _TERMINATE:
            service._termination_event.set()

    service._websocket.send = fake_send

    async def run():
        await _send_audio(service, _pcm(10))
        await service._disconnect()

    asyncio.run(run())

    assert sent == [_pcm(10) + b"\0" * 320, _TERMINATE]  # 10 ms, padded to 20 ms


def _empty_final() -> TurnMessage:
    return _turn("", end_of_turn=True)


def _handled_frames(service: AssemblyAISTTService, message: TurnMessage) -> list:
    pushed = []

    async def fake_push_frame(frame, direction=None):
        pushed.append(frame)

    service.push_frame = fake_push_frame
    asyncio.run(service._handle_transcription(message))
    return pushed


def test_empty_final_after_force_endpoint_finalizes_the_turn():
    """A ForceEndpoint with nothing to end gets an empty final. It still confirms
    the finalize, so the user turn doesn't wait out the STT timeout."""
    service = _manual_service()
    service.request_finalize()

    frames = _handled_frames(service, _empty_final())

    assert [type(f) for f in frames] == [TranscriptionFinalizedFrame]


def test_empty_final_without_force_endpoint_pushes_nothing():
    service = _manual_service()

    assert _handled_frames(service, _empty_final()) == []


def test_empty_final_in_assemblyai_turn_detection_mode_pushes_nothing():
    service = AssemblyAISTTService(api_key="test-key", vad_force_turn_endpoint=False)
    service.request_finalize()

    assert _handled_frames(service, _empty_final()) == []
