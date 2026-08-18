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

from pipecat.frames.frames import ProposedUserStartedSpeakingFrame
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


def test_default_model_is_universal_3_5_pro():
    # universal-3-5-pro is the default model sent to AssemblyAI.
    service = AssemblyAISTTService(api_key="test-key")
    assert _query(service)["speech_model"] == ["universal-3-5-pro"]


def test_continuous_partials_defaults_to_true_for_u3_pro():
    # universal-3-5-pro is the default U3 Pro model; continuous_partials should be on by default.
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
        settings=AssemblyAISTTService.Settings(language_codes=[Language.KO]),
    )
    assert _query(service)["language_codes"] == [json.dumps(["ko"])]


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
    # Older models keep the client-side mutual-exclusivity check.
    with pytest.raises(ValueError, match="only U3 Pro models"):
        AssemblyAISTTService(
            api_key="test-key",
            settings=AssemblyAISTTService.Settings(
                model="universal-streaming-english",
                prompt="Some context for the session.",
                keyterms_prompt=["alpha", "beta"],
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


if __name__ == "__main__":
    unittest.main()
