#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the typed settings infrastructure in pipecat.services.settings."""

import pytest
from deepgram import LiveOptions

from pipecat.services.deepgram.stt import DeepgramSTTSettings
from pipecat.services.deepgram.stt_sagemaker import DeepgramSageMakerSTTSettings
from pipecat.services.settings import (
    NOT_GIVEN,
    LLMSettings,
    ServiceSettings,
    STTSettings,
    TTSSettings,
    _NotGiven,
    is_given,
)

# ---------------------------------------------------------------------------
# NOT_GIVEN sentinel
# ---------------------------------------------------------------------------


class TestNotGiven:
    def test_singleton(self):
        """NOT_GIVEN is a singleton — every reference is the same object."""
        assert _NotGiven() is _NotGiven()
        assert NOT_GIVEN is _NotGiven()

    def test_repr(self):
        assert repr(NOT_GIVEN) == "NOT_GIVEN"

    def test_bool_is_false(self):
        assert not NOT_GIVEN
        assert bool(NOT_GIVEN) is False

    def test_is_given_with_not_given(self):
        assert is_given(NOT_GIVEN) is False

    def test_is_given_with_none(self):
        assert is_given(None) is True

    def test_is_given_with_values(self):
        assert is_given(0) is True
        assert is_given("") is True
        assert is_given(False) is True
        assert is_given(42) is True
        assert is_given("hello") is True


# ---------------------------------------------------------------------------
# ServiceSettings base
# ---------------------------------------------------------------------------


class TestServiceSettings:
    def test_default_fields_are_not_given(self):
        s = ServiceSettings()
        assert not is_given(s.model)
        assert s.extra == {}

    def test_given_fields_empty_by_default(self):
        s = ServiceSettings()
        assert s.given_fields() == {}

    def test_given_fields_includes_set_values(self):
        s = ServiceSettings(model="gpt-4o")
        assert s.given_fields() == {"model": "gpt-4o"}

    def test_given_fields_includes_extra(self):
        s = ServiceSettings(model="gpt-4o")
        s.extra = {"custom_key": 42}
        result = s.given_fields()
        assert result == {"model": "gpt-4o", "custom_key": 42}

    def test_copy_is_deep(self):
        s = ServiceSettings(model="gpt-4o")
        s.extra = {"nested": {"a": 1}}
        c = s.copy()
        assert c.model == "gpt-4o"
        assert c.extra == {"nested": {"a": 1}}
        # Mutating the copy shouldn't affect the original
        c.extra["nested"]["a"] = 999
        assert s.extra["nested"]["a"] == 1


# ---------------------------------------------------------------------------
# apply_update
# ---------------------------------------------------------------------------


class TestApplyUpdate:
    def test_apply_update_basic(self):
        current = TTSSettings(voice="alice", language="en")
        delta = TTSSettings(voice="bob")
        changed = current.apply_update(delta)
        assert changed.keys() == {"voice"}
        assert changed["voice"] == "alice"  # old value
        assert current.voice == "bob"
        assert current.language == "en"

    def test_apply_update_no_change(self):
        current = TTSSettings(voice="alice", language="en")
        delta = TTSSettings(voice="alice")
        changed = current.apply_update(delta)
        assert changed == {}
        assert current.voice == "alice"

    def test_apply_update_not_given_skipped(self):
        current = TTSSettings(voice="alice", language="en")
        delta = TTSSettings()  # all NOT_GIVEN
        changed = current.apply_update(delta)
        assert changed == {}
        assert current.voice == "alice"
        assert current.language == "en"

    def test_apply_update_multiple_fields(self):
        current = LLMSettings(temperature=0.7, max_tokens=100)
        delta = LLMSettings(temperature=0.9, max_tokens=200, top_p=0.95)
        changed = current.apply_update(delta)
        assert changed.keys() == {"temperature", "max_tokens", "top_p"}
        assert changed["temperature"] == 0.7
        assert changed["max_tokens"] == 100
        assert current.temperature == 0.9
        assert current.max_tokens == 200
        assert current.top_p == 0.95

    def test_apply_update_extra_merged(self):
        current = TTSSettings(voice="alice")
        current.extra = {"speed": 1.0, "stability": 0.5}
        delta = TTSSettings()
        delta.extra = {"speed": 1.2}
        changed = current.apply_update(delta)
        assert "speed" in changed
        assert changed["speed"] == 1.0  # old value
        assert current.extra == {"speed": 1.2, "stability": 0.5}

    def test_apply_update_extra_no_change(self):
        current = TTSSettings(voice="alice")
        current.extra = {"speed": 1.0}
        delta = TTSSettings()
        delta.extra = {"speed": 1.0}
        changed = current.apply_update(delta)
        assert changed == {}

    def test_apply_update_model_field(self):
        current = ServiceSettings(model="old-model")
        delta = ServiceSettings(model="new-model")
        changed = current.apply_update(delta)
        assert changed.keys() == {"model"}
        assert changed["model"] == "old-model"
        assert current.model == "new-model"

    def test_apply_update_none_is_a_valid_value(self):
        """Setting a field to None should be treated as a change from NOT_GIVEN."""
        current = TTSSettings()
        delta = TTSSettings(language=None)
        changed = current.apply_update(delta)
        assert "language" in changed
        assert current.language is None

    def test_apply_update_none_to_value(self):
        current = TTSSettings(language=None)
        delta = TTSSettings(language="en")
        changed = current.apply_update(delta)
        assert "language" in changed
        assert changed["language"] is None  # old value was None
        assert current.language == "en"


# ---------------------------------------------------------------------------
# from_mapping
# ---------------------------------------------------------------------------


class TestFromMapping:
    def test_basic_mapping(self):
        s = TTSSettings.from_mapping({"voice": "alice", "language": "en"})
        assert s.voice == "alice"
        assert s.language == "en"
        assert not is_given(s.model)

    def test_alias_resolution(self):
        """'voice_id' is an alias for 'voice' in TTSSettings."""
        s = TTSSettings.from_mapping({"voice_id": "alice"})
        assert s.voice == "alice"

    def test_unknown_keys_go_to_extra(self):
        s = TTSSettings.from_mapping({"voice": "alice", "speed": 1.2, "stability": 0.5})
        assert s.voice == "alice"
        assert s.extra == {"speed": 1.2, "stability": 0.5}

    def test_model_field(self):
        s = LLMSettings.from_mapping({"model": "gpt-4o", "temperature": 0.7})
        assert s.model == "gpt-4o"
        assert s.temperature == 0.7

    def test_empty_mapping(self):
        s = ServiceSettings.from_mapping({})
        assert s.given_fields() == {}

    def test_all_unknown_keys(self):
        s = ServiceSettings.from_mapping({"foo": 1, "bar": 2})
        assert not is_given(s.model)
        assert s.extra == {"foo": 1, "bar": 2}

    def test_llm_settings_from_mapping(self):
        s = LLMSettings.from_mapping({"temperature": 0.5, "max_tokens": 1000, "custom_param": True})
        assert s.temperature == 0.5
        assert s.max_tokens == 1000
        assert s.extra == {"custom_param": True}

    def test_stt_settings_from_mapping(self):
        s = STTSettings.from_mapping({"language": "fr", "model": "whisper-large"})
        assert s.language == "fr"
        assert s.model == "whisper-large"


# ---------------------------------------------------------------------------
# LLMSettings specifics
# ---------------------------------------------------------------------------


class TestLLMSettings:
    def test_all_fields_not_given_by_default(self):
        s = LLMSettings()
        for name in (
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "seed",
        ):
            assert not is_given(getattr(s, name)), f"{name} should be NOT_GIVEN"

    def test_given_fields(self):
        s = LLMSettings(temperature=0.7, seed=42)
        assert s.given_fields() == {"temperature": 0.7, "seed": 42}


# ---------------------------------------------------------------------------
# TTSSettings specifics
# ---------------------------------------------------------------------------


class TestTTSSettings:
    def test_all_fields_not_given_by_default(self):
        s = TTSSettings()
        for name in ("model", "voice", "language"):
            assert not is_given(getattr(s, name)), f"{name} should be NOT_GIVEN"

    def test_aliases_class_var(self):
        assert TTSSettings._aliases == {"voice_id": "voice"}

    def test_given_fields(self):
        s = TTSSettings(voice="alice")
        assert s.given_fields() == {"voice": "alice"}


# ---------------------------------------------------------------------------
# STTSettings specifics
# ---------------------------------------------------------------------------


class TestSTTSettings:
    def test_all_fields_not_given_by_default(self):
        s = STTSettings()
        for name in ("model", "language"):
            assert not is_given(getattr(s, name)), f"{name} should be NOT_GIVEN"

    def test_given_fields(self):
        s = STTSettings(language="en", model="whisper-large")
        assert s.given_fields() == {"language": "en", "model": "whisper-large"}


# ---------------------------------------------------------------------------
# Integration: roundtrip from_mapping → apply_update
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_from_mapping_then_apply_update(self):
        """Simulate the real flow: dict arrives via frame, gets converted, applied."""
        # Simulating current service state
        current = TTSSettings(model="eleven_turbo_v2_5", voice="alice", language="en")
        current.extra = {"stability": 0.5, "speed": 1.0}

        # Incoming dict-based update
        raw = {"voice_id": "bob", "speed": 1.2}
        delta = TTSSettings.from_mapping(raw)

        changed = current.apply_update(delta)
        assert changed.keys() == {"voice", "speed"}
        assert changed["voice"] == "alice"
        assert changed["speed"] == 1.0
        assert current.voice == "bob"
        assert current.language == "en"
        assert current.extra["speed"] == 1.2
        assert current.extra["stability"] == 0.5

    def test_from_mapping_preserves_model(self):
        current = LLMSettings(model="gpt-4o", temperature=0.7)
        delta = LLMSettings.from_mapping({"model": "gpt-4o-mini", "temperature": 0.9})
        changed = current.apply_update(delta)
        assert changed.keys() == {"model", "temperature"}
        assert changed["model"] == "gpt-4o"
        assert current.model == "gpt-4o-mini"
        assert current.temperature == 0.9


# ---------------------------------------------------------------------------
# DeepgramSTTSettings: live_options delta merge
# ---------------------------------------------------------------------------


class TestDeepgramSTTSettingsApplyUpdate:
    def _make_store(self, **lo_kwargs) -> DeepgramSTTSettings:
        """Helper to build a store-mode DeepgramSTTSettings."""
        defaults = dict(
            encoding="linear16",
            channels=1,
            interim_results=True,
            smart_format=False,
            punctuate=True,
            profanity_filter=True,
            vad_events=False,
        )
        defaults.update(lo_kwargs)
        s = DeepgramSTTSettings(
            model="nova-3-general",
            language="en",
            live_options=LiveOptions(**defaults),
        )
        return s

    def test_apply_update_merges_live_options_as_delta(self):
        """Only the given fields in the delta LiveOptions are merged."""
        current = self._make_store()
        assert current.live_options.punctuate is True

        delta = DeepgramSTTSettings(live_options=LiveOptions(punctuate=False))
        changed = current.apply_update(delta)

        assert current.live_options.punctuate is False
        assert "punctuate" in changed
        # Other fields are untouched
        assert current.live_options.encoding == "linear16"
        assert current.live_options.channels == 1

    def test_apply_update_syncs_model_from_live_options_to_top_level(self):
        """model inside live_options delta should sync to top-level model."""
        current = self._make_store()
        assert current.model == "nova-3-general"

        delta = DeepgramSTTSettings(live_options=LiveOptions(model="nova-2"))
        changed = current.apply_update(delta)

        assert current.model == "nova-2"
        assert "model" in changed

    def test_apply_update_syncs_language_from_live_options_to_top_level(self):
        """language inside live_options delta should sync to top-level language."""
        current = self._make_store()
        assert current.language == "en"

        delta = DeepgramSTTSettings(live_options=LiveOptions(language="es"))
        changed = current.apply_update(delta)

        assert current.language == "es"
        assert "language" in changed

    def test_apply_update_syncs_top_level_model_into_live_options(self):
        """Top-level model change should propagate into stored live_options."""
        current = self._make_store()
        assert current.model == "nova-3-general"

        delta = DeepgramSTTSettings(model="nova-2")
        changed = current.apply_update(delta)

        assert current.model == "nova-2"
        assert current.live_options.model == "nova-2"
        assert "model" in changed

    def test_apply_update_syncs_top_level_language_into_live_options(self):
        """Top-level language change should propagate into stored live_options."""
        current = self._make_store()

        delta = DeepgramSTTSettings(language="fr")
        changed = current.apply_update(delta)

        assert current.language == "fr"
        assert current.live_options.language == "fr"
        assert "language" in changed

    def test_apply_update_no_change(self):
        """Delta with same values should report no changes."""
        current = self._make_store()
        delta = DeepgramSTTSettings(live_options=LiveOptions(punctuate=True))
        changed = current.apply_update(delta)
        assert changed == {}

    def test_apply_update_top_level_model_takes_precedence_over_live_options(self):
        """When both top-level model and live_options.model are set, top-level wins."""
        current = self._make_store()
        assert current.model == "nova-3-general"

        delta = DeepgramSTTSettings(
            model="nova-2",
            live_options=LiveOptions(model="nova-3"),
        )
        changed = current.apply_update(delta)

        assert current.model == "nova-2"
        assert current.live_options.model == "nova-2"
        assert "model" in changed

    def test_apply_update_top_level_language_takes_precedence_over_live_options(self):
        """When both top-level language and live_options.language are set, top-level wins."""
        current = self._make_store()
        assert current.language == "en"

        delta = DeepgramSTTSettings(
            language="fr",
            live_options=LiveOptions(language="es"),
        )
        changed = current.apply_update(delta)

        assert current.language == "fr"
        assert current.live_options.language == "fr"
        assert "language" in changed


class TestDeepgramSTTSettingsFromMapping:
    def test_routes_live_options_kwargs(self):
        """LiveOptions-valid keys should be collected into live_options."""
        delta = DeepgramSTTSettings.from_mapping({"punctuate": False, "filler_words": True})
        assert is_given(delta.live_options)
        assert delta.live_options.punctuate is False
        assert delta.live_options.filler_words is True

    def test_routes_model_and_language_to_top_level(self):
        """model and language should be top-level fields, not in live_options."""
        delta = DeepgramSTTSettings.from_mapping({"model": "nova-2", "language": "es"})
        assert delta.model == "nova-2"
        assert delta.language == "es"
        assert not is_given(delta.live_options)

    def test_unknown_keys_go_to_extra(self):
        """Keys that aren't LiveOptions params or STT fields go to extra."""
        delta = DeepgramSTTSettings.from_mapping({"unknown_param": 42})
        assert delta.extra == {"unknown_param": 42}
        assert not is_given(delta.live_options)

    def test_mixed_keys(self):
        """model + LiveOptions keys + unknown keys are routed correctly."""
        delta = DeepgramSTTSettings.from_mapping(
            {"model": "nova-2", "punctuate": False, "unknown": "val"}
        )
        assert delta.model == "nova-2"
        assert delta.live_options.punctuate is False
        assert delta.extra == {"unknown": "val"}

    def test_roundtrip_from_mapping_apply_update(self):
        """Simulate dict-style update: from_mapping -> apply_update."""
        current = DeepgramSTTSettings(
            model="nova-3-general",
            language="en",
            live_options=LiveOptions(
                encoding="linear16",
                channels=1,
                interim_results=True,
                punctuate=True,
                profanity_filter=True,
                vad_events=False,
            ),
        )

        raw = {"punctuate": False, "filler_words": True}
        delta = DeepgramSTTSettings.from_mapping(raw)
        changed = current.apply_update(delta)

        assert current.live_options.punctuate is False
        assert current.live_options.filler_words is True
        # Unchanged fields stay put
        assert current.live_options.encoding == "linear16"
        assert current.model == "nova-3-general"
        assert "punctuate" in changed

    def test_roundtrip_model_via_dict(self):
        """Dict update with model should change top-level and NOT create live_options."""
        current = DeepgramSTTSettings(
            model="nova-3-general",
            language="en",
            live_options=LiveOptions(encoding="linear16", channels=1),
        )

        raw = {"model": "nova-2"}
        delta = DeepgramSTTSettings.from_mapping(raw)
        changed = current.apply_update(delta)

        assert current.model == "nova-2"
        assert current.live_options.model == "nova-2"
        assert "model" in changed


# ---------------------------------------------------------------------------
# DeepgramSageMakerSTTSettings: smoke test that the shared base is inherited
# ---------------------------------------------------------------------------


class TestDeepgramSageMakerSTTSettings:
    def test_inherits_live_options_behavior(self):
        """Smoke test: SageMaker settings inherit the shared base correctly."""
        store = DeepgramSageMakerSTTSettings(
            model="nova-3",
            language="en",
            live_options=LiveOptions(encoding="linear16", channels=1, punctuate=True),
        )
        delta = DeepgramSageMakerSTTSettings(live_options=LiveOptions(punctuate=False))
        changed = store.apply_update(delta)

        assert store.live_options.punctuate is False
        assert store.live_options.encoding == "linear16"
        assert "punctuate" in changed
