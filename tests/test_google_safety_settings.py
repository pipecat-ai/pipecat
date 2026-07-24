#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for safety settings support in GoogleLLMService."""

from google.genai.types import (
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
)

from pipecat.services.google.llm import GoogleLLMService, GoogleLLMSettings
from pipecat.services.settings import is_given


def _safety_setting(
    category: HarmCategory = HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    threshold: HarmBlockThreshold = HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
) -> SafetySetting:
    return SafetySetting(category=category, threshold=threshold)


def test_safety_settings_omitted_by_default():
    """Generation params carry no safety_settings key unless configured."""
    service = GoogleLLMService(api_key="test-key")

    params = service._build_generation_params()

    assert "safety_settings" not in params


def test_safety_settings_passed_to_generation_params():
    """Configured safety settings reach the generation params verbatim."""
    safety = [_safety_setting()]
    service = GoogleLLMService(
        api_key="test-key",
        settings=GoogleLLMService.Settings(safety_settings=safety),
    )

    params = service._build_generation_params()

    assert params["safety_settings"] == safety


def test_safety_settings_survive_config_construction():
    """The generation params build a GenerateContentConfig the SDK accepts."""
    service = GoogleLLMService(
        api_key="test-key",
        settings=GoogleLLMService.Settings(safety_settings=[_safety_setting()]),
    )

    config = GenerateContentConfig(**service._build_generation_params())

    assert config.safety_settings is not None
    assert config.safety_settings[0].category == HarmCategory.HARM_CATEGORY_HATE_SPEECH
    assert config.safety_settings[0].threshold == HarmBlockThreshold.BLOCK_LOW_AND_ABOVE


def test_safety_settings_updated_at_runtime():
    """A settings delta replaces the safety settings on the live service."""
    service = GoogleLLMService(
        api_key="test-key",
        settings=GoogleLLMService.Settings(safety_settings=[_safety_setting()]),
    )
    updated = [
        _safety_setting(
            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        )
    ]

    service._settings.apply_update(GoogleLLMService.Settings(safety_settings=updated))

    assert service._build_generation_params()["safety_settings"] == updated


def test_from_mapping_coerces_dict_entries():
    """Plain dicts from a dict-based settings update become SafetySetting objects."""
    settings = GoogleLLMSettings.from_mapping(
        {
            "safety_settings": [
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_LOW_AND_ABOVE",
                }
            ]
        }
    )

    assert settings.safety_settings == [_safety_setting()]


def test_from_mapping_leaves_safety_setting_objects_untouched():
    """Already-typed entries pass through from_mapping unchanged."""
    safety = [_safety_setting()]

    settings = GoogleLLMSettings.from_mapping({"safety_settings": safety})

    assert settings.safety_settings == safety


def test_from_mapping_without_safety_settings_is_not_given():
    """Omitting safety_settings leaves the delta field unset."""
    settings = GoogleLLMSettings.from_mapping({"temperature": 0.5})

    assert not is_given(settings.safety_settings)
