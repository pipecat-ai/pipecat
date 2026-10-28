#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Azure TTS in expressive conversation."""

import pytest

pytest.importorskip("azure.cognitiveservices.speech")

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter  # noqa: E402
from pipecat.adapters.tts.services.azure_adapter import AzureHDExpressionAdapter  # noqa: E402
from pipecat.services.azure.tts import AzureHttpTTSService, AzureTTSService  # noqa: E402


class TestAzureHDExpressionAdapter:
    def test_capabilities(self):
        capabilities = AzureHDExpressionAdapter().capabilities()
        assert not capabilities.allows_free_text_sounds
        assert not capabilities.allows_free_text_delivery
        assert not capabilities.emphasis
        assert capabilities.pause_range_ms is None
        assert capabilities.delivery_name("apologetic") == "remorseful"
        assert capabilities.sound_name("laugh") == "laughter"

    def test_a_marker_where_the_style_changes(self):
        adapter = AzureHDExpressionAdapter()
        assert adapter.format_text("Hi.", ["calm"], None) == "[calm] Hi."
        assert adapter.format_text("Hi.", ["calm"], ["calm"]) == "Hi."

    def test_one_style_at_a_time_the_one_added_last(self):
        adapter = AzureHDExpressionAdapter()
        assert adapter.format_text("A B 7", ["calm", "slow"], ["calm"]) == "[slow] A B 7"
        assert adapter.format_text(", okay.", ["calm"], ["calm", "slow"]) == "[calm] , okay."

    def test_a_return_to_the_baseline_writes_neutral(self):
        assert AzureHDExpressionAdapter().format_text("Bye.", [], ["calm"]) == "[Neutral] Bye."

    def test_sounds(self):
        assert AzureHDExpressionAdapter().format_sound("sighing") == " [sighing] "


@pytest.mark.parametrize("service_class", [AzureTTSService, AzureHttpTTSService])
@pytest.mark.parametrize(
    "voice, adapter",
    [
        ("en-US-Ava:DragonHDLatestNeural", AzureHDExpressionAdapter),
        ("en-US-Ava:DragonHDOmniLatestNeural", AzureHDExpressionAdapter),
        ("en-US-Tiana:DragonHDFlashLatestNeural", BaseExpressionAdapter),
        ("en-US-SaraNeural", BaseExpressionAdapter),
    ],
)
def test_expression_adapter_follows_the_voice(service_class, voice, adapter):
    service = service_class(
        api_key="key", region="eastus", settings=service_class.Settings(voice=voice)
    )
    assert type(service.get_expression_adapter()) is adapter


def test_markers_are_split_out_of_word_boundaries():
    service = AzureTTSService(
        api_key="key",
        region="eastus",
        settings=AzureTTSService.Settings(voice="en-US-Ava:DragonHDLatestNeural"),
    )
    # Azure reports markers across boundaries, joined to the words around them.
    boundaries = ["[", "sighing] [", "sympathetic] It's", "is", "[", "slow] A"]

    words = [word for boundary in boundaries for word in service._split_style_markers(boundary)]

    assert words == ["[sighing]", "[sympathetic]", "It's", "is", "[slow]", "A"]
