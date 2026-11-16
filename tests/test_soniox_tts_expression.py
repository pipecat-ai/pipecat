#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Soniox TTS in expressive conversation."""

import pytest

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.adapters.tts.services.soniox_adapter import SonioxExpressionAdapter
from pipecat.services.soniox.tts import SonioxTTSService


class TestSonioxExpressionAdapter:
    def test_capabilities(self):
        capabilities = SonioxExpressionAdapter().capabilities()
        assert not capabilities.allows_free_text_sounds
        assert not capabilities.allows_free_text_delivery
        assert capabilities.emphasis
        assert capabilities.delivery_name("sympathetic") == "warm"
        assert capabilities.delivery_name("soft") == "softly"
        assert capabilities.sound_name("laugh") == "laughs"

    def test_tags_where_delivery_changes(self):
        adapter = SonioxExpressionAdapter()
        assert adapter.format_text("Hi.", ["excited"], None) == "[excited] Hi."
        assert adapter.format_text("Hi.", ["excited"], ["excited"]) == "Hi."
        assert adapter.format_text("A B 7", ["calm", "slow"], ["calm"]) == "[calm] [slowly] A B 7"
        assert adapter.format_text("Hi.", ["fast"], None) == "[quickly] Hi."

    def test_a_return_to_the_baseline_writes_neutral(self):
        adapter = SonioxExpressionAdapter()
        # A tag lasts past its sentence, so one spoken in a delivery ends by
        # setting it back...
        assert adapter.format_text("Bye.", [], ["calm"]) == "[neutral] Bye."
        assert adapter.format_end(["calm"]) == " [neutral]"
        # ...and a sentence in the baseline has nothing to set back.
        assert adapter.format_text("Bye.", [], None) == "Bye."
        assert adapter.format_end([]) == ""

    def test_emphasis(self):
        adapter = SonioxExpressionAdapter()
        assert adapter.format_text("now", ["calm", "emphasis"], ["calm"]) == "*now*"

    def test_sounds_and_pauses(self):
        adapter = SonioxExpressionAdapter()
        assert adapter.format_sound("clears throat") == " [clears throat] "
        assert adapter.format_pause(600) == " [pause] "
        assert adapter.format_pause(1200) == " [long pause] "


@pytest.mark.parametrize(
    "model, adapter",
    [("tts-rt-v2", SonioxExpressionAdapter), ("tts-rt-v1", BaseExpressionAdapter)],
)
def test_expression_adapter_follows_the_model(model, adapter):
    service = SonioxTTSService(api_key="key", settings=SonioxTTSService.Settings(model=model))
    assert type(service.get_expression_adapter()) is adapter
