#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Inworld TTS in expressive conversation."""

import pytest

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.adapters.tts.services.inworld_adapter import (
    InworldExpressionAdapter,
    InworldFlashExpressionAdapter,
)
from pipecat.expressive.expression import ExpressionDelivery
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.pipeline.worker import PipelineParams
from pipecat.services.inworld.tts import InworldHttpTTSService, InworldTTSService
from pipecat.tests.utils import run_test


class _RejectingResponse:
    """A response that fails, so a request is recorded without audio."""

    status = 400

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        return "rejected"


class _RecordingSession:
    def __init__(self):
        self.payloads: list[dict] = []

    def post(self, url, json=None, headers=None):
        self.payloads.append(json)
        return _RejectingResponse()


class TestInworldExpressionAdapter:
    def test_performs_every_universal_delivery(self):
        capabilities = InworldExpressionAdapter().capabilities()
        assert capabilities.allows_free_text_delivery
        assert not capabilities.allows_free_text_sounds
        assert capabilities.emphasis
        assert capabilities.delivery_name("warm") == "warm"
        assert capabilities.sound_name("breath") == "breathe"
        # Delivery is free text, so a name that isn't universal is performed as written,
        # while sounds are a fixed list.
        assert capabilities.delivery_name("british accent") == "british accent"
        assert capabilities.sound_name("beatbox") is None

    def test_instructions_where_delivery_changes(self):
        adapter = InworldExpressionAdapter()
        assert adapter.format_text("Hi.", ["calm"], None) == "[calm] Hi."
        assert adapter.format_text("Hi.", ["calm"], ["calm"]) == "Hi."
        assert adapter.format_text("A B 7", ["calm", "slow"], ["calm"]) == "[calm and slow] A B 7"

    def test_a_return_to_the_baseline_resets(self):
        adapter = InworldExpressionAdapter()
        # An instruction lasts past its sentence, so one spoken in a delivery
        # ends by setting it back...
        assert adapter.format_text("Bye.", [], ["calm"]) == "[reset] Bye."
        assert adapter.format_end(["calm"]) == " [reset]"
        # ...and a sentence in the baseline has nothing to set back.
        assert adapter.format_text("Bye.", [], None) == "Bye."
        assert adapter.format_end([]) == ""
        assert adapter.format_end(["emphasis"]) == ""

    def test_emphasis_is_written_in_capitals(self):
        adapter = InworldExpressionAdapter()
        assert adapter.format_text("now", ["calm", "emphasis"], ["calm"]) == "NOW"

    def test_sounds_and_pauses(self):
        adapter = InworldExpressionAdapter()
        assert adapter.format_sound("laugh") == " [laugh] "
        assert adapter.format_pause(600) == ' <break time="600ms"/> '


class TestInworldFlashExpressionAdapter:
    def test_performs_sounds_and_pauses_without_delivery(self):
        capabilities = InworldFlashExpressionAdapter().capabilities()
        assert capabilities.delivery == []
        assert not capabilities.emphasis
        assert capabilities.sound_name("laugh") == "laugh"
        assert capabilities.pause_range_ms == (100, 10000)


@pytest.mark.parametrize("service_class", [InworldTTSService, InworldHttpTTSService])
@pytest.mark.parametrize(
    "model, adapter",
    [
        ("inworld-tts-2", InworldExpressionAdapter),
        ("inworld-tts-2-flash", InworldFlashExpressionAdapter),
        ("inworld-tts-1.5-max", BaseExpressionAdapter),
    ],
)
def test_expression_adapter_follows_the_model(service_class, model, adapter):
    kwargs = {} if service_class is InworldTTSService else {"aiohttp_session": object()}
    service = service_class(api_key="key", settings=service_class.Settings(model=model), **kwargs)
    assert type(service.get_expression_adapter()) is adapter


@pytest.mark.asyncio
async def test_each_request_writes_its_instructions():
    session = _RecordingSession()
    service = InworldHttpTTSService(
        api_key="key",
        aiohttp_session=session,
        settings=InworldHttpTTSService.Settings(model="inworld-tts-2"),
    )
    await run_test(
        service,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("<<sympathetic>> I'm sorry. [[sigh]] Your code is <<slow: A B 7>>."),
            LLMTextFrame(" <<neutral>> Anything ^^else^^?"),
            LLMFullResponseEndFrame(),
        ],
        pipeline_params=PipelineParams(enable_expressive_conversation=True),
    )

    assert [p["text"] for p in session.payloads] == [
        "[sympathetic] I'm sorry. [reset]",
        "[sigh] [sympathetic] Your code is [sympathetic and slow] A B 7. [reset]",
        "Anything ELSE?",
    ]


def test_a_pause_is_left_out_of_the_words_reported():
    service = InworldTTSService(api_key="key")
    alignment = {
        "words": ["there", ' <break time="500ms"/> ', "friend"],
        "wordStartTimeSeconds": [0.1, 0.4, 0.9],
        "wordEndTimeSeconds": [0.3, 0.9, 1.2],
    }

    word_times = service._calculate_word_times({"wordAlignment": alignment})

    assert [word for word, _ in word_times] == ["there", "  ", "friend"]
