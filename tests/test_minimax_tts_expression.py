#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for MiniMax TTS in expressive conversation."""

import pytest

from pipecat.expressive.adapter import ExpressionAdapter
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.pipeline.worker import PipelineParams
from pipecat.services.minimax.tts import MiniMaxExpressionAdapter, MiniMaxHttpTTSService
from pipecat.tests.utils import run_test


class _RejectingResponse:
    """A response that fails, so a request is recorded without audio."""

    status = 400

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _RecordingSession:
    def __init__(self):
        self.payloads: list[dict] = []

    def post(self, url, headers=None, json=None):
        self.payloads.append(json)
        return _RejectingResponse()


def _service(session, **settings) -> MiniMaxHttpTTSService:
    return MiniMaxHttpTTSService(
        api_key="key",
        group_id="group",
        aiohttp_session=session,
        settings=MiniMaxHttpTTSService.Settings(**settings),
    )


class TestMiniMaxExpressionAdapter:
    def test_events_and_pauses(self):
        adapter = MiniMaxExpressionAdapter()
        assert adapter.render_event("laughs") == "(laughs)"
        assert adapter.render_pause(600) == "<#0.6#>"
        assert adapter.render_text("Hi.", ["happy"], None) == "Hi."

    def test_vocabulary_is_fixed(self):
        # MiniMax reads an interjection outside its list aloud.
        capabilities = MiniMaxExpressionAdapter().capabilities()
        assert not capabilities.free_text_events and not capabilities.free_text_modes


@pytest.mark.parametrize(
    "model, adapter",
    [
        ("speech-2.8-turbo", MiniMaxExpressionAdapter),
        ("speech-2.8-hd", MiniMaxExpressionAdapter),
        ("speech-2.6-turbo", ExpressionAdapter),
    ],
)
def test_expression_adapter_follows_the_model(model, adapter):
    service = _service(_RecordingSession(), model=model)
    assert type(service.get_expression_adapter()) is adapter


@pytest.mark.asyncio
async def test_each_request_sends_the_emotion_of_its_modes():
    session = _RecordingSession()
    await run_test(
        _service(session, emotion="calm"),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("<<happy>> Great news. [[laughs]] It's back. <<neutral>> Anything else?"),
            LLMFullResponseEndFrame(),
        ],
        pipeline_params=PipelineParams(enable_expressive_conversation=True),
    )

    # The baseline is the configured emotion.
    assert [(p["text"], p["voice_setting"].get("emotion")) for p in session.payloads] == [
        ("Great news.", "happy"),
        ("(laughs)It's back.", "happy"),
        ("Anything else?", "calm"),
    ]
