#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Session handling of Deepgram Flux STT on SageMaker against a fake BiDi client."""

import asyncio
import json
from unittest.mock import patch

import pytest

pytest.importorskip("aws_sdk_sagemaker_runtime_http2")

from aws_sdk_sagemaker_runtime_http2.models import (  # noqa: E402
    ResponsePayloadPart,
    ResponseStreamEventPayloadPart,
)

from pipecat.frames.frames import (  # noqa: E402
    InputAudioRawFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
)
from pipecat.services.deepgram.flux.sagemaker.stt import (  # noqa: E402
    DeepgramFluxSageMakerSTTService,
)
from pipecat.tests.utils import SleepFrame, run_test  # noqa: E402

FRAME_SECONDS = 0.02


class FakeBidiClient:
    """A SageMaker BiDi client backed by a fake Flux STT endpoint.

    Confirms the session with ``Connected`` and, after ``open_turn_after`` audio
    chunks, opens a turn with StartOfTurn and an Update. With ``end_mid_turn``
    the session then ends on its own; otherwise it ends when the service sends
    CloseStream.
    """

    open_turn_after: int | None = None
    end_mid_turn = False

    def __init__(self, **kwargs):
        self.is_active = False
        self.audio_chunks = 0
        self._responses: asyncio.Queue = asyncio.Queue()

    async def start_session(self):
        self.is_active = True
        await self._send({"type": "Connected", "request_id": "test"})

    async def close_session(self):
        self.is_active = False
        self._end()

    async def send_audio_chunk(self, audio: bytes):
        self.audio_chunks += 1
        if self.audio_chunks == self.open_turn_after:
            for event, transcript in (("StartOfTurn", "hello"), ("Update", "hello there")):
                await self._send({"type": "TurnInfo", "event": event, "transcript": transcript})
            if self.end_mid_turn:
                self._end()

    async def send_json(self, data: dict):
        if data.get("type") == "CloseStream":
            self._end()

    async def receive_response(self):
        return await self._responses.get()

    async def _send(self, message: dict):
        payload = json.dumps(message).encode("utf-8")
        await self._responses.put(
            ResponseStreamEventPayloadPart(value=ResponsePayloadPart(bytes_=payload))
        )

    def _end(self):
        # The endpoint closing its stream; the client stays active until
        # close_session(), as SageMakerBidiClient does.
        self._responses.put_nowait(None)


@pytest.fixture
def fake_client():
    FakeBidiClient.open_turn_after = None
    FakeBidiClient.end_mid_turn = False
    with patch(
        "pipecat.services.deepgram.flux.sagemaker.stt.SageMakerBidiClient", new=FakeBidiClient
    ):
        yield FakeBidiClient


async def _run(seconds: float) -> list:
    service = DeepgramFluxSageMakerSTTService(endpoint_name="test-endpoint", region="us-east-2")
    frames = []
    for _ in range(int(seconds / FRAME_SECONDS)):
        frames.append(
            InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
        )
        frames.append(SleepFrame(sleep=FRAME_SECONDS))
    down, _ = await run_test(service, frames_to_send=frames)
    return down


def _turn_ends(down) -> tuple[list[tuple[str, bool]], list[int]]:
    """The transcriptions pushed and the positions of proposed turn stops."""
    transcripts = [(f.text, f.finalized) for f in down if isinstance(f, TranscriptionFrame)]
    stops = [i for i, f in enumerate(down) if isinstance(f, ProposedUserStoppedSpeakingFrame)]
    return transcripts, stops


@pytest.mark.asyncio
async def test_a_turn_cut_short_by_the_session_ending_ends_with_its_transcript_so_far(
    fake_client,
):
    fake_client.open_turn_after = 5
    fake_client.end_mid_turn = True

    down = await _run(seconds=0.4)

    transcripts, stops = _turn_ends(down)
    assert transcripts == [("hello there", False)]
    assert len(stops) == 1
    first_transcript = next(i for i, f in enumerate(down) if isinstance(f, TranscriptionFrame))
    assert first_transcript < stops[0]


@pytest.mark.asyncio
async def test_stopping_mid_turn_does_not_end_the_turn(fake_client):
    fake_client.open_turn_after = 5

    down = await _run(seconds=0.4)

    transcripts, stops = _turn_ends(down)
    assert transcripts == []
    assert stops == []
