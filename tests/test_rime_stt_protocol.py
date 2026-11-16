#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Real SDK, protobuf and gRPC behavior through Pipecat workers."""

import asyncio

import grpc
import numpy as np
import pytest

pytest.importorskip("rimelabs_sdk")

from rime_api import speech_to_text_pb2 as proto
from rimelabs_sdk import RimeInputError

from pipecat.frames.frames import (
    ErrorFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.rime.stt import RimeSTTService
from pipecat.utils.errors import ErrorCategory
from tests.rime_stt_helpers import audio, wait
from tests.rime_stt_peer import Failure, peer_server, pipeline


async def pcm_audio():
    yield bytes(320)


async def start_speech(worker, peer):
    await worker.queue_frames([VADUserStartedSpeakingFrame(), audio()])
    call = await peer.next_call()
    await wait(call.received_audio)
    return call


@pytest.mark.asyncio
@pytest.mark.parametrize("contract", ["revised_hypotheses", "append_fragments"])
@pytest.mark.parametrize("rate", [8000, 16000, 24000, 48000])
async def test_streaming_config_pcm_and_transcript_contract(monkeypatch, contract, rate):
    async with peer_server(monkeypatch) as peer:
        service = RimeSTTService(
            api_key="local-peer",
            sample_rate=rate,
            settings=RimeSTTService.Settings(
                language="es-MX",
                mode="verbatim",
                context_terms=["Pipecat", "Rime"],
                output_contract=contract,
            ),
        )
        async with pipeline(service) as (worker, out, _):
            samples = np.full(rate // 10, 1000, dtype="<i2")
            await worker.queue_frames(
                [
                    VADUserStartedSpeakingFrame(),
                    UserAudioRawFrame(samples.tobytes(), rate, 1, user_id="speaker"),
                ]
            )
            call = await peer.next_call()
            await wait(call.received_audio)
            config = call.request
            assert config.language == "es-MX"
            assert config.mode == proto.TRANSCRIPTION_MODE_VERBATIM
            assert list(config.context_terms) == ["Pipecat", "Rime"]
            assert config.output_contract == getattr(
                proto, "STREAMING_OUTPUT_CONTRACT_" + contract.upper()
            )
            assert not call.eof.is_set()
            append = contract == "append_fragments"
            call.update("hola ", append=append)
            await out.wait_for(InterimTranscriptionFrame)
            call.update("mundo" if append else "hola mundo", append=append)
            await out.wait_for(InterimTranscriptionFrame, 2)
            await worker.queue_frame(VADUserStoppedSpeakingFrame())
            await wait(call.eof)
            call.finish()
            final = (await out.wait_for(TranscriptionFrame))[0]
            assert final.text == "hola mundo" and final.finalized
            assert final.user_id == "speaker"
            assert [f.text for f in out.frames if isinstance(f, InterimTranscriptionFrame)] == [
                "hola ",
                "hola mundo",
            ]
            pcm = np.frombuffer(b"".join(call.audio), dtype="<i2")
            assert len(pcm) == pytest.approx(1600, abs=1)
            assert pcm[100:-100].mean() == pytest.approx(1000, abs=1)


@pytest.mark.asyncio
async def test_later_reply_waits_for_the_earlier_final(monkeypatch):
    async with peer_server(monkeypatch) as peer:
        service = RimeSTTService(api_key="local-peer")
        async with pipeline(service) as (worker, out, _):
            calls = []
            for _ in range(2):
                call = await start_speech(worker, peer)
                await worker.queue_frame(VADUserStoppedSpeakingFrame())
                await wait(call.eof)
                calls.append(call)
            first, second = calls
            second.update("second")
            second.finish()
            await wait(second.closed)
            await asyncio.sleep(0.05)
            assert not out.transcripts
            first.update("first")
            first.finish()
            finals = await out.wait_for(TranscriptionFrame, 2)
            assert [frame.text for frame in finals] == ["first", "second"]
            assert [frame.finalized for frame in finals] == [False, True]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,category,usable",
    [
        (grpc.StatusCode.UNAUTHENTICATED, ErrorCategory.AUTHENTICATION, False),
        (grpc.StatusCode.PERMISSION_DENIED, ErrorCategory.AUTHORIZATION, False),
        (grpc.StatusCode.INVALID_ARGUMENT, ErrorCategory.INVALID_REQUEST, False),
        (grpc.StatusCode.RESOURCE_EXHAUSTED, ErrorCategory.UNKNOWN, True),
        (grpc.StatusCode.UNAVAILABLE, ErrorCategory.CONNECTIVITY, True),
        (grpc.StatusCode.DEADLINE_EXCEEDED, ErrorCategory.CONNECTIVITY, True),
        (grpc.StatusCode.INTERNAL, ErrorCategory.UNKNOWN, True),
    ],
)
async def test_grpc_failure_category_and_request_id(monkeypatch, code, category, usable):
    async with peer_server(monkeypatch) as peer:
        service = RimeSTTService(api_key="local-peer")
        async with pipeline(service) as (worker, out, upstream):
            call = await start_speech(worker, peer)
            call.update("unfinished")
            await out.wait_for(InterimTranscriptionFrame)
            call.responses.put_nowait(Failure(code, "Private recognition hints"))
            error = (await upstream.wait_for(ErrorFrame))[0]
            assert error.category is category and call.request_id in error.error
            assert "Private recognition hints" not in error.error
            assert service.is_usable is usable
            assert not out.transcripts
            await wait(call.closed)
            assert len(peer.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", ["missing_final", "mismatched_final"])
async def test_invalid_completion_reports_an_error_without_a_final(monkeypatch, malformed):
    async with peer_server(monkeypatch) as peer:
        service = RimeSTTService(api_key="local-peer")
        async with pipeline(service) as (worker, out, upstream):
            call = await start_speech(worker, peer)
            call.update("partial")
            await out.wait_for(InterimTranscriptionFrame)
            await worker.queue_frame(VADUserStoppedSpeakingFrame())
            await wait(call.eof)
            if malformed == "missing_final":
                call.responses.put_nowait(None)
            else:
                call.finish(text="different")
            error = (await upstream.wait_for(ErrorFrame))[0]
            assert "RimeStreamError" in error.error
            assert not out.transcripts


@pytest.mark.asyncio
async def test_cleanup_closes_the_sdk_client(monkeypatch):
    async with peer_server(monkeypatch):
        service = RimeSTTService(api_key="local-peer")
        async with pipeline(service):
            client = service._client
        await service.cleanup()
        assert client is not None
        with pytest.raises(RimeInputError, match="closed"):
            client.stt.stream(pcm_audio(), language="en")
