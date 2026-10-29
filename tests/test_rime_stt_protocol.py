#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Real SDK, protobuf and gRPC behavior through Pipecat workers."""

import asyncio
import io
import wave

import grpc
import numpy as np
import pytest
from rime_api import speech_to_text_pb2 as proto
from rimelabs_sdk import Rime, RimeInputError

from pipecat.audio.utils import pcm_to_wav
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    MetricsFrame,
    TranscriptionFrame,
    UserAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.rime.stt import RimeSegmentedSTTService, RimeSTTService
from pipecat.utils.errors import ErrorCategory
from tests.rime_stt_helpers import audio, wait
from tests.rime_stt_peer import Failure, language, peer_server, pipeline


async def start_speech(worker, peer):
    await worker.queue_frames([VADUserStartedSpeakingFrame(), audio()])
    call = await peer.next_call()
    await wait(call.received_audio)
    return call


@pytest.mark.asyncio
@pytest.mark.parametrize("contract", ["revised_hypotheses", "append_fragments"])
@pytest.mark.parametrize("rate,channels", [(8000, 1), (16000, 2), (24000, 1), (48000, 2)])
async def test_streaming_config_pcm_and_transcript_contract(monkeypatch, contract, rate, channels):
    async with peer_server(monkeypatch) as peer, Rime(api_key="local-peer") as client:
        service = RimeSTTService(
            client=client,
            settings=RimeSTTService.Settings(
                language="es-MX",
                mode="verbatim",
                context_terms=["Pipecat", "Rime"],
                output_contract=contract,
            ),
        )
        async with pipeline(service) as (worker, out, _):
            samples = np.full((rate // 10, channels), 1000, dtype="<i2")
            if channels == 2:
                samples[:, 1] = 3000
            await worker.queue_frames(
                [
                    VADUserStartedSpeakingFrame(),
                    UserAudioRawFrame(samples.tobytes(), rate, channels, user_id="speaker"),
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
            assert final.result == {"language": "es-MX", "request_id": call.request_id}
            assert [f.text for f in out.frames if isinstance(f, InterimTranscriptionFrame)] == [
                "hola ",
                "hola mundo",
            ]
            pcm = np.frombuffer(b"".join(call.audio), dtype="<i2")
            assert len(pcm) == pytest.approx(1600, abs=1)
            assert pcm[100:-100].mean() == pytest.approx(2000 if channels == 2 else 1000, abs=1)


@pytest.mark.asyncio
async def test_unary_wav_and_options_reach_the_sdk_peer(monkeypatch):
    async with peer_server(monkeypatch) as peer, Rime(api_key="local-peer") as client:
        service = RimeSegmentedSTTService(
            client=client,
            trailing_silence_secs=0,
            settings=RimeSegmentedSTTService.Settings(
                language="es", mode="verbatim", context_terms=["Rime"]
            ),
        )
        async with pipeline(service) as (worker, out, _):
            samples = np.tile(np.array([1000, 3000], dtype="<i2"), (4800, 1)).tobytes()
            await worker.queue_frames(
                [
                    VADUserStartedSpeakingFrame(),
                    audio(samples[:3], 48000, 2),
                    audio(samples[3:], 48000, 2),
                ]
            )
            await out.wait_for(InputAudioRawFrame, 2)
            await worker.queue_frame(VADUserStoppedSpeakingFrame())
            call = await peer.next_call()
            assert call.request.options.language == "es"
            assert call.request.options.mode == proto.TRANSCRIPTION_MODE_VERBATIM
            assert list(call.request.options.context_terms) == ["Rime"]
            with wave.open(io.BytesIO(call.audio[0]), "rb") as wav:
                assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
                samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
            assert len(samples) == 1600
            assert samples[100:-100].mean() == pytest.approx(2000, abs=1)
            call.responses.put_nowait(
                proto.TranscriptionResponse(text="hola", language=language("es"))
            )
            final = (await out.wait_for(TranscriptionFrame))[0]
            assert final.text == "hola" and final.finalized
            assert final.result["request_id"] == call.request_id


@pytest.mark.asyncio
async def test_later_upload_finishes_before_earlier_reply_without_reordering(monkeypatch):
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
            await out.wait_for(MetricsFrame)
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
async def test_grpc_failure_preserves_category_and_request_id(monkeypatch, code, category, usable):
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
async def test_sdk_rejects_invalid_completion_without_committing_partial(monkeypatch, malformed):
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
@pytest.mark.parametrize("service_cls", [RimeSTTService, RimeSegmentedSTTService])
@pytest.mark.parametrize("stop_frame", [CancelFrame, EndFrame])
async def test_cancel_and_shutdown_deadline_preserve_injected_client(
    monkeypatch, service_cls, stop_frame
):
    async with peer_server(monkeypatch) as peer, Rime(api_key="local-peer") as client:
        service = service_cls(client=client, shutdown_timeout=0.05)
        async with pipeline(service) as (worker, out, _):
            await worker.queue_frames([VADUserStartedSpeakingFrame(), audio()])
            await out.wait_for(InputAudioRawFrame)
            await worker.queue_frame(VADUserStoppedSpeakingFrame())
            call = await peer.next_call()
            await wait(call.eof)
            await worker.queue_frame(stop_frame())
            await wait(call.closed)
            await out.wait_for(stop_frame)
            assert not out.transcripts
        await service.cleanup()
        async with asyncio.TaskGroup() as tasks:
            request = tasks.create_task(
                client.stt.transcribe(pcm_to_wav(bytes(320), 16000), language="en")
            )
            fresh = await peer.next_call()
            fresh.responses.put_nowait(
                proto.TranscriptionResponse(text="still open", language=language("en"))
            )
        assert request.result().text == "still open"


@pytest.mark.asyncio
@pytest.mark.parametrize("service_cls", [RimeSTTService, RimeSegmentedSTTService])
async def test_worker_cleanup_closes_owned_sdk_client(monkeypatch, service_cls):
    async with peer_server(monkeypatch):
        service = service_cls(api_key="local-peer")
        async with pipeline(service):
            client = service._recognition.client
        await service.cleanup()
        with pytest.raises(RimeInputError, match="closed"):
            await client.stt.transcribe(pcm_to_wav(bytes(320), 16000), language="en")
