#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The eval harness against a real accepted-WebSocket transport and bot pipeline.

Speech providers and the agent's responses are deterministic. The server uses
the actual eval transport, RTVI processor/observer and pipeline lifecycle;
neither end manufactures the other's wire messages.
"""

import asyncio
import json
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import websockets
from starlette.websockets import WebSocket

from pipecat.evals.fastapi_transport import EvalFastAPIWebsocketTransport
from pipecat.evals.scenario import EvalExpectation, EvalScriptScenario, EvalScriptTurn
from pipecat.evals.serializer import EvalSerializer, parse_eval_connect
from pipecat.evals.session import EvalSession, EvalSessionParams
from pipecat.evals.simulation import EvalSimulationMetric, EvalSimulationScenario
from pipecat.evals.transport import EvalTransportParams
from pipecat.evals.tts import CachingTTSService
from pipecat.frames.frames import (
    CancelFrame,
    ErrorFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMConfigureOutputFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.workers.runner import WorkerRunner
from tests.test_evals_session import _ScriptedPersonaLLM, _YesJudge

SAMPLE_RATE = 16000
TEXT_GREETING = "The agent intended to greet the caller."
TEXT_ANSWER = "The agent intended to answer the question."
SPOKEN_GREETING = "Hello, what is your question?"
SPOKEN_ANSWER = "The spoken answer is Berlin."
CALLER_TEXT = "What is Germany's capital?"


def _speech_pcm():
    with wave.open(str(Path(__file__).parent / "assets/continuous_speech_16k.wav"), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (SAMPLE_RATE, 1, 2)
        return wav.readframes(SAMPLE_RATE * 3 // 2)


class _CallerTTS(TTSService):
    def __init__(self):
        super().__init__(sample_rate=SAMPLE_RATE)
        self.calls = []

    async def run_tts(self, text, context_id):
        self.calls.append(text)
        yield TTSStartedFrame()
        yield TTSAudioRawFrame(audio=_speech_pcm(), sample_rate=SAMPLE_RATE, num_channels=1)
        yield TTSStoppedFrame()


class _StreamingRecognizer(STTService):
    """Emit a final transcript asynchronously after a complete incoming segment."""

    def __init__(self, replies, final_delay=0.05):
        super().__init__(settings=STTSettings(model="deterministic-stream"))
        self.replies = iter(replies)
        self.final_delay = final_delay
        self.segments = 0
        self.audio_bytes = 0
        self.active = False
        self.finals = []

    async def _final(self, text):
        await asyncio.sleep(self.final_delay)
        self.finals.append(text)
        await self.push_frame(
            TranscriptionFrame(text=text, user_id="", timestamp="2026-11-10T00:00:00Z")
        )

    async def run_stt(self, audio):
        if any(audio):
            self.audio_bytes += len(audio)
            self.active = True
        elif self.active:
            self.active = False
            self.segments += 1
            self.create_task(self._final(next(self.replies)), name="test-final-transcript")
        yield None


class _Agent(FrameProcessor):
    """A deterministic agent that handles real text/audio input frames."""

    def __init__(self):
        super().__init__()
        self.skip_tts = False
        self.audio_bytes = 0
        self.audio_received = asyncio.Event()
        self.speaking = False
        self.replies = 0
        self.cancelled = asyncio.Event()

    async def reply(self, text):
        self.replies += 1
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(text=text))
        await self.push_frame(LLMFullResponseEndFrame())
        if not self.skip_tts:
            await self.push_frame(TTSStartedFrame())
            await self.push_frame(
                TTSAudioRawFrame(audio=_speech_pcm(), sample_rate=SAMPLE_RATE, num_channels=1)
            )
            await self.push_frame(TTSStoppedFrame())

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, LLMConfigureOutputFrame):
            self.skip_tts = frame.skip_tts
        elif isinstance(frame, LLMMessagesAppendFrame) and frame.run_llm:
            self.create_task(self.reply(TEXT_ANSWER), name="test-agent-text-reply")
        elif isinstance(frame, InputAudioRawFrame):
            if any(frame.audio):
                self.audio_bytes += len(frame.audio)
                self.audio_received.set()
                if not self.speaking:
                    self.speaking = True
                    await self.push_frame(UserStartedSpeakingFrame())
            elif self.speaking:
                self.speaking = False
                await self.push_frame(UserStoppedSpeakingFrame())
                self.create_task(self.reply(TEXT_ANSWER), name="test-agent-audio-reply")
        elif isinstance(frame, CancelFrame):
            self.cancelled.set()
        await self.push_frame(frame, direction)


class _AcceptedWebSocketBot:
    def __init__(self, *, greeting, audio_input=True):
        self.greeting = greeting
        self.audio_input = audio_input
        self.agent = None
        self.flags = None
        self.closed = asyncio.Event()
        self.connected = asyncio.Event()
        self.messages = []

    async def handle(self, connection):
        first = True

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "websocket.connect"}
            try:
                payload = await connection.recv()
            except websockets.ConnectionClosed:
                return {"type": "websocket.disconnect", "code": connection.close_code or 1000}
            if isinstance(payload, str):
                self.messages.append(json.loads(payload))
                return {"type": "websocket.receive", "text": payload}
            return {"type": "websocket.receive", "bytes": payload}

        async def send(message):
            if message["type"] == "websocket.send":
                await connection.send(message.get("text", message.get("bytes")))
            elif message["type"] == "websocket.close":
                await connection.close(code=message.get("code", 1000))

        socket = WebSocket(
            {"type": "websocket", "path": "/", "query_string": b"", "headers": []},
            receive,
            send,
        )
        await socket.accept()
        self.flags = parse_eval_connect(await socket.receive_json())
        assert self.flags is not None
        transport = EvalFastAPIWebsocketTransport(
            socket,
            EvalTransportParams(
                serializer=EvalSerializer(),
                audio_in_enabled=self.audio_input,
                audio_out_enabled=True,
                audio_in_sample_rate=SAMPLE_RATE,
                audio_out_sample_rate=SAMPLE_RATE,
            ),
            self.flags,
        )
        self.agent = _Agent()
        self.connected.set()
        worker = PipelineWorker(
            Pipeline([transport.input(), self.agent, transport.output()]),
            params=PipelineParams(
                audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE
            ),
        )
        runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)

        @worker.rtvi.event_handler("on_client_ready")
        async def ready(_rtvi):
            if self.greeting:
                await self.agent.reply(TEXT_GREETING)

        @transport.event_handler("on_client_disconnected")
        async def disconnected(_transport, _websocket):
            await worker.cancel()

        try:
            await runner.add_workers(worker)
            await runner.run()
        finally:
            self.closed.set()


@asynccontextmanager
async def _bot(*, greeting=False, audio_input=True):
    bot = _AcceptedWebSocketBot(greeting=greeting, audio_input=audio_input)
    async with websockets.serve(bot.handle, "127.0.0.1", 0) as server:
        yield bot, f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"


def _session(kind, user_audio, judge_audio, url, tmp_path, final_delay=0.05):
    voice = _CallerTTS()
    speech = CachingTTSService(voice, cache_key="contract", cache_dir=tmp_path, use_cache=False)
    transcripts = [SPOKEN_GREETING, SPOKEN_ANSWER] if kind == "simulation" else [SPOKEN_ANSWER]
    recognizer = _StreamingRecognizer(transcripts, final_delay=final_delay)
    params = EvalSessionParams(
        default_timeout_ms=7000, bot_ready_timeout_s=5, require_audio_support=True
    )
    if kind == "script":
        scenario = EvalScriptScenario(
            name="audio-contract",
            user_audio=user_audio,
            user_speech={"sample_rate": SAMPLE_RATE} if user_audio else None,
            bot_audio=judge_audio,
            turns=[
                EvalScriptTurn(
                    user=CALLER_TEXT,
                    expect=[
                        EvalExpectation(
                            event="response" if judge_audio else "llm_response",
                            text_contains=SPOKEN_ANSWER if judge_audio else TEXT_ANSWER,
                        )
                    ],
                )
            ],
        )
        session = EvalSession.from_scenario(
            scenario,
            url,
            params=params,
            user_tts=speech if user_audio else None,
            bot_stt=recognizer if judge_audio else None,
        )
    else:
        greeting = SPOKEN_GREETING if judge_audio else TEXT_GREETING
        answer = SPOKEN_ANSWER if judge_audio else TEXT_ANSWER
        persona = _ScriptedPersonaLLM(
            {greeting: CALLER_TEXT, answer: ("end_call", {"succeeded": True, "reason": "answered"})}
        )
        scenario = EvalSimulationScenario(
            name="audio-contract",
            persona="An interested caller",
            goal="Get an answer",
            success="The agent answered",
            user_audio=user_audio,
            user_speech={"sample_rate": SAMPLE_RATE} if user_audio else None,
            bot_audio=judge_audio,
            max_duration_s=15,
            max_silence_s=8,
            metrics=[EvalSimulationMetric("latency", measure="latency", max_value=10)],
        )
        session = EvalSession.from_scenario(
            scenario,
            url,
            params=params,
            persona_llm=persona,
            judge=_YesJudge(),
            user_tts=speech if user_audio else None,
            bot_stt=recognizer if judge_audio else None,
        )
    return session, voice, recognizer


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["script", "simulation"])
@pytest.mark.parametrize("user_audio", [False, True])
@pytest.mark.parametrize("judge_audio", [False, True])
async def test_real_eval_transport_modality_contract(kind, user_audio, judge_audio, tmp_path):
    async with _bot(greeting=kind == "simulation") as (bot, url):
        session, voice, recognizer = _session(kind, user_audio, judge_audio, url, tmp_path)
        result = await asyncio.wait_for(session.run(), timeout=20)
        assert result.passed, result
        assert bot.flags.skip_tts is (not judge_audio)
        assert bot.flags.capture_bot_audio is judge_audio
        assert bool(voice.calls) is user_audio
        assert bool(bot.agent.audio_bytes) is user_audio
        assert bool(recognizer.finals) is judge_audio
        if kind == "simulation":
            assert result.metrics[0].value is not None
            assert result.metrics[0].passed
            expected = (
                [SPOKEN_GREETING, SPOKEN_ANSWER] if judge_audio else [TEXT_GREETING, TEXT_ANSWER]
            )
            assert [
                m["content"] for m in result.conversation if m["role"] == "assistant"
            ] == expected
        await asyncio.wait_for(bot.closed.wait(), timeout=3)
        assert any(m.get("data", {}).get("t") == "eval-cancel" for m in bot.messages)


@pytest.mark.asyncio
async def test_late_streaming_final_is_used_for_last_scripted_reply(tmp_path):
    async with _bot() as (bot, url):
        session, _voice, recognizer = _session(
            "script", False, True, url, tmp_path, final_delay=0.8
        )
        result = await asyncio.wait_for(session.run(), timeout=15)
        assert result.passed, result
        assert recognizer.finals == [SPOKEN_ANSWER]
        assert result.turns[0].expectations[0].matched == SPOKEN_ANSWER
        await asyncio.wait_for(bot.closed.wait(), timeout=3)


@pytest.mark.asyncio
async def test_cancel_during_caller_speech_closes_actual_target_pipeline(tmp_path):
    async with _bot() as (bot, url):
        session, voice, _recognizer = _session("script", True, True, url, tmp_path)
        task = asyncio.create_task(session.run())
        try:
            await asyncio.wait_for(bot.connected.wait(), timeout=5)
            await asyncio.wait_for(bot.agent.audio_received.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
            await asyncio.wait_for(bot.closed.wait(), timeout=3)
            assert voice.calls == [CALLER_TEXT]
            assert any(m.get("data", {}).get("t") == "eval-cancel" for m in bot.messages)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("service", ["tts", "stt"])
async def test_speech_pipeline_failure_is_unscored_and_closes_the_target(service, tmp_path):
    async with _bot() as (bot, url):
        session, voice, recognizer = _session(
            "script", service == "tts", service == "stt", url, tmp_path
        )
        if service == "tts":

            async def failed_tts(text, context_id):
                yield ErrorFrame(error="provider-diagnostic-canary")

            voice.run_tts = failed_tts
        else:

            async def failed_stt(audio):
                if any(audio):
                    yield ErrorFrame(error="provider-diagnostic-canary")
                else:
                    yield None

            recognizer.run_stt = failed_stt
        result = await asyncio.wait_for(session.run(), timeout=5)
        assert not result.passed
        assert [failure.kind for failure in result.failures] == ["error"]
        assert all("provider-diagnostic-canary" not in f.reason for f in result.failures)
        await asyncio.wait_for(bot.closed.wait(), timeout=3)
        assert any(m.get("data", {}).get("t") == "eval-cancel" for m in bot.messages)


class _InterimRecognizer(_StreamingRecognizer):
    async def _final(self, text):
        await self.push_frame(
            InterimTranscriptionFrame(
                text="An incorrect interim hypothesis", user_id="", timestamp="2026-11-10T00:00:00Z"
            )
        )
        await super()._final(text)


def _script_with_recognizer(url, turns, recognizer):
    return EvalSession.from_scenario(
        EvalScriptScenario(name="streaming-boundaries", bot_audio=True, turns=turns),
        url,
        params=EvalSessionParams(
            default_timeout_ms=7000, bot_ready_timeout_s=5, require_audio_support=True
        ),
        bot_stt=recognizer,
    )


@pytest.mark.asyncio
async def test_interim_recognition_does_not_become_judged_response():
    async with _bot() as (bot, url):
        recognizer = _InterimRecognizer([SPOKEN_ANSWER], final_delay=0.8)
        session = _script_with_recognizer(
            url,
            [
                EvalScriptTurn(
                    user=CALLER_TEXT,
                    expect=[EvalExpectation(event="response", text_contains=SPOKEN_ANSWER)],
                )
            ],
            recognizer,
        )
        result = await asyncio.wait_for(session.run(), timeout=15)
        assert result.passed, result
        assert result.turns[0].expectations[0].matched == SPOKEN_ANSWER
        assert recognizer.finals == [SPOKEN_ANSWER]
        await asyncio.wait_for(bot.closed.wait(), timeout=3)


@pytest.mark.asyncio
async def test_late_previous_transcript_does_not_satisfy_next_turn():
    fresh_answer = "This is the second turn's independently recognized answer."
    async with _bot() as (bot, url):
        recognizer = _StreamingRecognizer([SPOKEN_ANSWER, fresh_answer], final_delay=0.8)
        session = _script_with_recognizer(
            url,
            [
                EvalScriptTurn(
                    user=CALLER_TEXT, expect=[EvalExpectation(event="bot_started_speaking")]
                ),
                EvalScriptTurn(
                    user="Now answer a second question.",
                    expect=[EvalExpectation(event="response", text_contains=fresh_answer)],
                ),
            ],
            recognizer,
        )
        result = await asyncio.wait_for(session.run(), timeout=20)
        assert result.passed, result
        assert recognizer.finals == [SPOKEN_ANSWER, fresh_answer]
        assert result.turns[1].expectations[0].matched == fresh_answer
        assert bot.agent.replies == 2
        await asyncio.wait_for(bot.closed.wait(), timeout=3)


@pytest.mark.asyncio
async def test_unsupported_audio_target_cannot_activate_the_caller(tmp_path):
    async with _bot(greeting=True, audio_input=False) as (bot, url):
        session, voice, _recognizer = _session("simulation", True, False, url, tmp_path)
        result = await asyncio.wait_for(session.run(), timeout=5)
        assert result.error is not None
        assert not result.passed
        assert "audio transport support" in result.error
        assert voice.calls == []
        assert bot.agent.audio_bytes == 0
        assert not any(m.get("type") == "send-text" for m in bot.messages)
        await asyncio.wait_for(bot.closed.wait(), timeout=3)
        assert session._client._sink._pending_answer is None
