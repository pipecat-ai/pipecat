#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Integration tests for the VAD state :class:`TurnAnalyzerUserTurnStopStrategy` keeps.

VAD emits ``VADUserStartedSpeakingFrame`` / ``VADUserStoppedSpeakingFrame`` only
on transitions, so the strategy carries that state between them. A turn can
start from a transcript rather than from a VAD frame — mid-utterance, or for
speech VAD never reported at all — and the end-of-turn decision has to hold up
either way.

The tests drive real pipeline processors: a real input transport with its audio
filter, and the real aggregator, controller and strategies.
"""

import time
import unittest
from typing import Any

import numpy as np

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn, SmartTurnParams
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams, VADState
from pipecat.frames.frames import (
    FilterControlFrame,
    InputAudioRawFrame,
    LLMContextFrame,
    StartFrame,
    STTMetadataFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_start import (
    BaseUserTurnStartStrategy,
    MinWordsUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.time import time_now_iso8601

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

DEEPGRAM_TTFS_P99 = 0.35

# Kept short so a test can show a turn ending well inside it.
WATCHDOG_TIMEOUT = 2.0

# Audio is fed in short runs separated by sleeps, so the pipeline stays drained
# and frame ordering doesn't depend on how fast the host happens to be. VAD is
# frame-driven, so pausing the feed does not advance it toward a stop.
AUDIO_RUN_SECS = 0.5
DRAIN_SECS = 0.4

# One continuous utterance, split by the STT endpointer into four finalized
# transcripts.
FRAGMENTS = [
    "Yes. I am ready to get started. I have my coffee.",
    "I've got",
    "a bunch of notes around me.",
    "I've got all of my notes here.",
]


class PassthroughFilter(BaseAudioFilter):
    """Input filter that leaves audio untouched."""

    async def start(self, sample_rate: int):
        pass

    async def stop(self):
        pass

    async def process_frame(self, frame: FilterControlFrame):
        pass

    async def filter(self, audio: bytes) -> bytes:
        return audio


class SuppressingFilter(PassthroughFilter):
    """Input filter that attenuates everything to silence.

    Models noise cancellation that removes the speaker. With transport-side
    transcription the service still transcribes the published track, so
    transcripts keep arriving while the pipeline's audio goes quiet.
    """

    async def filter(self, audio: bytes) -> bytes:
        return bytes(len(audio))


class EnergyVADAnalyzer(VADAnalyzer):
    """VAD analyzer driven by frame energy instead of an ML model.

    Runs the real :class:`VADAnalyzer` state machine; only the per-frame
    confidence is deterministic.
    """

    async def analyze_audio(self, buffer: bytes) -> VADState:
        """Analyze inline rather than on the analyzer's thread-pool executor.

        A thread hop per 20 ms frame is enough to leave transcripts queued
        behind the audio on a busy host, which changes the frame ordering these
        tests depend on.
        """
        return self._run_analyzer(buffer)

    def num_frames_required(self) -> int:
        return FRAME_SAMPLES

    def voice_confidence(self, buffer: bytes) -> float:
        samples = np.frombuffer(buffer, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(samples**2))) / 32768.0
        return 1.0 if rms > 0.05 else 0.0


class StubSmartTurn(BaseSmartTurn):
    """Smart turn analyzer with a deterministic model call.

    Always predicts "incomplete", so any end-of-turn in these tests comes from
    the analyzer's silence timeout or from the strategy, never from the model.
    """

    def _predict_endpoint(self, audio_array: np.ndarray) -> dict[str, Any]:
        return {"prediction": 0, "probability": 0.0}


class CompletingSmartTurn(BaseSmartTurn):
    """Smart turn analyzer that always predicts the turn is complete."""

    def _predict_endpoint(self, audio_array: np.ndarray) -> dict[str, Any]:
        return {"prediction": 1, "probability": 1.0}


class ReadyInputTransport(BaseInputTransport):
    """Input transport that reports itself ready as soon as it starts.

    Concrete transports call :meth:`set_transport_ready` once connected; there
    is no connection to wait on here.
    """

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)


def _speech_audio(seconds: float) -> list[InputAudioRawFrame]:
    rng = np.random.default_rng(0)
    return [
        InputAudioRawFrame(
            audio=rng.integers(-9000, 9000, FRAME_SAMPLES, dtype=np.int16).tobytes(),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )
        for _ in range(int(seconds * 1000 / FRAME_MS))
    ]


def _silent_audio(seconds: float) -> list[InputAudioRawFrame]:
    return [
        InputAudioRawFrame(audio=bytes(FRAME_SAMPLES * 2), sample_rate=SAMPLE_RATE, num_channels=1)
        for _ in range(int(seconds * 1000 / FRAME_MS))
    ]


def _transcript(text: str) -> TranscriptionFrame:
    """A finalized transcript as a transport pushes it, bypassing the audio path."""
    frame = TranscriptionFrame(text, "participant", time_now_iso8601())
    frame.finalized = True
    return frame


class TestTurnStartVADState(unittest.IsolatedAsyncioTestCase):
    async def _run(
        self,
        start_strategies: list[BaseUserTurnStartStrategy],
        audio_filter: BaseAudioFilter | None = None,
    ) -> dict[str, Any]:
        context = LLMContext()
        aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(
                # min_volume is disabled so frame energy alone drives VAD.
                vad_analyzer=EnergyVADAnalyzer(
                    params=VADParams(confidence=0.7, start_secs=0.1, stop_secs=1.4, min_volume=0.0)
                ),
                user_turn_strategies=UserTurnStrategies(
                    start=start_strategies,
                    stop=[
                        TurnAnalyzerUserTurnStopStrategy(
                            turn_analyzer=StubSmartTurn(params=SmartTurnParams(stop_secs=3.0))
                        )
                    ],
                ),
                user_turn_stop_timeout=600.0,
            ),
        )

        turn_starts = 0
        turn_stops = 0

        @aggregator.event_handler("on_user_turn_started")
        async def _on_started(agg, strategy):
            nonlocal turn_starts
            turn_starts += 1

        @aggregator.event_handler("on_user_turn_stopped")
        async def _on_stopped(agg, strategy, message):
            nonlocal turn_stops
            turn_stops += 1

        transport = ReadyInputTransport(
            TransportParams(
                audio_in_enabled=True, audio_in_filter=audio_filter or PassthroughFilter()
            )
        )

        # The participant talks continuously, and the transcription service
        # finalizes a fragment part way through. The utterance ends with a real
        # pause.
        #
        # Audio arrives in short runs, each followed by a sleep, so a transcript
        # is never queued behind a long run of audio frames — the interruption a
        # turn start broadcasts would discard it.
        frames_to_send: list = [
            STTMetadataFrame(service_name="DailyTransport", ttfs_p99_latency=DEEPGRAM_TTFS_P99),
            *_speech_audio(AUDIO_RUN_SECS),
            SleepFrame(DRAIN_SECS),
        ]
        for fragment in FRAGMENTS:
            frames_to_send += [
                _transcript(fragment),
                SleepFrame(DRAIN_SECS),
                *_speech_audio(AUDIO_RUN_SECS),
                SleepFrame(DRAIN_SECS),
            ]
        frames_to_send += [*_silent_audio(2.0), SleepFrame(DRAIN_SECS)]

        received_down, _ = await run_test(
            Pipeline([transport, aggregator]),
            frames_to_send=frames_to_send,
            expected_down_frames=None,
            send_end_frame=True,
        )

        return {
            "turn_starts": turn_starts,
            "turn_stops": turn_stops,
            "llm_calls": sum(1 for f in received_down if isinstance(f, LLMContextFrame)),
            "user_messages": [m["content"] for m in context.messages if m.get("role") == "user"],
        }

    async def test_vad_driven_start_aggregates_the_utterance(self):
        """A VAD-driven turn start keeps the utterance in a single turn."""
        result = await self._run([VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()])

        self.assertEqual(result["turn_starts"], 1)
        self.assertEqual(result["llm_calls"], 1)
        self.assertEqual(len(result["user_messages"]), 1)
        for fragment in FRAGMENTS:
            self.assertIn(fragment, result["user_messages"][0])

    async def test_transcript_driven_start_aggregates_the_utterance(self):
        """A transcript-driven turn start reaches the same single turn.

        The turn starts mid-utterance, with the VAD stop still to come.
        """
        result = await self._run([MinWordsUserTurnStartStrategy(min_words=2)])

        self.assertEqual(result["turn_starts"], 1)
        self.assertEqual(result["llm_calls"], 1)
        self.assertEqual(len(result["user_messages"]), 1)
        for fragment in FRAGMENTS:
            self.assertIn(fragment, result["user_messages"][0])

    async def test_no_vad_signal_falls_back_to_a_turn_per_transcript(self):
        """Without any VAD signal, each transcript drives its own turn.

        An input filter that removes the speaker silences VAD while
        transport-side transcription keeps transcribing the published track, so
        the strategy has no VAD state to reason from and the transcript fallback
        decides every turn.
        """
        result = await self._run(
            [VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()],
            audio_filter=SuppressingFilter(),
        )

        self.assertEqual(result["llm_calls"], len(FRAGMENTS))
        self.assertEqual(result["user_messages"], FRAGMENTS)

    async def test_transcript_only_turn_after_a_vad_turn_still_completes_promptly(self):
        """A transcript VAD never saw still ends its turn on the STT budget.

        VAD reports only transitions, so a word too short or too quiet for VAD
        produces a transcript with no VAD stop behind it. That turn is decided by
        the transcript fallback, on the STT budget rather than the much longer
        stop watchdog — including after earlier speech in the session did drive
        VAD normally.
        """
        context = LLMContext()
        aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(
                vad_analyzer=EnergyVADAnalyzer(
                    params=VADParams(confidence=0.7, start_secs=0.1, stop_secs=0.2, min_volume=0.0)
                ),
                user_turn_strategies=UserTurnStrategies(
                    start=[VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()],
                    stop=[
                        TurnAnalyzerUserTurnStopStrategy(
                            turn_analyzer=CompletingSmartTurn(params=SmartTurnParams(stop_secs=3.0))
                        )
                    ],
                ),
                user_turn_stop_timeout=WATCHDOG_TIMEOUT,
            ),
        )
        transport = ReadyInputTransport(
            TransportParams(audio_in_enabled=True, audio_in_filter=PassthroughFilter())
        )

        llm_calls = []
        push_context_frame = aggregator.push_context_frame

        async def timestamped_push_context_frame(*args, **kwargs):
            llm_calls.append(time.monotonic())
            return await push_context_frame(*args, **kwargs)

        aggregator.push_context_frame = timestamped_push_context_frame

        frames_to_send: list = [
            STTMetadataFrame(service_name="DailyTransport", ttfs_p99_latency=DEEPGRAM_TTFS_P99),
            # Audible speech: VAD reports a start and, after the silence, a stop.
            *_speech_audio(1.0),
            SleepFrame(0.1),
            _transcript("This is a normal audible sentence."),
            SleepFrame(0.1),
            *_silent_audio(0.6),
            SleepFrame(0.6),
            # Too quiet for VAD, but the transcription service still hears it.
            _transcript("Okay."),
            # Long enough for both the STT budget and the stop watchdog to
            # elapse, so the assertion below shows which one released the turn.
            SleepFrame(WATCHDOG_TIMEOUT + 1.0),
        ]

        await run_test(
            Pipeline([transport, aggregator]),
            frames_to_send=frames_to_send,
            expected_down_frames=None,
            send_end_frame=True,
        )

        self.assertEqual(len(llm_calls), 2)
        # The second turn is decided well inside the watchdog, not by it.
        self.assertLess(llm_calls[1] - llm_calls[0], WATCHDOG_TIMEOUT)
        self.assertEqual(
            [m["content"] for m in context.messages if m.get("role") == "user"],
            ["This is a normal audible sentence.", "Okay."],
        )


if __name__ == "__main__":
    unittest.main()
