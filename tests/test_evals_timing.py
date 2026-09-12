#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the eval harness's turn timing observer."""

import types
import unittest

from pipecat.evals.client_transport import EvalClientInputTransport, EvalClientOutputTransport
from pipecat.evals.timing import EvalTimingObserver
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallInProgressFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.websocket.client import WebsocketClientParams


def _fake_session():
    return types.SimpleNamespace(is_closing=False, is_connected=True)


class TestEvalTimingObserver(unittest.IsolatedAsyncioTestCase):
    """A turn's timing is measured from its input anchor, on the observer's clock."""

    async def asyncSetUp(self):
        self.clock = 100.0
        self.observer = EvalTimingObserver(time_source=lambda: self.clock)
        # The processors a frame can come from: the bot's frames enter through
        # the input transport, the user's utterance leaves through the output,
        # and the rest (the user TTS, the bot-audio aggregator) is the harness's.
        self.bot = EvalClientInputTransport(
            None, _fake_session(), WebsocketClientParams(audio_in_enabled=True)
        )
        self.output = EvalClientOutputTransport(
            None, _fake_session(), WebsocketClientParams(audio_out_enabled=True)
        )
        self.harness = IdentityFilter(name="harness")
        self.sink = IdentityFilter(name="sink")

    def _at(self, t: float) -> None:
        """Set the clock to ``t`` seconds after it began."""
        self.clock = 100.0 + t

    async def _push(self, t: float, frame, source=None, direction=FrameDirection.DOWNSTREAM):
        """Feed one frame to the observer at ``t``, as a push from ``source`` would."""
        self._at(t)
        await self.observer.on_push_frame(
            FramePushed(
                source=source or self.bot,
                destination=self.sink,
                frame=frame,
                direction=direction,
                timestamp=0,
            )
        )

    async def _user_audio(self, t: float, seconds: float) -> None:
        """The user TTS pushing ``seconds`` of audio toward the output."""
        pcm = b"\x01\x00" * int(16000 * seconds)
        frame = TTSAudioRawFrame(audio=pcm, sample_rate=16000, num_channels=1)
        await self._push(t, frame, source=self.harness)

    async def _bot_heard(self, t: float) -> None:
        """The bot-audio aggregator broadcasting that the bot began to speak."""
        down, up = UserStartedSpeakingFrame(), UserStartedSpeakingFrame()
        down.broadcast_sibling_id, up.broadcast_sibling_id = up.id, down.id
        await self._push(t, up, source=self.harness, direction=FrameDirection.UPSTREAM)
        await self._push(t, down, source=self.harness)

    async def test_text_turn_is_anchored_at_the_send(self):
        self._at(10.0)
        timing = self.observer.begin_turn()
        await self._push(10.3, LLMFullResponseStartFrame())
        await self._push(10.5, LLMTextFrame(text="Paris"))
        await self._push(
            10.6, FunctionCallInProgressFrame(function_name="f", tool_call_id="1", arguments={})
        )
        await self._push(10.9, LLMFullResponseEndFrame())
        await self._push(11.0, BotStartedSpeakingFrame())
        await self._bot_heard(11.25)
        await self._push(12.0, BotStoppedSpeakingFrame())

        self.assertIs(timing, self.observer.timing)
        self.assertEqual(timing.input_duration_ms, 0)
        self.assertEqual(timing.llm_started_ms, 300)
        self.assertEqual(timing.first_token_ms, 500)
        self.assertEqual(timing.function_call_ms, 600)
        self.assertEqual(timing.llm_response_ms, 900)
        self.assertEqual(timing.bot_started_speaking_ms, 1000)
        self.assertEqual(timing.bot_speech_onset_ms, 1250)
        self.assertEqual(timing.bot_stopped_speaking_ms, 2000)
        self.assertEqual(timing.speech_padding_ms, 250)
        # The anchor is the send, so there is no voice to measure from.
        self.assertIsNone(timing.voice_to_voice_ms)
        self.assertEqual(timing.bot_metrics, [])

    async def test_spoken_turn_is_anchored_at_the_end_of_the_utterance(self):
        self._at(10.0)
        timing = self.observer.begin_turn()
        # The user's utterance goes out: 1.2s of audio in two frames, bracketed
        # by the output's own speaking frames as it is sent.
        await self._user_audio(10.01, 0.7)
        await self._user_audio(10.02, 0.5)
        await self._push(10.05, BotStartedSpeakingFrame(), source=self.output)
        # The bot's previous reply ends while the user is still speaking: it
        # predates the anchor, so it is not this turn's.
        await self._push(10.4, BotStartedSpeakingFrame())
        await self._push(10.8, BotStoppedSpeakingFrame())
        await self._push(11.2, BotStoppedSpeakingFrame(), source=self.output)
        await self._push(11.5, LLMFullResponseStartFrame())
        await self._push(11.7, LLMTextFrame(text="Hi"))
        await self._push(12.0, BotStartedSpeakingFrame())
        await self._bot_heard(12.1)

        self.assertEqual(timing.input_duration_ms, 1200)
        self.assertEqual(timing.llm_started_ms, 300)
        self.assertEqual(timing.first_token_ms, 500)
        self.assertEqual(timing.bot_started_speaking_ms, 800)
        self.assertEqual(timing.bot_speech_onset_ms, 900)
        self.assertEqual(timing.voice_to_voice_ms, 900)
        self.assertEqual(timing.speech_padding_ms, 100)
        self.assertIsNone(timing.bot_stopped_speaking_ms)

    async def test_the_outputs_stop_without_audio_moves_nothing(self):
        self._at(10.0)
        timing = self.observer.begin_turn()
        await self._push(10.2, LLMFullResponseStartFrame())
        await self._push(10.5, BotStoppedSpeakingFrame(), source=self.output)
        self.assertEqual(timing.input_duration_ms, 0)
        self.assertEqual(timing.llm_started_ms, 200)

    async def test_each_measure_takes_its_first_frame(self):
        self._at(10.0)
        timing = self.observer.begin_turn()
        # A filler response, then the answer: the turn's LLM measures are the filler's.
        await self._push(10.2, LLMFullResponseStartFrame())
        await self._push(10.3, LLMTextFrame(text="Let me check. "))
        await self._push(10.4, LLMFullResponseEndFrame())
        await self._push(11.0, LLMFullResponseStartFrame())
        await self._push(11.1, LLMTextFrame(text="Sunny."))
        await self._push(11.2, LLMFullResponseEndFrame())
        self.assertEqual(timing.llm_started_ms, 200)
        self.assertEqual(timing.first_token_ms, 300)
        self.assertEqual(timing.llm_response_ms, 400)

    async def test_a_response_begun_before_the_input_is_not_the_reply(self):
        # Its text and end land after the send, but its start did not: the
        # measures wait for a response that starts in the turn.
        await self._push(9.0, LLMFullResponseStartFrame())
        self._at(10.0)
        timing = self.observer.begin_turn()
        await self._push(10.1, LLMTextFrame(text="straggler"))
        await self._push(10.2, LLMFullResponseEndFrame())
        self.assertIsNone(timing.first_token_ms)
        self.assertIsNone(timing.llm_response_ms)
        await self._push(10.5, LLMFullResponseStartFrame())
        await self._push(10.6, LLMTextFrame(text="reply"))
        self.assertEqual(timing.first_token_ms, 600)

    async def test_a_stop_before_a_start_is_the_previous_reply_ending(self):
        self._at(10.0)
        timing = self.observer.begin_turn()
        await self._push(10.1, BotStoppedSpeakingFrame())
        self.assertIsNone(timing.bot_stopped_speaking_ms)
        await self._push(10.5, BotStartedSpeakingFrame())
        await self._push(11.0, BotStoppedSpeakingFrame())
        self.assertEqual(timing.bot_started_speaking_ms, 500)
        self.assertEqual(timing.bot_stopped_speaking_ms, 1000)

    async def test_a_frame_is_read_where_it_was_made(self):
        # Every processor pushes a frame along; only its first push says who
        # made it, and it is read once. The persona's own LLM text, and the
        # sink passing the bot's on, time nothing.
        self._at(10.0)
        timing = self.observer.begin_turn()
        started = LLMFullResponseStartFrame()
        await self._push(10.2, started)
        await self._push(10.4, started, source=self.harness)
        await self._push(10.5, LLMTextFrame(text="persona"), source=self.harness)
        self.assertEqual(timing.llm_started_ms, 200)
        self.assertIsNone(timing.first_token_ms)

    async def test_a_new_turn_gets_its_own_timing(self):
        self._at(10.0)
        first = self.observer.begin_turn()
        await self._push(10.3, LLMFullResponseStartFrame())
        self._at(20.0)
        second = self.observer.begin_turn()
        await self._push(20.4, LLMFullResponseStartFrame())
        self.assertIsNot(first, second)
        self.assertEqual(first.llm_started_ms, 300)
        self.assertEqual(second.llm_started_ms, 400)

    async def test_the_bots_metrics_are_kept_per_turn(self):
        self._at(10.0)
        timing = self.observer.begin_turn()
        await self._push(
            10.5,
            MetricsFrame(
                data=[
                    TTFBMetricsData(processor="OpenAILLMService#0", model="gpt", value=0.412),
                    ProcessingMetricsData(processor="OpenAILLMService#0", value=1.0),
                ]
            ),
        )
        await self._push(
            10.6,
            MetricsFrame(
                data=[
                    LLMUsageMetricsData(
                        processor="",
                        value=LLMTokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                    )
                ]
            ),
        )
        await self._push(
            10.7, MetricsFrame(data=[TTFBMetricsData(processor="CartesiaTTSService#0", value=0.1)])
        )
        self.assertEqual(
            timing.bot_metrics,
            [
                {
                    "processor": "OpenAILLMService#0",
                    "ttfb_ms": 412,
                    "processing_ms": 1000,
                    "tokens": None,
                },
                {
                    "processor": None,
                    "ttfb_ms": None,
                    "processing_ms": None,
                    "tokens": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
                {
                    "processor": "CartesiaTTSService#0",
                    "ttfb_ms": 100,
                    "processing_ms": None,
                    "tokens": None,
                },
            ],
        )
        # A report from before a spoken turn's end is not the reply's.
        await self._user_audio(10.62, 0.5)
        await self._push(10.65, BotStoppedSpeakingFrame(), source=self.output)
        self.assertEqual([m["processor"] for m in timing.bot_metrics], ["CartesiaTTSService#0"])


if __name__ == "__main__":
    unittest.main()
