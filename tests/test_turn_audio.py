#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.audio.turn_audio import TurnAudio, TurnAudioCollector
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.tests.utils import SleepFrame, run_test

SAMPLE_RATE = 16000

# 0.1s of mono audio. Every run of speech below is one chunk long unless it says
# otherwise, so run lengths are comparable by counting chunks.
CHUNK = b"\x11\x00" * 1600


def user_audio() -> Frame:
    return InputAudioRawFrame(audio=CHUNK, sample_rate=SAMPLE_RATE, num_channels=1)


def bot_audio() -> Frame:
    return OutputAudioRawFrame(audio=CHUNK, sample_rate=SAMPLE_RATE, num_channels=1)


def user_speaks() -> list[Frame]:
    """One run of user speech, with the pauses a real mic would produce."""
    return [
        UserStartedSpeakingFrame(),
        SleepFrame(sleep=0.05),
        user_audio(),
        SleepFrame(sleep=0.05),
        UserStoppedSpeakingFrame(),
        SleepFrame(sleep=0.2),
    ]


def bot_speaks() -> list[Frame]:
    """One run of bot speech.

    Bot audio is a DataFrame and queues, while the speaking frames are
    SystemFrames that skip the queue, so the audio needs time to drain before
    BotStoppedSpeakingFrame or it would be reported after its own run ended.
    """
    return [
        BotStartedSpeakingFrame(),
        SleepFrame(sleep=0.05),
        bot_audio(),
        SleepFrame(sleep=0.1),
        BotStoppedSpeakingFrame(),
        SleepFrame(sleep=0.2),
    ]


class TestTurnAudioCollector(unittest.IsolatedAsyncioTestCase):
    async def _collect(self, frames: list[Frame]) -> TurnAudioCollector:
        """Run frames through a recording processor and return the collector."""
        processor = AudioBufferProcessor(
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            enable_turn_audio=True,
            auto_start_recording=True,
        )
        turn_tracker = TurnTrackingObserver()
        collector = TurnAudioCollector()
        collector.attach(processor, turn_tracker)

        await run_test(
            processor,
            frames_to_send=frames,
            observers=[turn_tracker],
            pipeline_params=PipelineParams(
                audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE
            ),
        )
        return collector

    async def test_collects_nothing_without_audio(self):
        collector = await self._collect([SleepFrame(sleep=0.1)])
        self.assertEqual(collector.turns(), [])
        self.assertIsNone(collector.sample_rate)

    async def test_files_a_turn_under_its_number(self):
        collector = await self._collect([*user_speaks(), *bot_speaks()])

        # Turn 1 opens with the pipeline, so the first exchange lands there.
        self.assertEqual(collector.turn_numbers(), [1])
        turn = collector.turns()[0]
        self.assertEqual(turn.number, 1)
        self.assertEqual(turn.user, CHUNK)
        self.assertEqual(turn.bot, CHUNK)
        self.assertEqual(collector.sample_rate, SAMPLE_RATE)

    async def test_numbers_successive_turns(self):
        collector = await self._collect(
            [*user_speaks(), *bot_speaks(), *user_speaks(), *bot_speaks()]
        )

        self.assertEqual(collector.turn_numbers(), [1, 2])
        for turn in collector.turns():
            self.assertEqual(turn.user, CHUNK, f"turn {turn.number} user audio")
            self.assertEqual(turn.bot, CHUNK, f"turn {turn.number} bot audio")

    async def test_keeps_each_run_of_speech_in_a_turn(self):
        # The user pauses mid-thought before the bot has replied, so both runs
        # belong to the same turn.
        collector = await self._collect([*user_speaks(), *user_speaks(), *bot_speaks()])

        self.assertEqual(collector.turn_numbers(), [1])
        turn = collector.turns()[0]
        self.assertEqual(turn.user_runs, [CHUNK, CHUNK])
        self.assertEqual(turn.user, CHUNK * 2)

    async def test_keeps_each_run_of_bot_speech_in_a_turn(self):
        # A bot that resumes after a function call reports twice for one turn.
        collector = await self._collect([*user_speaks(), *bot_speaks(), *bot_speaks()])

        self.assertEqual(collector.turn_numbers(), [1])
        turn = collector.turns()[0]
        self.assertEqual(turn.bot_runs, [CHUNK, CHUNK])
        self.assertEqual(turn.bot, CHUNK * 2)

    async def test_files_barged_in_bot_audio_under_the_interrupted_turn(self):
        # The user barges in while the bot is speaking. That ends turn 1 and
        # starts turn 2, and only then is the bot's cut-off audio reported.
        collector = await self._collect(
            [
                *user_speaks(),
                BotStartedSpeakingFrame(),
                SleepFrame(sleep=0.05),
                bot_audio(),
                SleepFrame(sleep=0.1),
                UserStartedSpeakingFrame(),
                SleepFrame(sleep=0.05),
                BotStoppedSpeakingFrame(),
                SleepFrame(sleep=0.05),
                user_audio(),
                SleepFrame(sleep=0.05),
                UserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.2),
                *bot_speaks(),
            ]
        )

        self.assertEqual(collector.turn_numbers(), [1, 2])
        interrupted, replied = collector.turns()

        # The interrupted turn keeps the audio it was cut off in the middle of.
        self.assertEqual(interrupted.number, 1)
        self.assertEqual(interrupted.user, CHUNK)
        self.assertEqual(interrupted.bot, CHUNK)

        # The turn that interrupted keeps only its own.
        self.assertEqual(replied.number, 2)
        self.assertEqual(replied.user, CHUNK)
        self.assertEqual(replied.bot, CHUNK)


class TestTurnAudio(unittest.TestCase):
    def test_joins_runs(self):
        turn = TurnAudio(number=3, user_runs=[b"ab", b"cd"], bot_runs=[b"ef"])
        self.assertEqual(turn.user, b"abcd")
        self.assertEqual(turn.bot, b"ef")

    def test_empty_turn(self):
        turn = TurnAudio(number=1)
        self.assertEqual(turn.user, b"")
        self.assertEqual(turn.bot, b"")


if __name__ == "__main__":
    unittest.main()
