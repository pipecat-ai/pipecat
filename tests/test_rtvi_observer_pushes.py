#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for which pushes RTVIObserver handles."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import (
    AggregatedTextFrame,
    AggregatedTextProgressFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSTextFrame,
    UserBackchannelFrame,
    UserStartedSpeakingFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVIConfigureObserverFrame
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver, RTVIObserverParams
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.text.base_text_aggregator import TextType


class TestRTVIObserverPushes(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.observer = RTVIObserver(params=RTVIObserverParams())
        self.observer.send_rtvi_message = AsyncMock()
        self.source = FrameProcessor()

    async def _push(self, frame, *, first_push=True, source=None):
        source = source or self.source
        await self.observer.on_push_frame(
            FramePushed(
                source=source,
                destination=source,
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
                first_push=first_push,
            )
        )

    async def test_audio_is_skipped_unless_audio_levels_are_reported(self):
        for frame_type in (InputAudioRawFrame, TTSAudioRawFrame):
            await self._push(frame_type(audio=b"\0" * 320, sample_rate=16000, num_channels=1))
        self.observer.send_rtvi_message.assert_not_awaited()

        observer = RTVIObserver(
            params=RTVIObserverParams(user_audio_level_enabled=True, audio_level_period_secs=0)
        )
        observer.send_rtvi_message = AsyncMock()
        self.observer = observer
        await self._push(InputAudioRawFrame(audio=b"\0" * 320, sample_rate=16000, num_channels=1))
        observer.send_rtvi_message.assert_awaited_once()

    async def test_a_frame_is_handled_on_its_first_push_only(self):
        frame = UserStartedSpeakingFrame()

        await self._push(frame)
        await self._push(frame, first_push=False)

        self.observer.send_rtvi_message.assert_awaited_once()

    async def test_a_frame_disabled_on_its_first_push_is_never_handled(self):
        frame = VADUserStartedSpeakingFrame()

        await self._push(frame)
        self.observer._apply_config(RTVIConfigureObserverFrame(vad_user_speaking_enabled=True))
        await self._push(frame, first_push=False)

        self.observer.send_rtvi_message.assert_not_awaited()

    async def test_aggregated_text_is_handled_once_it_has_gone_through_the_transport(self):
        frame = AggregatedTextFrame(text="hello", text_type="sentence")
        transport = BaseOutputTransport(TransportParams())

        await self._push(frame)
        self.assertEqual(self.observer._queued_aggregated_text_frames, [])

        await self._push(frame, first_push=False, source=transport)
        self.assertEqual(self.observer._queued_aggregated_text_frames, [frame])

    def _sent_messages(self):
        return [call.args[0] for call in self.observer.send_rtvi_message.await_args_list]

    async def test_a_user_backchannel_is_sent_as_user_input(self):
        await self._push(UserBackchannelFrame(text="mhm", user_id="user", timestamp="now"))

        message = self.observer.send_rtvi_message.await_args.args[0]
        self.assertIsInstance(message, RTVI.UserInputMessage)
        self.assertEqual(
            message.data,
            RTVI.UserInputMessageData(
                text="mhm", input_type="backchannel", user_id="user", timestamp="now", final=True
            ),
        )

    async def test_a_transcription_is_sent_as_user_transcription_and_user_input(self):
        await self._push(InterimTranscriptionFrame(text="Hel", user_id="user", timestamp="t1"))
        await self._push(TranscriptionFrame(text="Hello.", user_id="user", timestamp="t2"))

        messages = self._sent_messages()
        self.assertEqual(
            [type(m) for m in messages],
            [
                RTVI.UserTranscriptionMessage,
                RTVI.UserInputMessage,
                RTVI.UserTranscriptionMessage,
                RTVI.UserInputMessage,
            ],
        )
        self.assertEqual(
            [messages[1].data, messages[3].data],
            [
                RTVI.UserInputMessageData(
                    text="Hel",
                    input_type="transcription",
                    user_id="user",
                    timestamp="t1",
                    final=False,
                ),
                RTVI.UserInputMessageData(
                    text="Hello.",
                    input_type="transcription",
                    user_id="user",
                    timestamp="t2",
                    final=True,
                ),
            ],
        )

    async def test_a_transcription_is_only_sent_as_user_input_without_user_transcription(self):
        self.observer._params.user_transcription_enabled = False
        await self._push(TranscriptionFrame(text="Hello.", user_id="user", timestamp="now"))

        self.assertEqual([type(m) for m in self._sent_messages()], [RTVI.UserInputMessage])

    async def test_user_input_is_kept_from_the_client_when_disabled(self):
        self.observer._params.user_input_enabled = False
        await self._push(TranscriptionFrame(text="Hello.", user_id="user", timestamp="now"))
        await self._push(UserBackchannelFrame(text="mhm", user_id="user", timestamp="now"))

        self.assertEqual([type(m) for m in self._sent_messages()], [RTVI.UserTranscriptionMessage])


class TestRTVIObserverSegments(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.observer = RTVIObserver(params=RTVIObserverParams(skip_text_types=["status"]))
        self.observer.send_rtvi_message = AsyncMock()
        # The bot is speaking, so text is sent right away instead of queued.
        self.observer._bot_is_speaking = True
        self.transport = BaseOutputTransport(TransportParams())

    async def _push(self, frame):
        await self.observer.on_push_frame(
            FramePushed(
                source=self.transport,
                destination=self.transport,
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
                first_push=True,
            )
        )

    def _segment(self, text, text_type):
        # All the segments of one LLM response share a TTS context.
        segment = AggregatedTextFrame(text, text_type, context_id="turn")
        segment.will_be_spoken = True
        return segment

    def _word(self, text, segment):
        return TTSTextFrame(text, TextType.WORD, context_id="turn", segment_id=segment.id)

    def _progress(self, segment, accumulated_text, remaining_text):
        return AggregatedTextProgressFrame(
            segment_id=segment.id,
            context_id="turn",
            text=segment.text,
            text_type=segment.text_type,
            accumulated_text=accumulated_text,
            remaining_text=remaining_text,
        )

    def _sent_texts(self):
        messages = [call.args[0] for call in self.observer.send_rtvi_message.await_args_list]
        return [m.data.text for m in messages if hasattr(getattr(m, "data", None), "text")]

    def _sent_messages(self):
        return [call.args[0] for call in self.observer.send_rtvi_message.await_args_list]

    def _sent_outputs(self):
        return [
            call.args[0].data
            for call in self.observer.send_rtvi_message.await_args_list
            if isinstance(call.args[0], RTVI.BotOutputMessage)
        ]

    async def test_a_segment_spoken_in_one_piece_completes_as_that_segment(self):
        sentence = self._segment("That sounds fun.", TextType.SENTENCE)
        spoken = TTSTextFrame(sentence.text, TextType.SENTENCE, segment_id=sentence.id)
        spoken.will_be_spoken = True
        await self._push(sentence)
        await self._push(spoken)

        self.assertEqual(
            [(data.spoken_status, data.segment_id) for data in self._sent_outputs()],
            [("new", sentence.id), ("completed", sentence.id)],
        )

    async def test_a_skipped_segment_keeps_its_progress_and_words_from_the_client(self):
        status = self._segment("One moment, please.", "status")
        await self._push(status)
        await self._push(self._word("One", status))
        await self._push(self._progress(status, "One", " moment, please."))

        self.observer.send_rtvi_message.assert_not_awaited()

    async def test_a_segment_after_a_skipped_one_is_sent(self):
        status = self._segment("One moment, please.", "status")
        sentence = self._segment("That sounds fun.", TextType.SENTENCE)
        await self._push(status)
        await self._push(self._word("One", status))
        await self._push(sentence)
        await self._push(self._word("That", sentence))

        self.assertEqual(self._sent_texts(), ["That sounds fun.", "That"])

    async def test_words_follow_their_segment_when_the_segments_come_first(self):
        # A turn's segments can all arrive before the first one is spoken.
        status = self._segment("One moment, please.", "status")
        sentence = self._segment("That sounds fun.", TextType.SENTENCE)
        await self._push(status)
        await self._push(sentence)
        await self._push(self._word("One", status))
        await self._push(self._word("That", sentence))

        self.assertEqual(self._sent_texts(), ["That sounds fun.", "That"])

    async def test_a_skipped_segment_is_forgotten_once_spoken(self):
        status = self._segment("One moment, please.", "status")
        await self._push(status)
        await self._push(self._progress(status, "One moment, please.", ""))

        self.assertEqual(self.observer._skipped_segment_ids, set())

    async def test_a_skipped_segment_cut_off_is_forgotten_when_the_bot_stops(self):
        status = self._segment("One moment, please.", "status")
        await self._push(status)
        await self._push(InterruptionFrame())
        # A word the output transport pushed before it handled the interruption.
        await self._push(self._word("One", status))
        await self._push(BotStoppedSpeakingFrame())

        self.assertNotIn("One", self._sent_texts())
        self.assertEqual(self.observer._skipped_segment_ids, set())

    async def test_a_skipped_segment_cut_off_in_a_pause_is_forgotten_when_the_bot_speaks(self):
        status = self._segment("One moment, please.", "status")
        await self._push(status)
        await self._push(BotStoppedSpeakingFrame())
        self.assertEqual(self.observer._skipped_segment_ids, {status.id})

        await self._push(InterruptionFrame())
        await self._push(BotStartedSpeakingFrame())
        self.assertEqual(self.observer._skipped_segment_ids, set())

    async def test_a_skipped_segment_spoken_in_one_piece_is_forgotten(self):
        status = self._segment("One moment, please.", "status")
        await self._push(status)
        await self._push(TTSTextFrame(status.text, "status", segment_id=status.id))

        self.observer.send_rtvi_message.assert_not_awaited()
        self.assertEqual(self.observer._skipped_segment_ids, set())

    def _connect_client(self, version):
        self.observer._rtvi = SimpleNamespace(client_version=version)

    async def test_a_backchannel_is_sent_as_output_and_tts_text(self):
        backchannel = self._segment("Mm-hmm, go on.", TextType.BACKCHANNEL)
        await self._push(backchannel)
        await self._push(self._word("Mm-hmm,", backchannel))
        await self._push(self._progress(backchannel, "Mm-hmm,", " go on."))

        messages = self._sent_messages()
        self.assertEqual(
            [type(m) for m in messages],
            [RTVI.BotOutputMessage, RTVI.BotTTSTextMessage, RTVI.BotOutputMessage],
        )
        self.assertEqual(
            [(data.text_type, data.spoken_status) for data in self._sent_outputs()],
            [(TextType.BACKCHANNEL, "new"), (TextType.BACKCHANNEL, "in-progress")],
        )
        self.assertEqual(messages[1].data.text, "Mm-hmm,")

    async def test_a_backchannel_spoken_in_one_piece_is_sent_as_output_and_tts_text(self):
        backchannel = self._segment("Mm-hmm.", TextType.BACKCHANNEL)
        spoken = TTSTextFrame(backchannel.text, TextType.BACKCHANNEL, segment_id=backchannel.id)
        spoken.will_be_spoken = True
        await self._push(backchannel)
        await self._push(spoken)

        self.assertEqual(
            [type(m) for m in self._sent_messages()],
            [RTVI.BotOutputMessage, RTVI.BotOutputMessage, RTVI.BotTTSTextMessage],
        )
        self.assertEqual(
            [data.spoken_status for data in self._sent_outputs()], ["new", "completed"]
        )

    async def test_a_backchannel_is_kept_from_older_clients(self):
        for version in ([2, 1, 0], [1, 4, 0]):
            with self.subTest(version=version):
                self._connect_client(version)
                backchannel = self._segment("Mm-hmm, go on.", TextType.BACKCHANNEL)
                await self._push(backchannel)
                await self._push(self._word("Mm-hmm,", backchannel))
                await self._push(self._progress(backchannel, "Mm-hmm,", " go on."))
                await self._push(self._progress(backchannel, "Mm-hmm, go on.", ""))

                self.observer.send_rtvi_message.assert_not_awaited()
                self.assertEqual(self.observer._skipped_segment_ids, set())

    async def test_a_backchannel_spoken_in_one_piece_is_kept_from_older_clients(self):
        self._connect_client([2, 1, 0])
        backchannel = self._segment("Mm-hmm.", TextType.BACKCHANNEL)
        spoken = TTSTextFrame(backchannel.text, TextType.BACKCHANNEL, segment_id=backchannel.id)
        spoken.will_be_spoken = True
        await self._push(backchannel)
        await self._push(spoken)

        self.observer.send_rtvi_message.assert_not_awaited()
        self.assertEqual(self.observer._skipped_segment_ids, set())

    async def test_a_segment_after_a_backchannel_is_sent_to_older_clients(self):
        self._connect_client([2, 1, 0])
        backchannel = self._segment("Mm-hmm.", TextType.BACKCHANNEL)
        sentence = self._segment("That sounds fun.", TextType.SENTENCE)
        await self._push(backchannel)
        await self._push(self._word("Mm-hmm.", backchannel))
        await self._push(sentence)
        await self._push(self._word("That", sentence))

        self.assertEqual(self._sent_texts(), ["That sounds fun.", "That"])

    async def test_a_backchannel_is_kept_from_the_client_when_skipped(self):
        self.observer._params.skip_text_types = ["status", TextType.BACKCHANNEL]
        backchannel = self._segment("Mm-hmm.", TextType.BACKCHANNEL)
        await self._push(backchannel)
        await self._push(self._word("Mm-hmm.", backchannel))

        self.observer.send_rtvi_message.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
