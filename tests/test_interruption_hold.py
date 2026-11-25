#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for opt-in ``InterruptionMode.HOLD`` interruption handling."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pipecat.audio.mixers.base_audio_mixer import BaseAudioMixer
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    InterruptionHoldFrame,
    InterruptionResumeFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    MixerControlFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.processors.interruption_hold import InterruptionHoldProcessor
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.empty_user_turn import EmptyUserTurnConfig
from pipecat.turns.types import InterruptionMode, UserTurnResumedParams
from pipecat.turns.user_start import (
    ExternalUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
    UserTurnStartedParams,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_start import (
    InterruptionMode as StartInterruptionMode,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import (
    EagerUserTurnStrategies,
    ExternalUserTurnStrategies,
    StrategyMode,
    UserTurnStrategies,
    UserTurnStrategiesParams,
)
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.text.base_text_aggregator import TextType
from tests.frame_processor_helpers import frame_processor_setup

USER_TURN_STOP_TIMEOUT = 0.15
USER_SPEECH_TIMEOUT = 0.08


class _PassthroughMixer(BaseAudioMixer):
    async def start(self, sample_rate: int):
        pass

    async def stop(self):
        pass

    async def process_frame(self, frame: MixerControlFrame):
        pass

    async def mix(self, audio: bytes) -> bytes:
        return audio


class _ContextRecorder(FrameProcessor):
    """Counts downstream LLMContextFrames, optionally consuming them like an LLM service."""

    def __init__(self, *, consume: bool = False):
        super().__init__()
        self.context_frames = 0
        self._consume = consume

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            self.context_frames += 1
            if self._consume:
                return
        await self.push_frame(frame, direction)


class _StubOutputTransport(BaseOutputTransport):
    """Output transport stub for full-pipeline tests with BaseOutputTransport."""

    def __init__(self, params: TransportParams):
        super().__init__(params)
        self.written_audio: list[bytes] = []

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame) -> bool:
        self.written_audio.append(frame.audio)
        return True


async def _clock_idle(sender):
    while sender._clock_frame or not sender._clock_queue.empty():
        await asyncio.sleep(0.01)


async def _make_transport(mixer: BaseAudioMixer | None = None) -> BaseOutputTransport:
    params = TransportParams(audio_out_enabled=True, audio_out_mixer=mixer)
    transport = BaseOutputTransport(params)
    transport.push_frame = AsyncMock()
    transport.write_audio_frame = AsyncMock(return_value=True)

    task_manager = TaskManager()
    await transport.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=task_manager,
            pipeline_worker=SimpleNamespace(app_resources=None),  # type: ignore[arg-type]
        )
    )
    start_frame = StartFrame(audio_out_sample_rate=16000)
    await transport.process_frame(start_frame, FrameDirection.DOWNSTREAM)
    await transport.set_transport_ready(start_frame)
    return transport


def _hold_params(
    *,
    user_speech_timeout: float = USER_SPEECH_TIMEOUT,
    user_turn_stop_timeout: float = USER_TURN_STOP_TIMEOUT,
    **kwargs,
) -> LLMUserAggregatorParams:
    return LLMUserAggregatorParams(
        user_turn_strategies=UserTurnStrategies(
            start=[
                VADUserTurnStartStrategy(interruptions=InterruptionMode.HOLD),
                TranscriptionUserTurnStartStrategy(interruptions=InterruptionMode.HOLD),
            ],
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=user_speech_timeout)],
        ),
        user_turn_stop_timeout=user_turn_stop_timeout,
        **kwargs,
    )


def _bot_speaking_start() -> list[Frame]:
    return [
        LLMRunFrame(),
        SleepFrame(),
        LLMFullResponseStartFrame(),
        BotStartedSpeakingFrame(),
        TTSTextFrame("Where would", text_type=TextType.WORD),
        SleepFrame(),
    ]


def _bot_speaking_tail() -> list[Frame]:
    return [
        TTSTextFrame("you like to go?", text_type=TextType.WORD),
        TTSStoppedFrame(),
        LLMFullResponseEndFrame(),
        SleepFrame(),
    ]


class TestInterruptionModeConfig(unittest.IsolatedAsyncioTestCase):
    async def test_exports_and_defaults(self):
        self.assertIs(InterruptionMode, StartInterruptionMode)

        vad_default = VADUserTurnStartStrategy()
        self.assertEqual(vad_default.interruption_mode, InterruptionMode.IMMEDIATE)

        vad_false = VADUserTurnStartStrategy(enable_interruptions=False)
        self.assertEqual(vad_false.interruption_mode, InterruptionMode.NONE)

        vad_true = VADUserTurnStartStrategy(enable_interruptions=True)
        self.assertEqual(vad_true.interruption_mode, InterruptionMode.IMMEDIATE)

        vad_hold = VADUserTurnStartStrategy(interruptions=InterruptionMode.HOLD)
        self.assertEqual(vad_hold.interruption_mode, InterruptionMode.HOLD)

        trans_hold = TranscriptionUserTurnStartStrategy(interruptions=InterruptionMode.HOLD)
        self.assertEqual(trans_hold.interruption_mode, InterruptionMode.HOLD)

        ext_hold = ExternalUserTurnStartStrategy(interruptions=InterruptionMode.HOLD)
        self.assertEqual(ext_hold.interruption_mode, InterruptionMode.HOLD)

        ext_compat = ExternalUserTurnStartStrategy(enable_interruptions=False)
        self.assertEqual(ext_compat.interruption_mode, InterruptionMode.NONE)

    async def test_conflicting_arguments_raise(self):
        with self.assertRaises(ValueError):
            VADUserTurnStartStrategy(
                interruptions=InterruptionMode.HOLD,
                enable_interruptions=True,
            )
        with self.assertRaises(ValueError):
            ExternalUserTurnStrategies(
                interruptions=InterruptionMode.HOLD,
                enable_interruptions=False,
            )

    async def test_external_and_eager_strategies_support_hold_and_bool(self):
        ext_hold = ExternalUserTurnStrategies(interruptions=InterruptionMode.HOLD)
        self.assertEqual(ext_hold.interruptions, InterruptionMode.HOLD)
        self.assertTrue(ext_hold.enable_interruptions)
        self.assertEqual(ext_hold.start[0].interruption_mode, InterruptionMode.HOLD)

        ext_off = ExternalUserTurnStrategies(enable_interruptions=False)
        self.assertEqual(ext_off.interruptions, InterruptionMode.NONE)
        self.assertFalse(ext_off.enable_interruptions)

        eager_hold = EagerUserTurnStrategies(interruptions=InterruptionMode.HOLD)
        self.assertEqual(eager_hold.interruptions, InterruptionMode.HOLD)
        self.assertTrue(eager_hold.enable_interruptions)

    async def test_params_synchronization(self):
        p1 = UserTurnStartedParams(enable_interruptions=True, enable_user_speaking_frames=True)
        self.assertEqual(p1.interruption_mode, InterruptionMode.IMMEDIATE)

        p2 = UserTurnStartedParams(enable_interruptions=False, enable_user_speaking_frames=True)
        self.assertEqual(p2.interruption_mode, InterruptionMode.NONE)

        p3 = UserTurnStartedParams(
            enable_interruptions=False,
            enable_user_speaking_frames=True,
            interruption_mode=InterruptionMode.HOLD,
        )
        self.assertTrue(p3.enable_interruptions)
        self.assertEqual(p3.interruption_mode, InterruptionMode.HOLD)

        r1 = UserTurnResumedParams(enable_interruptions=True, inference_triggered=False)
        self.assertEqual(r1.interruption_mode, InterruptionMode.IMMEDIATE)

        r2 = UserTurnResumedParams(
            enable_interruptions=True,
            inference_triggered=True,
            interruption_mode=InterruptionMode.HOLD,
        )
        self.assertEqual(r2.interruption_mode, InterruptionMode.HOLD)
        self.assertTrue(r2.enable_interruptions)

    async def test_controller_combines_modes_in_strategy_mode_all(self):
        task_manager = TaskManager()
        cases = [
            (
                [InterruptionMode.IMMEDIATE, InterruptionMode.HOLD],
                InterruptionMode.HOLD,
                True,
            ),
            (
                [InterruptionMode.HOLD, InterruptionMode.NONE],
                InterruptionMode.NONE,
                False,
            ),
            (
                [InterruptionMode.IMMEDIATE, InterruptionMode.IMMEDIATE],
                InterruptionMode.IMMEDIATE,
                True,
            ),
        ]

        async def run_case(modes: list[InterruptionMode]) -> list[UserTurnStartedParams]:
            strategies = [
                VADUserTurnStartStrategy(interruptions=modes[0]),
                TranscriptionUserTurnStartStrategy(interruptions=modes[1]),
            ]
            controller = UserTurnController(
                user_turn_strategies=UserTurnStrategies(start=strategies),
                user_turn_strategies_params=UserTurnStrategiesParams(start_mode=StrategyMode.ALL),
            )
            await controller.setup(frame_processor_setup(task_manager))
            await controller.start()

            started: list[UserTurnStartedParams] = []

            @controller.event_handler("on_user_turn_started")
            async def on_started(_ctrl, _strategy, params):
                started.append(params)

            await controller.process_frame(VADUserStartedSpeakingFrame())
            await controller.process_frame(
                TranscriptionFrame(text="hi", user_id="", timestamp="now")
            )
            await controller.cleanup()
            return started

        for modes, expected_mode, expected_flag in cases:
            with self.subTest(modes=modes):
                started = await run_case(modes)
                self.assertEqual(len(started), 1)
                self.assertEqual(started[0].interruption_mode, expected_mode)
                self.assertEqual(started[0].enable_interruptions, expected_flag)


class TestInterruptionHoldProcessor(unittest.IsolatedAsyncioTestCase):
    async def test_hold_and_resume_preserves_order_and_passes_system_frames(self):
        processor = InterruptionHoldProcessor()
        held_frames = [
            TTSAudioRawFrame(b"\x01\x02", sample_rate=16000, num_channels=1),
            TTSTextFrame("hello", text_type=TextType.WORD),
            TTSStoppedFrame(),
        ]
        frames_to_send = [
            InterruptionHoldFrame(),
            *held_frames,
            SleepFrame(),
            InterruptionResumeFrame(),
            TTSTextFrame("after", text_type=TextType.WORD),
        ]
        down, _ = await run_test(Pipeline([processor]), frames_to_send=frames_to_send)

        names = [type(f).__name__ for f in down]
        self.assertEqual(
            names,
            [
                "InterruptionHoldFrame",
                "InterruptionResumeFrame",
                "TTSAudioRawFrame",
                "TTSTextFrame",
                "TTSStoppedFrame",
                "TTSTextFrame",
            ],
        )
        text_frames = [f.text for f in down if isinstance(f, TTSTextFrame)]
        self.assertEqual(text_frames, ["hello", "after"])
        self.assertFalse(processor.is_holding)

    async def test_hold_and_commit_drops_interruptible_but_keeps_uninterruptible(self):
        processor = InterruptionHoldProcessor()
        uninterruptible = FunctionCallInProgressFrame(
            function_name="lookup", tool_call_id="1", arguments={}
        )
        frames_to_send = [
            InterruptionHoldFrame(),
            TTSAudioRawFrame(b"\x01\x02", sample_rate=16000, num_channels=1),
            TTSTextFrame("dropped", text_type=TextType.WORD),
            uninterruptible,
            TTSStoppedFrame(),
            SleepFrame(),
            InterruptionFrame(),
            TTSTextFrame("kept_after", text_type=TextType.WORD),
        ]
        down, _ = await run_test(Pipeline([processor]), frames_to_send=frames_to_send)

        names = [type(f).__name__ for f in down]
        self.assertEqual(
            names,
            [
                "InterruptionHoldFrame",
                "InterruptionFrame",
                "FunctionCallInProgressFrame",
                "TTSTextFrame",
            ],
        )
        self.assertEqual([f.text for f in down if isinstance(f, TTSTextFrame)], ["kept_after"])
        self.assertFalse(processor.is_holding)

    async def test_end_frame_while_held_flushes_without_hanging(self):
        processor = InterruptionHoldProcessor()
        frames_to_send = [
            InterruptionHoldFrame(),
            TTSTextFrame("flushed", text_type=TextType.WORD),
        ]
        down, _ = await run_test(Pipeline([processor]), frames_to_send=frames_to_send)
        self.assertEqual(
            [type(f).__name__ for f in down],
            ["InterruptionHoldFrame", "TTSTextFrame"],
        )
        self.assertFalse(processor.is_holding)


class TestBaseOutputTransportInterruptionHold(unittest.IsolatedAsyncioTestCase):
    async def test_audio_hold_and_resume_plays_buffered_audio_in_order(self):
        transport = await _make_transport()
        try:
            sender = transport._media_senders[None]
            chunk_size = sender.audio_chunk_size

            c1 = TTSAudioRawFrame(b"\x01" * chunk_size, sender.sample_rate, 1)
            c2 = TTSAudioRawFrame(b"\x02" * chunk_size, sender.sample_rate, 1)
            c3 = TTSAudioRawFrame(b"\x03" * chunk_size, sender.sample_rate, 1)

            await transport.process_frame(c1, FrameDirection.DOWNSTREAM)
            await asyncio.sleep(0.05)
            self.assertEqual(transport.write_audio_frame.call_count, 1)
            self.assertTrue(sender._bot_speaking)

            await transport.process_frame(InterruptionHoldFrame(), FrameDirection.DOWNSTREAM)
            await transport.process_frame(c2, FrameDirection.DOWNSTREAM)
            await transport.process_frame(c3, FrameDirection.DOWNSTREAM)
            await transport.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
            await asyncio.sleep(0.1)

            # Audio is paused and no BotStoppedSpeakingFrame fires while held.
            self.assertEqual(transport.write_audio_frame.call_count, 1)
            pushed = [type(c.args[0]) for c in transport.push_frame.call_args_list]
            self.assertNotIn(BotStoppedSpeakingFrame, pushed)

            await transport.process_frame(InterruptionResumeFrame(), FrameDirection.DOWNSTREAM)
            await asyncio.sleep(0.1)

            written = b"".join(c.args[0].audio for c in transport.write_audio_frame.call_args_list)
            self.assertEqual(written, c1.audio + c2.audio + c3.audio)
            pushed = [type(c.args[0]) for c in transport.push_frame.call_args_list]
            self.assertIn(BotStoppedSpeakingFrame, pushed)
        finally:
            await transport.cancel(CancelFrame())

    async def test_audio_hold_and_commit_drops_buffered_audio_and_keeps_uninterruptible(self):
        transport = await _make_transport()
        try:
            sender = transport._media_senders[None]
            chunk_size = sender.audio_chunk_size

            c1 = TTSAudioRawFrame(b"\x01" * chunk_size, sender.sample_rate, 1)
            c2 = TTSAudioRawFrame(b"\x02" * chunk_size, sender.sample_rate, 1)
            fn = FunctionCallInProgressFrame(function_name="fn", tool_call_id="1", arguments={})

            await transport.process_frame(c1, FrameDirection.DOWNSTREAM)
            await asyncio.sleep(0.05)

            await transport.process_frame(InterruptionHoldFrame(), FrameDirection.DOWNSTREAM)
            await transport.process_frame(c2, FrameDirection.DOWNSTREAM)
            await transport.process_frame(fn, FrameDirection.DOWNSTREAM)
            await asyncio.sleep(0.05)
            self.assertEqual(transport.write_audio_frame.call_count, 1)

            await transport.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
            await asyncio.sleep(0.05)

            written = b"".join(c.args[0].audio for c in transport.write_audio_frame.call_args_list)
            self.assertEqual(written, c1.audio)
            pushed_types = [type(c.args[0]) for c in transport.push_frame.call_args_list]
            self.assertIn(FunctionCallInProgressFrame, pushed_types)
            self.assertIn(BotStoppedSpeakingFrame, pushed_types)
        finally:
            await transport.cancel(CancelFrame())

    async def test_hold_before_first_audio_arrives_buffers_and_resumes_or_drops(self):
        for commit in (False, True):
            with self.subTest(commit=commit):
                transport = await _make_transport()
                try:
                    sender = transport._media_senders[None]
                    chunk = TTSAudioRawFrame(
                        b"\x05" * sender.audio_chunk_size, sender.sample_rate, 1
                    )
                    await transport.process_frame(
                        InterruptionHoldFrame(), FrameDirection.DOWNSTREAM
                    )
                    await transport.process_frame(chunk, FrameDirection.DOWNSTREAM)
                    await asyncio.sleep(0.05)
                    self.assertEqual(transport.write_audio_frame.call_count, 0)

                    if commit:
                        await transport.process_frame(
                            InterruptionFrame(), FrameDirection.DOWNSTREAM
                        )
                        await asyncio.sleep(0.05)
                        self.assertEqual(transport.write_audio_frame.call_count, 0)
                    else:
                        await transport.process_frame(
                            InterruptionResumeFrame(), FrameDirection.DOWNSTREAM
                        )
                        await asyncio.sleep(0.05)
                        self.assertEqual(transport.write_audio_frame.call_count, 1)
                        self.assertEqual(
                            transport.write_audio_frame.call_args_list[0].args[0].audio,
                            chunk.audio,
                        )
                finally:
                    await transport.cancel(CancelFrame())

    async def test_timed_frames_pause_and_resume_or_cancel(self):
        for commit in (False, True):
            with self.subTest(commit=commit):
                transport = await _make_transport()
                try:
                    sender = transport._media_senders[None]
                    now = transport.get_clock().get_time()
                    first = TTSTextFrame("first", "sentence")
                    first.pts = now + 80_000_000
                    second = TTSTextFrame("second", "sentence")
                    second.pts = now + 120_000_000
                    kept = TTSTextFrame("kept", "sentence")
                    kept.pts = now + 140_000_000
                    kept.interruptible = False

                    for f in (first, second, kept):
                        await transport.process_frame(f, FrameDirection.DOWNSTREAM)
                    await asyncio.sleep(0.01)
                    self.assertIs(sender._clock_frame, first)

                    await transport.process_frame(
                        InterruptionHoldFrame(), FrameDirection.DOWNSTREAM
                    )
                    await asyncio.sleep(0.18)

                    delivered = [
                        c.args[0].text
                        for c in transport.push_frame.call_args_list
                        if isinstance(c.args[0], TTSTextFrame)
                    ]
                    self.assertEqual(delivered, [])

                    if commit:
                        await transport.process_frame(
                            InterruptionFrame(), FrameDirection.DOWNSTREAM
                        )
                        await asyncio.wait_for(_clock_idle(sender), timeout=1)
                        delivered = [
                            c.args[0].text
                            for c in transport.push_frame.call_args_list
                            if isinstance(c.args[0], TTSTextFrame)
                        ]
                        self.assertEqual(delivered, ["kept"])
                    else:
                        await transport.process_frame(
                            InterruptionResumeFrame(), FrameDirection.DOWNSTREAM
                        )
                        await asyncio.wait_for(_clock_idle(sender), timeout=1)
                        delivered = [
                            c.args[0].text
                            for c in transport.push_frame.call_args_list
                            if isinstance(c.args[0], TTSTextFrame)
                        ]
                        self.assertEqual(delivered, ["first", "second", "kept"])
                finally:
                    await transport.cancel(CancelFrame())

    async def test_mixer_keeps_sending_background_while_bot_audio_held(self):
        transport = await _make_transport(mixer=_PassthroughMixer())
        try:
            sender = transport._media_senders[None]
            c1 = TTSAudioRawFrame(b"\x07" * sender.audio_chunk_size, sender.sample_rate, 1)

            await transport.process_frame(InterruptionHoldFrame(), FrameDirection.DOWNSTREAM)
            await transport.process_frame(c1, FrameDirection.DOWNSTREAM)
            await asyncio.sleep(0.1)

            # Background mixer silence keeps running while bot chunk stays queued.
            self.assertGreater(transport.write_audio_frame.call_count, 0)
            written_while_held = [
                c.args[0].audio for c in transport.write_audio_frame.call_args_list
            ]
            self.assertNotIn(c1.audio, written_while_held)

            await transport.process_frame(InterruptionResumeFrame(), FrameDirection.DOWNSTREAM)
            await asyncio.sleep(0.1)

            written_after_resume = [
                c.args[0].audio for c in transport.write_audio_frame.call_args_list
            ]
            self.assertIn(c1.audio, written_after_resume)
        finally:
            await transport.cancel(CancelFrame())

    async def test_end_frame_while_held_stops_cleanly(self):
        transport = await _make_transport()
        try:
            sender = transport._media_senders[None]
            chunk = TTSAudioRawFrame(b"\x09" * sender.audio_chunk_size, sender.sample_rate, 1)
            await transport.process_frame(InterruptionHoldFrame(), FrameDirection.DOWNSTREAM)
            await transport.process_frame(chunk, FrameDirection.DOWNSTREAM)
            await asyncio.wait_for(
                transport.process_frame(EndFrame(), FrameDirection.DOWNSTREAM),
                timeout=1.0,
            )
            self.assertGreaterEqual(transport.write_audio_frame.call_count, 1)
            self.assertEqual(
                transport.write_audio_frame.call_args_list[0].args[0].audio,
                chunk.audio,
            )
        finally:
            await transport.cleanup()


class TestInterruptionHoldPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_empty_turn_resumes_held_output_without_new_llm_call_or_recovery(self):
        context = LLMContext()
        user, assistant = LLMContextAggregatorPair(
            context,
            user_params=_hold_params(
                empty_user_turn=EmptyUserTurnConfig(
                    interrupted_prompt="should not run in hold mode"
                )
            ),
        )
        recorder = _ContextRecorder()
        hold_processor = InterruptionHoldProcessor()

        frames = [
            *_bot_speaking_start(),
            VADUserStartedSpeakingFrame(),
            *_bot_speaking_tail(),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=USER_TURN_STOP_TIMEOUT + 0.15),
        ]
        down, _ = await run_test(
            Pipeline([user, recorder, hold_processor, assistant]),
            frames_to_send=frames,
        )

        self.assertEqual(recorder.context_frames, 1)
        down_types = [type(f) for f in down]
        self.assertIn(InterruptionHoldFrame, down_types)
        self.assertIn(InterruptionResumeFrame, down_types)
        self.assertNotIn(InterruptionFrame, down_types)
        self.assertEqual(
            context.get_messages(),
            [{"role": "assistant", "content": "Where would you like to go?"}],
        )

    async def test_transcribed_turn_commits_interruption_and_preserves_context_order(self):
        context = LLMContext()
        user, assistant = LLMContextAggregatorPair(context, user_params=_hold_params())
        recorder = _ContextRecorder()
        hold_processor = InterruptionHoldProcessor()

        frames = [
            *_bot_speaking_start(),
            VADUserStartedSpeakingFrame(),
            *_bot_speaking_tail(),
            TranscriptionFrame(text="Actually, Boston.", user_id="", timestamp="now"),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=USER_SPEECH_TIMEOUT + 0.1),
        ]
        down, _ = await run_test(
            Pipeline([user, recorder, hold_processor, assistant]),
            frames_to_send=frames,
        )

        self.assertEqual(recorder.context_frames, 2)
        down_types = [type(f) for f in down]
        self.assertIn(InterruptionHoldFrame, down_types)
        self.assertIn(InterruptionFrame, down_types)
        self.assertNotIn(InterruptionResumeFrame, down_types)
        self.assertEqual(
            context.get_messages(),
            [
                {"role": "assistant", "content": "Where would"},
                {"role": "user", "content": "Actually, Boston."},
            ],
        )

    async def test_back_to_back_transcript_at_zero_timeout_preserves_assistant_before_user(self):
        context = LLMContext()
        user, assistant = LLMContextAggregatorPair(
            context,
            user_params=_hold_params(user_speech_timeout=0.0),
        )
        recorder = _ContextRecorder()
        hold_processor = InterruptionHoldProcessor()

        frames = [
            *_bot_speaking_start(),
            VADUserStartedSpeakingFrame(),
            TTSTextFrame("you like to go?", text_type=TextType.WORD),
            VADUserStoppedSpeakingFrame(),
            TranscriptionFrame(text="Skip to Boston.", user_id="", timestamp="now"),
            SleepFrame(sleep=0.1),
        ]
        await run_test(
            Pipeline([user, recorder, hold_processor, assistant]),
            frames_to_send=frames,
        )

        self.assertEqual(
            context.get_messages(),
            [
                {"role": "assistant", "content": "Where would"},
                {"role": "user", "content": "Skip to Boston."},
            ],
        )

    async def test_native_base_output_transport_empty_and_transcribed_turns(self):
        for transcribed in (False, True):
            with self.subTest(transcribed=transcribed):
                context = LLMContext()
                user, assistant = LLMContextAggregatorPair(
                    context,
                    user_params=_hold_params(empty_user_turn=EmptyUserTurnConfig()),
                )
                llm_stub = _ContextRecorder(consume=True)
                transport = _StubOutputTransport(
                    TransportParams(
                        audio_out_enabled=True,
                        audio_out_sample_rate=16000,
                        audio_out_end_silence_secs=0,
                    )
                )
                chunk_size = int(16000 / 100) * 1 * 2 * 4  # 40ms @ 16kHz mono

                c1 = TTSAudioRawFrame(b"\x01" * chunk_size, 16000, 1)
                c2 = TTSAudioRawFrame(b"\x02" * chunk_size, 16000, 1)

                frames: list[Frame] = [
                    LLMRunFrame(),
                    SleepFrame(),
                    LLMFullResponseStartFrame(),
                    c1,
                    TTSTextFrame("Where would", text_type=TextType.WORD),
                    SleepFrame(),
                    VADUserStartedSpeakingFrame(),
                    c2,
                    TTSTextFrame("you like to go?", text_type=TextType.WORD),
                    TTSStoppedFrame(),
                    LLMFullResponseEndFrame(),
                    SleepFrame(),
                ]
                if transcribed:
                    frames.extend(
                        [
                            TranscriptionFrame(text="Boston please.", user_id="", timestamp="now"),
                            VADUserStoppedSpeakingFrame(),
                            SleepFrame(sleep=USER_SPEECH_TIMEOUT + 0.1),
                        ]
                    )
                else:
                    frames.extend(
                        [
                            VADUserStoppedSpeakingFrame(),
                            SleepFrame(sleep=USER_TURN_STOP_TIMEOUT + 0.15),
                        ]
                    )

                await run_test(
                    Pipeline([user, llm_stub, transport, assistant]),
                    frames_to_send=frames,
                )

                if transcribed:
                    self.assertEqual(llm_stub.context_frames, 2)
                    self.assertEqual(transport.written_audio, [c1.audio])
                    self.assertEqual(
                        context.get_messages(),
                        [
                            {"role": "assistant", "content": "Where would"},
                            {"role": "user", "content": "Boston please."},
                        ],
                    )
                else:
                    self.assertEqual(llm_stub.context_frames, 1)
                    self.assertEqual(transport.written_audio, [c1.audio, c2.audio])
                    self.assertEqual(
                        context.get_messages(),
                        [{"role": "assistant", "content": "Where would you like to go?"}],
                    )

    async def test_function_call_in_progress_interrupts_immediately_even_in_hold_mode(self):
        context = LLMContext()
        user, assistant = LLMContextAggregatorPair(context, user_params=_hold_params())
        recorder = _ContextRecorder()

        frames = [
            LLMRunFrame(),
            SleepFrame(),
            FunctionCallsStartedFrame(
                function_calls=[
                    FunctionCallFromLLM(
                        function_name="get_weather",
                        tool_call_id="1",
                        arguments={},
                        context=None,
                    )
                ]
            ),
            SleepFrame(),
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=USER_TURN_STOP_TIMEOUT + 0.15),
        ]
        down, _ = await run_test(Pipeline([user, recorder, assistant]), frames_to_send=frames)

        down_types = [type(f) for f in down]
        self.assertIn(InterruptionFrame, down_types)
        self.assertNotIn(InterruptionHoldFrame, down_types)
        self.assertNotIn(InterruptionResumeFrame, down_types)

    async def test_bot_idle_empty_turn_ignores_empty_user_turn_and_does_not_hold(self):
        context = LLMContext()
        user, assistant = LLMContextAggregatorPair(
            context,
            user_params=_hold_params(
                empty_user_turn=EmptyUserTurnConfig(idle_prompt="are you there?"),
                user_idle_timeout=0.15,
            ),
        )
        idle_fired = 0

        @user.event_handler("on_user_turn_idle")
        async def on_idle(_agg):
            nonlocal idle_fired
            idle_fired += 1

        recorder = _ContextRecorder()
        frames = [
            BotStoppedSpeakingFrame(),
            SleepFrame(),
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=USER_TURN_STOP_TIMEOUT + 0.25),
        ]
        down, _ = await run_test(Pipeline([user, recorder, assistant]), frames_to_send=frames)

        down_types = [type(f) for f in down]
        self.assertNotIn(InterruptionHoldFrame, down_types)
        self.assertNotIn(InterruptionResumeFrame, down_types)
        self.assertNotIn(InterruptionFrame, down_types)
        self.assertEqual(recorder.context_frames, 0)
        self.assertEqual(context.get_messages(), [])
        self.assertGreaterEqual(idle_fired, 1)

    async def test_standalone_user_turn_processor_hold_resume_and_commit(self):
        for transcribed in (False, True):
            with self.subTest(transcribed=transcribed):
                utp = UserTurnProcessor(
                    user_turn_strategies=UserTurnStrategies(
                        start=[
                            VADUserTurnStartStrategy(interruptions=InterruptionMode.HOLD),
                            TranscriptionUserTurnStartStrategy(interruptions=InterruptionMode.HOLD),
                        ],
                        stop=[
                            SpeechTimeoutUserTurnStopStrategy(
                                user_speech_timeout=USER_SPEECH_TIMEOUT
                            )
                        ],
                    ),
                    user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
                )
                frames: list[Frame] = [
                    BotStartedSpeakingFrame(),
                    SleepFrame(),
                    VADUserStartedSpeakingFrame(),
                ]
                if transcribed:
                    frames.extend(
                        [
                            TranscriptionFrame(text="hold commit", user_id="", timestamp="now"),
                            VADUserStoppedSpeakingFrame(),
                            SleepFrame(sleep=USER_SPEECH_TIMEOUT + 0.1),
                        ]
                    )
                else:
                    frames.extend(
                        [
                            VADUserStoppedSpeakingFrame(),
                            SleepFrame(sleep=USER_TURN_STOP_TIMEOUT + 0.15),
                        ]
                    )

                down, _ = await run_test(Pipeline([utp]), frames_to_send=frames)
                down_types = [type(f) for f in down]
                self.assertIn(InterruptionHoldFrame, down_types)
                if transcribed:
                    self.assertIn(InterruptionFrame, down_types)
                    self.assertNotIn(InterruptionResumeFrame, down_types)
                else:
                    self.assertIn(InterruptionResumeFrame, down_types)
                    self.assertNotIn(InterruptionFrame, down_types)

    async def test_end_frame_during_held_turn_does_not_hang(self):
        context = LLMContext()
        user, assistant = LLMContextAggregatorPair(context, user_params=_hold_params())
        hold_processor = InterruptionHoldProcessor()
        frames = [
            *_bot_speaking_start(),
            VADUserStartedSpeakingFrame(),
            TTSTextFrame(" trailing", text_type=TextType.WORD),
        ]
        await asyncio.wait_for(
            run_test(Pipeline([user, hold_processor, assistant]), frames_to_send=frames),
            timeout=2.0,
        )


if __name__ == "__main__":
    unittest.main()
