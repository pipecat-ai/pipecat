#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""State-machine tests for VAD hysteresis.

Confidence and volume are scripted. The tests drive ``VADAnalyzer`` directly so
the thresholds, timers, and events do not depend on a VAD model.
"""

import unittest

from pydantic import ValidationError

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams, VADState
from pipecat.audio.vad.vad_controller import VADController
from pipecat.frames.frames import (
    InputAudioRawFrame,
    SpeechControlParamsFrame,
    UserSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.tests.utils import run_test
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup

# One Silero window is 32ms at both 8kHz (256 samples) and 16kHz (512 samples).
WINDOW_SECS = 0.032
OPEN = 0.55
HOLD = 0.35
MIN_VOLUME = 0.6


def _samples_per_window(sample_rate: int) -> int:
    return 512 if sample_rate == 16000 else 256


class ScriptedVADAnalyzer(VADAnalyzer):
    """VAD analyzer with caller-supplied confidence and volume.

    Each full window consumes one confidence and, when volumes are a list, one
    volume. Partial buffers consume neither.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        params: VADParams,
        confidences: list[float],
        volumes: float | list[float] = 1.0,
    ):
        super().__init__(sample_rate=sample_rate, params=params)
        self._confidences = confidences
        self._volumes = volumes
        self.confidence_reads = 0
        self.volume_reads = 0
        self.set_sample_rate(sample_rate)

    def num_frames_required(self) -> int:
        return _samples_per_window(self.sample_rate)

    def voice_confidence(self, buffer: bytes) -> float:
        confidence = self._confidences[self.confidence_reads]
        self.confidence_reads += 1
        return confidence

    def _get_smoothed_volume(self, audio: bytes) -> float:
        self.volume_reads += 1
        if isinstance(self._volumes, list):
            return self._volumes[self.volume_reads - 1]
        return self._volumes


class VADAnalyzerTestCase(unittest.TestCase):
    def setUp(self):
        self._analyzers: list[VADAnalyzer] = []

    def tearDown(self):
        for analyzer in self._analyzers:
            analyzer._executor.shutdown(wait=False)

    def make(
        self,
        confidences: list[float],
        volumes: float | list[float] = 1.0,
        *,
        sample_rate: int = 16000,
        start_windows: int = 1,
        stop_windows: int = 2,
        check_windows: bool = True,
        **overrides,
    ) -> ScriptedVADAnalyzer:
        params = dict(
            confidence=OPEN,
            hold_confidence=HOLD,
            start_secs=WINDOW_SECS * start_windows,
            stop_secs=WINDOW_SECS * stop_windows,
            min_volume=MIN_VOLUME,
        )
        params.update(overrides)
        analyzer = ScriptedVADAnalyzer(
            sample_rate=sample_rate,
            params=VADParams(**params),
            confidences=confidences,
            volumes=volumes,
        )
        self._analyzers.append(analyzer)
        if check_windows:
            self.assertEqual(analyzer._vad_start_frames, start_windows)
            self.assertEqual(analyzer._vad_stop_frames, stop_windows)
        return analyzer

    def push(self, analyzer: ScriptedVADAnalyzer, windows: int = 1) -> list[VADState]:
        """Feed one window at a time and return the state after each."""
        frame = b"\x00" * analyzer.num_frames_required() * 2
        return [analyzer._run_analyzer(frame) for _ in range(windows)]

    def push_batch(self, analyzer: ScriptedVADAnalyzer, windows: int) -> VADState:
        """Feed several windows in one buffer."""
        frame = b"\x00" * analyzer.num_frames_required() * 2
        return analyzer._run_analyzer(frame * windows)


class TestVADParamsHysteresis(VADAnalyzerTestCase):
    def test_accepts_hold_within_confidence(self):
        for confidence, hold in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (OPEN, HOLD), (OPEN, OPEN)):
            with self.subTest(confidence=confidence, hold=hold):
                params = VADParams(confidence=confidence, hold_confidence=hold)
                self.assertEqual(params.hold_confidence, hold)
                self.assertEqual(params.confidence, confidence)

    def test_disabled_hold_accepts_existing_confidence_range(self):
        """A single threshold keeps the range callers already pass."""
        params = VADParams(confidence=1.5, min_volume=-1)
        self.assertIsNone(params.hold_confidence)
        params.confidence = -0.2
        params.min_volume = 2.0
        self.assertEqual(params.confidence, -0.2)

    def test_rejects_invalid_hold_combinations(self):
        invalid = (
            dict(confidence=OPEN, hold_confidence=OPEN + 0.01),
            dict(confidence=OPEN, hold_confidence=-0.01),
            dict(confidence=1.01, hold_confidence=0.5),
            dict(confidence=-0.1, hold_confidence=0.0),
            dict(confidence=0.0, hold_confidence=0.01),
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValidationError) as caught:
                    VADParams(**values)
                self.assertIn("0.0 <= hold_confidence <= confidence <= 1.0", str(caught.exception))

    def test_assignment_rejects_a_hold_above_confidence(self):
        params = VADParams(confidence=OPEN, hold_confidence=HOLD)
        with self.assertRaises(ValidationError):
            params.hold_confidence = OPEN + 0.1
        self.assertEqual(params.hold_confidence, HOLD)
        with self.assertRaises(ValidationError):
            params.confidence = HOLD - 0.1
        self.assertEqual(params.confidence, OPEN)

    def test_set_params_rejects_a_constructed_invalid_instance(self):
        analyzer = self.make([OPEN])
        original = analyzer.params
        invalid = VADParams.model_construct(
            confidence=0.2,
            hold_confidence=0.9,
            start_secs=WINDOW_SECS,
            stop_secs=WINDOW_SECS * 2,
            min_volume=MIN_VOLUME,
        )
        with self.assertRaises(ValidationError):
            analyzer.set_params(invalid)
        self.assertIs(analyzer.params, original)
        self.assertEqual(analyzer.params.hold_confidence, HOLD)


class TestVADHysteresisStateMachine(VADAnalyzerTestCase):
    def test_speech_starts_at_the_open_threshold(self):
        for confidence, volume in ((OPEN, MIN_VOLUME), (0.9, 1.0)):
            with self.subTest(confidence=confidence, volume=volume):
                analyzer = self.make([confidence], volumes=volume)
                self.assertEqual(self.push(analyzer), [VADState.SPEAKING])

    def test_speech_does_not_start_between_the_thresholds(self):
        for confidence in (HOLD, (OPEN + HOLD) / 2, OPEN - 0.001):
            with self.subTest(confidence=confidence):
                analyzer = self.make([confidence], volumes=1.0)
                self.assertEqual(self.push(analyzer), [VADState.QUIET])

    def test_starting_is_cancelled_when_confidence_drops_below_open(self):
        analyzer = self.make([OPEN, HOLD, OPEN], start_windows=2)
        self.assertEqual(
            self.push(analyzer, 3),
            [VADState.STARTING, VADState.QUIET, VADState.STARTING],
        )

    def test_starting_is_cancelled_when_volume_drops_below_minimum(self):
        analyzer = self.make(
            [OPEN, OPEN],
            volumes=[MIN_VOLUME, MIN_VOLUME - 0.01],
            start_windows=2,
        )
        self.assertEqual(self.push(analyzer, 2), [VADState.STARTING, VADState.QUIET])

    def test_quiet_requires_volume(self):
        analyzer = self.make([OPEN], volumes=MIN_VOLUME - 0.01)
        self.assertEqual(self.push(analyzer), [VADState.QUIET])

    def test_speech_continues_between_the_thresholds(self):
        analyzer = self.make([OPEN, HOLD, OPEN - 0.001])
        self.assertEqual(
            self.push(analyzer, 3),
            [VADState.SPEAKING, VADState.SPEAKING, VADState.SPEAKING],
        )

    def test_speech_continues_when_volume_drops_above_the_hold_threshold(self):
        analyzer = self.make(
            [OPEN, HOLD],
            volumes=[MIN_VOLUME, 0.0],
        )
        self.assertEqual(self.push(analyzer, 2), [VADState.SPEAKING, VADState.SPEAKING])

    def test_speaking_stops_when_confidence_falls_below_hold(self):
        analyzer = self.make([OPEN, HOLD - 0.001], volumes=1.0, stop_windows=2)
        self.assertEqual(self.push(analyzer, 2), [VADState.SPEAKING, VADState.STOPPING])

    def test_stopping_returns_to_speaking_when_confidence_recovers(self):
        # start_windows stays above 1 so recovery cannot be a fresh start.
        analyzer = self.make(
            [OPEN, OPEN, HOLD - 0.01, HOLD],
            volumes=[1.0, 1.0, 1.0, 0.0],
            start_windows=2,
            stop_windows=3,
        )
        self.assertEqual(
            self.push(analyzer, 4),
            [VADState.STARTING, VADState.SPEAKING, VADState.STOPPING, VADState.SPEAKING],
        )

    def test_stopping_reaches_quiet_after_the_stop_duration(self):
        analyzer = self.make([OPEN, 0.0, 0.0], stop_windows=2)
        self.assertEqual(
            self.push(analyzer, 3),
            [VADState.SPEAKING, VADState.STOPPING, VADState.QUIET],
        )

    def test_stop_countdown_can_finish_inside_one_buffer(self):
        analyzer = self.make([OPEN, 0.0, 0.0], stop_windows=2)
        self.assertEqual(self.push(analyzer, 1), [VADState.SPEAKING])
        self.assertEqual(self.push_batch(analyzer, 2), VADState.QUIET)

    def test_recovery_inside_a_buffer_cancels_the_stop(self):
        analyzer = self.make([OPEN, 0.0, HOLD], stop_windows=2)
        self.assertEqual(self.push(analyzer, 1), [VADState.SPEAKING])
        self.assertEqual(self.push_batch(analyzer, 2), VADState.SPEAKING)

    def test_equal_thresholds_ignore_volume_only_after_speech_starts(self):
        analyzer = self.make(
            [OPEN, OPEN, OPEN - 0.001],
            volumes=[MIN_VOLUME, 0.0, 1.0],
            hold_confidence=OPEN,
            stop_windows=2,
        )
        self.assertEqual(
            self.push(analyzer, 3),
            [VADState.SPEAKING, VADState.SPEAKING, VADState.STOPPING],
        )
        # A later attempt to start still needs volume.
        analyzer = self.make([OPEN], volumes=MIN_VOLUME - 0.01, hold_confidence=OPEN)
        self.assertEqual(self.push(analyzer), [VADState.QUIET])

    def test_disabled_hysteresis_keeps_volume_and_one_threshold(self):
        analyzer = self.make(
            [OPEN, OPEN, HOLD, OPEN],
            volumes=[1.0, 0.0, 1.0, 1.0],
            hold_confidence=None,
            stop_windows=3,
        )
        # Volume starts the stop countdown, confidence below the single threshold
        # keeps it going, and confidence together with volume returns to speaking.
        self.assertEqual(
            self.push(analyzer, 4),
            [VADState.SPEAKING, VADState.STOPPING, VADState.STOPPING, VADState.SPEAKING],
        )

    def test_disabled_and_enabled_use_the_same_timer_counts(self):
        disabled = self.make(
            [],
            hold_confidence=None,
            check_windows=False,
            start_secs=0.2,
            stop_secs=0.8,
        )
        enabled = self.make([], check_windows=False, start_secs=0.2, stop_secs=0.8)
        self.assertEqual(
            (disabled._vad_start_frames, disabled._vad_stop_frames),
            (enabled._vad_start_frames, enabled._vad_stop_frames),
        )
        self.assertEqual(disabled._vad_start_frames, round(0.2 / WINDOW_SECS))
        self.assertEqual(enabled._vad_stop_frames, round(0.8 / WINDOW_SECS))

    def test_sample_rates_and_partial_buffers(self):
        for sample_rate in (8000, 16000):
            with self.subTest(sample_rate=sample_rate):
                analyzer = self.make(
                    [OPEN, HOLD, 0.0, 0.0],
                    sample_rate=sample_rate,
                    stop_windows=2,
                )
                frame = b"\x01\x00" * analyzer.num_frames_required()
                half = len(frame) // 2
                self.assertEqual(analyzer._run_analyzer(frame[:half]), VADState.QUIET)
                self.assertEqual(analyzer.confidence_reads, 0)
                self.assertEqual(analyzer._run_analyzer(frame[half:]), VADState.SPEAKING)
                self.assertEqual(
                    [analyzer._run_analyzer(frame) for _ in range(3)],
                    [VADState.SPEAKING, VADState.STOPPING, VADState.QUIET],
                )

    def test_dynamic_parameter_updates(self):
        analyzer = self.make([OPEN, 0.4, 0.4, 0.8], stop_windows=2)
        self.assertEqual(self.push(analyzer), [VADState.SPEAKING])

        # A live assignment takes effect on the next window and keeps the segment.
        analyzer.params.hold_confidence = 0.5
        self.assertEqual(self.push(analyzer), [VADState.STOPPING])
        with self.assertRaises(ValidationError):
            analyzer.params.hold_confidence = 0.9
        self.assertEqual(analyzer.params.hold_confidence, 0.5)

        # An invalid replacement does not reset the segment or the parameters.
        state_before = analyzer._vad_state
        with self.assertRaises(ValidationError):
            analyzer.set_params(VADParams(confidence=0.2, hold_confidence=0.9))
        self.assertEqual(analyzer._vad_state, state_before)
        self.assertEqual(analyzer.params.hold_confidence, 0.5)

        # A valid replacement starts a new segment under the new thresholds.
        analyzer.set_params(
            VADParams(
                confidence=0.8,
                hold_confidence=0.2,
                start_secs=WINDOW_SECS,
                stop_secs=WINDOW_SECS * 2,
                min_volume=MIN_VOLUME,
            )
        )
        self.assertEqual(analyzer._vad_state, VADState.QUIET)
        # 0.4 is below the new open threshold, so it does not continue anything.
        self.assertEqual(self.push(analyzer), [VADState.QUIET])
        self.assertEqual(self.push(analyzer), [VADState.SPEAKING])

    def test_reset_then_detects_the_next_segment(self):
        analyzer = self.make([OPEN, 0.0, 0.0, HOLD, OPEN], stop_windows=2)
        self.assertEqual(
            self.push(analyzer, 3),
            [VADState.SPEAKING, VADState.STOPPING, VADState.QUIET],
        )
        # Between the thresholds after a completed segment does not restart it.
        self.assertEqual(self.push(analyzer), [VADState.QUIET])
        self.assertEqual(self.push(analyzer), [VADState.SPEAKING])

        analyzer.set_params(analyzer.params)
        self.assertEqual(analyzer._vad_state, VADState.QUIET)
        self.assertEqual(analyzer._vad_starting_count, 0)
        self.assertEqual(analyzer._vad_stopping_count, 0)


class TestVADHysteresisEvents(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.task_manager = TaskManager()

    def _analyzer(self) -> ScriptedVADAnalyzer:
        return ScriptedVADAnalyzer(
            sample_rate=16000,
            params=VADParams(
                confidence=OPEN,
                hold_confidence=HOLD,
                start_secs=WINDOW_SECS * 2,
                stop_secs=WINDOW_SECS * 3,
                min_volume=MIN_VOLUME,
            ),
            # Open threshold for two windows, a dip that must not stop the
            # segment, a miss, a recovery, then three misses that do stop it.
            confidences=[OPEN, OPEN, HOLD, 0.0, HOLD, 0.0, 0.0, 0.0],
            volumes=[MIN_VOLUME, MIN_VOLUME, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        )

    async def test_events_fire_once_on_speaking_and_quiet(self):
        analyzer = self._analyzer()
        controller = VADController(analyzer)
        events: list[str] = []

        @controller.event_handler("on_speech_started")
        async def on_speech_started(_controller):
            events.append("started")

        @controller.event_handler("on_speech_stopped")
        async def on_speech_stopped(_controller):
            events.append("stopped")

        await controller.setup(frame_processor_setup(self.task_manager))
        frame = b"\x00" * analyzer.num_frames_required() * 2
        audio = InputAudioRawFrame(audio=frame, sample_rate=16000, num_channels=1)
        states = []
        for _ in range(len(analyzer._confidences)):
            await controller.process_frame(audio)
            states.append(analyzer._vad_state)

        self.assertEqual(
            states,
            [
                VADState.STARTING,
                VADState.SPEAKING,
                VADState.SPEAKING,
                VADState.STOPPING,
                VADState.SPEAKING,
                VADState.STOPPING,
                VADState.STOPPING,
                VADState.QUIET,
            ],
        )
        self.assertEqual(events, ["started", "stopped"])
        await controller.cleanup()

    async def test_processor_forwards_stopping_audio_before_a_single_stop(self):
        analyzer = ScriptedVADAnalyzer(
            sample_rate=16000,
            params=VADParams(
                confidence=OPEN,
                hold_confidence=HOLD,
                start_secs=WINDOW_SECS,
                stop_secs=WINDOW_SECS * 2,
                min_volume=MIN_VOLUME,
            ),
            confidences=[OPEN, HOLD, 0.0, 0.0],
            volumes=[1.0, 0.0, 0.0, 0.0],
        )
        processor = VADProcessor(
            vad_analyzer=analyzer,
            speech_activity_period=1_000_000,
            audio_idle_timeout=0,
        )
        audio = b"\x00" * analyzer.num_frames_required() * 2
        frames = [
            InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1) for _ in range(4)
        ]
        down, _up = await run_test(processor, frames_to_send=frames)
        types = [type(item) for item in down]
        self.assertEqual(
            types,
            [
                SpeechControlParamsFrame,
                InputAudioRawFrame,
                VADUserStartedSpeakingFrame,
                UserSpeakingFrame,
                InputAudioRawFrame,
                InputAudioRawFrame,
                InputAudioRawFrame,
                VADUserStoppedSpeakingFrame,
            ],
        )
        started = [item for item in down if isinstance(item, VADUserStartedSpeakingFrame)]
        stopped = [item for item in down if isinstance(item, VADUserStoppedSpeakingFrame)]
        self.assertEqual(len(started), 1)
        self.assertEqual(len(stopped), 1)
        self.assertEqual(started[0].start_secs, WINDOW_SECS)
        self.assertEqual(stopped[0].stop_secs, WINDOW_SECS * 2)
        # Audio is forwarded before VAD events, so the stopping window and the
        # window that reaches quiet both precede the single stop frame.
        self.assertLess(
            max(index for index, item in enumerate(down) if isinstance(item, InputAudioRawFrame)),
            down.index(stopped[0]),
        )
