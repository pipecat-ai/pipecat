#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    CancelFrame,
    EndFrame,
    InterruptionFrame,
    LLMContextFrame,
    MetricsFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.observers.base_observer import FramePushed
from pipecat.observers.latency_watchdog_observer import LatencyWatchdogObserver
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.tests.utils import SleepFrame, run_test


class MockLLM(IdentityFilter):
    @property
    def name(self):
        return "MockLLM"


class MockSTT(IdentityFilter):
    @property
    def name(self):
        return "MockSTT"


class MockTTS(IdentityFilter):
    @property
    def name(self):
        return "MockTTS"


def _push(watchdog, frame, destination=None, source=None):
    """Helper: push a frame through the observer without running a real pipeline."""
    return watchdog.on_push_frame(
        FramePushed(
            source=source,
            destination=destination,
            frame=frame,
            direction=FrameDirection.DOWNSTREAM,
            timestamp=0,
        )
    )


class TestLatencyWatchdogObserver(unittest.IsolatedAsyncioTestCase):
    """Tests for LatencyWatchdogObserver."""

    async def test_trigger_on_latency(self):
        """Callback fires when latency exceeds threshold."""
        watchdog = LatencyWatchdogObserver(cooldown_secs=0)
        processor = MockLLM()

        fired_events = []

        @watchdog.subscribe(
            start_processor=MockLLM, start_frame=LLMContextFrame, threshold_secs=0.1
        )
        async def on_slow(name, elapsed):
            fired_events.append((name, elapsed))

        frames_to_send = [LLMContextFrame(context=None), SleepFrame(sleep=0.2)]
        await run_test(processor, frames_to_send=frames_to_send, observers=[watchdog])

        self.assertEqual(len(fired_events), 1)
        self.assertEqual(fired_events[0][0], "MockLLM")

    async def test_disarm_on_metrics(self):
        """Callback is NOT fired if MetricsFrame arrives before threshold."""
        watchdog = LatencyWatchdogObserver(cooldown_secs=0)
        processor = MockLLM()

        fired_events = []

        @watchdog.subscribe(
            start_processor=MockLLM, start_frame=LLMContextFrame, threshold_secs=0.2
        )
        async def on_slow(name, elapsed):
            fired_events.append((name, elapsed))

        frames_to_send = [
            LLMContextFrame(context=None),
            SleepFrame(sleep=0.05),
            MetricsFrame(data=[TTFBMetricsData(processor="MockLLM", value=0.05)]),
            SleepFrame(sleep=0.3),
        ]
        await run_test(processor, frames_to_send=frames_to_send, observers=[watchdog])

        self.assertEqual(len(fired_events), 0)

    async def test_cancel_on_interruption(self):
        """InterruptionFrame cancels pending watchdog timers."""
        watchdog = LatencyWatchdogObserver(cooldown_secs=0)
        processor = MockLLM()

        fired_events = []

        @watchdog.subscribe(
            start_processor=MockLLM, start_frame=LLMContextFrame, threshold_secs=0.2
        )
        async def on_slow(name, elapsed):
            fired_events.append((name, elapsed))

        frames_to_send = [
            LLMContextFrame(context=None),
            SleepFrame(sleep=0.05),
            InterruptionFrame(),
            SleepFrame(sleep=0.3),
        ]
        await run_test(processor, frames_to_send=frames_to_send, observers=[watchdog])

        self.assertEqual(len(fired_events), 0)

    async def test_cancel_on_end_frame(self):
        """EndFrame cancels pending watchdog timers."""
        watchdog = LatencyWatchdogObserver(cooldown_secs=0)

        fired_events = []

        @watchdog.subscribe(start_frame=VADUserStartedSpeakingFrame, threshold_secs=0.2)
        async def on_slow(name, elapsed):
            fired_events.append(name)

        await _push(watchdog, VADUserStartedSpeakingFrame(), destination=MockSTT())
        await asyncio.sleep(0.05)
        await _push(watchdog, EndFrame(), destination=MockSTT())
        await asyncio.sleep(0.3)

        self.assertEqual(len(fired_events), 0)

    async def test_cancel_on_cancel_frame(self):
        """CancelFrame cancels pending watchdog timers."""
        watchdog = LatencyWatchdogObserver(cooldown_secs=0)

        fired_events = []

        @watchdog.subscribe(start_frame=VADUserStartedSpeakingFrame, threshold_secs=0.2)
        async def on_slow(name, elapsed):
            fired_events.append(name)

        await _push(watchdog, VADUserStartedSpeakingFrame(), destination=MockSTT())
        await asyncio.sleep(0.05)
        await _push(watchdog, CancelFrame(), destination=MockSTT())
        await asyncio.sleep(0.3)

        self.assertEqual(len(fired_events), 0)

    async def test_cooldown(self):
        """Cooldown prevents rapid re-triggering."""
        watchdog = LatencyWatchdogObserver(cooldown_secs=1.0)
        processor = MockLLM()

        fired_events = []

        @watchdog.subscribe(
            start_processor=MockLLM, start_frame=LLMContextFrame, threshold_secs=0.1
        )
        async def on_slow(name, elapsed):
            fired_events.append((name, elapsed))

        frames_to_send = [
            LLMContextFrame(context=None),
            SleepFrame(sleep=0.2),  # Triggers 1
            # Signal disarm so we can arm again
            MetricsFrame(data=[TTFBMetricsData(processor="MockLLM", value=0.1)]),
            LLMContextFrame(context=None),
            SleepFrame(sleep=0.2),  # Should be in cooldown
        ]
        await run_test(processor, frames_to_send=frames_to_send, observers=[watchdog])

        self.assertEqual(len(fired_events), 1)

    async def test_spanning_latency(self):
        """Monitoring across different processors (spanning)."""
        watchdog = LatencyWatchdogObserver(cooldown_secs=0)
        stt = MockSTT()
        tts = MockTTS()

        fired_events = []

        @watchdog.subscribe(
            start_processor=MockSTT,
            stop_processor=MockTTS,
            threshold_secs=0.1,
            start_frame=LLMContextFrame,
        )
        async def on_slow(name, elapsed):
            fired_events.append(name)

        # Arm at STT → timer expires before TTS emits TTFB
        await _push(watchdog, LLMContextFrame(context=None), destination=stt)
        await asyncio.sleep(0.2)
        self.assertIn("MockSTT -> MockTTS", fired_events)

        # Arm then disarm at TTS before threshold
        fired_events.clear()
        await _push(watchdog, LLMContextFrame(context=None), destination=stt)
        await asyncio.sleep(0.05)
        await _push(
            watchdog,
            MetricsFrame(data=[TTFBMetricsData(processor="MockTTS", value=0.01)]),
            source=tts,
        )
        await asyncio.sleep(0.1)
        self.assertEqual(len(fired_events), 0)

    async def test_global_latency(self):
        """Monitoring from a global frame (VAD start) to bot speaking."""
        watchdog = LatencyWatchdogObserver(cooldown_secs=0)

        fired_events = []

        @watchdog.subscribe(
            start_frame=VADUserStartedSpeakingFrame,
            stop_frame=BotStartedSpeakingFrame,
            threshold_secs=0.1,
        )
        async def on_slow(name, elapsed):
            fired_events.append(name)

        await _push(watchdog, VADUserStartedSpeakingFrame(), destination=MockSTT())
        await asyncio.sleep(0.2)
        self.assertEqual(len(fired_events), 1)

    async def test_no_reset(self):
        """Subsequent start frames are ignored (no timer reset)."""
        watchdog = LatencyWatchdogObserver(cooldown_secs=0)

        fired_events = []

        @watchdog.subscribe(start_frame=VADUserStartedSpeakingFrame, threshold_secs=0.2)
        async def on_slow(name, elapsed):
            fired_events.append(asyncio.get_running_loop().time())

        start_time = asyncio.get_running_loop().time()
        await _push(watchdog, VADUserStartedSpeakingFrame(), destination=MockSTT())

        await asyncio.sleep(0.1)
        # Send again at T=0.1. Should be IGNORED.
        await _push(watchdog, VADUserStartedSpeakingFrame(), destination=MockSTT())

        await asyncio.sleep(0.15)  # T=0.25 Total
        # If ignored, fired at T=0.2. If reset, wouldn't fire until T=0.3.
        self.assertEqual(len(fired_events), 1)
        self.assertLess(fired_events[0] - start_time, 0.25)

    async def test_inheritance_matching(self):
        """Subscribing to a base class matches subclass instances.

        A subscription on LLMService should fire when the processor is an instance
        of a subclass. We call on_push_frame directly to avoid running a real
        LLMService pipeline (which would try to connect to an API).
        """

        class ConcreteMyLLM(LLMService):
            def __init__(self):
                super().__init__()

            def can_generate_metrics(self):
                return True

        watchdog = LatencyWatchdogObserver(cooldown_secs=0)
        processor = ConcreteMyLLM()

        fired_events = []

        @watchdog.subscribe(
            start_processor=LLMService, start_frame=LLMContextFrame, threshold_secs=0.1
        )
        async def on_slow(name, elapsed):
            fired_events.append(name)

        await _push(watchdog, LLMContextFrame(context=None), destination=processor)
        await asyncio.sleep(0.2)
        self.assertEqual(len(fired_events), 1)

    async def test_handler_exception_does_not_break_other_subscriptions(self):
        """A handler that raises must not prevent other subscriptions from firing."""
        watchdog = LatencyWatchdogObserver(cooldown_secs=0)

        good_fires = []

        @watchdog.subscribe(start_frame=VADUserStartedSpeakingFrame, threshold_secs=0.1)
        async def bad_handler(name, elapsed):
            raise RuntimeError("handler blew up")

        @watchdog.subscribe(start_frame=BotStartedSpeakingFrame, threshold_secs=0.1)
        async def good_handler(name, elapsed):
            good_fires.append(name)

        # Fire the bad one first, then the good one.
        await _push(watchdog, VADUserStartedSpeakingFrame(), destination=MockSTT())
        await asyncio.sleep(0.15)  # bad_handler fires and raises
        await _push(watchdog, BotStartedSpeakingFrame(), destination=MockTTS())
        await asyncio.sleep(0.15)  # good_handler must still fire

        self.assertEqual(len(good_fires), 1)


if __name__ == "__main__":
    unittest.main()
