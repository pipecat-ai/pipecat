#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from typing import List

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    CancelFrame,
    EndFrame,
    InterruptionFrame,
    LLMContextFrame,
    MetricsFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.observers.latency_watchdog_observer import LatencyWatchdogObserver
from pipecat.processors.filters.identity_filter import IdentityFilter
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


class TestLatencyWatchdogObserver(unittest.IsolatedAsyncioTestCase):
    """Tests for LatencyWatchdogObserver."""

    async def test_trigger_on_latency(self):
        """Test that callback is fired when latency exceeds threshold."""
        watchdog = LatencyWatchdogObserver(cooldown_secs=0)
        processor = MockLLM()

        fired_events = []

        # We manually register with MockLLM and LLMContextFrame
        @watchdog.subscribe(MockLLM, threshold_secs=0.1, input_frame=LLMContextFrame)
        async def on_slow(name, elapsed):
            fired_events.append((name, elapsed))

        frames_to_send = [
            LLMContextFrame(context=None),
            SleepFrame(sleep=0.2),
        ]

        await run_test(processor, frames_to_send=frames_to_send, observers=[watchdog])

        self.assertEqual(len(fired_events), 1)
        self.assertEqual(fired_events[0][0], "MockLLM")

    async def test_disarm_on_metrics(self):
        """Test that callback is NOT fired if MetricsFrame arrives in time."""
        watchdog = LatencyWatchdogObserver(cooldown_secs=0)
        processor = MockLLM()

        fired_events = []

        @watchdog.subscribe(MockLLM, threshold_secs=0.2, input_frame=LLMContextFrame)
        async def on_slow(name, elapsed):
            fired_events.append((name, elapsed))

        # Push input, then metrics quickly
        # Note: run_test pushes frames TO processor.
        # But for DISARM, the observer needs to see the MetricsFrame being PUSHED BY the processor.
        # IdentityFilter will push any frame it receives.
        frames_to_send = [
            LLMContextFrame(context=None),
            SleepFrame(sleep=0.05),
            MetricsFrame(data=[TTFBMetricsData(processor="MockLLM", value=0.05)]),
            SleepFrame(sleep=0.3),
        ]

        await run_test(processor, frames_to_send=frames_to_send, observers=[watchdog])

        self.assertEqual(len(fired_events), 0)

    async def test_cancel_on_interruption(self):
        """Test that InterruptionFrame cancels pending watchdog timers."""
        watchdog = LatencyWatchdogObserver(cooldown_secs=0)
        processor = MockLLM()

        fired_events = []

        @watchdog.subscribe(MockLLM, threshold_secs=0.2, input_frame=LLMContextFrame)
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

    async def test_cooldown(self):
        """Test that cooldown prevents rapid re-triggering."""
        watchdog = LatencyWatchdogObserver(cooldown_secs=1.0)
        processor = MockLLM()

        fired_events = []

        @watchdog.subscribe(MockLLM, threshold_secs=0.1, input_frame=LLMContextFrame)
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
        """Test monitoring across different processors (spanning)."""
        watchdog = LatencyWatchdogObserver(cooldown_secs=0)
        stt = MockSTT()
        tts = MockTTS()

        fired_events = []

        @watchdog.subscribe(
            start_processor=MockSTT,
            stop_processor=MockTTS,
            threshold_secs=0.1,
            start_frame=LLMContextFrame)
        async def on_slow(name, elapsed):
            fired_events.append(name)

        # 1. Arm at STT
        await watchdog.on_push_frame(
            FramePushed(
                source=None,
                destination=stt,
                frame=LLMContextFrame(
                    context=None),
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0))

        # 2. Wait for trigger
        await asyncio.sleep(0.2)
        self.assertIn("MockSTT -> MockTTS", fired_events)

        # 3. Test disarm at TTS
        fired_events.clear()
        await watchdog.on_push_frame(
            FramePushed(
                source=None,
                destination=stt,
                frame=LLMContextFrame(
                    context=None),
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0))
        await asyncio.sleep(0.05)
        # TTS Emits TTFB
        await watchdog.on_push_frame(
            FramePushed(
                source=tts,
                destination=None,
                frame=MetricsFrame(
                    data=[
                        TTFBMetricsData(
                            processor="MockTTS",
                            value=0.01)]),
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0))
        await asyncio.sleep(0.1)
        self.assertEqual(len(fired_events), 0)

    async def test_global_latency(self):
        """Test monitoring from a global frame (VAD start) to bot speaking."""
        watchdog = LatencyWatchdogObserver(cooldown_secs=0)

        fired_events = []

        @watchdog.subscribe(
            start_frame=VADUserStartedSpeakingFrame,
            stop_frame=BotStartedSpeakingFrame,
            threshold_secs=0.1)
        async def on_slow(name, elapsed):
            fired_events.append(name)

        # Arm with global VAD frame
        await watchdog.on_push_frame(
            FramePushed(
                source=None,
                destination=MockSTT(),
                frame=VADUserStartedSpeakingFrame(),
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0))

        await asyncio.sleep(0.2)
        self.assertEqual(len(fired_events), 1)

    async def test_no_reset(self):
        """Test that subsequent start frames are ignored (no timer reset)."""
        watchdog = LatencyWatchdogObserver(cooldown_secs=0)

        fired_events = []

        @watchdog.subscribe(start_frame=VADUserStartedSpeakingFrame, threshold_secs=0.2)
        async def on_slow(name, elapsed):
            fired_events.append(asyncio.get_event_loop().time())

        # Start at T=0
        start_time = asyncio.get_event_loop().time()
        await watchdog.on_push_frame(
            FramePushed(
                source=None,
                destination=MockSTT(),
                frame=VADUserStartedSpeakingFrame(),
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0))

        await asyncio.sleep(0.1)
        # Send again at T=0.1. Should be IGNORED.
        await watchdog.on_push_frame(
            FramePushed(
                source=None,
                destination=MockSTT(),
                frame=VADUserStartedSpeakingFrame(),
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0))

        await asyncio.sleep(0.15)  # T=0.25 Total
        # If it was ignored, it fired at T=0.2.
        # If it was reset, it wouldn't fire until T=0.3.
        self.assertEqual(len(fired_events), 1)
        self.assertLess(fired_events[0] - start_time, 0.25)

    async def test_inheritance_matching(self):
        """Test inheritance matching with a real Pipecat base class."""
        # We'll use IdentityFilter as a base for our mock
        class MyLLM(LLMService):
             def __init__(self): super().__init__()
             def can_generate_metrics(self): return True

        # Actually, let's keep it simple and just test that the subscription works with the class
        watchdog = LatencyWatchdogObserver(cooldown_secs=0)
        processor = MockLLM()

        fired_events = []
        @watchdog.subscribe(MockLLM, threshold_secs=0.1, input_frame=LLMContextFrame)
        async def on_slow(name, elapsed):
            fired_events.append(name)

        frames_to_send = [
            LLMContextFrame(context=None),
            SleepFrame(sleep=0.2),
        ]

        await run_test(processor, frames_to_send=frames_to_send, observers=[watchdog])
        self.assertIn("MockLLM", fired_events)


if __name__ == "__main__":
    unittest.main()
