import asyncio
import unittest

from pipecat.frames.frames import ClientConnectedFrame, Frame, StartFrame, TextFrame
from pipecat.observers.startup_timing_observer import (
    StartupTimingObserver,
    StartupTimingReport,
    TransportReadinessReport,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import run_test


class SlowStartProcessor(FrameProcessor):
    """A processor that sleeps during start to simulate slow initialization."""

    def __init__(self, delay: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self._delay = delay

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await asyncio.sleep(self._delay)
        await self.push_frame(frame, direction)


class FastProcessor(FrameProcessor):
    """A processor with no start delay."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class TestStartupTimingObserver(unittest.IsolatedAsyncioTestCase):
    """Tests for StartupTimingObserver."""

    async def test_timing_reported(self):
        """Test that startup timing is measured and reported."""
        observer = StartupTimingObserver()
        processor = SlowStartProcessor(delay=0.1)

        reports = []

        @observer.event_handler("on_startup_timing_report")
        async def on_report(obs, report):
            reports.append(report)

        frames_to_send = [TextFrame(text="hello")]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=[TextFrame],
            observers=[observer],
        )

        self.assertEqual(len(reports), 1)
        report = reports[0]
        self.assertGreater(report.total_duration_secs, 0)
        self.assertGreater(len(report.processor_timings), 0)

        # Find our slow processor in the timings.
        slow_timings = [
            t for t in report.processor_timings if "SlowStartProcessor" in t.processor_name
        ]
        self.assertEqual(len(slow_timings), 1)
        self.assertGreaterEqual(slow_timings[0].duration_secs, 0.05)

    async def test_processor_types_filter(self):
        """Test that processor_types filter limits which processors appear."""
        observer = StartupTimingObserver(processor_types=(SlowStartProcessor,))
        processor = SlowStartProcessor(delay=0.05)

        reports = []

        @observer.event_handler("on_startup_timing_report")
        async def on_report(obs, report):
            reports.append(report)

        frames_to_send = [TextFrame(text="hello")]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=[TextFrame],
            observers=[observer],
        )

        self.assertEqual(len(reports), 1)
        report = reports[0]

        # Only SlowStartProcessor should be in the timings.
        for t in report.processor_timings:
            self.assertIn("SlowStartProcessor", t.processor_name)

    async def test_report_emits_once(self):
        """Test that the report is emitted only once even with multiple frames."""
        observer = StartupTimingObserver()
        processor = FastProcessor()

        reports = []

        @observer.event_handler("on_startup_timing_report")
        async def on_report(obs, report):
            reports.append(report)

        frames_to_send = [
            TextFrame(text="first"),
            TextFrame(text="second"),
            TextFrame(text="third"),
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=[TextFrame, TextFrame, TextFrame],
            observers=[observer],
        )

        self.assertEqual(len(reports), 1)

    async def test_event_handler_receives_report(self):
        """Test that the event handler receives a proper StartupTimingReport."""
        observer = StartupTimingObserver()
        processor = SlowStartProcessor(delay=0.05)

        reports = []

        @observer.event_handler("on_startup_timing_report")
        async def on_report(obs, report):
            reports.append(report)

        frames_to_send = [TextFrame(text="hello")]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=[TextFrame],
            observers=[observer],
        )

        self.assertEqual(len(reports), 1)
        report = reports[0]
        self.assertIsInstance(report, StartupTimingReport)
        self.assertIsInstance(report.total_duration_secs, float)
        for timing in report.processor_timings:
            self.assertIsInstance(timing.processor_name, str)
            self.assertIsInstance(timing.duration_secs, float)

    async def test_excludes_internal_processors(self):
        """Test that internal pipeline processors are excluded by default."""
        observer = StartupTimingObserver()
        processor = FastProcessor()

        reports = []

        @observer.event_handler("on_startup_timing_report")
        async def on_report(obs, report):
            reports.append(report)

        frames_to_send = [TextFrame(text="hello")]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=[TextFrame],
            observers=[observer],
        )

        self.assertEqual(len(reports), 1)
        report = reports[0]

        # No internal processors (PipelineSource, PipelineSink, Pipeline) in the report.
        internal_names = ("Pipeline#", "PipelineTask#")
        for t in report.processor_timings:
            for prefix in internal_names:
                self.assertNotIn(
                    prefix,
                    t.processor_name,
                    f"Internal processor {t.processor_name} should be excluded by default",
                )

    async def test_transport_readiness_measured(self):
        """Test that ClientConnectedFrame after startup emits on_transport_readiness_measured."""
        observer = StartupTimingObserver()
        processor = FastProcessor()

        readiness_reports = []

        @observer.event_handler("on_transport_readiness_measured")
        async def on_readiness(obs, report):
            readiness_reports.append(report)

        frames_to_send = [ClientConnectedFrame(), TextFrame(text="hello")]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=[ClientConnectedFrame, TextFrame],
            observers=[observer],
        )

        self.assertEqual(len(readiness_reports), 1)
        report = readiness_reports[0]
        self.assertIsInstance(report, TransportReadinessReport)
        self.assertGreater(report.readiness_secs, 0)

    async def test_transport_readiness_only_first(self):
        """Test that only the first ClientConnectedFrame triggers the event."""
        observer = StartupTimingObserver()
        processor = FastProcessor()

        readiness_reports = []

        @observer.event_handler("on_transport_readiness_measured")
        async def on_readiness(obs, report):
            readiness_reports.append(report)

        frames_to_send = [
            ClientConnectedFrame(),
            ClientConnectedFrame(),
            TextFrame(text="hello"),
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=[ClientConnectedFrame, ClientConnectedFrame, TextFrame],
            observers=[observer],
        )

        self.assertEqual(len(readiness_reports), 1)

    async def test_transport_readiness_without_start_frame(self):
        """Test that ClientConnectedFrame before StartFrame does not crash."""
        observer = StartupTimingObserver()

        # Directly call on_push_frame with a ClientConnectedFrame before any
        # StartFrame has been seen. This should be a no-op (no crash).
        from pipecat.observers.base_observer import FramePushed

        processor = FastProcessor()
        destination = FastProcessor()
        data = FramePushed(
            source=processor,
            destination=destination,
            frame=ClientConnectedFrame(),
            direction=FrameDirection.DOWNSTREAM,
            timestamp=1000,
        )
        await observer.on_push_frame(data)

        # No event should have been emitted.
        self.assertFalse(observer._transport_readiness_measured)


if __name__ == "__main__":
    unittest.main()
