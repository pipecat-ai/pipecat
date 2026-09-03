import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pipecat.frames.frames import (
    BotConnectedFrame,
    ClientConnectedFrame,
    Frame,
    StartFrame,
    TextFrame,
)
from pipecat.observers import startup_timing_observer
from pipecat.observers.base_observer import (
    FrameProcessed,
    FramePushed,
    ProcessorSetUp,
    StartupWarmup,
)
from pipecat.observers.startup_timing_observer import (
    StartupTimingObserver,
    StartupTimingReport,
    TransportTimingReport,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import run_test
from pipecat.utils.asyncio.task_manager import TaskManager


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


class SlowSetupProcessor(FrameProcessor):
    """A processor that sleeps while being set up, as a connecting one does."""

    def __init__(self, delay: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self._delay = delay

    async def setup(self, setup):
        await super().setup(setup)
        await asyncio.sleep(self._delay)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class ConcurrentSetupProcessor(FrameProcessor):
    """A processor that records whether its setup overlapped with its peers'.

    The barrier releases only once every processor sharing it is inside
    ``setup()``, so setting up one at a time leaves each waiting for peers that
    never arrive.
    """

    OVERLAP_TIMEOUT_SECS = 1.0

    def __init__(self, barrier: asyncio.Barrier, timeout: float = OVERLAP_TIMEOUT_SECS, **kwargs):
        super().__init__(**kwargs)
        self._barrier = barrier
        self._timeout = timeout
        self.overlapped = False

    async def setup(self, setup):
        await super().setup(setup)
        try:
            await asyncio.wait_for(self._barrier.wait(), timeout=self._timeout)
            self.overlapped = True
        except (TimeoutError, asyncio.BrokenBarrierError):
            # A setup that raises is reported as a processor error and the
            # pipeline carries on, so a missed overlap has to reach the test
            # as a value to be asserted on.
            self.overlapped = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


def _ns(seconds: float) -> int:
    """Convert a point on a test timeline to the nanoseconds an event carries."""
    return int(seconds * 1_000_000_000)


class _FakeClock:
    """A clock the test moves by hand.

    The observer takes every timing it reports from the timestamps its events
    carry, so standing in for the two readings it makes of its own leaves a
    report that can be asserted exactly.
    """

    def __init__(self):
        self._now_ns = 0

    def set(self, seconds: float):
        self._now_ns = _ns(seconds)

    def as_time_module(self) -> SimpleNamespace:
        """Stand in for the ``time`` module the observer reads."""
        return SimpleNamespace(monotonic_ns=lambda: self._now_ns, time=lambda: 1_000.0)


def warming_for(duration_secs: float):
    """Build a stand-in for the framework's deferred-import warming.

    Warming is loaded once per process, so a test that used the real one would
    measure a full load or an already-cached no-op depending on what ran before
    it.
    """

    def warm():
        time.sleep(duration_secs)

    return warm


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

    async def test_setup_time_counts_towards_what_a_processor_cost(self):
        """A processor that connects while being set up is measured for it.

        Services connect during setup rather than while handling the
        StartFrame, so a report that only measured start() would show a fast
        startup for a pipeline that spent its time connecting.
        """
        observer = StartupTimingObserver()
        processor = SlowSetupProcessor(delay=0.1)

        reports = []

        @observer.event_handler("on_startup_timing_report")
        async def on_report(obs, report):
            reports.append(report)

        await run_test(
            processor,
            frames_to_send=[TextFrame(text="hello")],
            expected_down_frames=[TextFrame],
            observers=[observer],
        )

        timings = [
            t for t in reports[0].processor_timings if "SlowSetupProcessor" in t.processor_name
        ]
        self.assertEqual(len(timings), 1)
        timing = timings[0]

        self.assertGreaterEqual(timing.setup_duration_secs, 0.05)
        # Setting up is what this processor cost, and start() added nothing.
        self.assertGreaterEqual(timing.duration_secs, timing.setup_duration_secs)

    async def test_a_started_pipeline_reports_its_phases_and_warming(self):
        """The report a real pipeline produces covers both phases and warming.

        The barrier holds all three processors inside setup() together, so the
        concurrency the setup phase is built on holds however slowly the
        machine runs them.
        """
        observer = StartupTimingObserver()
        barrier = asyncio.Barrier(3)
        processors = [ConcurrentSetupProcessor(barrier) for _ in range(3)]

        reports = []

        @observer.event_handler("on_startup_timing_report")
        async def on_report(obs, report):
            reports.append(report)

        with patch("pipecat.pipeline.worker.warm_deferred_imports", warming_for(0.0)):
            await run_test(
                Pipeline(processors),
                frames_to_send=[TextFrame(text="hello")],
                expected_down_frames=[TextFrame],
                observers=[observer],
                # A setup that does not overlap spends a timeout per processor
                # before reaching the assertion below, which the default start
                # timeout would cut short and report as a failure to start.
                start_timeout=5.0,
            )

        self.assertTrue(
            all(p.overlapped for p in processors),
            "the three processors were not all inside setup() at once",
        )

        report = reports[0]
        self.assertEqual(
            len(
                [
                    t
                    for t in report.processor_timings
                    if "ConcurrentSetupProcessor" in t.processor_name
                ]
            ),
            3,
        )
        # Warming reaches the report from the worker that ran it, through the
        # observer the pipeline fans its events out to.
        self.assertIsNotNone(report.warmup)
        self.assertGreaterEqual(report.total_duration_secs, report.setup_phase_secs)

    async def test_phases_split_the_span_where_setting_up_ends(self):
        """Setting up ends with the last of the concurrent work, warming included.

        Three processors connect together and warming outlasts them, so the
        concurrent phase runs to warming and the StartFrame's trip through the
        pipeline is what follows.
        """
        clock = _FakeClock()
        with patch.object(startup_timing_observer, "time", clock.as_time_module()):
            observer = StartupTimingObserver()
            await observer.setup(TaskManager())
            reports = []

            @observer.event_handler("on_startup_timing_report")
            async def on_report(obs, report):
                reports.append(report)

            processors = [FastProcessor() for _ in range(3)]
            for processor in processors:
                await observer.on_processor_setup(
                    ProcessorSetUp(processor=processor, started_at_ns=0, finished_at_ns=_ns(0.2))
                )
            await observer.on_startup_warmup(
                StartupWarmup(started_at_ns=0, finished_at_ns=_ns(0.5))
            )
            await self._walk_start_frame(observer, processors, first_arrival=0.5, each=0.1)

            clock.set(0.9)
            await observer.on_pipeline_started()
            await asyncio.sleep(0)

        report = reports[0]
        self.assertAlmostEqual(report.total_duration_secs, 0.9, places=6)
        # Warming outlasted the processors, so it decides where setting up ends.
        self.assertAlmostEqual(report.setup_phase_secs, 0.5, places=6)
        # Everything after that is the frame's trip, the 0.3s the processors
        # spent on it and the 0.1s the pipeline spent carrying it between them.
        self.assertAlmostEqual(report.start_phase_secs, 0.4, places=6)
        self.assertAlmostEqual(report.warmup.duration_secs, 0.5, places=6)
        self.assertAlmostEqual(report.warmup.blocking_duration_secs, 0.3, places=6)

        # The frame reached them one after another, so their offsets step.
        for index, timing in enumerate(report.processor_timings):
            self.assertAlmostEqual(timing.setup_duration_secs, 0.2, places=6)
            self.assertAlmostEqual(timing.start_duration_secs, 0.1, places=6)
            self.assertAlmostEqual(timing.start_offset_secs, 0.1 * index, places=6)

    async def test_warming_a_slower_setup_hides_costs_nothing(self):
        """Warming runs alongside setup, so a pipeline slower to connect waits no longer."""
        clock = _FakeClock()
        with patch.object(startup_timing_observer, "time", clock.as_time_module()):
            observer = StartupTimingObserver()
            await observer.setup(TaskManager())
            reports = []

            @observer.event_handler("on_startup_timing_report")
            async def on_report(obs, report):
                reports.append(report)

            processor = FastProcessor()
            await observer.on_processor_setup(
                ProcessorSetUp(processor=processor, started_at_ns=0, finished_at_ns=_ns(0.5))
            )
            await observer.on_startup_warmup(
                StartupWarmup(started_at_ns=0, finished_at_ns=_ns(0.2))
            )
            await self._walk_start_frame(observer, [processor], first_arrival=0.5, each=0.1)

            clock.set(0.6)
            await observer.on_pipeline_started()
            await asyncio.sleep(0)

        report = reports[0]
        self.assertAlmostEqual(report.warmup.duration_secs, 0.2, places=6)
        self.assertEqual(report.warmup.blocking_duration_secs, 0.0)
        # Connecting outlasted warming, so it alone decides the phase.
        self.assertAlmostEqual(report.setup_phase_secs, 0.5, places=6)

    async def _walk_start_frame(self, observer, processors, *, first_arrival, each):
        """Send a StartFrame through the processors one after another."""
        frame = StartFrame()
        arrival = first_arrival
        for processor in processors:
            await observer.on_process_frame(
                FrameProcessed(
                    processor=processor,
                    frame=frame,
                    direction=FrameDirection.DOWNSTREAM,
                    timestamp=_ns(arrival),
                )
            )
            arrival += each
            await observer.on_push_frame(
                FramePushed(
                    source=processor,
                    destination=processor,
                    frame=frame,
                    direction=FrameDirection.DOWNSTREAM,
                    timestamp=_ns(arrival),
                )
            )

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
        self.assertGreater(report.start_time, 0)
        for timing in report.processor_timings:
            self.assertIsInstance(timing.processor_name, str)
            self.assertIsInstance(timing.duration_secs, float)
            self.assertGreaterEqual(timing.start_offset_secs, 0)

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
        internal_names = ("Pipeline#", "PipelineWorker#")
        for t in report.processor_timings:
            for prefix in internal_names:
                self.assertNotIn(
                    prefix,
                    t.processor_name,
                    f"Internal processor {t.processor_name} should be excluded by default",
                )

    async def test_transport_timing_client_only(self):
        """Test that ClientConnectedFrame emits on_transport_timing_report."""
        observer = StartupTimingObserver()
        processor = FastProcessor()

        transport_reports = []

        @observer.event_handler("on_transport_timing_report")
        async def on_transport(obs, report):
            transport_reports.append(report)

        frames_to_send = [ClientConnectedFrame(), TextFrame(text="hello")]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=[ClientConnectedFrame, TextFrame],
            observers=[observer],
        )

        self.assertEqual(len(transport_reports), 1)
        report = transport_reports[0]
        self.assertIsInstance(report, TransportTimingReport)
        self.assertGreater(report.start_time, 0)
        self.assertGreater(report.client_connected_secs, 0)
        self.assertIsNone(report.bot_connected_secs)

    async def test_transport_timing_only_first_client(self):
        """Test that only the first ClientConnectedFrame triggers the event."""
        observer = StartupTimingObserver()
        processor = FastProcessor()

        transport_reports = []

        @observer.event_handler("on_transport_timing_report")
        async def on_transport(obs, report):
            transport_reports.append(report)

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

        self.assertEqual(len(transport_reports), 1)

    async def test_transport_timing_before_the_start_frame(self):
        """A client that connects before the StartFrame is still measured.

        A transport connects while it is being set up, so it can report a
        connection before the StartFrame is pushed. Timings run from the
        pipeline starting to set up, so there is nothing to wait for.
        """
        observer = StartupTimingObserver()
        await observer.setup(TaskManager())

        reports = []

        @observer.event_handler("on_transport_timing_report")
        async def on_report(obs, report):
            reports.append(report)

        processor = FastProcessor()
        data = FramePushed(
            source=processor,
            destination=FastProcessor(),
            frame=ClientConnectedFrame(),
            direction=FrameDirection.DOWNSTREAM,
            timestamp=250_000_000,
        )
        await observer.on_push_frame(data)
        await asyncio.sleep(0)

        self.assertTrue(observer._transport_timing_reported)
        self.assertEqual(len(reports), 1)
        self.assertAlmostEqual(reports[0].client_connected_secs, 0.25, places=3)

    async def test_bot_and_client_connected(self):
        """Test that BotConnectedFrame timing is included in the transport report."""
        observer = StartupTimingObserver()
        processor = FastProcessor()

        transport_reports = []

        @observer.event_handler("on_transport_timing_report")
        async def on_transport(obs, report):
            transport_reports.append(report)

        frames_to_send = [
            BotConnectedFrame(),
            ClientConnectedFrame(),
            TextFrame(text="hello"),
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=[BotConnectedFrame, ClientConnectedFrame, TextFrame],
            observers=[observer],
        )

        self.assertEqual(len(transport_reports), 1)
        report = transport_reports[0]
        self.assertGreater(report.client_connected_secs, 0)
        self.assertIsNotNone(report.bot_connected_secs)
        self.assertGreater(report.bot_connected_secs, 0)

        # Client connected should be >= bot connected.
        self.assertGreaterEqual(report.client_connected_secs, report.bot_connected_secs)

    async def test_bot_connected_only_first(self):
        """Test that only the first BotConnectedFrame is recorded."""
        observer = StartupTimingObserver()
        processor = FastProcessor()

        transport_reports = []

        @observer.event_handler("on_transport_timing_report")
        async def on_transport(obs, report):
            transport_reports.append(report)

        frames_to_send = [
            BotConnectedFrame(),
            BotConnectedFrame(),
            ClientConnectedFrame(),
            TextFrame(text="hello"),
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=[
                BotConnectedFrame,
                BotConnectedFrame,
                ClientConnectedFrame,
                TextFrame,
            ],
            observers=[observer],
        )

        # Only one transport report, with bot timing from first frame.
        self.assertEqual(len(transport_reports), 1)
        self.assertIsNotNone(transport_reports[0].bot_connected_secs)


if __name__ == "__main__":
    unittest.main()
