import asyncio

from pipecat.frames.frames import EndFrame, Frame, InputDTMFFrame, KeypadEntry, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.time import time_now_iso8601


class DTMFAggregator(FrameProcessor):
    """
    Aggregates DTMF frames using idle wait logic.

    The aggregator accumulates digits from incoming InputDTMFFrame instances.
    It flushes the aggregated digits by emitting a TranscriptionFrame when:
      - No new digit arrives within the specified timeout period, or
      - The termination digit (“#”) is received.
    """

    def __init__(self, timeout: float = 1.0, **kwargs):
        """
        :param timeout: Idle timeout in seconds before flushing the aggregated digits.
        """
        super().__init__(**kwargs)
        self._aggregation = ""
        self._idle_timeout = timeout
        self._digit_event = asyncio.Event()
        self._digit_aggregate_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # Handle DTMF frames.
        await super().process_frame(frame, direction)
        if isinstance(frame, InputDTMFFrame):
            # Start the digit aggregation task if it's not running yet.
            if self._digit_aggregate_task is None:
                self._digit_aggregate_task = self.create_task(self._digit_agg_handler(direction))

            # Append the incoming digit.
            self._aggregation += frame.button.value

            # If the digit is the termination character, flush immediately.
            if frame.button == KeypadEntry.POUND:
                await self.flush_aggregation(direction)
            else:
                # Signal the digit aggregation task that a new digit has arrived.
                self._digit_event.set()
        elif isinstance(frame, EndFrame):
            # For EndFrame, flush any pending aggregation and stop the digit aggregation task.
            if self._aggregation:
                await self.flush_aggregation(direction)
            await self.push_frame(frame, direction)
            if self._digit_aggregate_task:
                await self._stop_digit_aggregate_task()
        else:
            # For any other frame, simply pass it downstream.
            await self.push_frame(frame, direction)

    async def _digit_agg_handler(self, direction: FrameDirection):
        """
        Idle task that waits for new DTMF activity. If no new digit is received within
        the timeout period, the current aggregation is flushed.
        """
        while True:
            try:
                # Wait for a new digit signal with a timeout.
                await asyncio.wait_for(self._digit_event.wait(), timeout=self._idle_timeout)
            except asyncio.TimeoutError:
                # No new digit arrived within the timeout period; flush aggregation if non-empty.
                if self._aggregation:
                    await self.flush_aggregation(direction)
            finally:
                # Clear the event for the next cycle.
                self._digit_event.clear()

    async def flush_aggregation(self, direction: FrameDirection):
        """
        Flush the aggregated digits by emitting a TranscriptionFrame downstream.
        """
        if self._aggregation:
            # Todo: Change to different frame type if we decide to handle it in llm processor separately.
            aggregated_frame = TranscriptionFrame(self._aggregation, "", time_now_iso8601())
            await self.push_frame(aggregated_frame, direction)
            self._aggregation = ""

    async def _stop_digit_aggregate_task(self):
        """
        Cancels the digit aggregation task if it exists.
        """
        if self._digit_aggregate_task:
            await self.cancel_task(self._digit_aggregate_task)
            self._digit_aggregate_task = None

    async def cleanup(self) -> None:
        """
        Cleans up resources, ensuring that the digit aggregation task is cancelled.
        """
        await super().cleanup()
        if self._digit_aggregate_task:
            await self._stop_digit_aggregate_task()
