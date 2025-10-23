import time

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.base_llm import LLMService
from pipecat.transports.base_output import BaseOutputTransport


class TurnDetectorObserver(BaseObserver):
    """Observer ... of turns."""

    def __init__(self):
        super().__init__()

        self._turn_observer = None
        self._arrow = "→"

        self._turn_number = 1
        self._endframe_queued = False

    def init(self):
        """
        Set ...
        """
        pass

    def set_turn_observer_event_handlers(self, turn_observer):
        self._turn_observer = turn_observer
        self.set_turn_observer_event_handlers(self._turn_observer)

    def get_turn_observer(self):
        return self._turn_observer

    def set_turn_observer_event_handlers(self, turn_observer):
        """Sets the Turn Observer event handlers `on_turn_started` and `on_turn_ended`.

        Args:
            turn_observer: The turn tracking observer of the pipeline task
        """

        @turn_observer.event_handler("on_turn_started")
        async def on_turn_started(observer, turn_number):
            self._turn_number = turn_number
            current_time = time.time()
            logger.info(f"🔄 Turn {turn_number} started")

            # 🫆🫆🫆🫆
            # code to start conversation turn here
            # 🫆🫆🫆🫆
            # 🫆🫆🫆🫆
            # 🫆🫆🫆🫆

        @turn_observer.event_handler("on_turn_ended")
        async def on_turn_ended(observer, turn_number, duration, was_interrupted):
            current_time = time.time()

            if was_interrupted:
                logger.info(f"🔄 Turn {turn_number} interrupted after {duration:.2f}s")
            else:
                logger.info(f"🏁 Turn {turn_number} completed in {duration:.2f}s")

            # 🫆🫆🫆🫆
            # code to end conversation turn here
            # 🫆🫆🫆🫆
            # 🫆🫆🫆🫆
            # 🫆🫆🫆🫆

    ########
    # everything past here isn't needed, just nice to have logging
    ########
    async def on_push_frame(self, data: FramePushed):
        """Runs when any frame is pushed through pipeline.
        Determines based on what type of frame and where it came from
        what metrics to update.

        Args:
            data: the pushed frame
        """
        src = data.source
        dst = data.destination
        frame = data.frame
        direction = data.direction
        timestamp = data.timestamp

        # Convert timestamp to milliseconds for readability
        time_sec = timestamp / 1_000_000
        # Convert timestamp to seconds for readability
        # time_sec = timestamp / 1_000_000_000

        # only log downstream frames
        if direction == FrameDirection.UPSTREAM:
            return

        if isinstance(src, Pipeline) or isinstance(dst, Pipeline):
            if isinstance(frame, StartFrame):
                self._handle_StartFrame(src, dst, frame, time_sec)
            elif isinstance(frame, EndFrame):
                self._handle_EndFrame(src, dst, frame, time_sec)

        if isinstance(src, BaseOutputTransport):
            if isinstance(frame, BotStartedSpeakingFrame):
                self._handle_BotStartedSpeakingFrame(src, dst, frame, time_sec)
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._handle_BotStoppedSpeakingFrame(src, dst, frame, time_sec)

            elif isinstance(frame, UserStartedSpeakingFrame):
                self._handle_UserStartedSpeakingFrame(src, dst, frame, time_sec)
            elif isinstance(frame, UserStoppedSpeakingFrame):
                self._handle_UserStoppedSpeakingFrame(src, dst, frame, time_sec)

        if isinstance(src, LLMService):
            if isinstance(frame, LLMFullResponseStartFrame):
                self._handle_LLMFullResponseStartFrame(src, dst, frame, time_sec)
            elif isinstance(frame, LLMFullResponseEndFrame):
                self._handle_LLMFullResponseEndFrame(src, dst, frame, time_sec)
            elif isinstance(frame, FunctionCallsStartedFrame):
                self._handle_FunctionCallsStartedFrame(src, dst, frame, time_sec)
            elif isinstance(frame, FunctionCallResultFrame):
                self._handle_FunctionCallResultFrame(src, dst, frame, time_sec)

    # ------------ FRAME HANDLERS ------------

    def _handle_StartFrame(self, src, dst, frame, time_sec):
        if isinstance(dst, Pipeline):
            logger.info(f"🟢🟢🟢 StartFrame: {src} {self._arrow} {dst} at {time_sec:.2f}s")

    def _handle_EndFrame(self, src, dst, frame, time_sec):
        if isinstance(dst, Pipeline):
            logger.info(f"Queueing 🔴🔴🔴 EndFrame: {src} {self._arrow} {dst} at {time_sec:.2f}s")
            self._endframe_queued = True

        if isinstance(src, Pipeline):
            logger.info(f"🔴🔴🔴 EndFrame: {src} {self._arrow} {dst} at {time_sec:.2f}s")

            current_time = time.time()
            end_state_info = {
                "turn_number": self._turn_number,
            }

    def _handle_BotStartedSpeakingFrame(self, src, dst, frame, time_sec):
        logger.info(f"🤖🟢 BotStartedSpeakingFrame: {src} {self._arrow} {dst} at {time_sec:.2f}s")

    def _handle_BotStoppedSpeakingFrame(self, src, dst, frame, time_sec):
        logger.info(f"🤖🔴 BotStoppedSpeakingFrame: {src} {self._arrow} {dst} at {time_sec:.2f}s")

    def _handle_LLMFullResponseStartFrame(self, src, dst, frame, time_sec):
        logger.info(f"🧠🟢 LLMFullResponseStartFrame: {src} {self._arrow} {dst} at {time_sec:.2f}s")

    def _handle_LLMFullResponseEndFrame(self, src, dst, frame, time_sec):
        logger.info(f"🧠🔴 LLMFullResponseEndFrame: {src} {self._arrow} {dst} at {time_sec:.2f}s")

    def _handle_UserStartedSpeakingFrame(self, src, dst, frame, time_sec):
        logger.info(f"🙂🟢 UserStartedSpeakingFrame: {src} {self._arrow} {dst} at {time_sec:.2f}s")

    def _handle_UserStoppedSpeakingFrame(self, src, dst, frame, time_sec):
        logger.info(f"🙂🔴 UserStoppedSpeakingFrame: {src} {self._arrow} {dst} at {time_sec:.2f}s")

    def _handle_FunctionCallsStartedFrame(self, src, dst, frame, time_sec):
        logger.info(
            f"📐🟢 {frame.function_calls[0].function_name} FunctionCallsStartedFrame: {src} {self._arrow} {dst} at {time_sec:.2f}s"
        )

    def _handle_FunctionCallResultFrame(self, src, dst, frame, time_sec):
        logger.info(
            f"📐🔴 {frame.function_name} FunctionCallResultFrame: {src} {self._arrow} {dst} at {time_sec:.2f}s"
        )
