import time

from loguru import logger

from pipecat.frames.frames import BotStartedSpeakingFrame, Frame, UserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame


class LatencyProcessor(FrameProcessor):
    """Measure user-stop → bot-start and send it as metadata to the client."""

    def __init__(self, transport, rtvi):
        super().__init__()
        self.transport = transport
        self.rtvi = rtvi
        self._user_stop_ts: float | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # record when user stops speaking
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStoppedSpeakingFrame):
            self._user_stop_ts = time.monotonic()

        # on bot start, compute & ship metadata
        elif isinstance(frame, BotStartedSpeakingFrame) and self._user_stop_ts is not None:
            now = time.monotonic()
            latency_ms = (now - self._user_stop_ts) * 1000
            payload = {"latency_ms": latency_ms}
            logger.debug(f"[Latency] injecting RTVIServerMessageFrame {payload}")
            await self.push_frame(RTVIServerMessageFrame(data=payload))
            self._user_stop_ts = None

        # always pass frames through

        await self.push_frame(frame, direction)
