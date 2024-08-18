#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json

from pipecat.frames.frames import Frame, StartInterruptionFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from loguru import logger

try:
    from fastapi import WebSocket
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use FastAPI websockets, you need to `pip install pipecat-ai[websocket]`.")
    raise Exception(f"Missing module: {e}")


class TwilioProcessor(FrameProcessor):
    def __init__(self, websocket_client: WebSocket, stream_sid: str):
        super().__init__()
        self.websocket_client = websocket_client
        self.stream_sid = stream_sid

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await self.push_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            answer = {"event": "clear", "streamSid": self.stream_sid}
            await self.websocket_client.send_text(json.dumps(answer))
