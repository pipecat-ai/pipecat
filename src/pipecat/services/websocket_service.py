#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Optional

import websockets
from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import ErrorFrame
from pipecat.utils.network import exponential_backoff_time


class WebsocketService(ABC):
    """Base class for websocket-based services with reconnection logic."""

    def __init__(self, *, reconnect_on_error: bool = True, **kwargs):
        """Initialize websocket attributes."""
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._reconnect_on_error = reconnect_on_error

    async def _verify_connection(self) -> bool:
        """Verify websocket connection is working.

        Returns:
            bool: True if connection is verified working, False otherwise
        """
        try:
            if not self._websocket:
                return False
            await self._websocket.ping()
            return True
        except Exception as e:
            logger.error(f"{self} connection verification failed: {e}")
            return False

    async def _reconnect_websocket(self, attempt_number: int) -> bool:
        """Reconnect the websocket.

        Args:
            attempt_number: Current retry attempt number

        Returns:
            bool: True if reconnection and verification successful, False otherwise
        """
        logger.warning(f"{self} reconnecting (attempt: {attempt_number})")
        await self._disconnect_websocket()
        await self._connect_websocket()
        return await self._verify_connection()

    async def _receive_task_handler(self, report_error: Callable[[ErrorFrame], Awaitable[None]]):
        """Handles WebSocket message receiving with automatic retry logic.

        Args:
            report_error: Callback to report errors
        """
        retry_count = 0
        MAX_RETRIES = 3

        while True:
            try:
                await self._receive_messages()
                retry_count = 0  # Reset counter on successful message receive
                if self._websocket and self._websocket.state == State.CLOSED:
                    raise websockets.ConnectionClosedOK(
                        self._websocket.close_rcvd,
                        self._websocket.close_sent,
                        self._websocket.close_rcvd_then_sent,
                    )
            except Exception as e:
                message = f"{self} error receiving messages: {e}"
                logger.error(message)

                if self._reconnect_on_error:
                    retry_count += 1
                    if retry_count >= MAX_RETRIES:
                        await report_error(ErrorFrame(message, fatal=True))
                        break

                    logger.warning(f"{self} connection error, will retry: {e}")
                    await report_error(ErrorFrame(message))

                    try:
                        if await self._reconnect_websocket(retry_count):
                            retry_count = 0  # Reset counter on successful reconnection
                        wait_time = exponential_backoff_time(retry_count)
                        await asyncio.sleep(wait_time)
                    except Exception as reconnect_error:
                        logger.error(f"{self} reconnection failed: {reconnect_error}")
                else:
                    await report_error(ErrorFrame(message))
                    break

    @abstractmethod
    async def _connect(self):
        """Implement service-specific connection logic. This function will
        connect to the websocket via _connect_websocket() among other connection
        logic."""
        pass

    @abstractmethod
    async def _disconnect(self):
        """Implement service-specific disconnection logic. This function will
        disconnect to the websocket via _connect_websocket() among other
        connection logic.

        """
        pass

    @abstractmethod
    async def _connect_websocket(self):
        """Implement service-specific websocket connection logic. This function
        should only connect to the websocket."""
        pass

    @abstractmethod
    async def _disconnect_websocket(self):
        """Implement service-specific websocket disconnection logic. This
        function should only disconnect from the websocket."""
        pass

    @abstractmethod
    async def _receive_messages(self):
        """Implement service-specific message receiving logic."""
        pass
