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

from pipecat.frames.frames import ErrorFrame


class WebsocketService(ABC):
    """Base class for websocket-based services with reconnection logic."""

    def __init__(self):
        """Initialize websocket attributes."""
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None

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

    def _calculate_wait_time(
        self, attempt: int, min_wait: float = 4, max_wait: float = 10, multiplier: float = 1
    ) -> float:
        """Calculate exponential backoff wait time.

        Args:
            attempt: Current attempt number (1-based)
            min_wait: Minimum wait time in seconds
            max_wait: Maximum wait time in seconds
            multiplier: Base multiplier for exponential calculation

        Returns:
            Wait time in seconds
        """
        try:
            exp = 2 ** (attempt - 1) * multiplier
            result = max(0, min(exp, max_wait))
            return max(min_wait, result)
        except (ValueError, ArithmeticError):
            return max_wait

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
                logger.debug(f"{self} connection established successfully")
                retry_count = 0  # Reset counter on successful message receive

            except asyncio.CancelledError:
                break

            except Exception as e:
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    message = f"{self} error receiving messages: {e}"
                    logger.error(message)
                    await report_error(ErrorFrame(message, fatal=True))
                    break

                logger.warning(f"{self} connection error, will retry: {e}")

                try:
                    if await self._reconnect_websocket(retry_count):
                        retry_count = 0  # Reset counter on successful reconnection
                    wait_time = self._calculate_wait_time(retry_count)
                    await asyncio.sleep(wait_time)
                except Exception as reconnect_error:
                    logger.error(f"{self} reconnection failed: {reconnect_error}")
                    continue

    @abstractmethod
    async def _connect_websocket(self):
        """Implement service-specific websocket connection logic."""
        pass

    @abstractmethod
    async def _disconnect_websocket(self):
        """Implement service-specific websocket disconnection logic."""
        pass

    @abstractmethod
    async def _receive_messages(self):
        """Implement service-specific message receiving logic."""
        pass
