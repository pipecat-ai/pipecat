#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base websocket service with automatic reconnection and error handling."""

import asyncio
from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Optional

import websockets
from loguru import logger
from websockets.exceptions import ConnectionClosedOK
from websockets.protocol import State

from pipecat.frames.frames import ErrorFrame
from pipecat.utils.network import exponential_backoff_time


class WebsocketService(ABC):
    """Base class for websocket-based services with automatic reconnection.

    Provides websocket connection management, automatic reconnection with
    exponential backoff, connection verification, and error handling.
    Subclasses implement service-specific connection and message handling logic.
    """

    def __init__(self, *, reconnect_on_error: bool = True, **kwargs):
        """Initialize the websocket service.

        Args:
            reconnect_on_error: Whether to automatically reconnect on connection errors.
            **kwargs: Additional arguments (unused, for compatibility).
        """
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._reconnect_on_error = reconnect_on_error

    async def _verify_connection(self) -> bool:
        """Verify the websocket connection is active and responsive.

        Returns:
            True if connection is verified working, False otherwise.
        """
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                return False
            await self._websocket.ping()
            return True
        except Exception as e:
            logger.error(f"{self} connection verification failed: {e}")
            return False

    async def _reconnect_websocket(self, attempt_number: int) -> bool:
        """Reconnect the websocket with the current attempt number.

        Args:
            attempt_number: Current retry attempt number for logging.

        Returns:
            True if reconnection and verification successful, False otherwise.
        """
        logger.warning(f"{self} reconnecting (attempt: {attempt_number})")
        await self._disconnect_websocket()
        await self._connect_websocket()
        return await self._verify_connection()

    async def _receive_task_handler(self, report_error: Callable[[ErrorFrame], Awaitable[None]]):
        """Handle websocket message receiving with automatic retry logic.

        Continuously receives messages with automatic reconnection on errors.
        Uses exponential backoff between retry attempts and reports fatal errors
        after maximum retries are exhausted.

        Args:
            report_error: Callback function to report connection errors.
        """
        retry_count = 0
        MAX_RETRIES = 3

        while True:
            try:
                await self._receive_messages()
                retry_count = 0  # Reset counter on successful message receive
            except ConnectionClosedOK as e:
                # Normal closure, don't retry
                logger.debug(f"{self} connection closed normally: {e}")
                break
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
        """Connect to the service.

        Implement service-specific connection logic including websocket connection
        via _connect_websocket() and any additional setup required.
        """
        pass

    @abstractmethod
    async def _disconnect(self):
        """Disconnect from the service.

        Implement service-specific disconnection logic including websocket
        disconnection via _disconnect_websocket() and any cleanup required.
        """
        pass

    @abstractmethod
    async def _connect_websocket(self):
        """Establish the websocket connection.

        Implement the low-level websocket connection logic specific to the service.
        Should only handle websocket connection, not additional service setup.
        """
        pass

    @abstractmethod
    async def _disconnect_websocket(self):
        """Close the websocket connection.

        Implement the low-level websocket disconnection logic specific to the service.
        Should only handle websocket disconnection, not additional service cleanup.
        """
        pass

    @abstractmethod
    async def _receive_messages(self):
        """Receive and process websocket messages.

        Implement service-specific logic for receiving and handling messages
        from the websocket connection. Called continuously by the receive task handler.
        """
        pass
