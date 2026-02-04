#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base websocket service with automatic reconnection and error handling."""

import asyncio
from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Optional

import websockets
from loguru import logger
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
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
        self._reconnect_in_progress: bool = False
        self._disconnecting: bool = False

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

    async def _try_reconnect(
        self,
        max_retries: int = 3,
        report_error: Optional[Callable[[ErrorFrame], Awaitable[None]]] = None,
    ) -> bool:
        # Prevent concurrent reconnection attempts
        if self._reconnect_in_progress:
            logger.warning(f"{self} reconnect attempt aborted: already in progress")
            return False

        self._reconnect_in_progress = True
        last_exception: Optional[Exception] = None
        try:
            for attempt in range(1, max_retries + 1):
                try:
                    logger.warning(f"{self} reconnecting, attempt {attempt}")
                    if await self._reconnect_websocket(attempt):
                        logger.info(f"{self} reconnected successfully on attempt {attempt}")
                        return True
                except Exception as e:
                    last_exception = e
                    logger.error(f"{self} reconnection attempt {attempt} failed: {e}")
                    if report_error:
                        await report_error(
                            ErrorFrame(f"{self} reconnection attempt {attempt} failed: {e}")
                        )
                wait_time = exponential_backoff_time(attempt)
                await asyncio.sleep(wait_time)
            fatal_msg = f"{self} failed to reconnect after {max_retries} attempts"
            if last_exception:
                fatal_msg += f": {last_exception}"
            logger.error(fatal_msg)
            if report_error:
                await report_error(ErrorFrame(fatal_msg, fatal=True))
            return False
        finally:
            self._reconnect_in_progress = False

    async def send_with_retry(self, message, report_error: Callable[[ErrorFrame], Awaitable[None]]):
        """Attempt to send a message, retrying after reconnect if necessary."""
        try:
            await self._websocket.send(message)
        except Exception as e:
            logger.error(f"{self} send failed: {e}, will try to reconnect")
            # Try to reconnect before retrying
            success = await self._try_reconnect(report_error=report_error)
            if success:
                logger.info(f"{self} reconnected successfully, will retry send the message")
                # trying to send the message one more time
                await self._websocket.send(message)
            else:
                logger.error(f"{self} send failed; unable to reconnect")

    async def _maybe_try_reconnect(
        self,
        error_message: str,
        report_error: Callable[[ErrorFrame], Awaitable[None]],
        error: Optional[Exception] = None,
    ) -> bool:
        """Check if reconnection should be attempted and try if appropriate.

        Args:
            error_message: Human-readable error message for logging.
            report_error: Callback function to report connection errors.
            error: The exception that occurred (optional, may be None for graceful closes).

        Returns:
            True if should continue the receive loop, False if should break.
        """
        # Don't reconnect if we're intentionally disconnecting
        if self._disconnecting:
            if error:
                logger.warning(f"{self} error during disconnect: {error}")
            else:
                logger.debug(f"{self} receive loop ended during disconnect")
            return False

        # Log the message
        logger.warning(error_message)

        # Try to reconnect if enabled
        if self._reconnect_on_error:
            success = await self._try_reconnect(report_error=report_error)
            return success
        else:
            # Reconnection disabled
            await report_error(ErrorFrame(error_message))
            return False

    async def _receive_task_handler(self, report_error: Callable[[ErrorFrame], Awaitable[None]]):
        """Handle websocket message receiving with automatic retry logic.

        Continuously receives messages with automatic reconnection on errors.
        Uses exponential backoff between retry attempts and reports fatal errors
        after maximum retries are exhausted.

        Args:
            report_error: Callback function to report connection errors.
        """
        while True:
            try:
                await self._receive_messages()
                # _receive_messages() returned normally. This happens when the websocket
                # closes gracefully (server sent close frame). The async for loop over
                # the websocket exits without raising an exception in this case.
                # We must handle this to avoid an infinite loop.
                message = f"{self} connection closed by server"
                should_continue = await self._maybe_try_reconnect(message, report_error)
                if not should_continue:
                    break
            except ConnectionClosedOK as e:
                # Normal closure, don't retry
                logger.debug(f"{self} connection closed normally: {e}")
                break
            except ConnectionClosedError as e:
                # Connection closed with error (e.g., no close frame received/sent)
                # This often indicates network issues, server problems, or abrupt disconnection
                message = f"{self} connection closed, but with an error: {e}"
                should_continue = await self._maybe_try_reconnect(message, report_error, e)
                if not should_continue:
                    break
            except Exception as e:
                # General error during message receiving
                message = f"{self} error receiving messages: {e}"
                should_continue = await self._maybe_try_reconnect(message, report_error, e)
                if not should_continue:
                    break

    async def _connect(self):
        """Connect to the service and reset disconnecting flag.

        Manages the disconnecting flag to enable reconnection. Subclasses should
        call super()._connect() first, then implement their specific connection
        logic including websocket connection via _connect_websocket() and any
        additional setup required.
        """
        self._disconnecting = False

    async def _disconnect(self):
        """Disconnect from the service and set disconnecting flag.

        Manages the disconnecting flag to prevent reconnection during intentional
        disconnect. Subclasses should call super()._disconnect() first, then
        implement their specific disconnection logic including websocket
        disconnection via _disconnect_websocket() and any cleanup required.
        """
        self._disconnecting = True

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
