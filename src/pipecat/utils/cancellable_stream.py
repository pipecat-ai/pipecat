"""Provides a wrapper class for making async streams cancellable.

This module implements a CancellableStream class that wraps any async iterator,
adding the ability to safely cancel iteration at any point. This is particularly
useful for handling cleanup of async streams in scenarios where early termination
is required.

The module provides functionality for:
- Wrapping any async iterator with cancellation capabilities
- Safe termination of async streams
- Proper cleanup and synchronization during cancellation
"""

import asyncio
from typing import AsyncIterator, Generic, TypeVar

T = TypeVar("T")


class CancellableStream(Generic[T]):
    """A wrapper around an async stream that can be cancelled."""

    def __init__(self, stream: AsyncIterator[T]) -> None:
        """Initialize a cancellable stream wrapper.

        Creates a wrapper around an async iterator that can be cancelled mid-iteration.
        The wrapper maintains state about cancellation requests and iteration status.

        Args:
            stream: The async iterator to wrap. This is the source stream that will
                   be iterated over until either exhaustion or cancellation.

        Attributes:
            _stream: The wrapped async iterator
            _cancel_future: A future that completes when cancellation is acknowledged
            _cancel_requested: Flag indicating if cancellation has been requested
            _iter_started: Flag indicating if iteration has begun
        """
        self._stream: AsyncIterator[T] = stream
        self._cancel_future: asyncio.Future[None] | None = None
        self._cancel_requested: bool = False
        self._iter_started: bool = False

    async def cancel(self) -> None:
        """Request stream cancellation and wait for acknowledgment.

        Sets up cancellation state and waits until the cancellation is acknowledged
        by the next iteration attempt. If iteration hasn't started yet, the
        cancellation is acknowledged immediately.

        The method will:
        1. Create a cancellation future if one doesn't exist
        2. Mark the stream as cancelled
        3. If iteration hasn't started, complete the future immediately
        4. Otherwise, wait for the next iteration to acknowledge cancellation

        Returns:
            None: The method returns when cancellation is acknowledged.
        """
        if self._cancel_future is None:
            self._cancel_future = asyncio.get_event_loop().create_future()
        self._cancel_requested = True

        # If iteration has not started, we complete the future immediately
        if not self._iter_started:
            self._cancel_future.set_result(None)

        await self._cancel_future

    def __aiter__(self) -> "CancellableStream[T]":
        self._iter_started = True
        return self

    async def __anext__(self) -> T:
        if self._cancel_requested:
            # Complete the future if cancellation was requested
            if self._cancel_future and not self._cancel_future.done():
                self._cancel_future.set_result(None)
            raise StopAsyncIteration

        try:
            return await self._stream.__anext__()
        except StopAsyncIteration:
            # also complete cancel future if iteration naturally ends
            if self._cancel_future and not self._cancel_future.done():
                self._cancel_future.set_result(None)
            raise
