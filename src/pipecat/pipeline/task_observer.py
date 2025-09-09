#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Task observer for managing pipeline frame observers.

This module provides a proxy observer system that manages multiple observers
for pipeline frame events, ensuring that observer processing doesn't block
the main pipeline execution.
"""

import asyncio
import inspect
from typing import Any, Dict, List, Optional

from attr import dataclass

from pipecat.observers.base_observer import BaseObserver, FrameProcessed, FramePushed
from pipecat.utils.asyncio.task_manager import BaseTaskManager


@dataclass
class Proxy:
    """Proxy data for managing observer tasks and queues.

    This represents is the data received from the main observer that
    is queued for later processing.

    Parameters:
        queue: Queue for frame data awaiting observer processing.
        task: Asyncio task running the observer's frame processing loop.
        observer: The actual observer instance being proxied.
    """

    queue: asyncio.Queue
    task: asyncio.Task
    observer: BaseObserver


class TaskObserver(BaseObserver):
    """Proxy observer that manages multiple observers without blocking the pipeline.

    This is a pipeline frame observer that is meant to be used as a proxy to
    the user provided observers. That is, this is the observer that should be
    passed to the frame processors. Then, every time a frame is pushed this
    observer will call all the observers registered to the pipeline task.

    This observer makes sure that passing frames to observers doesn't block the
    pipeline by creating a queue and a task for each user observer. When a frame
    is received, it will be put in a queue for efficiency and later processed by
    each task.
    """

    def __init__(
        self,
        *,
        observers: Optional[List[BaseObserver]] = None,
        task_manager: BaseTaskManager,
        **kwargs,
    ):
        """Initialize the TaskObserver.

        Args:
            observers: List of observers to manage. Defaults to empty list.
            task_manager: Task manager for creating and managing observer tasks.
            **kwargs: Additional arguments passed to the base observer.
        """
        super().__init__(**kwargs)
        self._observers = observers or []
        self._task_manager = task_manager
        self._proxies: Optional[Dict[BaseObserver, Proxy]] = (
            None  # Becomes a dict after start() is called
        )

    def add_observer(self, observer: BaseObserver):
        """Add a new observer to the managed list.

        Args:
            observer: The observer to add.
        """
        # Add the observer to the list.
        self._observers.append(observer)

        # If we already started, create a new proxy for the observer.
        # Otherwise, it will be created in start().
        if self._proxies:
            proxy = self._create_proxy(observer)
            self._proxies[observer] = proxy

    async def remove_observer(self, observer: BaseObserver):
        """Remove an observer and clean up its resources.

        Args:
            observer: The observer to remove.
        """
        # If the observer has a proxy, remove it.
        if self._proxies and observer in self._proxies:
            proxy = self._proxies[observer]
            # Remove the proxy so it doesn't get called anymore.
            del self._proxies[observer]
            # Cancel the proxy task right away.
            await self._task_manager.cancel_task(proxy.task)

        # Remove the observer from the list.
        if observer in self._observers:
            self._observers.remove(observer)

    async def start(self):
        """Start all proxy observer tasks."""
        self._proxies = self._create_proxies(self._observers)

    async def stop(self):
        """Stop all proxy observer tasks."""
        if not self._proxies:
            return

        for proxy in self._proxies.values():
            await self._task_manager.cancel_task(proxy.task)

    async def cleanup(self):
        """Cleanup all proxy observers."""
        await super().cleanup()

        if not self._proxies:
            return

        for proxy in self._proxies:
            await proxy.cleanup()

    async def on_process_frame(self, data: FramePushed):
        """Queue frame data for all managed observers.

        Args:
            data: The frame push event data to distribute to observers.
        """
        await self._send_to_proxy(data)

    async def on_push_frame(self, data: FramePushed):
        """Queue frame data for all managed observers.

        Args:
            data: The frame push event data to distribute to observers.
        """
        await self._send_to_proxy(data)

    def _create_proxy(self, observer: BaseObserver) -> Proxy:
        """Create a proxy for a single observer."""
        queue = asyncio.Queue()
        task = self._task_manager.create_task(
            self._proxy_task_handler(queue, observer),
            f"TaskObserver::{observer}::_proxy_task_handler",
        )
        proxy = Proxy(queue=queue, task=task, observer=observer)
        return proxy

    def _create_proxies(self, observers: List[BaseObserver]) -> Dict[BaseObserver, Proxy]:
        """Create proxies for all observers."""
        proxies = {}
        for observer in observers:
            proxy = self._create_proxy(observer)
            proxies[observer] = proxy
        return proxies

    async def _send_to_proxy(self, data: Any):
        for proxy in self._proxies.values():
            await proxy.queue.put(data)

    async def _proxy_task_handler(self, queue: asyncio.Queue, observer: BaseObserver):
        """Handle frame processing for a single observer."""
        on_push_frame_deprecated = False
        signature = inspect.signature(observer.on_push_frame)
        if len(signature.parameters) > 1:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Observer `on_push_frame(source, destination, frame, direction, timestamp)` is deprecated, us `on_push_frame(data: FramePushed)` instead.",
                    DeprecationWarning,
                )

            on_push_frame_deprecated = True

        while True:
            data = await queue.get()

            if isinstance(data, FramePushed):
                if on_push_frame_deprecated:
                    await observer.on_push_frame(
                        data.src, data.dst, data.frame, data.direction, data.timestamp
                    )
                else:
                    await observer.on_push_frame(data)
            elif isinstance(data, FrameProcessed):
                await observer.on_process_frame(data)

            queue.task_done()
