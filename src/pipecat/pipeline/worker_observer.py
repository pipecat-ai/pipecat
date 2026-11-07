#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Worker observer for managing pipeline frame observers.

This module provides a proxy observer system that manages multiple observers
for pipeline frame events, ensuring that observer processing doesn't block
the main pipeline execution.
"""

import asyncio
import traceback
import weakref
from typing import Any

from attr import dataclass
from loguru import logger

from pipecat.frames.frames import Frame
from pipecat.observers.base_observer import (
    BaseObserver,
    FrameProcessed,
    FramePushed,
    ProcessorSetUp,
    StartupWarmup,
)
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.deprecation import deprecated


def _handles_processing(observer: BaseObserver) -> bool:
    """Whether the observer overrides ``on_process_frame``, on its class or on the instance."""
    return (
        "on_process_frame" in getattr(observer, "__dict__", ())
        or getattr(type(observer), "on_process_frame", None) is not BaseObserver.on_process_frame
    )


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


class _PipelineStartedSignal:
    """Internal sentinel queued to observers when the pipeline has started."""

    pass


class WorkerObserver(BaseObserver):
    """Proxy observer that manages multiple observers without blocking the pipeline.

    This is a pipeline frame observer that is meant to be used as a proxy to
    the user provided observers. That is, this is the observer that should be
    passed to the frame processors. Then, every time a frame is pushed this
    observer will call all the observers registered to the pipeline worker.

    This observer makes sure that passing frames to observers doesn't block the
    pipeline by creating a queue and a worker for each user observer. When a frame
    is received, it will be put in a queue for efficiency and later processed by
    each worker.

    It also tells a frame's first push from the ones that follow, and only
    passes the first one to observers that want a frame once.

    A frame being processed is only passed to the observers that override
    :meth:`~pipecat.observers.base_observer.BaseObserver.on_process_frame`,
    on their class or on the instance.

    A frame is only passed to the observers that observe its class (see
    :meth:`~pipecat.observers.base_observer.BaseObserver.observes_frame_type`).
    """

    def __init__(
        self,
        *,
        observers: list[BaseObserver] | None = None,
        **kwargs,
    ):
        """Initialize the WorkerObserver.

        Args:
            observers: List of observers to manage. Defaults to empty list.
            **kwargs: Additional arguments passed to the base observer.
        """
        super().__init__(**kwargs)
        self._observers = observers or []
        self._proxies: dict[BaseObserver, Proxy] | None = (
            None  # Becomes a dict after setup() is called
        )
        # Frames pushed so far, held weakly: an entry goes away with its
        # frame, so this tracks the frames in flight, not every frame ever
        # pushed.
        self._frames_pushed: weakref.WeakValueDictionary[int, Frame] = weakref.WeakValueDictionary()
        # The proxies that observe each frame class, worked out when the first
        # frame of the class arrives and forgotten when the observers change.
        self._frame_targets: dict[type[Frame], list[Proxy]] = {}

    def add_observer(self, observer: BaseObserver):
        """Add a new observer to the managed list.

        Args:
            observer: The observer to add.
        """
        # Add the observer to the list.
        self._observers.append(observer)

        # After setup(), create the proxy immediately; setup() handles earlier additions.
        if self._proxies is not None:
            proxy = self._create_proxy(observer)
            self._proxies[observer] = proxy
            self._frame_targets.clear()

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
            self._frame_targets.clear()
            # Cancel the proxy worker right away. An observer removing itself
            # runs on that worker, which can't be cancelled from inside, so
            # the worker ends itself instead.
            if proxy.task is not asyncio.current_task():
                await self.cancel_task(proxy.task)
            # Shutdown only cleans up the observers in _proxies.
            await observer.cleanup()

        # Remove the observer from the list.
        if observer in self._observers:
            self._observers.remove(observer)

    async def setup(self, task_manager: BaseTaskManager):
        """Set up a proxy for every managed observer.

        Processors report their own setup to observers, so the proxies are in
        place before any of them is set up.

        Args:
            task_manager: The task manager the proxies run their tasks on.
        """
        await super().setup(task_manager)
        self._proxies = self._create_proxies(self._observers)
        self._frame_targets.clear()

        for observer in self._proxies:
            await observer.setup(task_manager)

    async def cleanup(self):
        """Cleanup all proxy observers.

        Tearing down is best effort: an observer that raises is logged and the
        rest are still cleaned up.
        """
        await super().cleanup()
        self._frame_targets.clear()

        if not self._proxies:
            return

        for proxy in self._proxies.values():
            await self.cancel_task(proxy.task)

        for observer in self._proxies:
            try:
                await observer.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up {observer}: {e}")

    async def on_pipeline_started(self):
        """Forward pipeline started signal to all managed observers."""
        await self._send_to_proxy(_PipelineStartedSignal())

    async def on_process_frame(self, data: FrameProcessed):
        """Queue frame data for the managed observers that handle the frame's class.

        Args:
            data: The frame processing event data to distribute to observers.
        """
        if not self._proxies:
            return

        for proxy in self._targets(type(data.frame)):
            if _handles_processing(proxy.observer):
                await proxy.queue.put(data)

    async def on_push_frame(self, data: FramePushed):
        """Queue frame data for the managed observers that observe the frame's class.

        A repeated push only reaches the observers that want every push. Every
        push counts toward telling a frame's first push from the rest, whoever
        observes it.

        Args:
            data: The frame push event data to distribute to observers.
        """
        if not self._proxies:
            return

        frame = data.frame
        data.first_push = frame.id not in self._frames_pushed
        if data.first_push:
            self._frames_pushed[frame.id] = frame

        for proxy in self._targets(type(frame)):
            if data.first_push or proxy.observer.observe_every_push:
                await proxy.queue.put(data)

    async def on_processor_setup(self, data: ProcessorSetUp):
        """Queue processor setup timing for all managed observers.

        Args:
            data: The processor setup event data to distribute to observers.
        """
        await self._send_to_proxy(data)

    @deprecated(
        "`WorkerObserver.on_startup_warmup` is deprecated since 1.12.0 and will be removed in "
        "2.0.0. No replacement."
    )
    async def on_startup_warmup(self, data: StartupWarmup):
        """Queue deferred-import warming timing for all managed observers.

        .. deprecated:: 1.12.0
            No replacement. Nothing warms deferred imports at startup, so this
            is never called. Will be removed in 2.0.0.

        Args:
            data: The startup warmup event data to distribute to observers.
        """
        await self._send_to_proxy(data)

    def _create_proxy(self, observer: BaseObserver) -> Proxy:
        """Create a proxy for a single observer."""
        queue = asyncio.Queue()
        task = self.create_task(self._proxy_task_handler(queue, observer))
        proxy = Proxy(queue=queue, task=task, observer=observer)
        return proxy

    def _create_proxies(self, observers: list[BaseObserver]) -> dict[BaseObserver, Proxy]:
        """Create proxies for all observers."""
        proxies = {}
        for observer in observers:
            proxy = self._create_proxy(observer)
            proxies[observer] = proxy
        return proxies

    def _targets(self, frame_type: type[Frame]) -> list[Proxy]:
        """The proxies whose observers observe frames of the given class."""
        targets = self._frame_targets.get(frame_type)
        if targets is None:
            proxies = self._proxies.values() if self._proxies else ()
            targets = self._frame_targets[frame_type] = [
                proxy for proxy in proxies if self._observes(proxy.observer, frame_type)
            ]
        return targets

    def _observes(self, observer: BaseObserver, frame_type: type[Frame]) -> bool:
        """Whether the observer observes frames of the given class.

        An observer whose check raises observes the class, as it would without
        the check. The error is logged, once per class while the answer is kept.
        """
        try:
            return observer.observes_frame_type(frame_type)
        except Exception as e:
            self._log_observer_error(observer, e)
            return True

    async def _send_to_proxy(self, data: Any):
        if not self._proxies:
            return
        for proxy in self._proxies.values():
            await proxy.queue.put(data)

    async def _proxy_task_handler(self, queue: asyncio.Queue, observer: BaseObserver):
        """Handle frame processing for a single observer."""
        while True:
            data = await queue.get()

            # A failing observer must not end this task, or it would get
            # nothing more while its queue keeps growing.
            try:
                await self._notify_observer(observer, data)
            except Exception as e:
                # Cancellation still ends the task, even when the observer turns it into an error.
                if (task := asyncio.current_task()) and task.cancelling():
                    raise
                self._log_observer_error(observer, e)

            queue.task_done()

            # Stop once the observer has removed itself (see remove_observer()).
            proxy = self._proxies.get(observer) if self._proxies else None
            if not proxy or proxy.queue is not queue:
                break

    def _log_observer_error(self, observer: BaseObserver, e: Exception):
        """Log an exception raised by an observer, with where it was raised."""
        last = traceback.extract_tb(e.__traceback__)[-1]
        logger.error(
            f"{self}: uncaught exception in observer {observer} "
            f"({last.filename}:{last.lineno}): {e}"
        )

    async def _notify_observer(self, observer: BaseObserver, data: Any):
        """Call the observer callback that matches the queued data."""
        if isinstance(data, _PipelineStartedSignal):
            await observer.on_pipeline_started()
        elif isinstance(data, FramePushed):
            await observer.on_push_frame(data)
        elif isinstance(data, FrameProcessed):
            await observer.on_process_frame(data)
        elif isinstance(data, ProcessorSetUp):
            await observer.on_processor_setup(data)
        elif isinstance(data, StartupWarmup):
            await observer.on_startup_warmup(data)
