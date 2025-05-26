#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import inspect
from typing import Dict, List, Optional

from attr import dataclass

from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.utils.asyncio import BaseTaskManager


@dataclass
class Proxy:
    """This is the data we receive from the main observer and that we put into
    a queue for later processing.

    """

    queue: asyncio.Queue
    task: asyncio.Task
    observer: BaseObserver


class TaskObserver(BaseObserver):
    """This is a pipeline frame observer that is meant to be used as a proxy to
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
        super().__init__(**kwargs)
        self._observers = observers or []
        self._task_manager = task_manager
        self._proxies: Dict[BaseObserver, Proxy] = {}

    async def add_observer(self, observer: BaseObserver):
        proxy = self._create_proxy(observer)
        self._proxies[observer] = proxy
        self._observers.append(observer)

    async def remove_observer(self, observer: BaseObserver):
        if observer in self._proxies:
            proxy = self._proxies[observer]
            # Remove the proxy so it doesn't get called anymore.
            del self._proxies[observer]
            # Cancel the proxy task right away.
            await self._task_manager.cancel_task(proxy.task)
            # Remove the observer.
            self._observers.remove(observer)

    async def start(self):
        """Starts all proxy observer tasks."""
        self._proxies = self._create_proxies(self._observers)

    async def stop(self):
        """Stops all proxy observer tasks."""
        for proxy in self._proxies.values():
            await self._task_manager.cancel_task(proxy.task)

    async def on_push_frame(self, data: FramePushed):
        for proxy in self._proxies.values():
            await proxy.queue.put(data)

    def _create_proxy(self, observer: BaseObserver) -> Proxy:
        queue = asyncio.Queue()
        task = self._task_manager.create_task(
            self._proxy_task_handler(queue, observer),
            f"TaskObserver::{observer}::_proxy_task_handler",
        )
        proxy = Proxy(queue=queue, task=task, observer=observer)
        return proxy

    def _create_proxies(self, observers: List[BaseObserver]) -> Dict[BaseObserver, Proxy]:
        proxies = {}
        for observer in observers:
            proxy = self._create_proxy(observer)
            proxies[observer] = proxy
        return proxies

    async def _proxy_task_handler(self, queue: asyncio.Queue, observer: BaseObserver):
        warning_reported = False
        while True:
            data = await queue.get()

            signature = inspect.signature(observer.on_push_frame)
            if len(signature.parameters) > 1:
                if not warning_reported:
                    import warnings

                    with warnings.catch_warnings():
                        warnings.simplefilter("always")
                        warnings.warn(
                            "Observer `on_push_frame(source, destination, frame, direction, timestamp)` is deprecated, us `on_push_frame(data: FramePushed)` instead.",
                            DeprecationWarning,
                        )
                    warning_reported = True
                await observer.on_push_frame(
                    data.src, data.dst, data.frame, data.direction, data.timestamp
                )
            else:
                await observer.on_push_frame(data)

            queue.task_done()
