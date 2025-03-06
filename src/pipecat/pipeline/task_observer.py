#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import List

from attr import dataclass

from pipecat.frames.frames import Frame
from pipecat.observers.base_observer import BaseObserver
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.asyncio import BaseTaskManager


@dataclass
class Proxy:
    """This is the data we receive from the main observer and that we put into
    a queue for later processing.

    """

    queue: asyncio.Queue
    task: asyncio.Task
    observer: BaseObserver


@dataclass
class ObserverData:
    """This is the data we receive from the main observer and that we put into a
    proxy queue for later processing.

    """

    src: FrameProcessor
    dst: FrameProcessor
    frame: Frame
    direction: FrameDirection
    timestamp: int


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

    def __init__(self, *, observers: List[BaseObserver] = [], task_manager: BaseTaskManager):
        self._observers = observers
        self._task_manager = task_manager
        self._proxies: List[Proxy] = []

    async def start(self):
        """Starts all proxy observer tasks."""
        self._proxies = self._create_proxies(self._observers)

    async def stop(self):
        """Stops all proxy observer tasks."""
        for proxy in self._proxies:
            await self._task_manager.cancel_task(proxy.task)

    async def on_push_frame(
        self,
        src: FrameProcessor,
        dst: FrameProcessor,
        frame: Frame,
        direction: FrameDirection,
        timestamp: int,
    ):
        for proxy in self._proxies:
            await proxy.queue.put(
                ObserverData(
                    src=src, dst=dst, frame=frame, direction=direction, timestamp=timestamp
                )
            )

    def _create_proxies(self, observers) -> List[Proxy]:
        proxies = []
        for observer in observers:
            queue = asyncio.Queue()
            task = self._task_manager.create_task(
                self._proxy_task_handler(queue, observer),
                f"TaskObserver::{observer.__class__.__name__}::_proxy_task_handler",
            )
            proxy = Proxy(queue=queue, task=task, observer=observer)
            proxies.append(proxy)
        return proxies

    async def _proxy_task_handler(self, queue: asyncio.Queue, observer: BaseObserver):
        while True:
            data = await queue.get()
            await observer.on_push_frame(
                data.src, data.dst, data.frame, data.direction, data.timestamp
            )
