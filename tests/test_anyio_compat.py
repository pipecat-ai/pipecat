#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the anyio compatibility layer.

These tests use the ``anyio`` pytest plugin to run each test under both
the ``asyncio`` and ``trio`` backends, proving the compat primitives and
:class:`AnyioTaskManager` behave identically on each.
"""

import anyio
import pytest

from pipecat.utils.asyncio.anyio_task_manager import AnyioTaskManager, TaskHandle

pytestmark = pytest.mark.anyio
from pipecat.utils.asyncio.compat import (
    Event,
    PriorityQueue,
    Queue,
    current_backend,
    get_cancelled_exc_class,
    sleep,
)


@pytest.fixture(params=["asyncio", "trio"])
def anyio_backend(request):
    """Parametrize every test in this module over both async backends."""
    return request.param


class TestCompatPrimitives:
    async def test_current_backend_matches_fixture(self, anyio_backend):
        assert current_backend() == anyio_backend

    async def test_sleep(self, anyio_backend):
        await sleep(0)

    async def test_event(self, anyio_backend):
        ev = Event()
        assert not ev.is_set()
        ev.set()
        assert ev.is_set()
        await ev.wait()

    async def test_cancelled_exc_class(self, anyio_backend):
        exc = get_cancelled_exc_class()
        if anyio_backend == "asyncio":
            import asyncio

            assert exc is asyncio.CancelledError
        else:
            import trio

            assert exc is trio.Cancelled


class TestQueue:
    async def test_put_get(self, anyio_backend):
        q: Queue[int] = Queue()
        await q.put(1)
        await q.put(2)
        assert q.qsize() == 2
        assert await q.get() == 1
        assert await q.get() == 2
        assert q.empty()

    async def test_nowait(self, anyio_backend):
        q: Queue[str] = Queue()
        q.put_nowait("hello")
        assert q.get_nowait() == "hello"
        with pytest.raises(anyio.WouldBlock):
            q.get_nowait()

    async def test_bounded_queue_blocks(self, anyio_backend):
        q: Queue[int] = Queue(maxsize=1)
        await q.put(1)
        with pytest.raises(anyio.WouldBlock):
            q.put_nowait(2)


class TestPriorityQueue:
    async def test_priority_ordering(self, anyio_backend):
        pq: PriorityQueue[tuple[int, str]] = PriorityQueue()
        await pq.put((2, "b"))
        await pq.put((1, "a"))
        await pq.put((3, "c"))
        assert await pq.get() == (1, "a")
        assert await pq.get() == (2, "b")
        assert await pq.get() == (3, "c")
        assert pq.empty()

    async def test_nowait(self, anyio_backend):
        pq: PriorityQueue[int] = PriorityQueue()
        pq.put_nowait(5)
        pq.put_nowait(1)
        assert pq.get_nowait() == 1
        assert pq.get_nowait() == 5
        with pytest.raises(anyio.WouldBlock):
            pq.get_nowait()

    async def test_concurrent_producer_consumer(self, anyio_backend):
        pq: PriorityQueue[int] = PriorityQueue()
        received: list[int] = []

        async def producer():
            for i in (3, 1, 2):
                await pq.put(i)
                await sleep(0)

        async def consumer():
            for _ in range(3):
                received.append(await pq.get())

        async with anyio.create_task_group() as tg:
            tg.start_soon(producer)
            tg.start_soon(consumer)

        assert sorted(received) == [1, 2, 3]


class TestAnyioTaskManager:
    async def test_create_and_await_task(self, anyio_backend):
        async with AnyioTaskManager() as tm:

            async def worker():
                await sleep(0)
                return 42

            handle = tm.create_task(worker(), name="worker")
            assert isinstance(handle, TaskHandle)
            assert handle.get_name() == "worker"
            result = await handle
            assert result == 42
            assert handle.done()
            assert handle.result() == 42

    async def test_cancel_task(self, anyio_backend):
        async with AnyioTaskManager() as tm:

            async def forever():
                while True:
                    await sleep(0.01)

            handle = tm.create_task(forever(), name="forever")
            await sleep(0)  # let it start
            assert not handle.done()
            await tm.cancel_task(handle)
            assert handle.done()
            assert handle.cancelled()

    async def test_cancel_with_timeout(self, anyio_backend):
        async with AnyioTaskManager() as tm:

            async def stubborn():
                while True:
                    await sleep(0.01)

            handle = tm.create_task(stubborn(), name="stubborn")
            await sleep(0)
            await tm.cancel_task(handle, timeout=1.0)
            assert handle.done()

    async def test_task_exception_captured(self, anyio_backend):
        async with AnyioTaskManager() as tm:

            async def boom():
                raise ValueError("kaboom")

            handle = tm.create_task(boom(), name="boom")
            await handle._done.wait()
            assert handle.done()
            assert isinstance(handle.exception(), ValueError)
            with pytest.raises(ValueError, match="kaboom"):
                handle.result()

    async def test_current_tasks(self, anyio_backend):
        async with AnyioTaskManager() as tm:

            async def idle():
                await sleep(10)

            h1 = tm.create_task(idle(), name="t1")
            h2 = tm.create_task(idle(), name="t2")
            await sleep(0)
            names = {t.get_name() for t in tm.current_tasks()}
            assert names == {"t1", "t2"}
            await tm.cancel_task(h1)
            await tm.cancel_task(h2)

    async def test_context_exit_cancels_children(self, anyio_backend):
        handle_holder: list[TaskHandle] = []

        async with AnyioTaskManager() as tm:

            async def idle():
                await sleep(10)

            handle_holder.append(tm.create_task(idle(), name="child"))
            await sleep(0)

        assert handle_holder[0].done()

    async def test_create_task_outside_context_raises(self, anyio_backend):
        tm = AnyioTaskManager()

        async def noop():
            pass

        coro = noop()
        with pytest.raises(RuntimeError, match="async context manager"):
            tm.create_task(coro, name="bad")
        coro.close()

    async def test_get_event_loop_on_trio_raises(self, anyio_backend):
        async with AnyioTaskManager() as tm:
            if anyio_backend == "trio":
                with pytest.raises(RuntimeError, match="not available under trio"):
                    tm.get_event_loop()
            else:
                loop = tm.get_event_loop()
                assert loop is not None
