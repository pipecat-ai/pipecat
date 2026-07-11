#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for the PgmqBus backend layer.

Covers:

- ``IsolatedPgmqBackend`` issues the expected ``public.bus_*`` SQL with the
  expected positional args, and normalizes the read result shape.
- ``PgmqBus`` correctly delegates to an injected backend (the
  ``backend=`` constructor path) and rejects ambiguous / empty construction.

``DirectPgmqBackend`` is exercised end-to-end via ``test_pgmq_bus.py``
against an in-memory ``FakePgmq``; no separate tests for it here.
"""

import asyncio
import json
import unittest
from dataclasses import dataclass

from pipecat.bus import BusDataMessage
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

try:
    from pipecat.bus.network.pgmq import PgmqBus
    from pipecat.bus.network.pgmq_backends import (
        BackendMessage,
        IsolatedPgmqBackend,
        PgmqBackend,
    )
except Exception:
    raise unittest.SkipTest('pgmq extra not installed (`uv add "pipecat-ai[pgmq]"`)')

# ---------------------------------------------------------------------------
# IsolatedPgmqBackend: assert it speaks the right SQL to asyncpg.
# ---------------------------------------------------------------------------


@dataclass
class _Call:
    method: str  # 'fetchval' | 'execute' | 'fetch'
    sql: str
    args: tuple


class _FakeConn:
    def __init__(self, recorder, fetchval_return=None, fetch_return=None):
        self._recorder = recorder
        self._fetchval_return = fetchval_return
        self._fetch_return = fetch_return or []

    async def fetchval(self, sql, *args):
        self._recorder.append(_Call("fetchval", sql, args))
        return self._fetchval_return

    async def execute(self, sql, *args):
        self._recorder.append(_Call("execute", sql, args))
        return "OK"

    async def fetch(self, sql, *args):
        self._recorder.append(_Call("fetch", sql, args))
        return self._fetch_return


class _FakePool:
    def __init__(self, fetchval_return=None, fetch_return=None):
        self.calls: list[_Call] = []
        self._fetchval_return = fetchval_return
        self._fetch_return = fetch_return or []

    def acquire(self):
        recorder = self.calls
        fetchval_return = self._fetchval_return
        fetch_return = self._fetch_return

        class _Ctx:
            async def __aenter__(self_inner):
                return _FakeConn(recorder, fetchval_return, fetch_return)

            async def __aexit__(self_inner, exc_type, exc, tb):
                return False

        return _Ctx()


class TestIsolatedPgmqBackend(unittest.IsolatedAsyncioTestCase):
    async def test_join_calls_bus_join_and_returns_queue_name(self):
        pool = _FakePool(fetchval_return="q_abc123")
        backend = IsolatedPgmqBackend(pool=pool)

        queue = await backend.join("ch_xyz")

        self.assertEqual(queue, "q_abc123")
        self.assertEqual(len(pool.calls), 1)
        self.assertIn("public.bus_join", pool.calls[0].sql)
        self.assertEqual(pool.calls[0].args, ("ch_xyz",))

    async def test_join_raises_when_wrapper_returns_empty(self):
        pool = _FakePool(fetchval_return=None)
        backend = IsolatedPgmqBackend(pool=pool)
        with self.assertRaises(RuntimeError):
            await backend.join("ch_xyz")

    async def test_publish_serializes_payload_as_jsonb(self):
        pool = _FakePool()
        backend = IsolatedPgmqBackend(pool=pool)

        await backend.publish("ch_xyz", "q_self", {"k": "v", "n": 1})

        self.assertEqual(len(pool.calls), 1)
        call = pool.calls[0]
        self.assertIn("public.bus_publish", call.sql)
        # (channel, my_queue, json_text)
        self.assertEqual(call.args[0], "ch_xyz")
        self.assertEqual(call.args[1], "q_self")
        self.assertEqual(json.loads(call.args[2]), {"k": "v", "n": 1})

    async def test_read_normalizes_rows_and_decodes_string_jsonb(self):
        # asyncpg can hand back jsonb as either a str (when no codec is
        # registered) or a dict (with a codec). Both should normalize.
        pool = _FakePool(
            fetch_return=[
                {"msg_id": 1, "message": {"hello": "world"}},
                {"msg_id": 2, "message": json.dumps({"hello": "again"})},
            ]
        )
        backend = IsolatedPgmqBackend(pool=pool)

        msgs = await backend.read(
            "q_self",
            channel="ch_xyz",
            vt=30,
            qty=10,
            max_poll_seconds=5,
            poll_interval_ms=100,
        )

        self.assertEqual(len(msgs), 2)
        self.assertIsInstance(msgs[0], BackendMessage)
        self.assertEqual(msgs[0].msg_id, 1)
        self.assertEqual(msgs[0].message, {"hello": "world"})
        self.assertEqual(msgs[1].msg_id, 2)
        self.assertEqual(msgs[1].message, {"hello": "again"})

        self.assertEqual(len(pool.calls), 1)
        call = pool.calls[0]
        self.assertIn("public.bus_subscribe", call.sql)
        # (my_queue, channel, vt, qty, max_poll_seconds)
        self.assertEqual(call.args, ("q_self", "ch_xyz", 30, 10, 5))

    async def test_archive_and_leave_call_the_right_wrappers(self):
        pool = _FakePool(fetchval_return=True)
        backend = IsolatedPgmqBackend(pool=pool)

        archived = await backend.archive("q_self", channel="ch_xyz", msg_id=42)
        self.assertTrue(archived)

        await backend.leave("q_self", channel="ch_xyz")

        self.assertEqual(len(pool.calls), 2)
        self.assertIn("public.bus_archive", pool.calls[0].sql)
        self.assertEqual(pool.calls[0].args, ("q_self", "ch_xyz", 42))
        self.assertIn("public.bus_leave", pool.calls[1].sql)
        self.assertEqual(pool.calls[1].args, ("q_self", "ch_xyz"))

    def test_satisfies_pgmq_backend_protocol(self):
        # Runtime-checkable Protocol: confirm structural compliance.
        backend = IsolatedPgmqBackend(pool=_FakePool())
        self.assertIsInstance(backend, PgmqBackend)


# ---------------------------------------------------------------------------
# PgmqBus(backend=...): orchestrator delegates to whatever backend it's given.
# ---------------------------------------------------------------------------


class _RecordingBackend:
    """A backend that records every call PgmqBus makes against it."""

    def __init__(self):
        self.calls: list[tuple[str, tuple, dict]] = []
        self._inbound: asyncio.Queue[BackendMessage] = asyncio.Queue()

    async def join(self, channel):
        self.calls.append(("join", (channel,), {}))
        return "q_injected"

    async def publish(self, channel, my_queue, payload):
        self.calls.append(("publish", (channel, my_queue, payload), {}))

    async def read(self, queue, *, channel, vt, qty, max_poll_seconds, poll_interval_ms):
        self.calls.append(
            (
                "read",
                (queue,),
                dict(
                    channel=channel,
                    vt=vt,
                    qty=qty,
                    max_poll_seconds=max_poll_seconds,
                    poll_interval_ms=poll_interval_ms,
                ),
            )
        )
        try:
            msg = await asyncio.wait_for(self._inbound.get(), timeout=0.05)
            return [msg]
        except TimeoutError:
            return []

    async def archive(self, queue, *, channel, msg_id):
        self.calls.append(("archive", (queue,), dict(channel=channel, msg_id=msg_id)))
        return True

    async def leave(self, queue, *, channel):
        self.calls.append(("leave", (queue,), dict(channel=channel)))


class TestPgmqBusBackendInjection(unittest.IsolatedAsyncioTestCase):
    def test_rejects_construction_with_neither_pgmq_nor_backend(self):
        with self.assertRaises(ValueError):
            PgmqBus(channel="ch_xyz")

    def test_rejects_construction_with_both_pgmq_and_backend(self):
        with self.assertRaises(ValueError):
            PgmqBus(pgmq=object(), backend=_RecordingBackend(), channel="ch_xyz")

    async def test_lifecycle_and_publish_delegate_to_backend(self):
        backend = _RecordingBackend()
        bus = PgmqBus(
            backend=backend,
            channel="ch_xyz",
            poll_interval_ms=10,
            max_poll_seconds=1,
        )
        tm = TaskManager()
        tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        await bus.setup(tm)

        await bus.start()
        try:
            self.assertEqual(bus._queue_name, "q_injected")
            await bus.send(BusDataMessage(source="task_a"))
            # Give the (mostly idle) reader a chance to record a read call.
            await asyncio.sleep(0.05)
        finally:
            await bus.stop()

        # join + publish (at least 1) + leave must all be observed; read may
        # have fired multiple times, archive never (no inbound msg).
        methods = [c[0] for c in backend.calls]
        self.assertEqual(methods.count("join"), 1)
        self.assertGreaterEqual(methods.count("publish"), 1)
        self.assertEqual(methods.count("leave"), 1)
        self.assertEqual(methods.count("archive"), 0)

        # The publish call passed (channel, my_queue, payload-dict).
        publish_call = next(c for c in backend.calls if c[0] == "publish")
        channel, my_queue, payload = publish_call[1]
        self.assertEqual(channel, "ch_xyz")
        self.assertEqual(my_queue, "q_injected")
        self.assertIsInstance(payload, dict)


if __name__ == "__main__":
    unittest.main()
