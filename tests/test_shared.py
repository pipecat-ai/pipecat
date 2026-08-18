#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the shared resource decorators."""

import asyncio
import unittest

from pipecat.utils.shared import acquires, releases


class _Client:
    """Stands in for a client an input and an output transport share."""

    def __init__(self):
        self.setups = 0
        self.cleanups = 0
        self.joins = 0
        self.leaves = 0
        self.connected = False
        self.seen_connected = []

    @acquires("client")
    async def setup(self, delay: float = 0.05):
        await asyncio.sleep(delay)
        self.setups += 1
        self.connected = True

    @releases("client")
    async def cleanup(self):
        self.cleanups += 1
        self.connected = False

    @acquires("room")
    async def join(self) -> str:
        self.joins += 1
        return "joined"

    @releases("room")
    async def leave(self):
        self.leaves += 1

    async def owner_setup(self):
        """Set up as one of several owners, recording what it then sees."""
        await self.setup()
        self.seen_connected.append(self.connected)


class _FailingClient:
    """Stands in for a shared client that cannot be built."""

    def __init__(self):
        self.setups = 0
        self.cleanups = 0

    @acquires("client")
    async def setup(self, delay: float = 0.05):
        await asyncio.sleep(delay)
        self.setups += 1
        raise RuntimeError("could not connect")

    @releases("client")
    async def cleanup(self):
        self.cleanups += 1


class TestSharedResource(unittest.IsolatedAsyncioTestCase):
    async def test_only_the_first_owner_runs_the_body(self):
        client = _Client()

        await asyncio.gather(client.setup(), client.setup(), client.setup())

        self.assertEqual(client.setups, 1)

    async def test_later_owners_wait_for_the_first(self):
        """A caller must not continue against a half-built resource.

        Processors are set up concurrently, so an owner can arrive while
        another is still setting the resource up.
        """
        client = _Client()

        await asyncio.gather(client.owner_setup(), client.owner_setup())

        self.assertEqual(client.seen_connected, [True, True])

    async def test_only_the_last_owner_undoes_it(self):
        client = _Client()
        await asyncio.gather(client.setup(), client.setup())

        await client.cleanup()
        self.assertEqual(client.cleanups, 0, "the first owner leaving must not tear down")

        await client.cleanup()
        self.assertEqual(client.cleanups, 1)
        self.assertFalse(client.connected)

    async def test_release_without_acquire_does_nothing(self):
        client = _Client()

        await client.cleanup()
        await client.cleanup()

        self.assertEqual(client.cleanups, 0)

    async def test_resources_are_counted_separately(self):
        """setup/cleanup and join/leave are independent on the same object."""
        client = _Client()
        await asyncio.gather(client.setup(), client.setup())

        await asyncio.gather(client.join(), client.join())
        await client.leave()

        self.assertEqual(client.joins, 1)
        self.assertEqual(client.leaves, 0, "one owner still holds the room")
        self.assertEqual(client.cleanups, 0, "releasing a room must not clean up the client")

        await client.leave()
        self.assertEqual(client.leaves, 1)

    async def test_instances_count_their_own_owners(self):
        first = _Client()
        second = _Client()

        await asyncio.gather(first.setup(), first.setup())
        await second.setup()

        self.assertEqual(first.setups, 1)
        self.assertEqual(second.setups, 1)

    async def test_first_owner_gets_the_return_value(self):
        client = _Client()

        self.assertEqual(await client.join(), "joined")
        self.assertIsNone(await client.join())

    async def test_acquiring_again_after_release_runs_the_body(self):
        client = _Client()

        await client.setup()
        await client.cleanup()
        await client.setup()

        self.assertEqual(client.setups, 2)
        self.assertTrue(client.connected)

    async def test_every_owner_of_a_resource_that_failed_is_told(self):
        """A sibling that carried on regardless would run against nothing."""
        client = _FailingClient()

        with self.assertRaises(RuntimeError):
            await client.setup()
        with self.assertRaises(RuntimeError):
            await client.setup()

    async def test_a_body_that_fails_is_not_attempted_again(self):
        client = _FailingClient()

        for _ in range(2):
            with self.assertRaises(RuntimeError):
                await client.setup()

        self.assertEqual(client.setups, 1)

    async def test_concurrent_owners_of_a_failed_resource_all_fail(self):
        client = _FailingClient()

        results = await asyncio.gather(client.setup(), client.setup(), return_exceptions=True)

        self.assertEqual(client.setups, 1)
        self.assertTrue(all(isinstance(r, RuntimeError) for r in results))

    async def test_nothing_is_released_when_acquiring_failed(self):
        """The undo would otherwise run against a resource that was never built."""
        client = _FailingClient()

        with self.assertRaises(RuntimeError):
            await client.setup()
        await client.cleanup()

        self.assertEqual(client.cleanups, 0)


if __name__ == "__main__":
    unittest.main()
