#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for ``pipecat.bus.network`` lazy imports."""

import unittest

import pipecat.bus.network as network

try:
    import pgmq  # noqa: F401

    PGMQ_AVAILABLE = True
except ImportError:
    PGMQ_AVAILABLE = False

try:
    import redis  # noqa: F401

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class TestNetworkPackageLazyImports(unittest.TestCase):
    @unittest.skipUnless(PGMQ_AVAILABLE, "pgmq extra not installed")
    def test_pgmq_bus_lazy_import(self):
        """``PgmqBus`` is resolved on first attribute access."""
        from pipecat.bus.network.pgmq import PgmqBus as Direct

        self.assertIs(network.PgmqBus, Direct)

    @unittest.skipUnless(REDIS_AVAILABLE, "redis extra not installed")
    def test_redis_bus_lazy_import(self):
        """``RedisBus`` is resolved on first attribute access."""
        from pipecat.bus.network.redis import RedisBus as Direct

        self.assertIs(network.RedisBus, Direct)

    def test_unknown_attribute_raises(self):
        """Unknown attributes raise a clear ``AttributeError``."""
        with self.assertRaises(AttributeError) as ctx:
            network.NotARealBus  # noqa: B018
        self.assertIn("NotARealBus", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
