#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.utils.network import exponential_backoff_time


class TestUtilsNetwork(unittest.IsolatedAsyncioTestCase):
    async def test_exponential_backoff_time(self):
        # min_wait=4, max_wait=10, multiplier=1
        assert exponential_backoff_time(attempt=1, min_wait=4, max_wait=10, multiplier=1) == 4
        assert exponential_backoff_time(attempt=2, min_wait=4, max_wait=10, multiplier=1) == 4
        assert exponential_backoff_time(attempt=3, min_wait=4, max_wait=10, multiplier=1) == 4
        assert exponential_backoff_time(attempt=4, min_wait=4, max_wait=10, multiplier=1) == 8
        assert exponential_backoff_time(attempt=5, min_wait=4, max_wait=10, multiplier=1) == 10
        assert exponential_backoff_time(attempt=6, min_wait=4, max_wait=10, multiplier=1) == 10
        # min_wait=1, max_wait=10, multiplier=1
        assert exponential_backoff_time(attempt=1, min_wait=1, max_wait=10, multiplier=1) == 1
        assert exponential_backoff_time(attempt=2, min_wait=1, max_wait=10, multiplier=1) == 2
        assert exponential_backoff_time(attempt=3, min_wait=1, max_wait=10, multiplier=1) == 4
        assert exponential_backoff_time(attempt=4, min_wait=1, max_wait=10, multiplier=1) == 8
        assert exponential_backoff_time(attempt=5, min_wait=1, max_wait=10, multiplier=1) == 10
        assert exponential_backoff_time(attempt=6, min_wait=1, max_wait=10, multiplier=1) == 10
        # min_wait=1, max_wait=20, multiplier=2
        assert exponential_backoff_time(attempt=1, min_wait=1, max_wait=20, multiplier=2) == 2
        assert exponential_backoff_time(attempt=2, min_wait=1, max_wait=20, multiplier=2) == 4
        assert exponential_backoff_time(attempt=3, min_wait=1, max_wait=20, multiplier=2) == 8
        assert exponential_backoff_time(attempt=4, min_wait=1, max_wait=20, multiplier=2) == 16
        assert exponential_backoff_time(attempt=5, min_wait=1, max_wait=20, multiplier=2) == 20
        assert exponential_backoff_time(attempt=6, min_wait=1, max_wait=20, multiplier=2) == 20
