#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Google RTVI processor and observer."""

import unittest

from pipecat.processors.frameworks.rtvi.observer import RTVIObserverParams
from pipecat.services.google.rtvi import GoogleRTVIObserver, GoogleRTVIProcessor


class TestGoogleRTVIProcessor(unittest.TestCase):
    def test_observer_gets_the_params(self):
        params = RTVIObserverParams(bot_llm_enabled=False, metrics_enabled=False)
        observer = GoogleRTVIProcessor().create_rtvi_observer(params=params)
        self.assertIsInstance(observer, GoogleRTVIObserver)
        self.assertIs(observer._params, params)


if __name__ == "__main__":
    unittest.main()
