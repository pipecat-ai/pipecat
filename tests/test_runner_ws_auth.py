#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for PIPECAT_WEBSOCKET_AUTH / --ws-auth normalization."""

import unittest

from pipecat.runner.run import _normalize_ws_auth_mode


class TestNormalizeWsAuthMode(unittest.TestCase):
    def test_token_is_case_and_whitespace_insensitive(self):
        self.assertEqual(_normalize_ws_auth_mode("token"), "token")
        self.assertEqual(_normalize_ws_auth_mode("TOKEN"), "token")
        self.assertEqual(_normalize_ws_auth_mode(" Token "), "token")

    def test_blank_or_missing_defaults_to_none(self):
        self.assertEqual(_normalize_ws_auth_mode(None), "none")
        self.assertEqual(_normalize_ws_auth_mode(""), "none")
        self.assertEqual(_normalize_ws_auth_mode("   "), "none")

    def test_none_is_accepted(self):
        self.assertEqual(_normalize_ws_auth_mode("none"), "none")
        self.assertEqual(_normalize_ws_auth_mode("NONE"), "none")

    def test_invalid_value_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _normalize_ws_auth_mode("hmac")
        self.assertIn("hmac", str(ctx.exception))
