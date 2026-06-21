#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for TwilioFrameSerializer REST base-URL resolution (auto hang-up)."""

import unittest

from pipecat.serializers.twilio import _build_call_resource_url


class TestBuildCallResourceURL(unittest.TestCase):
    def test_default_host(self):
        # No base_url, no region/edge -> Twilio's default host (unchanged behavior).
        self.assertEqual(
            _build_call_resource_url("ACxxxx", "CAyyyy"),
            "https://api.twilio.com/2010-04-01/Accounts/ACxxxx/Calls/CAyyyy.json",
        )

    def test_region_edge_host(self):
        self.assertEqual(
            _build_call_resource_url("ACxxxx", "CAyyyy", region="au1", edge="sydney"),
            "https://api.sydney.au1.twilio.com/2010-04-01/Accounts/ACxxxx/Calls/CAyyyy.json",
        )

    def test_base_url_override(self):
        # base_url targets a Twilio-API-compatible backend; region/edge ignored.
        self.assertEqual(
            _build_call_resource_url(
                "ACxxxx", "CAyyyy", base_url="https://api.example.test", region="au1", edge="sydney"
            ),
            "https://api.example.test/2010-04-01/Accounts/ACxxxx/Calls/CAyyyy.json",
        )

    def test_base_url_trailing_slash_normalized(self):
        self.assertEqual(
            _build_call_resource_url("ACxxxx", "CAyyyy", base_url="https://api.example.test/"),
            "https://api.example.test/2010-04-01/Accounts/ACxxxx/Calls/CAyyyy.json",
        )


if __name__ == "__main__":
    unittest.main()
