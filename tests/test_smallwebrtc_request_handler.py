#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for `SmallWebRTCRequestHandler.handle_patch_request`.

Firefox emits an RFC 8840 end-of-candidates marker per media line: a real
`RTCIceCandidate` whose `.candidate` string is empty. `aiortc.sdp.candidate_from_sdp`
cannot parse an empty string, and aiortc's own `addIceCandidate` treats `None`
(not an empty-string candidate) as the end-of-candidates signal. These tests
confirm the handler forwards `None` for empty markers instead of erroring out
or silently dropping them.
"""

import unittest
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("aiortc")

from pipecat.transports.smallwebrtc.request_handler import (  # noqa: E402
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequestHandler,
)

REAL_CANDIDATE_SDP = "candidate:0 1 UDP 2121990399 192.168.1.1 35628 typ host"


class TestHandlePatchRequest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.handler = SmallWebRTCRequestHandler()
        self.peer_connection = AsyncMock()
        self.handler._pcs_map["pc-1"] = self.peer_connection

    async def test_empty_candidate_forwarded_as_none(self):
        """A single empty-string marker becomes a `None` candidate, not an error."""
        request = SmallWebRTCPatchRequest(
            pc_id="pc-1",
            candidates=[IceCandidate(candidate="", sdp_mid="0", sdp_mline_index=0)],
        )

        await self.handler.handle_patch_request(request)

        self.peer_connection.add_ice_candidate.assert_awaited_once_with(None)

    async def test_mixed_real_and_empty_candidates_across_mids(self):
        """Real candidates parse normally; empty markers become `None`, matching
        Firefox's batch of real candidates followed by a per-mid end-of-candidates
        marker for audio (mid 0), video (mid 1), and data (mid 3).
        """
        request = SmallWebRTCPatchRequest(
            pc_id="pc-1",
            candidates=[
                IceCandidate(candidate=REAL_CANDIDATE_SDP, sdp_mid="0", sdp_mline_index=0),
                IceCandidate(candidate="", sdp_mid="0", sdp_mline_index=0),
                IceCandidate(candidate=REAL_CANDIDATE_SDP, sdp_mid="1", sdp_mline_index=1),
                IceCandidate(candidate="", sdp_mid="1", sdp_mline_index=1),
                IceCandidate(candidate="", sdp_mid="3", sdp_mline_index=3),
            ],
        )

        await self.handler.handle_patch_request(request)

        self.assertEqual(self.peer_connection.add_ice_candidate.await_count, 5)
        calls = self.peer_connection.add_ice_candidate.await_args_list

        for call, expected_mid, expected_mline in [
            (calls[0], "0", 0),
            (calls[2], "1", 1),
        ]:
            (candidate,), _ = call.args, call.kwargs
            self.assertIsNotNone(candidate)
            self.assertEqual(candidate.sdpMid, expected_mid)
            self.assertEqual(candidate.sdpMLineIndex, expected_mline)

        for call in [calls[1], calls[3], calls[4]]:
            (candidate,), _ = call.args, call.kwargs
            self.assertIsNone(candidate)


if __name__ == "__main__":
    unittest.main()
