#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for `SmallWebRTCRequestHandler.handle_patch_request`.

Covers trickle-ICE candidate patch handling:

1. **End-of-candidates markers are skipped** — per RFC 8838, a peer signals
   "done gathering for this media line" with a patch whose ``candidate``
   string is empty. Browsers send one per media line on every connection.
   The marker must not be parsed as SDP (aiortc's ``candidate_from_sdp``
   asserts on it) and must not abort the rest of the batch.

2. **Real candidates are applied** — non-empty candidates in the same batch
   reach ``add_ice_candidate`` with ``sdpMid``/``sdpMLineIndex`` set.

3. **Unknown ``pc_id`` raises a 404** ``HTTPException``.
"""

import unittest
from unittest.mock import AsyncMock

import pytest

# The `webrtc` extra and the runner's HTTP dependencies are optional; skip the
# whole module when they are unavailable, matching the default CI unit test
# environment which does not install extras.
pytest.importorskip("aiortc")
pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402

from pipecat.transports.smallwebrtc.request_handler import (  # noqa: E402
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequestHandler,
)

HOST_CANDIDATE = "candidate:840965716 1 udp 2122260223 192.168.1.7 54321 typ host"


class TestHandlePatchRequest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.handler = SmallWebRTCRequestHandler()
        self.connection = AsyncMock()
        self.handler._pcs_map["pc-1"] = self.connection

    async def test_end_of_candidates_markers_are_skipped(self):
        request = SmallWebRTCPatchRequest(
            pc_id="pc-1",
            candidates=[
                IceCandidate(candidate="", sdp_mid="0", sdp_mline_index=0),
                IceCandidate(candidate=HOST_CANDIDATE, sdp_mid="0", sdp_mline_index=0),
                IceCandidate(candidate="", sdp_mid="1", sdp_mline_index=1),
            ],
        )

        await self.handler.handle_patch_request(request)

        self.assertEqual(self.connection.add_ice_candidate.await_count, 1)
        (candidate,) = self.connection.add_ice_candidate.await_args.args
        self.assertEqual(candidate.port, 54321)
        self.assertEqual(candidate.sdpMid, "0")
        self.assertEqual(candidate.sdpMLineIndex, 0)

    async def test_marker_only_batch_is_a_no_op(self):
        request = SmallWebRTCPatchRequest(
            pc_id="pc-1",
            candidates=[IceCandidate(candidate="", sdp_mid="0", sdp_mline_index=0)],
        )

        await self.handler.handle_patch_request(request)

        self.connection.add_ice_candidate.assert_not_awaited()

    async def test_unknown_pc_id_raises_404(self):
        request = SmallWebRTCPatchRequest(pc_id="missing", candidates=[])

        with self.assertRaises(HTTPException) as ctx:
            await self.handler.handle_patch_request(request)

        self.assertEqual(ctx.exception.status_code, 404)
