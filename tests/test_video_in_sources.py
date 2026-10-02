#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pydantic import ValidationError

from pipecat.transports.base_transport import TransportParams, VideoInSourceParams


class TestVideoInSourcesParams(unittest.TestCase):
    def test_defaults_to_no_sources(self):
        self.assertEqual(TransportParams().video_in_sources, {})

    def test_sources_keyed_by_source(self):
        params = TransportParams(
            video_in_enabled=True,
            video_in_sources={
                "camera": VideoInSourceParams(framerate=0),
                "screenVideo": VideoInSourceParams(framerate=1),
            },
        )
        self.assertEqual(params.video_in_sources["camera"].framerate, 0)
        self.assertEqual(params.video_in_sources["screenVideo"].framerate, 1)

    def test_source_params_from_dict(self):
        params = TransportParams(video_in_enabled=True, video_in_sources={"camera": {}})
        self.assertEqual(params.video_in_sources["camera"], VideoInSourceParams())

    def test_default_framerate(self):
        self.assertEqual(VideoInSourceParams().framerate, 30)

    def test_negative_framerate_rejected(self):
        with self.assertRaises(ValidationError):
            VideoInSourceParams(framerate=-1)

    def test_sources_require_video_in_enabled(self):
        with self.assertRaises(ValidationError):
            TransportParams(video_in_sources={"camera": VideoInSourceParams()})

    def test_sources_survive_model_dump(self):
        params = TransportParams(
            video_in_enabled=True, video_in_sources={"screenVideo": VideoInSourceParams()}
        )
        self.assertEqual(TransportParams(**params.model_dump()), params)


if __name__ == "__main__":
    unittest.main()
