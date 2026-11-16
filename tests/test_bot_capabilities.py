#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.pipeline.capabilities import BotCapabilities
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor
from pipecat.transports import base_input
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams, VideoInSourceParams
from pipecat.transports.heygen.transport import HeyGenOutputTransport


class TestBotCapabilities(unittest.TestCase):
    def test_combine_true_wins(self):
        a = BotCapabilities(audio_in=False, video_out=True)
        b = BotCapabilities(audio_in=True, video_out=False)
        self.assertEqual(a.combine(b), BotCapabilities(audio_in=True, video_out=True))

    def test_combine_keeps_false_and_unknown(self):
        a = BotCapabilities(video_in=False)
        b = BotCapabilities()
        self.assertEqual(a.combine(b), BotCapabilities(video_in=False))

    def test_override_applies_only_known_fields(self):
        base = BotCapabilities(audio_in=True, video_out=False, metrics=True)
        result = base.override(BotCapabilities(video_out=True))
        self.assertEqual(result, BotCapabilities(audio_in=True, video_out=True, metrics=True))


class _SourcesInputTransport(BaseInputTransport):
    """An input transport that captures the camera and screen share from ``video_in_sources``."""

    def _supports_video_in_source(self, video_source: str) -> bool:
        return video_source in ("camera", "screenVideo")


def _transport(**params) -> tuple[BaseInputTransport, BaseOutputTransport]:
    transport_params = TransportParams(**params)
    return _SourcesInputTransport(transport_params), BaseOutputTransport(transport_params)


class TestPipelineWorkerCapabilities(unittest.TestCase):
    def test_derived_from_transport_params(self):
        input, output = _transport(audio_in_enabled=True, audio_out_enabled=True)
        worker = PipelineWorker(Pipeline([input, output]))
        self.assertEqual(
            worker.capabilities,
            BotCapabilities(
                audio_in=True,
                audio_out=True,
                video_in=False,
                screen_in=False,
                video_out=False,
                metrics=False,
            ),
        )

    def test_screen_in_when_sources_include_screen_video(self):
        input, output = _transport(
            video_in_enabled=True,
            video_in_sources={
                "camera": VideoInSourceParams(),
                "screenVideo": VideoInSourceParams(),
            },
        )
        worker = PipelineWorker(Pipeline([input, output]))
        self.assertTrue(worker.capabilities.screen_in)

    def test_no_screen_in_when_sources_exclude_screen_video(self):
        input, output = _transport(
            video_in_enabled=True, video_in_sources={"camera": VideoInSourceParams()}
        )
        worker = PipelineWorker(Pipeline([input, output]))
        self.assertIs(worker.capabilities.screen_in, False)

    def test_screen_in_unknown_without_sources(self):
        input, output = _transport(video_in_enabled=True)
        worker = PipelineWorker(Pipeline([input, output]))
        self.assertTrue(worker.capabilities.video_in)
        self.assertIsNone(worker.capabilities.screen_in)

    def test_unsupported_sources_warned_and_screen_in_unknown(self):
        params = TransportParams(
            video_in_enabled=True, video_in_sources={"screenVideo": VideoInSourceParams()}
        )
        with patch.object(base_input, "logger") as logger:
            input = BaseInputTransport(params)
        logger.warning.assert_called_once()
        self.assertIn("screenVideo", logger.warning.call_args.args[0])

        worker = PipelineWorker(Pipeline([input, BaseOutputTransport(params)]))
        self.assertIsNone(worker.capabilities.screen_in)

    def test_supported_sources_not_warned(self):
        with patch.object(base_input, "logger") as logger:
            _transport(video_in_enabled=True, video_in_sources={"camera": VideoInSourceParams()})
        logger.warning.assert_not_called()

    def test_metrics_from_pipeline_params(self):
        input, output = _transport()
        worker = PipelineWorker(
            Pipeline([input, output]), params=PipelineParams(enable_metrics=True)
        )
        self.assertTrue(worker.capabilities.metrics)

    def test_transports_nested_in_parallel_pipeline(self):
        input, output = _transport(video_in_enabled=True, video_out_enabled=True)
        worker = PipelineWorker(Pipeline([ParallelPipeline([input], [output])]))
        self.assertTrue(worker.capabilities.video_in)
        self.assertTrue(worker.capabilities.video_out)

    def test_no_transport_leaves_media_unknown(self):
        worker = PipelineWorker(Pipeline([]))
        self.assertEqual(worker.capabilities, BotCapabilities(metrics=False))

    def test_constructor_override(self):
        input, output = _transport(audio_in_enabled=True, audio_out_enabled=True)
        worker = PipelineWorker(
            Pipeline([input, output]), capabilities=BotCapabilities(video_out=True)
        )
        self.assertTrue(worker.capabilities.audio_in)
        self.assertTrue(worker.capabilities.video_out)

    def test_avatar_output_reports_video_out(self):
        output = HeyGenOutputTransport(
            client=Mock(), params=TransportParams(audio_out_enabled=True)
        )
        worker = PipelineWorker(Pipeline([output]))
        self.assertTrue(worker.capabilities.audio_out)
        self.assertTrue(worker.capabilities.video_out)


class TestBotReadyCapabilities(unittest.IsolatedAsyncioTestCase):
    async def test_bot_ready_carries_worker_capabilities(self):
        capabilities = BotCapabilities(audio_in=True, audio_out=True, video_in=False)
        processor = RTVIProcessor()
        processor._setup = SimpleNamespace(
            pipeline_worker=SimpleNamespace(capabilities=capabilities)
        )
        processor.push_transport_message = AsyncMock()

        await processor.set_bot_ready()

        message = processor.push_transport_message.call_args[0][0]
        self.assertIsInstance(message, RTVI.BotReady)
        self.assertEqual(message.data.capabilities, capabilities)
        # Unknown fields are left out of the wire message.
        self.assertEqual(
            message.model_dump(exclude_none=True)["data"]["capabilities"],
            {"audio_in": True, "audio_out": True, "video_in": False},
        )
        await processor.cleanup()


if __name__ == "__main__":
    unittest.main()
