#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    StartFrame,
    SystemFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    from gi.repository import Gst, GstApp
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use GStreamer, you need to `pip install pipecat-ai[gstreamer]`. Also, you need to install GStreamer in your system."
    )
    raise Exception(f"Missing module: {e}")


class GStreamerPipelineSource(FrameProcessor):
    class OutputParams(BaseModel):
        video_width: int = 1280
        video_height: int = 720
        audio_sample_rate: Optional[int] = None
        audio_channels: int = 1
        clock_sync: bool = True

    def __init__(self, *, pipeline: str, out_params: Optional[OutputParams] = None, **kwargs):
        super().__init__(**kwargs)

        self._out_params = out_params or GStreamerPipelineSource.OutputParams()
        self._sample_rate = 0

        Gst.init()

        self._player = Gst.Pipeline.new("player")

        source = Gst.parse_bin_from_description(pipeline, True)

        decodebin = Gst.ElementFactory.make("decodebin", None)
        decodebin.connect("pad-added", self._decodebin_callback)

        self._player.add(source)
        self._player.add(decodebin)
        source.link(decodebin)

        bus = self._player.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_gstreamer_message)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Specific system frames
        if isinstance(frame, StartFrame):
            # Push StartFrame before start(), because we want StartFrame to be
            # processed by every processor before any other frame is processed.
            await self.push_frame(frame, direction)
            await self._start(frame)
        elif isinstance(frame, CancelFrame):
            await self._cancel(frame)
            await self.push_frame(frame, direction)
        # All other system frames
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        # Control frames
        elif isinstance(frame, EndFrame):
            # Push EndFrame before stop(), because stop() waits on the task to
            # finish and the task finishes when EndFrame is processed.
            await self.push_frame(frame, direction)
            await self._stop(frame)
        # Other frames
        else:
            await self.push_frame(frame, direction)

    async def _start(self, frame: StartFrame):
        self._sample_rate = self._out_params.audio_sample_rate or frame.audio_out_sample_rate
        self._player.set_state(Gst.State.PLAYING)

    async def _stop(self, frame: EndFrame):
        self._player.set_state(Gst.State.NULL)

    async def _cancel(self, frame: CancelFrame):
        self._player.set_state(Gst.State.NULL)

    #
    # GStreamer
    #

    def _on_gstreamer_message(self, bus: Gst.Bus, message: Gst.Message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"{self} error: {err} : {debug}")
        return True

    def _decodebin_callback(self, decodebin: Gst.Element, pad: Gst.Pad):
        caps_string = pad.get_current_caps().to_string()
        if caps_string.startswith("audio"):
            self._decodebin_audio(pad)
        elif caps_string.startswith("video"):
            self._decodebin_video(pad)

    def _decodebin_audio(self, pad: Gst.Pad):
        queue_audio = Gst.ElementFactory.make("queue", None)
        audioconvert = Gst.ElementFactory.make("audioconvert", None)
        audioresample = Gst.ElementFactory.make("audioresample", None)
        audiocapsfilter = Gst.ElementFactory.make("capsfilter", None)
        audiocaps = Gst.Caps.from_string(
            f"audio/x-raw,format=S16LE,rate={self._sample_rate},channels={self._out_params.audio_channels},layout=interleaved"
        )
        audiocapsfilter.set_property("caps", audiocaps)
        appsink_audio = Gst.ElementFactory.make("appsink", None)
        appsink_audio.set_property("emit-signals", True)
        appsink_audio.set_property("sync", self._out_params.clock_sync)
        appsink_audio.connect("new-sample", self._appsink_audio_new_sample)

        self._player.add(queue_audio)
        self._player.add(audioconvert)
        self._player.add(audioresample)
        self._player.add(audiocapsfilter)
        self._player.add(appsink_audio)
        queue_audio.sync_state_with_parent()
        audioconvert.sync_state_with_parent()
        audioresample.sync_state_with_parent()
        audiocapsfilter.sync_state_with_parent()
        appsink_audio.sync_state_with_parent()

        queue_audio.link(audioconvert)
        audioconvert.link(audioresample)
        audioresample.link(audiocapsfilter)
        audiocapsfilter.link(appsink_audio)

        queue_pad = queue_audio.get_static_pad("sink")
        pad.link(queue_pad)

    def _decodebin_video(self, pad: Gst.Pad):
        queue_video = Gst.ElementFactory.make("queue", None)
        videoconvert = Gst.ElementFactory.make("videoconvert", None)
        videoscale = Gst.ElementFactory.make("videoscale", None)
        videocapsfilter = Gst.ElementFactory.make("capsfilter", None)
        videocaps = Gst.Caps.from_string(
            f"video/x-raw,format=RGB,width={self._out_params.video_width},height={self._out_params.video_height}"
        )
        videocapsfilter.set_property("caps", videocaps)

        appsink_video = Gst.ElementFactory.make("appsink", None)
        appsink_video.set_property("emit-signals", True)
        appsink_video.set_property("sync", self._out_params.clock_sync)
        appsink_video.connect("new-sample", self._appsink_video_new_sample)

        self._player.add(queue_video)
        self._player.add(videoconvert)
        self._player.add(videoscale)
        self._player.add(videocapsfilter)
        self._player.add(appsink_video)
        queue_video.sync_state_with_parent()
        videoconvert.sync_state_with_parent()
        videoscale.sync_state_with_parent()
        videocapsfilter.sync_state_with_parent()
        appsink_video.sync_state_with_parent()

        queue_video.link(videoconvert)
        videoconvert.link(videoscale)
        videoscale.link(videocapsfilter)
        videocapsfilter.link(appsink_video)

        queue_pad = queue_video.get_static_pad("sink")
        pad.link(queue_pad)

    def _appsink_audio_new_sample(self, appsink: GstApp.AppSink):
        buffer = appsink.pull_sample().get_buffer()
        (_, info) = buffer.map(Gst.MapFlags.READ)
        frame = OutputAudioRawFrame(
            audio=info.data,
            sample_rate=self._sample_rate,
            num_channels=self._out_params.audio_channels,
        )
        asyncio.run_coroutine_threadsafe(self.push_frame(frame), self.get_event_loop())
        buffer.unmap(info)
        return Gst.FlowReturn.OK

    def _appsink_video_new_sample(self, appsink: GstApp.AppSink):
        buffer = appsink.pull_sample().get_buffer()
        (_, info) = buffer.map(Gst.MapFlags.READ)
        frame = OutputImageRawFrame(
            image=info.data,
            size=(self._out_params.video_width, self._out_params.video_height),
            format="RGB",
        )
        asyncio.run_coroutine_threadsafe(self.push_frame(frame), self.get_event_loop())
        buffer.unmap(info)
        return Gst.FlowReturn.OK
