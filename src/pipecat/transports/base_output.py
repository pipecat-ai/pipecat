#
# Copyright (c) 2024, Daily

#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import itertools
import queue
import threading
import time

from typing import List

from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    SpriteFrame,
    StartFrame,
    EndFrame,
    Frame,
    ImageRawFrame,
    TransportMessageFrame)
from pipecat.transports.base_transport import TransportParams

from loguru import logger


class BaseOutputTransport(FrameProcessor):

    def __init__(self, params: TransportParams):
        super().__init__()

        self._params = params

        self._running = False

        # These are the images that we should send to the camera at our desired
        # framerate.
        self._camera_images = None

        # Start media threads.
        if self._params.camera_out_enabled:
            self._camera_out_queue = queue.Queue()
            self._camera_out_thread = threading.Thread(target=self._camera_out_thread_handler)

        self._sink_queue = queue.Queue()
        self._sink_thread = threading.Thread(target=self._sink_thread_handler)

        self._stopped_event = asyncio.Event()

    async def start(self):
        if self._running:
            return

        self._running = True

        if self._params.camera_out_enabled:
            self._camera_out_thread.start()

        self._sink_thread.start()

    async def stop(self):
        # This will exit all threads.
        self._running = False

        self._stopped_event.set()

    def send_message(self, frame: TransportMessageFrame):
        pass

    def write_frame_to_camera(self, frame: ImageRawFrame):
        pass

    def write_raw_audio_frames(self, frames: bytes):
        pass

    #
    # Frame processor
    #

    async def cleanup(self):
        if self._params.camera_out_enabled:
            self._camera_out_thread.join()

        self._sink_thread.join()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self.start()
        # EndFrame is managed in the queue handler.
        elif isinstance(frame, CancelFrame):
            await self.push_frame(frame, direction)
            await self.stop()
        elif self._frame_managed_by_sink(frame):
            self._sink_queue.put(frame)
        else:
            await self.push_frame(frame, direction)

        # If we are finishing, wait here until we have stopped, otherwise we might
        # close things too early upstream.
        if isinstance(frame, CancelFrame) or isinstance(frame, EndFrame):
            await self._stopped_event.wait()

    def _frame_managed_by_sink(self, frame: Frame):
        return (isinstance(frame, AudioRawFrame)
                or isinstance(frame, ImageRawFrame)
                or isinstance(frame, SpriteFrame)
                or isinstance(frame, TransportMessageFrame)
                or isinstance(frame, CancelFrame)
                or isinstance(frame, EndFrame))

    def _sink_thread_handler(self):
        buffer = bytearray()
        bytes_size_10ms = int(self._params.audio_out_sample_rate / 100) * \
            self._params.audio_out_channels * 2
        while self._running:
            try:
                frame = self._sink_queue.get(timeout=1)
                if isinstance(frame, CancelFrame) or isinstance(frame, EndFrame):
                    # Send all remaining audio before stopping (multiple of 10ms of audio).
                    self._send_audio_truncated(buffer, bytes_size_10ms)
                    future = asyncio.run_coroutine_threadsafe(self.stop(), self.get_event_loop())
                    future.result()
                elif isinstance(frame, AudioRawFrame):
                    if self._params.audio_out_enabled:
                        buffer.extend(frame.audio)
                        buffer = self._send_audio_truncated(buffer, bytes_size_10ms)
                elif isinstance(frame, ImageRawFrame) and self._params.camera_out_enabled:
                    self._set_camera_image(frame)
                elif isinstance(frame, SpriteFrame) and self._params.camera_out_enabled:
                    self._set_camera_images(frame.images)
                elif isinstance(frame, TransportMessageFrame):
                    self.send_message(frame)
            except queue.Empty:
                pass
            except BaseException as e:
                logger.error(f"Error processing sink queue: {e}")

    #
    # Camera out
    #

    async def send_image(self, frame: ImageRawFrame | SpriteFrame):
        await self.process_frame(frame, FrameDirection.DOWNSTREAM)

    def _draw_image(self, image: ImageRawFrame):
        desired_size = (self._params.camera_out_width, self._params.camera_out_height)

        if image.size != desired_size:
            logger.warning(
                f"{image} does not have the expected size {desired_size}, ignoring")
            return

        self.write_frame_to_camera(image)

    def _set_camera_image(self, image: ImageRawFrame):
        if self._params.camera_out_is_live:
            self._camera_out_queue.put(image)
        else:
            self._camera_images = itertools.cycle([image])

    def _set_camera_images(self, images: List[ImageRawFrame]):
        self._camera_images = itertools.cycle(images)

    def _camera_out_thread_handler(self):
        while self._running:
            try:
                if self._params.camera_out_is_live:
                    image = self._camera_out_queue.get(timeout=1)
                    self._draw_image(image)
                elif self._camera_images:
                    image = next(self._camera_images)
                    self._draw_image(image)
                    time.sleep(1.0 / self._params.camera_out_framerate)
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Error writing to camera: {e}")

    #
    # Audio out
    #

    async def send_audio(self, frame: AudioRawFrame):
        await self.process_frame(frame, FrameDirection.DOWNSTREAM)

    def _send_audio_truncated(self, buffer: bytearray, smallest_write_size: int) -> bytearray:
        try:
            truncated_length: int = len(buffer) - (len(buffer) % smallest_write_size)
            if truncated_length:
                self.write_raw_audio_frames(bytes(buffer[:truncated_length]))
                buffer = buffer[truncated_length:]
            return buffer
        except BaseException as e:
            logger.error(f"Error writing audio frames: {e}")
            return buffer
