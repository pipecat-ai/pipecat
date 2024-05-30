#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import itertools

from PIL import Image
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
    StartInterruptionFrame,
    StopInterruptionFrame,
    TransportMessageFrame)
from pipecat.transports.base_transport import TransportParams

from loguru import logger


class BaseOutputTransport(FrameProcessor):

    def __init__(self, params: TransportParams):
        super().__init__()

        self._params = params

        self._allow_interruptions = False

        # These are the images that we should send to the camera at our desired
        # framerate.
        self._camera_images = None

        # Create media task queues.
        if self._params.camera_out_enabled:
            self._camera_out_queue = asyncio.Queue()
        self._sink_queue = asyncio.Queue()

        self._stopped_event = asyncio.Event()
        self._is_interrupted = asyncio.Event()

    async def start(self, frame: StartFrame):
        # Make sure we have the latest params. Note that this transport might
        # have been started on another task that might not need interruptions,
        # for example.
        self._allow_interruptions = frame.allow_interruptions

        loop = self.get_event_loop()

        if self._params.camera_out_enabled:
            self._camera_out_task = loop.create_task(self._camera_out_task_handler())

        self._sink_task = loop.create_task(self._sink_task_handler())

        # Create push frame task. This is the task that will push frames in
        # order. We also guarantee that all frames are pushed in the same task.
        self._create_push_task()

    async def stop(self):
        self._stopped_event.set()

    async def cleanup(self):
        if self._params.camera_out_enabled:
            self._camera_out_task.cancel()

        self._sink_task.cancel()
        self._push_frame_task.cancel()

    async def send_message(self, frame: TransportMessageFrame):
        pass

    async def write_frame_to_camera(self, frame: ImageRawFrame):
        pass

    async def write_raw_audio_frames(self, frames: bytes):
        pass

    #
    # Frame processor
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        #
        # Out-of-band frames like (CancelFrame or StartInterruptionFrame) are
        # pushed immediately. Other frames require order so they are put in the
        # sink queue.
        #
        if isinstance(frame, StartFrame):
            await self.start(frame)
            await self._sink_queue.put(frame)
        # EndFrame is managed in the queue handler.
        elif isinstance(frame, CancelFrame):
            await self.stop()
            await self.push_frame(frame, direction)
        elif isinstance(frame, StartInterruptionFrame) or isinstance(frame, StopInterruptionFrame):
            await self._handle_interruptions(frame)
            await self.push_frame(frame, direction)
        else:
            await self._sink_queue.put(frame)

        # If we are finishing, wait here until we have stopped, otherwise we might
        # close things too early upstream. We need this event because we don't
        # know when the internal tasks will finish.
        if isinstance(frame, CancelFrame) or isinstance(frame, EndFrame):
            await self._stopped_event.wait()

    async def _handle_interruptions(self, frame: Frame):
        if not self._allow_interruptions:
            return

        if isinstance(frame, StartInterruptionFrame):
            self._is_interrupted.set()
            self._push_frame_task.cancel()
            self._create_push_task()
        elif isinstance(frame, StopInterruptionFrame):
            self._is_interrupted.clear()

    async def _sink_task_handler(self):
        # 10ms bytes
        bytes_size_10ms = int(self._params.audio_out_sample_rate / 100) * \
            self._params.audio_out_channels * 2

        # We will send at least 100ms bytes.
        smallest_write_size = bytes_size_10ms * 10

        # Audio accumlation buffer
        buffer = bytearray()
        while True:
            try:
                frame = await self._sink_queue.get()
                if not self._is_interrupted.is_set():
                    if isinstance(frame, AudioRawFrame):
                        if self._params.audio_out_enabled:
                            buffer.extend(frame.audio)
                            buffer = await self._send_audio_truncated(buffer, smallest_write_size)
                    elif isinstance(frame, ImageRawFrame) and self._params.camera_out_enabled:
                        await self._set_camera_image(frame)
                    elif isinstance(frame, SpriteFrame) and self._params.camera_out_enabled:
                        await self._set_camera_images(frame.images)
                    elif isinstance(frame, TransportMessageFrame):
                        await self.send_message(frame)
                    else:
                        await self._internal_push_frame(frame)
                else:
                    # If we get interrupted just clear the output buffer.
                    buffer = bytearray()

                if isinstance(frame, EndFrame):
                    # Send all remaining audio before stopping (multiple of 10ms of audio).
                    await self._send_audio_truncated(buffer, bytes_size_10ms)
                    await self.stop()

                self._sink_queue.task_done()
            except asyncio.CancelledError:
                break
            except BaseException as e:
                logger.error(f"Error processing sink queue: {e}")

    #
    # Push frames task
    #

    def _create_push_task(self):
        loop = self.get_event_loop()
        self._push_frame_task = loop.create_task(self._push_frame_task_handler())
        self._push_queue = asyncio.Queue()

    async def _internal_push_frame(
            self,
            frame: Frame | None,
            direction: FrameDirection | None = FrameDirection.DOWNSTREAM):
        await self._push_queue.put((frame, direction))

    async def _push_frame_task_handler(self):
        while True:
            try:
                (frame, direction) = await self._push_queue.get()
                await self.push_frame(frame, direction)
            except asyncio.CancelledError:
                break

    #
    # Camera out
    #

    async def send_image(self, frame: ImageRawFrame | SpriteFrame):
        await self.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def _draw_image(self, frame: ImageRawFrame):
        desired_size = (self._params.camera_out_width, self._params.camera_out_height)

        if frame.size != desired_size:
            image = Image.frombytes(frame.format, frame.size, frame.image)
            resized_image = image.resize(desired_size)
            logger.warning(
                f"{frame} does not have the expected size {desired_size}, resizing")
            frame = ImageRawFrame(resized_image.tobytes(), resized_image.size, resized_image.format)

        await self.write_frame_to_camera(frame)

    async def _set_camera_image(self, image: ImageRawFrame):
        if self._params.camera_out_is_live:
            await self._camera_out_queue.put(image)
        else:
            self._camera_images = itertools.cycle([image])

    async def _set_camera_images(self, images: List[ImageRawFrame]):
        self._camera_images = itertools.cycle(images)

    async def _camera_out_task_handler(self):
        while True:
            try:
                if self._params.camera_out_is_live:
                    image = await self._camera_out_queue.get()
                    await self._draw_image(image)
                    self._camera_out_queue.task_done()
                elif self._camera_images:
                    image = next(self._camera_images)
                    await self._draw_image(image)
                    await asyncio.sleep(1.0 / self._params.camera_out_framerate)
                else:
                    await asyncio.sleep(1.0 / self._params.camera_out_framerate)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error writing to camera: {e}")

    #
    # Audio out
    #

    async def send_audio(self, frame: AudioRawFrame):
        await self.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def _send_audio_truncated(self, buffer: bytearray, smallest_write_size: int) -> bytearray:
        truncated_length: int = len(buffer) - (len(buffer) % smallest_write_size)
        if truncated_length:
            await self.write_raw_audio_frames(bytes(buffer[:truncated_length]))
            buffer = buffer[truncated_length:]
        return buffer
