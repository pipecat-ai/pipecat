#
# Copyright (c) 2024, Daily

#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import itertools
import time
import sys

from PIL import Image
from typing import List

from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (
    AudioRawFrame,
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    MetricsFrame,
    SpriteFrame,
    StartFrame,
    EndFrame,
    Frame,
    ImageRawFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    SystemFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TextFrame,
    TransportMessageFrame)
from pipecat.transports.base_transport import TransportParams

from loguru import logger

from pipecat.utils.time import nanoseconds_to_seconds


class BaseOutputTransport(FrameProcessor):

    def __init__(self, params: TransportParams, **kwargs):
        super().__init__(sync=False, **kwargs)

        self._params = params

        # These are the images that we should send to the camera at our desired
        # framerate.
        self._camera_images = None

        # We will write 20ms audio at a time. If we receive long audio frames we
        # will chunk them. This will help with interruption handling.
        audio_bytes_10ms = int(self._params.audio_out_sample_rate / 100) * \
            self._params.audio_out_channels * 2
        self._audio_chunk_size = audio_bytes_10ms * 2
        self._audio_buffer = bytearray()

        self._stopped_event = asyncio.Event()

        # Indicates if the bot is currently speaking. This is useful when we
        # have an interruption since all the queued messages will be thrown
        # away and we would lose the TTSStoppedFrame.
        self._bot_speaking = False

        # Create sink frame task. This is the task that will actually write
        # audio or video frames. We write audio/video in a task so we can keep
        # generating frames upstream while, for example, the audio is playing.
        self._create_sink_tasks()

    async def start(self, frame: StartFrame):
        # Create camera output queue and task if needed.
        if self._params.camera_out_enabled:
            self._camera_out_queue = asyncio.Queue()
            self._camera_out_task = self.get_event_loop().create_task(self._camera_out_task_handler())
        # Create audio output queue and task if needed.
        if self._params.audio_out_enabled and self._params.audio_out_is_live:
            self._audio_out_queue = asyncio.Queue()
            self._audio_out_task = self.get_event_loop().create_task(self._audio_out_task_handler())

    async def stop(self, frame: EndFrame):
        # Cancel and wait for the camera output task to finish.
        if self._params.camera_out_enabled:
            self._camera_out_task.cancel()
            await self._camera_out_task

        # Cancel and wait for the audio output task to finish.
        if self._params.audio_out_enabled and self._params.audio_out_is_live:
            self._audio_out_task.cancel()
            await self._audio_out_task

        # Wait for the sink task to finish. They will finish when the EndFrame
        # is actually processed.
        await self._sink_task

    async def cancel(self, frame: CancelFrame):
        # Cancel all the tasks and wait for them to finish.

        if self._params.camera_out_enabled:
            self._camera_out_task.cancel()
            await self._camera_out_task

        self._sink_task.cancel()
        await self._sink_task

    async def send_message(self, frame: TransportMessageFrame):
        pass

    async def send_metrics(self, frame: MetricsFrame):
        pass

    async def write_frame_to_camera(self, frame: ImageRawFrame):
        pass

    async def write_raw_audio_frames(self, frames: bytes):
        pass

    #
    # Frame processor
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        #
        # System frames (like StartInterruptionFrame) are pushed
        # immediately. Other frames require order so they are put in the sink
        # queue.
        #
        if isinstance(frame, CancelFrame):
            await self.cancel(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, StartInterruptionFrame) or isinstance(frame, StopInterruptionFrame):
            await self.push_frame(frame, direction)
            await self._handle_interruptions(frame)
        elif isinstance(frame, MetricsFrame):
            await self.push_frame(frame, direction)
            await self.send_metrics(frame)
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        # Control frames.
        elif isinstance(frame, StartFrame):
            await self._sink_queue.put(frame)
            await self.start(frame)
        elif isinstance(frame, EndFrame):
            await self._sink_clock_queue.put((sys.maxsize, frame.id, frame))
            await self._sink_queue.put(frame)
            await self.stop(frame)
        # Other frames.
        elif isinstance(frame, AudioRawFrame):
            await self._handle_audio(frame)
        elif isinstance(frame, ImageRawFrame) or isinstance(frame, SpriteFrame):
            await self._handle_image(frame)
        elif isinstance(frame, TransportMessageFrame) and frame.urgent:
            await self.send_message(frame)
        # TODO(aleix): Images and audio should support presentation timestamps.
        elif frame.pts:
            await self._sink_clock_queue.put((frame.pts, frame.id, frame))
        else:
            await self._sink_queue.put(frame)

    async def _handle_interruptions(self, frame: Frame):
        if not self.interruptions_allowed:
            return

        if isinstance(frame, StartInterruptionFrame):
            # Stop sink tasks.
            self._sink_task.cancel()
            await self._sink_task
            # Stop sink clock tasks.
            self._sink_clock_task.cancel()
            await self._sink_clock_task
            # Create sink tasks.
            self._create_sink_tasks()
            # Let's send a bot stopped speaking if we have to.
            if self._bot_speaking:
                await self._bot_stopped_speaking()

    async def _handle_audio(self, frame: AudioRawFrame):
        if not self._params.audio_out_enabled:
            return

        if self._params.audio_out_is_live:
            await self._audio_out_queue.put(frame)
        else:
            self._audio_buffer.extend(frame.audio)
            while len(self._audio_buffer) >= self._audio_chunk_size:
                chunk = AudioRawFrame(bytes(self._audio_buffer[:self._audio_chunk_size]),
                                      sample_rate=frame.sample_rate, num_channels=frame.num_channels)
                await self._sink_queue.put(chunk)
                self._audio_buffer = self._audio_buffer[self._audio_chunk_size:]

    async def _handle_image(self, frame: ImageRawFrame | SpriteFrame):
        if not self._params.camera_out_enabled:
            return

        if self._params.camera_out_is_live:
            await self._camera_out_queue.put(frame)
        else:
            await self._sink_queue.put(frame)

    #
    # Sink tasks
    #

    def _create_sink_tasks(self):
        loop = self.get_event_loop()
        self._sink_queue = asyncio.Queue()
        self._sink_task = loop.create_task(self._sink_task_handler())
        self._sink_clock_queue = asyncio.PriorityQueue()
        self._sink_clock_task = loop.create_task(self._sink_clock_task_handler())

    async def _sink_frame_handler(self, frame: Frame):
        if isinstance(frame, AudioRawFrame):
            await self.write_raw_audio_frames(frame.audio)
            await self.push_frame(frame)
            await self.push_frame(BotSpeakingFrame(), FrameDirection.UPSTREAM)
        elif isinstance(frame, ImageRawFrame):
            await self._set_camera_image(frame)
        elif isinstance(frame, SpriteFrame):
            await self._set_camera_images(frame.images)
        elif isinstance(frame, TransportMessageFrame):
            await self.send_message(frame)
        elif isinstance(frame, TTSStartedFrame):
            await self._bot_started_speaking()
            await self.push_frame(frame)
        elif isinstance(frame, TTSStoppedFrame):
            await self._bot_stopped_speaking()
            await self.push_frame(frame)
        else:
            await self.push_frame(frame)

    async def _sink_task_handler(self):
        running = True
        while running:
            try:
                frame = await self._sink_queue.get()
                await self._sink_frame_handler(frame)
                running = not isinstance(frame, EndFrame)
                self._sink_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"{self} error processing sink queue: {e}")

    async def _sink_clock_frame_handler(self, frame: Frame):
        # TODO(aleix): For now we just process TextFrame. But we should process
        # audio and video as well.
        if isinstance(frame, TextFrame):
            await self.push_frame(frame)

    async def _sink_clock_task_handler(self):
        running = True
        while running:
            try:
                timestamp, _, frame = await self._sink_clock_queue.get()

                # If we hit an EndFrame, we cna finish right away.
                running = not isinstance(frame, EndFrame)

                # If we have a frame we check it's presentation timestamp. If it
                # has already passed we process it, otherwise we wait until it's
                # time to process it.
                if running:
                    current_time = self.get_clock().get_time()
                    if timestamp <= current_time:
                        await self._sink_clock_frame_handler(frame)
                    else:
                        wait_time = nanoseconds_to_seconds(timestamp - current_time)
                        await asyncio.sleep(wait_time)
                        await self._sink_frame_handler(frame)

                self._sink_clock_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"{self} error processing sink clock queue: {e}")

    async def _bot_started_speaking(self):
        logger.debug("Bot started speaking")
        self._bot_speaking = True
        await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)

    async def _bot_stopped_speaking(self):
        logger.debug("Bot stopped speaking")
        self._bot_speaking = False
        await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

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
        self._camera_images = itertools.cycle([image])

    async def _set_camera_images(self, images: List[ImageRawFrame]):
        self._camera_images = itertools.cycle(images)

    async def _camera_out_task_handler(self):
        self._camera_out_start_time = None
        self._camera_out_frame_index = 0
        self._camera_out_frame_duration = 1 / self._params.camera_out_framerate
        self._camera_out_frame_reset = self._camera_out_frame_duration * 5
        while True:
            try:
                if self._params.camera_out_is_live:
                    await self._camera_out_is_live_handler()
                elif self._camera_images:
                    image = next(self._camera_images)
                    await self._draw_image(image)
                    await asyncio.sleep(1.0 / self._params.camera_out_framerate)
                else:
                    await asyncio.sleep(1.0 / self._params.camera_out_framerate)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"{self} error writing to camera: {e}")

    async def _camera_out_is_live_handler(self):
        image = await self._camera_out_queue.get()

        # We get the start time as soon as we get the first image.
        if not self._camera_out_start_time:
            self._camera_out_start_time = time.time()
            self._camera_out_frame_index = 0

        # Calculate how much time we need to wait before rendering next image.
        real_elapsed_time = time.time() - self._camera_out_start_time
        real_render_time = self._camera_out_frame_index * self._camera_out_frame_duration
        delay_time = self._camera_out_frame_duration + real_render_time - real_elapsed_time

        if abs(delay_time) > self._camera_out_frame_reset:
            self._camera_out_start_time = time.time()
            self._camera_out_frame_index = 0
        elif delay_time > 0:
            await asyncio.sleep(delay_time)
            self._camera_out_frame_index += 1

        # Render image
        await self._draw_image(image)

        self._camera_out_queue.task_done()

    #
    # Audio out
    #

    async def send_audio(self, frame: AudioRawFrame):
        await self.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def _audio_out_task_handler(self):
        while True:
            try:
                frame = await self._audio_out_queue.get()
                await self.write_raw_audio_frames(frame.audio)
                await self.push_frame(frame)
                await self.push_frame(BotSpeakingFrame(), FrameDirection.UPSTREAM)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"{self} error writing to camera: {e}")
