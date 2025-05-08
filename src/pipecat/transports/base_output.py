#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import itertools
import sys
import time
from typing import Any, AsyncGenerator, Dict, List, Mapping, Optional

from loguru import logger
from PIL import Image

from pipecat.audio.mixers.base_audio_mixer import BaseAudioMixer
from pipecat.audio.utils import create_default_resampler
from pipecat.frames.frames import (
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    MixerControlFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    SpriteFrame,
    StartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    SystemFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.time import nanoseconds_to_seconds

BOT_VAD_STOP_SECS = 0.35


class BaseOutputTransport(FrameProcessor):
    def __init__(self, params: TransportParams, **kwargs):
        super().__init__(**kwargs)

        self._params = params

        # Output sample rate. It will be initialized on StartFrame.
        self._sample_rate = 0

        # We write 10ms*CHUNKS of audio at a time (where CHUNKS is the
        # `audio_out_10ms_chunks` parameter). If we receive long audio frames we
        # will chunk them. This helps with interruption handling. It will be
        # initialized on StartFrame.
        self._audio_chunk_size = 0

        # We will have one media sender per output frame destination. This allow
        # us to send multiple streams at the same time if the transport allows
        # it.
        self._media_senders: Dict[Any, "BaseOutputTransport.MediaSender"] = {}

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def audio_chunk_size(self) -> int:
        return self._audio_chunk_size

    async def start(self, frame: StartFrame):
        self._sample_rate = self._params.audio_out_sample_rate or frame.audio_out_sample_rate

        # We will write 10ms*CHUNKS of audio at a time (where CHUNKS is the
        # `audio_out_10ms_chunks` parameter). If we receive long audio frames we
        # will chunk them. This will help with interruption handling.
        audio_bytes_10ms = int(self._sample_rate / 100) * self._params.audio_out_channels * 2
        self._audio_chunk_size = audio_bytes_10ms * self._params.audio_out_10ms_chunks

    async def stop(self, frame: EndFrame):
        for _, sender in self._media_senders.items():
            await sender.stop(frame)

    async def cancel(self, frame: CancelFrame):
        for _, sender in self._media_senders.items():
            await sender.cancel(frame)

    async def set_transport_ready(self, frame: StartFrame):
        """To be called when the transport is ready to stream."""
        # Register destinations.
        for destination in self._params.audio_out_destinations:
            await self.register_audio_destination(destination)

        for destination in self._params.video_out_destinations:
            await self.register_video_destination(destination)

        # Start default media sender.
        self._media_senders[None] = BaseOutputTransport.MediaSender(
            self,
            destination=None,
            sample_rate=self.sample_rate,
            audio_chunk_size=self.audio_chunk_size,
            params=self._params,
        )
        await self._media_senders[None].start(frame)

        # Media senders already send both audio and video, so make sure we only
        # have one media server per shared name.
        destinations = list(
            set(self._params.audio_out_destinations + self._params.video_out_destinations)
        )

        # Start media senders.
        for destination in destinations:
            self._media_senders[destination] = BaseOutputTransport.MediaSender(
                self,
                destination=destination,
                sample_rate=self.sample_rate,
                audio_chunk_size=self.audio_chunk_size,
                params=self._params,
            )
            await self._media_senders[destination].start(frame)

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        pass

    async def register_video_destination(self, destination: str):
        pass

    async def register_audio_destination(self, destination: str):
        pass

    async def write_raw_video_frame(
        self, frame: OutputImageRawFrame, destination: Optional[str] = None
    ):
        pass

    async def write_raw_audio_frames(self, frames: bytes, destination: Optional[str] = None):
        pass

    async def send_audio(self, frame: OutputAudioRawFrame):
        await self.queue_frame(frame, FrameDirection.DOWNSTREAM)

    async def send_image(self, frame: OutputImageRawFrame | SpriteFrame):
        await self.queue_frame(frame, FrameDirection.DOWNSTREAM)

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
        if isinstance(frame, StartFrame):
            # Push StartFrame before start(), because we want StartFrame to be
            # processed by every processor before any other frame is processed.
            await self.push_frame(frame, direction)
            await self.start(frame)
        elif isinstance(frame, CancelFrame):
            await self.cancel(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, (StartInterruptionFrame, StopInterruptionFrame)):
            await self.push_frame(frame, direction)
            await self._handle_frame(frame)
        elif isinstance(frame, TransportMessageUrgentFrame):
            await self.send_message(frame)
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        # Control frames.
        elif isinstance(frame, EndFrame):
            await self.stop(frame)
            # Keep pushing EndFrame down so all the pipeline stops nicely.
            await self.push_frame(frame, direction)
        elif isinstance(frame, MixerControlFrame):
            await self._handle_frame(frame)
        # Other frames.
        elif isinstance(frame, OutputAudioRawFrame):
            await self._handle_frame(frame)
        elif isinstance(frame, (OutputImageRawFrame, SpriteFrame)):
            await self._handle_frame(frame)
        # TODO(aleix): Images and audio should support presentation timestamps.
        elif frame.pts:
            await self._handle_frame(frame)
        elif direction == FrameDirection.UPSTREAM:
            await self.push_frame(frame, direction)
        else:
            await self._handle_frame(frame)

    async def _handle_frame(self, frame: Frame):
        if frame.transport_destination not in self._media_senders:
            logger.warning(
                f"{self} destination [{frame.transport_destination}] not registered for frame {frame}"
            )
            return

        sender = self._media_senders[frame.transport_destination]

        if isinstance(frame, StartInterruptionFrame):
            await sender.handle_interruptions(frame)
        elif isinstance(frame, OutputAudioRawFrame):
            await sender.handle_audio_frame(frame)
        elif isinstance(frame, (OutputImageRawFrame, SpriteFrame)):
            await sender.handle_image_frame(frame)
        elif isinstance(frame, MixerControlFrame):
            await sender.handle_mixer_control_frame(frame)
        elif frame.pts:
            await sender.handle_timed_frame(frame)
        else:
            await sender.handle_sync_frame(frame)

    #
    # Media Sender
    #

    class MediaSender:
        def __init__(
            self,
            transport: "BaseOutputTransport",
            *,
            destination: Optional[str],
            sample_rate: int,
            audio_chunk_size: int,
            params: TransportParams,
        ):
            self._transport = transport
            self._destination = destination
            self._sample_rate = sample_rate
            self._audio_chunk_size = audio_chunk_size
            self._params = params

            # Buffer to keep track of incoming audio.
            self._audio_buffer = bytearray()

            # This will be used to resample incoming audio to the output sample rate.
            self._resampler = create_default_resampler()

            # The user can provide a single mixer, to be used by the default
            # destination, or a destination/mixer mapping.
            self._mixer: Optional[BaseAudioMixer] = None

            # These are the images that we should send at our desired framerate.
            self._video_images = None

            # Indicates if the bot is currently speaking.
            self._bot_speaking = False

            self._audio_task: Optional[asyncio.Task] = None
            self._video_task: Optional[asyncio.Task] = None
            self._clock_task: Optional[asyncio.Task] = None

        @property
        def sample_rate(self) -> int:
            return self._sample_rate

        @property
        def audio_chunk_size(self) -> int:
            return self._audio_chunk_size

        async def start(self, frame: StartFrame):
            self._audio_buffer = bytearray()

            # Create all tasks.
            self._create_video_task()
            self._create_clock_task()
            self._create_audio_task()

            # Check if we have an audio mixer for our destination.
            if self._params.audio_out_mixer:
                if isinstance(self._params.audio_out_mixer, Mapping):
                    self._mixer = self._params.audio_out_mixer.get(self._destination, None)
                elif not self._destination:
                    # Only use the default mixer if we are the default destination.
                    self._mixer = self._params.audio_out_mixer

            # Start audio mixer.
            if self._mixer:
                await self._mixer.start(self._sample_rate)

        async def stop(self, frame: EndFrame):
            # Let the sink tasks process the queue until they reach this EndFrame.
            await self._clock_queue.put((sys.maxsize, frame.id, frame))
            await self._audio_queue.put(frame)

            # At this point we have enqueued an EndFrame and we need to wait for
            # that EndFrame to be processed by the audio and clock tasks. We
            # also need to wait for these tasks before cancelling the video task
            # because it might be still rendering.
            if self._audio_task:
                await self._transport.wait_for_task(self._audio_task)
            if self._clock_task:
                await self._transport.wait_for_task(self._clock_task)

            # Stop audio mixer.
            if self._mixer:
                await self._mixer.stop()

            # We can now cancel the video task.
            await self._cancel_video_task()

        async def cancel(self, frame: CancelFrame):
            # Since we are cancelling everything it doesn't matter what task we cancel first.
            await self._cancel_audio_task()
            await self._cancel_clock_task()
            await self._cancel_video_task()

        async def handle_interruptions(self, _: StartInterruptionFrame):
            if not self._transport.interruptions_allowed:
                return

            # Cancel tasks.
            await self._cancel_audio_task()
            await self._cancel_clock_task()
            await self._cancel_video_task()
            # Create tasks.
            self._create_video_task()
            self._create_clock_task()
            self._create_audio_task()
            # Let's send a bot stopped speaking if we have to.
            await self._bot_stopped_speaking()

        async def handle_audio_frame(self, frame: OutputAudioRawFrame):
            if not self._params.audio_out_enabled:
                return

            # We might need to resample if incoming audio doesn't match the
            # transport sample rate.
            resampled = await self._resampler.resample(
                frame.audio, frame.sample_rate, self._sample_rate
            )

            cls = type(frame)
            self._audio_buffer.extend(resampled)
            while len(self._audio_buffer) >= self._audio_chunk_size:
                chunk = cls(
                    bytes(self._audio_buffer[: self._audio_chunk_size]),
                    sample_rate=self._sample_rate,
                    num_channels=frame.num_channels,
                )
                await self._audio_queue.put(chunk)
                self._audio_buffer = self._audio_buffer[self._audio_chunk_size :]

        async def handle_image_frame(self, frame: OutputImageRawFrame | SpriteFrame):
            if not self._params.video_out_enabled:
                return

            if self._params.video_out_is_live and isinstance(frame, OutputImageRawFrame):
                await self._video_queue.put(frame)
            elif isinstance(frame, OutputImageRawFrame):
                await self._set_video_image(frame)
            else:
                await self._set_video_images(frame.images)

        async def handle_timed_frame(self, frame: Frame):
            await self._clock_queue.put((frame.pts, frame.id, frame))

        async def handle_sync_frame(self, frame: Frame):
            await self._audio_queue.put(frame)

        async def handle_mixer_control_frame(self, frame: MixerControlFrame):
            if self._mixer:
                await self._mixer.process_frame(frame)

        #
        # Audio handling
        #

        def _create_audio_task(self):
            if not self._audio_task:
                self._audio_queue = asyncio.Queue()
                self._audio_task = self._transport.create_task(self._audio_task_handler())

        async def _cancel_audio_task(self):
            if self._audio_task:
                await self._transport.cancel_task(self._audio_task)
                self._audio_task = None

        async def _bot_started_speaking(self):
            if not self._bot_speaking:
                logger.debug(
                    f"Bot{f' [{self._destination}]' if self._destination else ''} started speaking"
                )

                downstream_frame = BotStartedSpeakingFrame()
                downstream_frame.transport_destination = self._destination
                upstream_frame = BotStartedSpeakingFrame()
                upstream_frame.transport_destination = self._destination
                await self._transport.push_frame(downstream_frame)
                await self._transport.push_frame(upstream_frame, FrameDirection.UPSTREAM)

                self._bot_speaking = True

        async def _bot_stopped_speaking(self):
            if self._bot_speaking:
                logger.debug(
                    f"Bot{f' [{self._destination}]' if self._destination else ''} stopped speaking"
                )

                downstream_frame = BotStoppedSpeakingFrame()
                downstream_frame.transport_destination = self._destination
                upstream_frame = BotStoppedSpeakingFrame()
                upstream_frame.transport_destination = self._destination
                await self._transport.push_frame(downstream_frame)
                await self._transport.push_frame(upstream_frame, FrameDirection.UPSTREAM)

                self._bot_speaking = False

                # Clean audio buffer (there could be tiny left overs if not multiple
                # to our output chunk size).
                self._audio_buffer = bytearray()

        async def _handle_frame(self, frame: Frame):
            if isinstance(frame, OutputImageRawFrame):
                await self._set_video_image(frame)
            elif isinstance(frame, SpriteFrame):
                await self._set_video_images(frame.images)
            elif isinstance(frame, TransportMessageFrame):
                await self._transport.send_message(frame)

        def _next_frame(self) -> AsyncGenerator[Frame, None]:
            async def without_mixer(vad_stop_secs: float) -> AsyncGenerator[Frame, None]:
                while True:
                    try:
                        frame = await asyncio.wait_for(
                            self._audio_queue.get(), timeout=vad_stop_secs
                        )
                        yield frame
                    except asyncio.TimeoutError:
                        # Notify the bot stopped speaking upstream if necessary.
                        await self._bot_stopped_speaking()

            async def with_mixer(vad_stop_secs: float) -> AsyncGenerator[Frame, None]:
                last_frame_time = 0
                silence = b"\x00" * self._audio_chunk_size
                while True:
                    try:
                        frame = self._audio_queue.get_nowait()
                        if isinstance(frame, OutputAudioRawFrame):
                            frame.audio = await self._mixer.mix(frame.audio)
                        last_frame_time = time.time()
                        yield frame
                    except asyncio.QueueEmpty:
                        # Notify the bot stopped speaking upstream if necessary.
                        diff_time = time.time() - last_frame_time
                        if diff_time > vad_stop_secs:
                            await self._bot_stopped_speaking()
                        # Generate an audio frame with only the mixer's part.
                        frame = OutputAudioRawFrame(
                            audio=await self._mixer.mix(silence),
                            sample_rate=self._sample_rate,
                            num_channels=self._params.audio_out_channels,
                        )
                        yield frame

            if self._mixer:
                return with_mixer(BOT_VAD_STOP_SECS)
            else:
                return without_mixer(BOT_VAD_STOP_SECS)

        async def _audio_task_handler(self):
            # Push a BotSpeakingFrame every 200ms, we don't really need to push it
            # at every audio chunk. If the audio chunk is bigger than 200ms, push at
            # every audio chunk.
            TOTAL_CHUNK_MS = self._params.audio_out_10ms_chunks * 10
            BOT_SPEAKING_CHUNK_PERIOD = max(int(200 / TOTAL_CHUNK_MS), 1)
            bot_speaking_counter = 0
            async for frame in self._next_frame():
                # Notify the bot started speaking upstream if necessary and that
                # it's actually speaking.
                if isinstance(frame, TTSAudioRawFrame):
                    await self._bot_started_speaking()
                    if bot_speaking_counter % BOT_SPEAKING_CHUNK_PERIOD == 0:
                        await self._transport.push_frame(BotSpeakingFrame())
                        await self._transport.push_frame(
                            BotSpeakingFrame(), FrameDirection.UPSTREAM
                        )
                        bot_speaking_counter = 0
                    bot_speaking_counter += 1

                # No need to push EndFrame, it's pushed from process_frame().
                if isinstance(frame, EndFrame):
                    break

                # Handle frame.
                await self._handle_frame(frame)

                # Also, push frame downstream in case anyone else needs it.
                await self._transport.push_frame(frame)

                # Send audio.
                if isinstance(frame, OutputAudioRawFrame):
                    await self._transport.write_raw_audio_frames(frame.audio, self._destination)

        #
        # Video handling
        #

        def _create_video_task(self):
            if not self._video_task and self._params.video_out_enabled:
                self._video_queue = asyncio.Queue()
                self._video_task = self._transport.create_task(self._video_task_handler())

        async def _cancel_video_task(self):
            # Stop video output task.
            if self._video_task:
                await self._transport.cancel_task(self._video_task)
                self._video_task = None

        async def _set_video_image(self, image: OutputImageRawFrame):
            self._video_images = itertools.cycle([image])

        async def _set_video_images(self, images: List[OutputImageRawFrame]):
            self._video_images = itertools.cycle(images)

        async def _video_task_handler(self):
            self._video_start_time = None
            self._video_frame_index = 0
            self._video_frame_duration = 1 / self._params.video_out_framerate
            self._video_frame_reset = self._video_frame_duration * 5
            while True:
                if self._params.video_out_is_live:
                    await self._video_is_live_handler()
                elif self._video_images:
                    image = next(self._video_images)
                    await self._draw_image(image)
                    await asyncio.sleep(self._video_frame_duration)
                else:
                    await asyncio.sleep(self._video_frame_duration)

        async def _video_is_live_handler(self):
            image = await self._video_queue.get()

            # We get the start time as soon as we get the first image.
            if not self._video_start_time:
                self._video_start_time = time.time()
                self._video_frame_index = 0

            # Calculate how much time we need to wait before rendering next image.
            real_elapsed_time = time.time() - self._video_start_time
            real_render_time = self._video_frame_index * self._video_frame_duration
            delay_time = self._video_frame_duration + real_render_time - real_elapsed_time

            if abs(delay_time) > self._video_frame_reset:
                self._video_start_time = time.time()
                self._video_frame_index = 0
            elif delay_time > 0:
                await asyncio.sleep(delay_time)
                self._video_frame_index += 1

            # Render image
            await self._draw_image(image)

            self._video_queue.task_done()

        async def _draw_image(self, frame: OutputImageRawFrame):
            desired_size = (self._params.video_out_width, self._params.video_out_height)

            # TODO: we should refactor in the future to support dynamic resolutions
            # which is kind of what happens in P2P connections.
            # We need to add support for that inside the DailyTransport
            if frame.size != desired_size:
                image = Image.frombytes(frame.format, frame.size, frame.image)
                resized_image = image.resize(desired_size)
                # logger.warning(f"{frame} does not have the expected size {desired_size}, resizing")
                frame = OutputImageRawFrame(
                    resized_image.tobytes(), resized_image.size, resized_image.format
                )

            await self._transport.write_raw_video_frame(frame, self._destination)

        #
        # Clock handling
        #

        def _create_clock_task(self):
            if not self._clock_task:
                self._clock_queue = asyncio.PriorityQueue()
                self._clock_task = self._transport.create_task(self._clock_task_handler())

        async def _cancel_clock_task(self):
            if self._clock_task:
                await self._transport.cancel_task(self._clock_task)
                self._clock_task = None

        async def _clock_task_handler(self):
            running = True
            while running:
                timestamp, _, frame = await self._clock_queue.get()

                # If we hit an EndFrame, we can finish right away.
                running = not isinstance(frame, EndFrame)

                # If we have a frame we check it's presentation timestamp. If it
                # has already passed we process it, otherwise we wait until it's
                # time to process it.
                if running:
                    current_time = self._transport.get_clock().get_time()
                    if timestamp > current_time:
                        wait_time = nanoseconds_to_seconds(timestamp - current_time)
                        await asyncio.sleep(wait_time)

                    # Push frame downstream.
                    await self._transport.push_frame(frame)

                self._clock_queue.task_done()
