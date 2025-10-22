#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base output transport implementation for Pipecat.

This module provides the BaseOutputTransport class which handles audio and video
output processing, including frame buffering, mixing, timing, and media streaming.
"""

import asyncio
import itertools
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator, Dict, List, Mapping, Optional

from loguru import logger
from PIL import Image

from pipecat.audio.dtmf.utils import load_dtmf_audio
from pipecat.audio.mixers.base_audio_mixer import BaseAudioMixer
from pipecat.audio.utils import create_stream_resampler, is_silence
from pipecat.frames.frames import (
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    MixerControlFrame,
    OutputAudioRawFrame,
    OutputDTMFFrame,
    OutputDTMFUrgentFrame,
    OutputImageRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    OutputTransportReadyFrame,
    SpeechOutputAudioRawFrame,
    SpriteFrame,
    StartFrame,
    SystemFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.time import nanoseconds_to_seconds

BOT_VAD_STOP_SECS = 0.35


class BaseOutputTransport(FrameProcessor):
    """Base class for output transport implementations.

    Handles audio and video output processing including frame buffering, audio mixing,
    timing coordination, and media streaming. Supports multiple output destinations
    and provides interruption handling for real-time communication.
    """

    def __init__(self, params: TransportParams, **kwargs):
        """Initialize the base output transport.

        Args:
            params: Transport configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
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
        """Get the current audio sample rate.

        Returns:
            The sample rate in Hz.
        """
        return self._sample_rate

    @property
    def audio_chunk_size(self) -> int:
        """Get the audio chunk size for output processing.

        Returns:
            The size of audio chunks in bytes.
        """
        return self._audio_chunk_size

    async def start(self, frame: StartFrame):
        """Start the output transport and initialize components.

        Args:
            frame: The start frame containing initialization parameters.
        """
        self._sample_rate = self._params.audio_out_sample_rate or frame.audio_out_sample_rate

        # We will write 10ms*CHUNKS of audio at a time (where CHUNKS is the
        # `audio_out_10ms_chunks` parameter). If we receive long audio frames we
        # will chunk them. This will help with interruption handling.
        audio_bytes_10ms = int(self._sample_rate / 100) * self._params.audio_out_channels * 2
        self._audio_chunk_size = audio_bytes_10ms * self._params.audio_out_10ms_chunks

    async def stop(self, frame: EndFrame):
        """Stop the output transport and cleanup resources.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        for _, sender in self._media_senders.items():
            await sender.stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the output transport and stop all processing.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        for _, sender in self._media_senders.items():
            await sender.cancel(frame)

    async def set_transport_ready(self, frame: StartFrame):
        """Called when the transport is ready to stream.

        Args:
            frame: The start frame containing initialization parameters.
        """
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

        # Sending a frame indicating that the output transport is ready and able to receive frames.
        await self.push_frame(OutputTransportReadyFrame(), FrameDirection.UPSTREAM)

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send a transport message.

        Args:
            frame: The transport message frame to send.
        """
        pass

    async def register_video_destination(self, destination: str):
        """Register a video output destination.

        Args:
            destination: The destination identifier to register.
        """
        pass

    async def register_audio_destination(self, destination: str):
        """Register an audio output destination.

        Args:
            destination: The destination identifier to register.
        """
        pass

    async def write_video_frame(self, frame: OutputImageRawFrame) -> bool:
        """Write a video frame to the transport.

        Args:
            frame: The output video frame to write.

        Returns:
            True if the video frame was written successfully, False otherwise.
        """
        return False

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the transport.

        Args:
            frame: The output audio frame to write.

        Returns:
            True if the audio frame was written successfully, False otherwise.
        """
        return False

    async def write_dtmf(self, frame: OutputDTMFFrame | OutputDTMFUrgentFrame):
        """Write a DTMF tone using the transport's preferred method.

        Args:
            frame: The DTMF frame to write.
        """
        if self._supports_native_dtmf():
            await self._write_dtmf_native(frame)
        else:
            await self._write_dtmf_audio(frame)

    def _supports_native_dtmf(self) -> bool:
        """Override in transport implementations that support native DTMF.

        Returns:
            True if the transport supports native DTMF, False otherwise.
        """
        return False

    async def _write_dtmf_native(self, frame: OutputDTMFFrame | OutputDTMFUrgentFrame):
        """Override in transport implementations for native DTMF.

        Args:
            frame: The DTMF frame to write.
        """
        raise NotImplementedError("Transport claims native DTMF support but doesn't implement it")

    async def _write_dtmf_audio(self, frame: OutputDTMFFrame | OutputDTMFUrgentFrame):
        """Generate and send audio tones for DTMF.

        Args:
            frame: The DTMF frame to write.
        """
        dtmf_audio = await load_dtmf_audio(frame.button, sample_rate=self._sample_rate)
        dtmf_audio_frame = OutputAudioRawFrame(
            audio=dtmf_audio, sample_rate=self._sample_rate, num_channels=1
        )
        await self.write_audio_frame(dtmf_audio_frame)

    async def send_audio(self, frame: OutputAudioRawFrame):
        """Send an audio frame downstream.

        Args:
            frame: The audio frame to send.
        """
        await self.queue_frame(frame, FrameDirection.DOWNSTREAM)

    async def send_image(self, frame: OutputImageRawFrame | SpriteFrame):
        """Send an image frame downstream.

        Args:
            frame: The image frame to send.
        """
        await self.queue_frame(frame, FrameDirection.DOWNSTREAM)

    #
    # Frame processor
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle transport-specific logic.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            # Push StartFrame before start(), because we want StartFrame to be
            # processed by every processor before any other frame is processed.
            await self.push_frame(frame, direction)
            await self.start(frame)
        elif isinstance(frame, EndFrame):
            await self.stop(frame)
            # Keep pushing EndFrame down so all the pipeline stops nicely.
            await self.push_frame(frame, direction)
        elif isinstance(frame, CancelFrame):
            await self.cancel(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            await self.push_frame(frame, direction)
            await self._handle_frame(frame)
        elif isinstance(frame, OutputTransportMessageUrgentFrame):
            await self.send_message(frame)
        elif isinstance(frame, OutputDTMFUrgentFrame):
            await self.write_dtmf(frame)
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        elif direction == FrameDirection.UPSTREAM:
            await self.push_frame(frame, direction)
        else:
            await self._handle_frame(frame)

    async def _handle_frame(self, frame: Frame):
        """Handle frames by routing them to appropriate media senders."""
        if frame.transport_destination not in self._media_senders:
            logger.warning(
                f"{self} destination [{frame.transport_destination}] not registered for frame {frame}"
            )
            return

        sender = self._media_senders[frame.transport_destination]

        if isinstance(frame, InterruptionFrame):
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
        """Handles media streaming for a specific destination.

        Manages audio and video output processing including buffering, timing,
        mixing, and frame delivery for a single output destination.
        """

        def __init__(
            self,
            transport: "BaseOutputTransport",
            *,
            destination: Optional[str],
            sample_rate: int,
            audio_chunk_size: int,
            params: TransportParams,
        ):
            """Initialize the media sender.

            Args:
                transport: The parent transport instance.
                destination: The destination identifier for this sender.
                sample_rate: The audio sample rate in Hz.
                audio_chunk_size: The size of audio chunks in bytes.
                params: Transport configuration parameters.
            """
            self._transport = transport
            self._destination = destination
            self._sample_rate = sample_rate
            self._audio_chunk_size = audio_chunk_size
            self._params = params

            # This is to resize images. We only need to resize one image at a time.
            self._executor = ThreadPoolExecutor(max_workers=1)

            # Buffer to keep track of incoming audio.
            self._audio_buffer = bytearray()

            # This will be used to resample incoming audio to the output sample rate.
            self._resampler = create_stream_resampler()

            # The user can provide a single mixer, to be used by the default
            # destination, or a destination/mixer mapping.
            self._mixer: Optional[BaseAudioMixer] = None

            # These are the images that we should send at our desired framerate.
            self._video_images = None

            # Indicates if the bot is currently speaking.
            self._bot_speaking = False
            # Last time a BotSpeakingFrame was pushed.
            self._bot_speaking_frame_time = 0
            # How often a BotSpeakingFrame should be pushed (value should be
            # lower than the audio chunks).
            self._bot_speaking_frame_period = 0.2
            # Last time the bot actually spoke.
            self._bot_speech_last_time = 0

            self._audio_task: Optional[asyncio.Task] = None
            self._video_task: Optional[asyncio.Task] = None
            self._clock_task: Optional[asyncio.Task] = None

        @property
        def sample_rate(self) -> int:
            """Get the audio sample rate.

            Returns:
                The sample rate in Hz.
            """
            return self._sample_rate

        @property
        def audio_chunk_size(self) -> int:
            """Get the audio chunk size.

            Returns:
                The size of audio chunks in bytes.
            """
            return self._audio_chunk_size

        async def start(self, frame: StartFrame):
            """Start the media sender and initialize components.

            Args:
                frame: The start frame containing initialization parameters.
            """
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
            """Stop the media sender and cleanup resources.

            Args:
                frame: The end frame signaling sender shutdown.
            """
            # Let the sink tasks process the queue until they reach this EndFrame.
            await self._clock_queue.put((float("inf"), frame.id, frame))
            await self._audio_queue.put(frame)

            # At this point we have enqueued an EndFrame and we need to wait for
            # that EndFrame to be processed by the audio and clock tasks. We
            # also need to wait for these tasks before cancelling the video task
            # because it might be still rendering.
            if self._audio_task:
                await self._audio_task
            if self._clock_task:
                await self._clock_task

            # Stop audio mixer.
            if self._mixer:
                await self._mixer.stop()

            # We can now cancel the video task.
            await self._cancel_video_task()

        async def cancel(self, frame: CancelFrame):
            """Cancel the media sender and stop all processing.

            Args:
                frame: The cancel frame signaling immediate cancellation.
            """
            # Since we are cancelling everything it doesn't matter what task we cancel first.
            await self._cancel_audio_task()
            await self._cancel_clock_task()
            await self._cancel_video_task()

        async def handle_interruptions(self, _: InterruptionFrame):
            """Handle interruption events by restarting tasks and clearing buffers.

            Args:
                _: The start interruption frame (unused).
            """
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
            """Handle incoming audio frames by buffering and chunking.

            Args:
                frame: The output audio frame to handle.
            """
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
                chunk.transport_destination = self._destination
                await self._audio_queue.put(chunk)
                self._audio_buffer = self._audio_buffer[self._audio_chunk_size :]

        async def handle_image_frame(self, frame: OutputImageRawFrame | SpriteFrame):
            """Handle incoming image frames for video output.

            Args:
                frame: The output image or sprite frame to handle.
            """
            if not self._params.video_out_enabled:
                return

            if self._params.video_out_is_live and isinstance(frame, OutputImageRawFrame):
                await self._video_queue.put(frame)
            elif isinstance(frame, OutputImageRawFrame):
                await self._set_video_image(frame)
            else:
                await self._set_video_images(frame.images)

        async def handle_timed_frame(self, frame: Frame):
            """Handle frames with presentation timestamps.

            Args:
                frame: The frame with timing information to handle.
            """
            await self._clock_queue.put((frame.pts, frame.id, frame))

        async def handle_sync_frame(self, frame: Frame):
            """Handle frames that need synchronized processing.

            Args:
                frame: The frame to handle synchronously.
            """
            await self._audio_queue.put(frame)

        async def handle_mixer_control_frame(self, frame: MixerControlFrame):
            """Handle audio mixer control frames.

            Args:
                frame: The mixer control frame to handle.
            """
            if self._mixer:
                await self._mixer.process_frame(frame)

        #
        # Audio handling
        #

        def _create_audio_task(self):
            """Create the audio processing task."""
            if not self._audio_task:
                self._audio_queue = asyncio.Queue()
                self._audio_task = self._transport.create_task(self._audio_task_handler())

        async def _cancel_audio_task(self):
            """Cancel and cleanup the audio processing task."""
            if self._audio_task:
                await self._transport.cancel_task(self._audio_task)
                self._audio_task = None

        async def _bot_started_speaking(self):
            """Handle bot started speaking event."""
            if self._bot_speaking:
                return

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
            """Handle bot stopped speaking event."""
            if not self._bot_speaking:
                return

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

        async def _bot_currently_speaking(self):
            """Handle bot speaking event."""
            await self._bot_started_speaking()

            diff_time = time.time() - self._bot_speaking_frame_time
            if diff_time >= self._bot_speaking_frame_period:
                await self._transport.push_frame(BotSpeakingFrame())
                await self._transport.push_frame(BotSpeakingFrame(), FrameDirection.UPSTREAM)
                self._bot_speaking_frame_time = time.time()

            self._bot_speech_last_time = time.time()

        async def _maybe_bot_currently_speaking(self, frame: SpeechOutputAudioRawFrame):
            if not is_silence(frame.audio):
                await self._bot_currently_speaking()
            else:
                silence_duration = time.time() - self._bot_speech_last_time
                if silence_duration > BOT_VAD_STOP_SECS:
                    await self._bot_stopped_speaking()

        async def _handle_bot_speech(self, frame: Frame):
            # TTS case.
            if isinstance(frame, TTSAudioRawFrame):
                await self._bot_currently_speaking()
            # Speech stream case.
            elif isinstance(frame, SpeechOutputAudioRawFrame):
                await self._maybe_bot_currently_speaking(frame)

        async def _handle_frame(self, frame: Frame):
            """Handle various frame types with appropriate processing.

            Args:
                frame: The frame to handle.
            """
            if isinstance(frame, OutputAudioRawFrame):
                await self._handle_bot_speech(frame)
            elif isinstance(frame, OutputImageRawFrame):
                await self._set_video_image(frame)
            elif isinstance(frame, SpriteFrame):
                await self._set_video_images(frame.images)
            elif isinstance(frame, OutputTransportMessageFrame):
                await self._transport.send_message(frame)
            elif isinstance(frame, OutputDTMFFrame):
                await self._transport.write_dtmf(frame)

        def _next_frame(self) -> AsyncGenerator[Frame, None]:
            """Generate the next frame for audio processing.

            Returns:
                An async generator yielding frames for processing.
            """

            async def without_mixer(vad_stop_secs: float) -> AsyncGenerator[Frame, None]:
                while True:
                    try:
                        frame = await asyncio.wait_for(
                            self._audio_queue.get(), timeout=vad_stop_secs
                        )
                        yield frame
                        self._audio_queue.task_done()
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
                        self._audio_queue.task_done()
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
                        # Allow other asyncio tasks to execute by adding a small sleep
                        # Without this sleep, in task cancellation scenarios, this loop would
                        # continuously return without any delay, leading to 100% CPU utilization
                        # and preventing cancel/stop signals from being processed properly
                        await asyncio.sleep(0)

            if self._mixer:
                return with_mixer(BOT_VAD_STOP_SECS)
            else:
                return without_mixer(BOT_VAD_STOP_SECS)

        async def _audio_task_handler(self):
            """Main audio processing task handler."""
            async for frame in self._next_frame():
                # No need to push EndFrame, it's pushed from process_frame().
                if isinstance(frame, EndFrame):
                    break

                # Handle frame.
                await self._handle_frame(frame)

                # If we are not able to write to the transport we shouldn't
                # pushb downstream.
                push_downstream = True

                # Try to send audio to the transport.
                try:
                    if isinstance(frame, OutputAudioRawFrame):
                        push_downstream = await self._transport.write_audio_frame(frame)
                except Exception as e:
                    logger.error(f"{self} Error writing {frame} to transport: {e}")
                    push_downstream = False

                # If we were able to send to the transport, push the frame
                # downstream in case anyone else needs it.
                if push_downstream:
                    await self._transport.push_frame(frame)

        #
        # Video handling
        #

        def _create_video_task(self):
            """Create the video processing task if video output is enabled."""
            if not self._video_task and self._params.video_out_enabled:
                self._video_queue = asyncio.Queue()
                self._video_task = self._transport.create_task(self._video_task_handler())

        async def _cancel_video_task(self):
            """Cancel and cleanup the video processing task."""
            # Stop video output task.
            if self._video_task:
                await self._transport.cancel_task(self._video_task)
                self._video_task = None

        async def _set_video_image(self, image: OutputImageRawFrame):
            """Set a single video image for cycling output.

            Args:
                image: The image frame to cycle for video output.
            """
            self._video_images = itertools.cycle([image])

        async def _set_video_images(self, images: List[OutputImageRawFrame]):
            """Set multiple video images for cycling output.

            Args:
                images: The list of image frames to cycle for video output.
            """
            self._video_images = itertools.cycle(images)

        async def _video_task_handler(self):
            """Main video processing task handler."""
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
            """Handle live video streaming with frame timing."""
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
            """Draw/render an image frame with resizing if needed.

            Args:
                frame: The image frame to draw.
            """

            def resize_frame(frame: OutputImageRawFrame) -> OutputImageRawFrame:
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

                return frame

            frame = await self._transport.get_event_loop().run_in_executor(
                self._executor, resize_frame, frame
            )
            await self._transport.write_video_frame(frame)

        #
        # Clock handling
        #

        def _create_clock_task(self):
            """Create the clock/timing processing task."""
            if not self._clock_task:
                self._clock_queue = asyncio.PriorityQueue()
                self._clock_task = self._transport.create_task(self._clock_task_handler())

        async def _cancel_clock_task(self):
            """Cancel and cleanup the clock processing task."""
            if self._clock_task:
                await self._transport.cancel_task(self._clock_task)
                self._clock_task = None

        async def _clock_task_handler(self):
            """Main clock/timing task handler for timed frame delivery."""
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
