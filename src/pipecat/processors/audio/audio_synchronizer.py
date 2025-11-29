#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Audio synchronizer for merging input and output audio streams."""

import asyncio
from typing import Callable, Dict, List

from loguru import logger

from pipecat.audio.utils import create_stream_resampler, interleave_stereo_audio, mix_audio


class AudioSynchronizer:
    """Synchronizes audio from separate input and output processors without being in the pipeline.

    This class subscribes to events from AudioBufferInputProcessor and AudioBufferOutputProcessor,
    buffers the audio data, and emits merged audio when both buffers have sufficient data.

    Events:
        on_merged_audio: Triggered when synchronized audio is available

    Args:
        sample_rate (int): The sample rate for audio processing
        buffer_size (int): Size of buffer before triggering merged audio events
        num_channels (int): Number of channels for merged output (1 for mono, 2 for stereo)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        buffer_size: int = 8000,  # 0.5 seconds at 16kHz
        num_channels: int = 1,
    ):
        """Initialize audio synchronizer.

        Args:
            sample_rate: The sample rate for audio processing in Hz.
            buffer_size: Size of buffer before triggering merged audio events.
            num_channels: Number of channels for merged output (1 for mono, 2 for stereo).
        """
        self._sample_rate = sample_rate
        self._buffer_size = buffer_size
        self._num_channels = num_channels

        self._input_buffer = bytearray()
        self._output_buffer = bytearray()

        self._event_handlers: Dict[str, List[Callable]] = {"on_merged_audio": []}
        self._resampler = create_stream_resampler()

        self._input_processor = None
        self._output_processor = None

        # Track if we're currently recording
        self._recording = False

        # Track total bytes received
        self._total_input_bytes = 0
        self._total_output_bytes = 0

    def register_processors(self, input_processor, output_processor):
        """Register input and output processors to synchronize.

        Args:
            input_processor: AudioBufferInputProcessor instance
            output_processor: AudioBufferOutputProcessor instance
        """
        logger.info(f"AudioSynchronizer: Registering processors")
        self._input_processor = input_processor
        self._output_processor = output_processor

        # Subscribe to events from both processors
        if input_processor:
            input_processor.add_event_handler("on_input_audio_data", self._handle_input_audio)
        else:
            logger.warning(f"AudioSynchronizer: No input processor provided!")

        if output_processor:
            output_processor.add_event_handler("on_output_audio_data", self._handle_output_audio)
        else:
            logger.warning(f"AudioSynchronizer: No output processor provided!")

    async def start_recording(self):
        """Start recording and synchronizing audio."""
        self._recording = True
        self._reset_buffers()
        # Reset byte counters on new recording
        self._total_input_bytes = 0
        self._total_output_bytes = 0
        logger.info("AudioSynchronizer: Started recording, reset byte counters")

    async def stop_recording(self):
        """Stop recording and flush remaining audio."""
        await self._call_audio_handler(final_flush=True)
        self._recording = False
        logger.info(
            f"AudioSynchronizer: Stopped recording - "
            f"total_input_bytes={self._total_input_bytes}, "
            f"total_output_bytes={self._total_output_bytes}"
        )

    def _has_audio(self) -> bool:
        """Check if both buffers contain audio data."""
        return self._buffer_has_audio(self._input_buffer) and self._buffer_has_audio(
            self._output_buffer
        )

    def _buffer_has_audio(self, buffer: bytearray) -> bool:
        """Check if a buffer contains audio data."""
        return buffer is not None and len(buffer) > 0

    async def _handle_input_audio(self, processor, pcm: bytes, sample_rate: int, num_channels: int):
        """Handle incoming input audio data."""
        if not self._recording:
            return

        # Track total bytes received
        bytes_received = len(pcm)
        self._total_input_bytes += bytes_received

        # Add audio
        self._input_buffer.extend(pcm)
        # logger.debug(
        #     f"AudioSynchronizer: Input - bytes_received={bytes_received}, "
        #     f"total_input_bytes={self._total_input_bytes}, "
        #     f"buffer_size={len(self._input_buffer)}"
        # )

        # Check if we should emit
        if self._buffer_size > 0 and len(self._input_buffer) > self._buffer_size:
            await self._call_audio_handler()

    async def _handle_output_audio(
        self, processor, pcm: bytes, sample_rate: int, num_channels: int
    ):
        """Handle incoming output audio data."""
        if not self._recording:
            return

        # Track total bytes received
        bytes_received = len(pcm)
        self._total_output_bytes += bytes_received

        # Add audio
        self._output_buffer.extend(pcm)
        # logger.debug(
        #     f"AudioSynchronizer: Output - bytes_received={bytes_received}, "
        #     f"total_output_bytes={self._total_output_bytes}, "
        #     f"buffer_size={len(self._output_buffer)}"
        # )

    async def _call_audio_handler(self, final_flush: bool = False):
        """Call the audio data event handler with merged audio.

        Args:
            final_flush: If True, use max of both buffers and pad shorter one with silence.
                        This is used when stopping recording to flush all remaining audio.
        """
        if not self._has_audio() or not self._recording:
            logger.trace(
                f"AudioSynchronizer: Not calling handler - has_audio={self._has_audio()}, recording={self._recording}"
            )
            return

        if final_flush:
            # For final flush, use max of both buffers and pad shorter one with silence
            merge_size = max(len(self._input_buffer), len(self._output_buffer))
        else:
            # Normal merge: use minimum of both buffers
            merge_size = min(len(self._input_buffer), len(self._output_buffer))

        # Ensure even size for 16-bit audio
        if merge_size % 2 != 0:
            merge_size -= 1

        if merge_size == 0:
            return

        # Extract chunks from both buffers
        input_chunk = bytes(self._input_buffer[:merge_size])
        output_chunk = bytes(self._output_buffer[:merge_size])

        # Pad with silence (zero bytes) if needed for final flush
        if final_flush:
            if len(input_chunk) < merge_size:
                input_chunk = input_chunk + bytes(merge_size - len(input_chunk))
            if len(output_chunk) < merge_size:
                output_chunk = output_chunk + bytes(merge_size - len(output_chunk))

        # Remove processed data from buffers
        if final_flush:
            # Clear all buffers on final flush
            self._input_buffer = bytearray()
            self._output_buffer = bytearray()
        else:
            self._input_buffer = self._input_buffer[merge_size:]
            self._output_buffer = self._output_buffer[merge_size:]

        # Merge the chunks
        if self._num_channels == 1:
            merged_audio = mix_audio(input_chunk, output_chunk)
        elif self._num_channels == 2:
            merged_audio = interleave_stereo_audio(input_chunk, output_chunk)
        else:
            merged_audio = b""

        await self._emit_event(
            "on_merged_audio", merged_audio, self._sample_rate, self._num_channels
        )

    def _reset_buffers(self):
        """Reset all audio buffers to empty state."""
        self._input_buffer = bytearray()
        self._output_buffer = bytearray()

    async def _emit_event(self, event_name: str, *args):
        """Emit an event to all registered handlers."""
        if event_name in self._event_handlers:
            for handler in self._event_handlers[event_name]:
                try:
                    await handler(self, *args)
                except Exception as e:
                    logger.error(f"Error in {event_name} handler: {e}")

    def add_event_handler(self, event_name: str, handler: Callable):
        """Add an event handler for the specified event.

        Args:
            event_name: Name of the event ("on_merged_audio")
            handler: Async callable to handle the event
        """
        if event_name not in self._event_handlers:
            logger.warning(f"AudioSynchronizer: Unknown event: {event_name}")
            return

        logger.debug(f"AudioSynchronizer: Adding handler for event '{event_name}'")
        self._event_handlers[event_name].append(handler)

    def event_handler(self, event_name: str):
        """Decorator for registering event handlers.

        Example:
            ```python
            synchronizer = AudioSynchronizer()

            @synchronizer.event_handler("on_merged_audio")
            async def handle_merged_audio(sync, pcm, sample_rate, num_channels):
                # Process merged audio
                pass
            ```
        """

        def decorator(handler):
            self.add_event_handler(event_name, handler)
            return handler

        return decorator

    def clear_buffers(self):
        """Clear all internal audio buffers."""
        self._input_buffer.clear()
        self._output_buffer.clear()
