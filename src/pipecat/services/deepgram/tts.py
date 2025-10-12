#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deepgram text-to-speech service implementation.

This module provides integration with Deepgram's text-to-speech API
for generating speech from text using various voice models.
"""

from typing import AsyncGenerator, Optional

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from deepgram import DeepgramClient, DeepgramClientOptions, SpeakOptions
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Deepgram, you need to `pip install pipecat-ai[deepgram]`.")
    raise Exception(f"Missing module: {e}")


class DeepgramTTSService(TTSService):
    """Deepgram text-to-speech service.

    Provides text-to-speech synthesis using Deepgram's streaming API.
    Supports various voice models and audio encoding formats with
    configurable sample rates and quality settings.
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice: str = "aura-2-helena-en",
        base_url: str = "",
        sample_rate: Optional[int] = None,
        encoding: str = "linear16",
        **kwargs,
    ):
        """Initialize the Deepgram TTS service.

        Args:
            api_key: Deepgram API key for authentication.
            voice: Voice model to use for synthesis. Defaults to "aura-2-helena-en".
            base_url: Custom base URL for Deepgram API. Uses default if empty.
            sample_rate: Audio sample rate in Hz. If None, uses service default.
            encoding: Audio encoding format. Defaults to "linear16".
            **kwargs: Additional arguments passed to parent TTSService class.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._settings = {
            "encoding": encoding,
        }
        self.set_voice(voice)

        client_options = DeepgramClientOptions(url=base_url)
        self._deepgram_client = DeepgramClient(api_key, config=client_options)

        # State tracking for <think> tag filtering
        self._think_buffer = ""
        self._inside_think = False

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True, as Deepgram TTS service supports metrics generation.
        """
        return True

    # Removing think tags for now
    def _filter_think_tags(self, text: str) -> str:
        """Filter out content between <think> and </think> tags.

        This method handles streaming text and maintains state across calls.
        Content between <think> tags is not spoken.

        Args:
            text: Input text that may contain <think> tags.

        Returns:
            str: Text with <think> blocks removed, ready for TTS.
        """
        # Add incoming text to buffer
        self._think_buffer += text
        result = ""

        while self._think_buffer:
            if self._inside_think:
                # Look for closing tag
                close_pos = self._think_buffer.find("</think>")
                if close_pos >= 0:
                    # Found closing tag - skip everything up to and including it
                    print(
                        f"🧠 [TTS] Skipping thinking content: {self._think_buffer[:close_pos][:50]}..."
                    )
                    self._think_buffer = self._think_buffer[close_pos + 8 :]  # len("</think>") = 8
                    self._inside_think = False
                else:
                    # Still inside think block, buffer everything
                    print(f"🧠 [TTS] Still thinking... (buffered {len(self._think_buffer)} chars)")
                    break
            else:
                # Look for opening tag
                open_pos = self._think_buffer.find("<think>")
                if open_pos >= 0:
                    # Found opening tag - output everything before it
                    result += self._think_buffer[:open_pos]
                    self._think_buffer = self._think_buffer[open_pos + 7 :]  # len("<think>") = 7
                    self._inside_think = True
                    print(f"🧠 [TTS] Entering think mode")
                else:
                    # No tags found - output everything and clear buffer
                    result += self._think_buffer
                    self._think_buffer = ""
                    break

        if result:
            print(f"💬 [TTS] Outputting text: {result[:100]}...")

        return result

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Deepgram's TTS API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech, plus start/stop frames.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        # Filter out <think> tags and their content
        filtered_text = self._filter_think_tags(text)

        # If all text was filtered out (inside think block), don't generate audio
        if not filtered_text or not filtered_text.strip():
            logger.debug(f"{self}: No speakable text after filtering <think> tags")
            return

        options = SpeakOptions(
            model=self._voice_id,
            encoding=self._settings["encoding"],
            sample_rate=self.sample_rate,
            container="none",
        )

        try:
            await self.start_ttfb_metrics()
            print(f"🔊 [TTS] Sending to Deepgram: {filtered_text}")

            response = await self._deepgram_client.speak.asyncrest.v("1").stream_raw(
                {"text": filtered_text}, options
            )

            await self.start_tts_usage_metrics(filtered_text)
            yield TTSStartedFrame()

            async for data in response.aiter_bytes():
                await self.stop_ttfb_metrics()
                if data:
                    yield TTSAudioRawFrame(audio=data, sample_rate=self.sample_rate, num_channels=1)

            yield TTSStoppedFrame()

        except Exception as e:
            logger.exception(f"{self} exception: {e}")
            yield ErrorFrame(f"Error getting audio: {str(e)}")
