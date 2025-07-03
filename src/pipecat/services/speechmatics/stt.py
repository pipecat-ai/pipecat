#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Speechmatics STT service integration."""

import asyncio
import datetime
import re
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional
from urllib.parse import urlencode

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from speechmatics.rt import (
        AsyncClient,
        AudioEncoding,
        AudioFormat,
        ConversationConfig,
        OperatingPoint,
        ServerMessageType,
        SpeakerDiarizationConfig,
        TranscriptionConfig,
        __version__,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Speechmatics, you need to `pip install pipecat-ai[speechmatics]`."
    )
    raise Exception(f"Missing module: {e}")


class AudioBuffer:
    """Audio buffer for STT clients.

    The Python SDK expects audio in a pre-defined number of frames. This
    buffer will accumulate the data from the pipeline and provide it to the
    STT client in the correct lengths, waiting for the number of frames to
    be available.
    """

    def __init__(self, maxsize: int = 0):
        """Initialize the audio buffer.

        Args:
            maxsize: Maximum size of the buffer.
        """
        self._queue = asyncio.Queue(maxsize=maxsize)
        self._current_chunk = b""
        self._position = 0
        self._closed = False

    def write_audio(self, data: bytes) -> None:
        """Write audio data to the buffer (thread-safe).

        Args:
            data: Audio data to write.
        """
        if data:
            try:
                self._queue.put_nowait(data)
            except asyncio.QueueFull:
                pass

    async def read(self, size: int) -> bytes:
        """Read exactly `size` bytes from the buffer (thread-safe).

        This process will block until the required number of bytes are available
        in the buffer. Audio is received from the pipeline in varying sizes, so
        this buffer will accumulate the data and provide it to the STT client in
        the correct lengths, waiting for the number of frames to be available.

        Calling stop() will close the buffer and release the blocking read
        process.

        Args:
            size: Number of bytes to read.

        Returns:
            bytes: Audio data read from the buffer.
        """
        result = b""
        bytes_needed = size

        while bytes_needed > 0 and not self._closed:
            # Use data from current chunk if available
            if self._position < len(self._current_chunk):
                available = len(self._current_chunk) - self._position
                take = min(bytes_needed, available)
                result += self._current_chunk[self._position : self._position + take]
                self._position += take
                bytes_needed -= take
                continue

            # Get next chunk
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                if chunk is None:
                    continue
                self._current_chunk = chunk
                self._position = 0
            except asyncio.TimeoutError:
                await asyncio.sleep(0)
                continue

        return result

    def stop(self) -> None:
        """Close the audio buffer."""
        self._closed = True


@dataclass
class SpeechFragment:
    """Fragment of an utterance.

    Parameters:
        start_time: Start time of the fragment in seconds (from session start).
        end_time: End time of the fragment in seconds (from session start).
        language: Language of the fragment. Defaults to `Language.EN`.
        is_eos: Whether the fragment is the end of a sentence. Defaults to `False`.
        is_final: Whether the fragment is the final fragment. Defaults to `False`.
        attaches_to: Whether the fragment attaches to the previous or next fragment (punctuation). Defaults to empty string.
        content: Content of the fragment. Defaults to empty string.
        speaker: Speaker of the fragment (if diarization is enabled). Defaults to `None`.
        confidence: Confidence of the fragment (0.0 to 1.0). Defaults to `1.0`.
        result: Raw result of the fragment from the TTS.
    """

    start_time: float
    end_time: float
    language: Language = Language.EN
    is_eos: bool = False
    is_final: bool = False
    attaches_to: str = ""
    content: str = ""
    speaker: Optional[str] = None
    confidence: float = 1.0
    result: Optional[Any] = None


@dataclass
class SpeakerFragments:
    """SpeechFragment items grouped by speaker_id.

    Parameters:
        speaker_id: The ID of the speaker.
        timestamp: The timestamp of the frame.
        language: The language of the frame.
        fragments: The list of SpeechFragment items.
    """

    speaker_id: Optional[str] = None
    timestamp: Optional[str] = None
    language: Optional[Language] = None
    fragments: list[SpeechFragment] = field(default_factory=list)

    def __str__(self):
        """Return a string representation of the object."""
        return f"SpeakerFragments(speaker_id: {self.speaker_id}, timestamp: {self.timestamp}, language: {self.language}, text: {self._format_text()})"

    def _format_text(self, format: Optional[str] = None) -> str:
        """Wrap text with speaker ID in an optional f-string format.

        Args:
            format: Format to wrap the text with.

        Returns:
            str: The wrapped text.
        """
        # Cumulative contents
        content = ""

        # Assemble the text
        for frag in self.fragments:
            if content == "" or frag.attaches_to == "previous":
                content += frag.content
            else:
                content += " " + frag.content

        # Format the text, if format is provided
        if format is None or self.speaker_id is None:
            return content
        return format.format(**{"speaker_id": self.speaker_id, "text": content})

    def _as_frame_attributes(self, format: Optional[str] = None) -> dict[str, Any]:
        """Return a dictionary of attributes for a TranscriptionFrame.

        Args:
            format: Format to wrap the text with.

        Returns:
            dict[str, Any]: The dictionary of attributes.
        """
        return {
            "text": self._format_text(format),
            "user_id": self.speaker_id,
            "timestamp": self.timestamp,
            "language": self.language,
            "result": [frag.result for frag in self.fragments],
        }


class SpeechmaticsSTTService(STTService):
    """Speechmatics STT service implementation.

    This service provides real-time speech-to-text transcription using the Speechmatics API.
    It supports partial and final transcriptions, multiple languages, various audio formats,
    and speaker diarization.
    """

    def __init__(
        self,
        *,
        api_key: str,
        language: Optional[Language] = None,
        language_code: Optional[str] = None,
        base_url: str = "wss://eu2.rt.speechmatics.com/v2",
        domain: Optional[str] = None,
        output_locale: Optional[Language] = None,
        output_locale_code: Optional[str] = None,
        enable_partials: bool = True,
        max_delay: float = 1.5,
        sample_rate: Optional[int] = 16000,
        chunk_size: int = 256,
        audio_encoding: AudioEncoding = AudioEncoding.PCM_S16LE,
        end_of_utterance_silence_trigger: float = 0.5,
        operating_point: OperatingPoint = OperatingPoint.ENHANCED,
        enable_speaker_diarization: bool = False,
        text_format: str = "<{speaker_id}>{text}</{speaker_id}>",
        max_speakers: Optional[int] = None,
        transcription_config: Optional[TranscriptionConfig] = None,
        **kwargs,
    ):
        """Initialize the Speechmatics STT service.

        Args:
            api_key: Speechmatics API key for authentication.
            language: Language code for transcription. Defaults to `None`.
            language_code: Language code string for transcription. Defaults to `None`.
            base_url: Base URL for Speechmatics API. Defaults to `wss://eu2.rt.speechmatics.com/v2`.
            domain: Domain for Speechmatics API. Defaults to `None`.
            output_locale: Output locale for transcription, e.g. `Language.EN_GB`. Defaults to `None`.
            output_locale_code: Output locale code for transcription. Defaults to `None`.
            enable_partials: Enable partial transcription results. Defaults to `True`.
            max_delay: Maximum delay for transcription in seconds. Defaults to `1.5`.
            sample_rate: Audio sample rate in Hz. Defaults to `16000`.
            chunk_size: Audio chunk size for streaming. Defaults to `256`.
            audio_encoding: Audio encoding format. Defaults to `pcm_s16le`.
            end_of_utterance_silence_trigger: Silence duration in seconds to trigger end of utterance detection. Defaults to `0.5`.
            operating_point: Operating point for transcription accuracy vs. latency tradeoff. Defaults to `enhanced`.
            enable_speaker_diarization: Enable speaker diarization to identify different speakers. Defaults to `False`.
            text_format: Wrapper for speaker ID. Defaults to `<{speaker_id}>{text}</{speaker_id}>`.
            max_speakers: Maximum number of speakers to detect. Defaults to `None` (auto-detect).
            transcription_config: Custom transcription configuration (other set parameters are merged). Defaults to `None`.
            **kwargs: Additional arguments passed to STTService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        # Client configuration
        self._api_key: str = api_key
        self._language: Optional[Language] = language
        self._language_code: Optional[str] = language_code
        self._base_url: str = base_url
        self._domain: Optional[str] = domain
        self._output_locale: Optional[Language] = output_locale
        self._output_locale_code: Optional[str] = output_locale_code
        self._enable_partials: bool = enable_partials
        self._max_delay: float = max_delay
        self._sample_rate: int = sample_rate
        self._chunk_size: int = chunk_size
        self._audio_encoding: AudioEncoding = audio_encoding
        self._end_of_utterance_silence_trigger: Optional[float] = end_of_utterance_silence_trigger
        self._operating_point: OperatingPoint = operating_point
        self._enable_speaker_diarization: bool = enable_speaker_diarization
        self._text_format: str = text_format
        self._max_speakers: Optional[int] = max_speakers

        # Check we have required attributes
        if not self._api_key:
            raise ValueError("Missing Speechmatics API key")
        if not self._base_url:
            raise ValueError("Missing Speechmatics base URL")

        # Validate the language code
        if self._language and self._language_code:
            raise ValueError("Language and language code cannot both be specified")
        elif self._language:
            self._language_code = _language_to_speechmatics_language(self._language)

        # Validate the output locale code
        if self._output_locale and self._output_locale_code:
            raise ValueError("Output locale and output locale code cannot both be specified")
        elif self._output_locale:
            self._output_locale_code = _locale_to_speechmatics_locale(
                self._language_code, self._output_locale
            )

        # Complete configuration objects
        self._transcription_config: TranscriptionConfig = None
        self._process_config(transcription_config)

        # STT client
        self._client: Optional[AsyncClient] = None
        self._client_task: Optional[asyncio.Task] = None
        self._audio_buffer: AudioBuffer = AudioBuffer(maxsize=10)
        self._start_time: Optional[datetime.datetime] = None

        # Current utterance speech data
        self._speech_fragments: list[SpeechFragment] = []

    async def start(self, frame: StartFrame):
        """Called when the new session starts."""
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Called when the session ends."""
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Called when the session is cancelled."""
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Adds audio to the audio buffer and yields None."""
        self._audio_buffer.write_audio(audio)
        yield None

    async def _run_client(self) -> None:
        """Runs the Speechmatics client in a thread."""
        await self._client.transcribe(
            self._audio_buffer,
            transcription_config=self._transcription_config,
            audio_format=AudioFormat(
                encoding=self._audio_encoding,
                sample_rate=self.sample_rate,
                chunk_size=self._chunk_size,
            ),
        )

    async def _connect(self) -> None:
        """Connect to the STT service."""
        # Create new STT RT client
        self._client = AsyncClient(
            api_key=self._api_key,
            url=_get_endpoint_url(self._base_url),
        )

        # Log the event
        logger.debug("Connected to Speechmatics STT service")

        # Recognition started event
        @self._client.on(ServerMessageType.RECOGNITION_STARTED)
        def _evt_on_recognition_started(message: dict[str, Any]):
            logger.debug(f"Recognition started (session: {message.get('id')})")
            self._start_time = datetime.datetime.now(datetime.timezone.utc)

        # Partial transcript event
        @self._client.on(ServerMessageType.ADD_PARTIAL_TRANSCRIPT)
        def _evt_on_partial_transcript(message: dict[str, Any]):
            self._handle_transcript(message, is_final=False)

        # Final transcript event
        @self._client.on(ServerMessageType.ADD_TRANSCRIPT)
        def _evt_on_final_transcript(message: dict[str, Any]):
            self._handle_transcript(message, is_final=True)

        # End of Utterance
        @self._client.on(ServerMessageType.END_OF_UTTERANCE)
        def _evt_on_end_of_utterance(message: dict[str, Any]):
            logger.debug("End of utterance received from STT")
            asyncio.run_coroutine_threadsafe(
                self._send_frames(finalized=True), self.get_event_loop()
            )

        # Start the client in a thread
        self._client_task = self.create_task(self._run_client())

    async def _disconnect(self) -> None:
        """Disconnect from the STT service."""
        # Stop the audio buffer
        self._audio_buffer.stop()

        # Disconnect the client
        try:
            if self._client:
                await asyncio.wait_for(self._client.close(), timeout=1.0)
        except asyncio.TimeoutError:
            logger.warning("Timeout while closing Speechmatics client connection")
        except Exception as e:
            logger.error(f"Error closing Speechmatics client: {e}")
        finally:
            self._client = None

        # Cancel the client task
        if self._client_task:
            await self.cancel_task(self._client_task)
            self._client_task = None

        # Log the event
        logger.debug("Disconnected from Speechmatics STT service")

    def _process_config(self, transcription_config: Optional[TranscriptionConfig] = None) -> None:
        """Create a formatted STT transcription config.

        This takes an optional TranscriptionConfig object and populates it with the
        values from the STT service. Individual parameters take priority over those
        within the config object.

        Args:
            transcription_config: Optional transcription config to use.
        """
        # Transcription config
        if not transcription_config:
            transcription_config = TranscriptionConfig(
                language=self._language_code or "en",
                domain=self._domain,
                output_locale=self._output_locale_code,
                operating_point=self._operating_point,
                diarization="speaker" if self._enable_speaker_diarization else None,
                enable_partials=self._enable_partials,
                max_delay=self._max_delay or 2.0,
            )
        else:
            if self._language_code:
                transcription_config.language = self._language_code
            if self._domain:
                transcription_config.domain = self._domain
            if self._output_locale_code:
                transcription_config.output_locale = self._output_locale_code
            if self._operating_point:
                transcription_config.operating_point = self._operating_point
            if self._enable_speaker_diarization:
                transcription_config.diarization = "speaker"
            if self._enable_partials:
                transcription_config.enable_partials = self._enable_partials
            if self._max_delay:
                transcription_config.max_delay = self._max_delay

        # Diarization
        if self._enable_speaker_diarization and self._max_speakers:
            transcription_config.speaker_diarization_config = SpeakerDiarizationConfig(
                max_speakers=self._max_speakers,
            )

        # End of Utterance
        if self._end_of_utterance_silence_trigger:
            transcription_config.conversation_config = ConversationConfig(
                end_of_utterance_silence_trigger=self._end_of_utterance_silence_trigger,
            )

        # Set config
        self._transcription_config = transcription_config

    def _handle_transcript(self, message: dict[str, Any], is_final: bool) -> None:
        """Handle the partial and final transcript events.

        Args:
            message: The new Partial or Final from the STT engine.
            is_final: Whether the data is final or partial.
        """
        # Add the speech fragments
        has_changed = self._add_speech_fragments(
            message=message,
            is_final=is_final,
        )

        # Skip if unchanged
        if not has_changed:
            return

        # Send frames
        asyncio.run_coroutine_threadsafe(self._send_frames(), self.get_event_loop())

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _send_frames(self, finalized: bool = False) -> None:
        """Send frames to the pipeline.

        Send speech frames to the pipeline. If VAD is enabled, then this will
        also send an interruption and user started speaking frames. When the
        final transcript is received, then this will send a user stopped speaking
        and stop interruption frames.

        Args:
            finalized: Whether the data is final or partial.
        """
        # Get speech frames (InterimTranscriptionFrame)
        speech_frames = self._get_frames_from_fragments()

        # Skip if no frames
        if not speech_frames:
            return

        # If final, then re=parse into TranscriptionFrame
        if finalized:
            # Reset the speech fragments
            self._speech_fragments.clear()

            # Transform frames
            frames = [
                TranscriptionFrame(**frame._as_frame_attributes(self._text_format))
                for frame in speech_frames
            ]

            # Log transcript(s)
            logger.debug(f"Finalized transcript: {[f.text for f in frames]}")

        # Return as interim results
        else:
            frames = [
                InterimTranscriptionFrame(**frame._as_frame_attributes()) for frame in speech_frames
            ]

        # Send the frames back to pipecat
        for frame in frames:
            await self._handle_transcription(
                transcript=frame.text,
                is_final=finalized,
                language=frame.language,
            )
            await self.push_frame(frame)

    def _add_speech_fragments(self, message: dict[str, Any], is_final: bool = False) -> bool:
        """Takes a new Partial or Final from the STT engine.

        Accumulates it into the _speech_data list. As new final data is added, all
        partials are removed from the list.

        Note: If a known speaker is `__[A-Z0-9_]{2,}__`, then the words are skipped,
        as this is used to protect against self-interruption by the assistant or to
        block out specific known voices.

        Args:
            message: The new Partial or Final from the STT engine.
            is_final: Whether the data is final or partial.

        Returns:
            bool: True if the speech data was updated, False otherwise.
        """
        # Parsed new speech data from the STT engine
        fragments: list[SpeechFragment] = []

        # Current length of the speech data
        current_length = len(self._speech_fragments)

        # Iterate over the results in the payload
        for result in message.get("results", []):
            alt = result.get("alternatives", [{}])[0]
            if alt.get("content", None):
                # Create the new fragment
                fragment = SpeechFragment(
                    start_time=result.get("start_time", 0),
                    end_time=result.get("end_time", 0),
                    language=alt.get("language", Language.EN),
                    is_eos=alt.get("is_eos", False),
                    is_final=is_final,
                    attaches_to=result.get("attaches_to", ""),
                    content=alt.get("content", ""),
                    speaker=alt.get("speaker", None),
                    confidence=alt.get("confidence", 1.0),
                    result=result,
                )

                # Drop `__XX__` speakers
                if fragment.speaker and re.match(r"^__[A-Z0-9_]{2,}__$", fragment.speaker):
                    continue

                # Add the fragment
                fragments.append(fragment)

        # Remove existing partials, as new partials and finals are provided
        self._speech_fragments = [frag for frag in self._speech_fragments if frag.is_final]

        # Return if no new fragments and length of the existing data is unchanged
        if not fragments and len(self._speech_fragments) == current_length:
            return False

        # Add the fragments to the speech data
        self._speech_fragments.extend(fragments)

        # Data was updated
        return True

    def _get_frames_from_fragments(self) -> list[SpeakerFragments]:
        """Get speech data objects for the current fragment list.

        Each speech fragments is grouped by contiguous speaker and then
        returned as internal SpeakerFragments objects with the `speaker_id` field
        set to the current speaker (string). An utterance may contain speech from
        more than one speaker (e.g. S1, S2, S1, S3, ...), so they are kept
        in strict order for the context of the conversation.

        Returns:
            list[SpeakerFragments]: The list of objects.
        """
        # Speaker groups
        current_speaker: str | None = None
        speaker_groups: list[list[SpeechFragment]] = [[]]

        # Group by speakers
        for frag in self._speech_fragments:
            if frag.speaker != current_speaker:
                current_speaker = frag.speaker
                if speaker_groups[-1]:
                    speaker_groups.append([])
            speaker_groups[-1].append(frag)

        # Create SpeakerFragments objects
        speaker_fragments: list[SpeakerFragments] = []
        for group in speaker_groups:
            sd = self._get_speaker_fragments_from_fragment_group(group)
            if sd:
                speaker_fragments.append(sd)

        # Return the grouped SpeakerFragments objects
        return speaker_fragments

    def _get_speaker_fragments_from_fragment_group(
        self,
        group: list[SpeechFragment],
    ) -> SpeakerFragments | None:
        """Take a group of fragments and piece together into SpeakerFragments.

        Each fragment for a given speaker is assembled into a string,
        taking into consideration whether words are attached to the
        previous or next word (notably punctuation). This ensures that
        the text does not have extra spaces. This will also check for
        any straggling punctuation from earlier utterances that should
        be removed.

        Args:
            group: List of SpeechFragment objects.

        Returns:
            SpeakerFragments: The object for the group.
        """
        # Check for starting fragments that are attached to previous
        if group and group[0].attaches_to == "previous":
            group = group[1:]

        # Check for trailing fragments that are attached to next
        if group and group[-1].attaches_to == "next":
            group = group[:-1]

        # Check there are results
        if not group:
            return None

        # Get the timing extremes
        start_time = min(frag.start_time for frag in group)

        # Timestamp
        ts = (self._start_time + datetime.timedelta(seconds=start_time)).isoformat(
            timespec="milliseconds"
        )

        # Return the SpeakerFragments object
        return SpeakerFragments(
            speaker_id=group[0].speaker,
            timestamp=ts,
            language=group[0].language,
            fragments=group,
        )


def _get_endpoint_url(url: str) -> str:
    """Format the endpoint URL with the SDK and app versions.

    Args:
        url: The base URL for the endpoint.

    Returns:
        str: The formatted endpoint URL.
    """
    query_params = dict()
    query_params["sm-app"] = f"pipecat/{__version__}"
    query = urlencode(query_params)

    return f"{url}?{query}"


def _language_to_speechmatics_language(language: Language) -> str:
    """Convert a Language enum to a Speechmatics language code.

    Args:
        language: The Language enum to convert.

    Returns:
        str: The Speechmatics language code, if found.
    """
    # List of supported input languages
    BASE_LANGUAGES = {
        Language.AR: "ar",
        Language.BA: "ba",
        Language.EU: "eu",
        Language.BE: "be",
        Language.BG: "bg",
        Language.BN: "bn",
        Language.YUE: "yue",
        Language.CA: "ca",
        Language.HR: "hr",
        Language.CS: "cs",
        Language.DA: "da",
        Language.NL: "nl",
        Language.EN: "en",
        Language.EO: "eo",
        Language.ET: "et",
        Language.FA: "fa",
        Language.FI: "fi",
        Language.FR: "fr",
        Language.GL: "gl",
        Language.DE: "de",
        Language.EL: "el",
        Language.HE: "he",
        Language.HI: "hi",
        Language.HU: "hu",
        Language.IT: "it",
        Language.ID: "id",
        Language.GA: "ga",
        Language.JA: "ja",
        Language.KO: "ko",
        Language.LV: "lv",
        Language.LT: "lt",
        Language.MS: "ms",
        Language.MT: "mt",
        Language.CMN: "cmn",
        Language.MR: "mr",
        Language.MN: "mn",
        Language.NO: "no",
        Language.PL: "pl",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SK: "sk",
        Language.SL: "sl",
        Language.ES: "es",
        Language.SV: "sv",
        Language.SW: "sw",
        Language.TA: "ta",
        Language.TH: "th",
        Language.TR: "tr",
        Language.UG: "ug",
        Language.UK: "uk",
        Language.UR: "ur",
        Language.VI: "vi",
        Language.CY: "cy",
    }

    # Get the language code
    result = BASE_LANGUAGES.get(language)

    # Fail if language is not supported
    if not result:
        raise ValueError(f"Unsupported language: {language}")

    # Return the language code
    return result


def _locale_to_speechmatics_locale(language_code: str, locale: Language) -> Optional[str]:
    """Convert a Language enum to a Speechmatics language code.

    Args:
        language_code: The language code.
        locale: The Language enum to convert.

    Returns:
        str: The Speechmatics language code, if found.
    """
    # Languages and output locales
    LOCALES = {
        "en": {
            Language.EN_GB: "en-GB",
            Language.EN_US: "en-US",
            Language.EN_AU: "en-AU",
        },
    }

    # Get the locale code
    result = LOCALES.get(language_code, {}).get(locale)

    # Fail if locale is not supported
    if not result:
        logger.warning(f"Unsupported output locale: {locale}, defaulting to {language_code}")

    # Return the locale code
    return result
