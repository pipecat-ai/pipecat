#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Daily transport implementation for Pipecat.

This module provides comprehensive Daily video conferencing integration including
audio/video streaming, transcription, recording, dial-in/out functionality, and
real-time communication features.
"""

import asyncio
import time
from concurrent.futures import CancelledError as FuturesCancelledError
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.frames.frames import (
    CancelFrame,
    ControlFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    InterimTranscriptionFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    SpriteFrame,
    StartFrame,
    TranscriptionFrame,
    UserAudioRawFrame,
    UserImageRawFrame,
    UserImageRequestFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.transcriptions.language import Language
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.utils.asyncio.task_manager import BaseTaskManager

try:
    from daily import (
        AudioData,
        CallClient,
        CustomAudioSource,
        CustomAudioTrack,
        Daily,
        EventHandler,
        VideoFrame,
        VirtualCameraDevice,
        VirtualSpeakerDevice,
    )
    from daily import LogLevel as DailyLogLevel
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use the Daily transport, you need to `pip install pipecat-ai[daily]`."
    )
    raise Exception(f"Missing module: {e}")

VAD_RESET_PERIOD_MS = 2000


@dataclass
class DailyOutputTransportMessageFrame(OutputTransportMessageFrame):
    """Frame for transport messages in Daily calls.

    Parameters:
        participant_id: Optional ID of the participant this message is for/from.
    """

    participant_id: Optional[str] = None


@dataclass
class DailyOutputTransportMessageUrgentFrame(OutputTransportMessageUrgentFrame):
    """Frame for urgent transport messages in Daily calls.

    Parameters:
        participant_id: Optional ID of the participant this message is for/from.
    """

    participant_id: Optional[str] = None


@dataclass
class DailyTransportMessageFrame(DailyOutputTransportMessageFrame):
    """Frame for transport messages in Daily calls.

    .. deprecated:: 0.0.87
        This frame is deprecated and will be removed in a future version.
        Instead, use `DailyOutputTransportMessageFrame`.

    Parameters:
        participant_id: Optional ID of the participant this message is for/from.
    """

    def __post_init__(self):
        super().__post_init__()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "DailyTransportMessageFrame is deprecated and will be removed in a future version. "
                "Instead, use DailyOutputTransportMessageFrame.",
                DeprecationWarning,
                stacklevel=2,
            )


@dataclass
class DailyTransportMessageUrgentFrame(DailyOutputTransportMessageUrgentFrame):
    """Frame for urgent transport messages in Daily calls.

    .. deprecated:: 0.0.87
        This frame is deprecated and will be removed in a future version.
        Instead, use `DailyOutputTransportMessageUrgentFrame`.

    Parameters:
        participant_id: Optional ID of the participant this message is for/from.
    """

    def __post_init__(self):
        super().__post_init__()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "DailyTransportMessageUrgentFrame is deprecated and will be removed in a future version. "
                "Instead, use DailyOutputTransportMessageUrgentFrame.",
                DeprecationWarning,
                stacklevel=2,
            )


@dataclass
class DailyInputTransportMessageFrame(InputTransportMessageFrame):
    """Frame for input urgent transport messages in Daily calls.

    Parameters:
        participant_id: Optional ID of the participant this message is for/from.
    """

    participant_id: Optional[str] = None


class DailyInputTransportMessageUrgentFrame(DailyInputTransportMessageFrame):
    """Frame for input urgent transport messages in Daily calls.

    .. deprecated:: 0.0.87
        This frame is deprecated and will be removed in a future version.
        Instead, use `DailyInputTransportMessageFrame`.

    Parameters:
        participant_id: Optional ID of the participant this message is for/from.
    """

    def __post_init__(self):
        super().__post_init__()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "DailyInputTransportMessageUrgentFrame is deprecated and will be removed in a future version. "
                "Instead, use DailyInputTransportMessageFrame.",
                DeprecationWarning,
                stacklevel=2,
            )


@dataclass
class DailyUpdateRemoteParticipantsFrame(ControlFrame):
    """Frame to update remote participants in Daily calls.

    .. deprecated:: 0.0.87
        `DailyUpdateRemoteParticipantsFrame` is deprecated and will be removed in a future version.
        Create your own custom frame and use a custom processor to handle it or use, for example,
        `on_after_push_frame` event instead in the output transport.

    Parameters:
        remote_participants: See https://reference-python.daily.co/api_reference.html#daily.CallClient.update_remote_participants.
    """

    remote_participants: Mapping[str, Any] = None

    def __post_init__(self):
        super().__post_init__()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "DailyUpdateRemoteParticipantsFrame is deprecated and will be removed in a future version."
                "Instead, create your own custom frame and handle it in the "
                '`@transport.output().event_handler("on_after_push_frame")` event handler or a '
                "custom processor.",
                DeprecationWarning,
                stacklevel=2,
            )


class WebRTCVADAnalyzer(VADAnalyzer):
    """Voice Activity Detection analyzer using WebRTC.

    Implements voice activity detection using Daily's native WebRTC VAD.
    """

    def __init__(self, *, sample_rate: Optional[int] = None, params: Optional[VADParams] = None):
        """Initialize the WebRTC VAD analyzer.

        Args:
            sample_rate: Audio sample rate in Hz.
            params: VAD configuration parameters.
        """
        super().__init__(sample_rate=sample_rate, params=params)

        self._webrtc_vad = Daily.create_native_vad(
            reset_period_ms=VAD_RESET_PERIOD_MS, sample_rate=self.sample_rate, channels=1
        )
        logger.debug("Loaded native WebRTC VAD")

    def num_frames_required(self) -> int:
        """Get the number of audio frames required for VAD analysis.

        Returns:
            The number of frames needed (equivalent to 10ms of audio).
        """
        return int(self.sample_rate / 100.0)

    def voice_confidence(self, buffer) -> float:
        """Analyze audio buffer and return voice confidence score.

        Args:
            buffer: Audio buffer to analyze.

        Returns:
            Voice confidence score between 0.0 and 1.0.
        """
        confidence = 0
        if len(buffer) > 0:
            confidence = self._webrtc_vad.analyze_frames(buffer)
        return confidence


class DailyDialinSettings(BaseModel):
    """Settings for Daily's dial-in functionality.

    Parameters:
        call_id: CallId is represented by UUID and represents the sessionId in the SIP Network.
        call_domain: Call Domain is represented by UUID and represents your Daily Domain on the SIP Network.
    """

    call_id: str = ""
    call_domain: str = ""


class DailyTranscriptionSettings(BaseModel):
    """Configuration settings for Daily's transcription service.

    Parameters:
        language: ISO language code for transcription (e.g. "en").
        model: Transcription model to use (e.g. "nova-2-general").
        profanity_filter: Whether to filter profanity from transcripts.
        redact: Whether to redact sensitive information.
        endpointing: Whether to use endpointing to determine speech segments.
        punctuate: Whether to add punctuation to transcripts.
        includeRawResponse: Whether to include raw response data.
        extra: Additional parameters passed to the Deepgram transcription service.
    """

    language: str = "en"
    model: str = "nova-2-general"
    profanity_filter: bool = True
    redact: bool = False
    endpointing: bool = True
    punctuate: bool = True
    includeRawResponse: bool = True
    extra: Mapping[str, Any] = {"interim_results": True}


class DailyParams(TransportParams):
    """Configuration parameters for Daily transport.

    Parameters:
        api_url: Daily API base URL.
        api_key: Daily API authentication key.
        audio_in_user_tracks: Receive users' audio in separate tracks
        dialin_settings: Optional settings for dial-in functionality.
        camera_out_enabled: Whether to enable the main camera output track.
        microphone_out_enabled: Whether to enable the main microphone track.
        transcription_enabled: Whether to enable speech transcription.
        transcription_settings: Configuration for transcription service.
    """

    api_url: str = "https://api.daily.co/v1"
    api_key: str = ""
    audio_in_user_tracks: bool = True
    dialin_settings: Optional[DailyDialinSettings] = None
    camera_out_enabled: bool = True
    microphone_out_enabled: bool = True
    transcription_enabled: bool = False
    transcription_settings: DailyTranscriptionSettings = DailyTranscriptionSettings()


class DailyCallbacks(BaseModel):
    """Callback handlers for Daily events.

    Parameters:
        on_active_speaker_changed: Called when the active speaker of the call has changed.
        on_joined: Called when bot successfully joined a room.
        on_left: Called when bot left a room.
        on_before_leave: Called when bot is about to leave the room.
        on_error: Called when an error occurs.
        on_app_message: Called when receiving an app message.
        on_call_state_updated: Called when call state changes.
        on_client_connected: Called when a client (participant) connects.
        on_client_disconnected: Called when a client (participant) disconnects.
        on_dialin_connected: Called when dial-in is connected.
        on_dialin_ready: Called when dial-in is ready.
        on_dialin_stopped: Called when dial-in is stopped.
        on_dialin_error: Called when dial-in encounters an error.
        on_dialin_warning: Called when dial-in has a warning.
        on_dialout_answered: Called when dial-out is answered.
        on_dialout_connected: Called when dial-out is connected.
        on_dialout_stopped: Called when dial-out is stopped.
        on_dialout_error: Called when dial-out encounters an error.
        on_dialout_warning: Called when dial-out has a warning.
        on_participant_joined: Called when a participant joins.
        on_participant_left: Called when a participant leaves.
        on_participant_updated: Called when participant info is updated.
        on_transcription_message: Called when receiving transcription.
        on_transcription_stopped: Called when transcription is stopped.
        on_transcription_error: Called when transcription encounters an error.
        on_recording_started: Called when recording starts.
        on_recording_stopped: Called when recording stops.
        on_recording_error: Called when recording encounters an error.
    """

    on_active_speaker_changed: Callable[[Mapping[str, Any]], Awaitable[None]]
    on_joined: Callable[[Mapping[str, Any]], Awaitable[None]]
    on_left: Callable[[], Awaitable[None]]
    on_before_leave: Callable[[], Awaitable[None]]
    on_error: Callable[[str], Awaitable[None]]
    on_app_message: Callable[[Any, str], Awaitable[None]]
    on_call_state_updated: Callable[[str], Awaitable[None]]
    on_client_connected: Callable[[Mapping[str, Any]], Awaitable[None]]
    on_client_disconnected: Callable[[Mapping[str, Any]], Awaitable[None]]
    on_dialin_connected: Callable[[Any], Awaitable[None]]
    on_dialin_ready: Callable[[str], Awaitable[None]]
    on_dialin_stopped: Callable[[Any], Awaitable[None]]
    on_dialin_error: Callable[[Any], Awaitable[None]]
    on_dialin_warning: Callable[[Any], Awaitable[None]]
    on_dialout_answered: Callable[[Any], Awaitable[None]]
    on_dialout_connected: Callable[[Any], Awaitable[None]]
    on_dialout_stopped: Callable[[Any], Awaitable[None]]
    on_dialout_error: Callable[[Any], Awaitable[None]]
    on_dialout_warning: Callable[[Any], Awaitable[None]]
    on_participant_joined: Callable[[Mapping[str, Any]], Awaitable[None]]
    on_participant_left: Callable[[Mapping[str, Any], str], Awaitable[None]]
    on_participant_updated: Callable[[Mapping[str, Any]], Awaitable[None]]
    on_transcription_message: Callable[[Mapping[str, Any]], Awaitable[None]]
    on_transcription_stopped: Callable[[str, bool], Awaitable[None]]
    on_transcription_error: Callable[[str], Awaitable[None]]
    on_recording_started: Callable[[Mapping[str, Any]], Awaitable[None]]
    on_recording_stopped: Callable[[str], Awaitable[None]]
    on_recording_error: Callable[[str, str], Awaitable[None]]


def completion_callback(future):
    """Create a completion callback for Daily API calls.

    Args:
        future: The asyncio Future to set the result on.

    Returns:
        A callback function that sets the future result.
    """

    def _callback(*args):
        def set_result(future, *args):
            try:
                if len(args) > 1:
                    future.set_result(args)
                else:
                    future.set_result(*args)
            except asyncio.InvalidStateError:
                pass

        future.get_loop().call_soon_threadsafe(set_result, future, *args)

    return _callback


@dataclass
class DailyAudioTrack:
    """Container for Daily audio track components.

    Parameters:
        source: The custom audio source for the track.
        track: The custom audio track instance.
    """

    source: CustomAudioSource
    track: CustomAudioTrack


class DailyTransportClient(EventHandler):
    """Core client for interacting with Daily's API.

    Manages the connection to Daily rooms and handles all low-level API interactions
    including room management, media streaming, transcription, and event handling.
    """

    _daily_initialized: bool = False

    def __new__(cls, *args, **kwargs):
        """Override EventHandler's __new__ method to ensure Daily is initialized only once."""
        return super().__new__(cls)

    def __init__(
        self,
        room_url: str,
        token: Optional[str],
        bot_name: str,
        params: DailyParams,
        callbacks: DailyCallbacks,
        transport_name: str,
    ):
        """Initialize the Daily transport client.

        Args:
            room_url: URL of the Daily room to connect to.
            token: Optional authentication token for the room.
            bot_name: Display name for the bot in the call.
            params: Configuration parameters for the transport.
            callbacks: Event callback handlers.
            transport_name: Name identifier for the transport.
        """
        super().__init__()

        if not DailyTransportClient._daily_initialized:
            DailyTransportClient._daily_initialized = True
            Daily.init()

        self._room_url: str = room_url
        self._token: Optional[str] = token
        self._bot_name: str = bot_name
        self._params: DailyParams = params
        self._callbacks = callbacks
        self._transport_name = transport_name

        self._participant_id: str = ""
        self._audio_renderers = {}
        self._video_renderers = {}
        self._transcription_ids = []
        self._transcription_status = None
        self._dial_out_session_id: str = ""
        self._dial_in_session_id: str = ""

        self._joining = False
        self._joined = False
        self._joined_event = asyncio.Event()
        self._leave_counter = 0

        self._task_manager: Optional[BaseTaskManager] = None

        # We use the executor to cleanup the client. We just do it from one
        # place, so only one thread is really needed.
        self._executor = ThreadPoolExecutor(max_workers=1)

        self._client: CallClient = CallClient(event_handler=self)

        # We use separate tasks to execute callbacks (events, audio or
        # video). In the case of events, if we call a `CallClient` function
        # inside the callback and wait for its completion this will result in a
        # deadlock (because we haven't exited the event callback). The deadlocks
        # occur because `daily-python` is holding the GIL when calling the
        # callbacks. So, if our callback handler makes a `CallClient` call and
        # waits for it to finish using completions (and a future) we will
        # deadlock because completions use event handlers (which are holding the
        # GIL).
        self._event_task = None
        self._audio_task = None
        self._video_task = None

        # Input and ouput sample rates. They will be initialize on setup().
        self._in_sample_rate = 0
        self._out_sample_rate = 0

        self._camera: Optional[VirtualCameraDevice] = None
        self._speaker: Optional[VirtualSpeakerDevice] = None
        self._microphone_track: Optional[DailyAudioTrack] = None
        self._custom_audio_tracks: Dict[str, DailyAudioTrack] = {}

    def _camera_name(self):
        """Generate a unique virtual camera name for this client instance."""
        return f"camera-{self}"

    def _speaker_name(self):
        """Generate a unique virtual speaker name for this client instance."""
        return f"speaker-{self}"

    @property
    def room_url(self) -> str:
        """Get the Daily room URL.

        Returns:
            The room URL this client is connected to.
        """
        return self._room_url

    @property
    def participant_id(self) -> str:
        """Get the participant ID for this client.

        Returns:
            The participant ID assigned by Daily.
        """
        return self._participant_id

    @property
    def in_sample_rate(self) -> int:
        """Get the input audio sample rate.

        Returns:
            The input sample rate in Hz.
        """
        return self._in_sample_rate

    @property
    def out_sample_rate(self) -> int:
        """Get the output audio sample rate.

        Returns:
            The output sample rate in Hz.
        """
        return self._out_sample_rate

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send an application message to participants.

        Args:
            frame: The message frame to send.
        """
        if not self._joined:
            return

        participant_id = None
        if isinstance(
            frame, (DailyOutputTransportMessageFrame, DailyOutputTransportMessageUrgentFrame)
        ):
            participant_id = frame.participant_id

        future = self._get_event_loop().create_future()
        self._client.send_app_message(
            frame.message, participant_id, completion=completion_callback(future)
        )
        await future

    async def read_next_audio_frame(self) -> Optional[InputAudioRawFrame]:
        """Reads the next 20ms audio frame from the virtual speaker."""
        if not self._speaker:
            return None

        sample_rate = self._in_sample_rate
        num_channels = self._params.audio_in_channels
        num_frames = int(sample_rate / 100) * 2  # 20ms of audio

        future = self._get_event_loop().create_future()
        self._speaker.read_frames(num_frames, completion=completion_callback(future))
        audio = await future

        if len(audio) > 0:
            return InputAudioRawFrame(
                audio=audio, sample_rate=sample_rate, num_channels=num_channels
            )
        else:
            # If we don't read any audio it could be there's no participant
            # connected. daily-python will return immediately if that's the
            # case, so let's sleep for a little bit (i.e. busy wait).
            await asyncio.sleep(0.01)
            return None

    async def register_audio_destination(self, destination: str):
        """Register a custom audio destination for multi-track output.

        Args:
            destination: The destination identifier to register.
        """
        self._custom_audio_tracks[destination] = await self.add_custom_audio_track(destination)
        self._client.update_publishing({"customAudio": {destination: True}})

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the appropriate audio track.

        Args:
            frame: The audio frame to write.

        Returns:
            True if the audio frame was written successfully, False otherwise.
        """
        future = self._get_event_loop().create_future()

        destination = frame.transport_destination
        audio_source: Optional[CustomAudioSource] = None
        if not destination and self._microphone_track:
            audio_source = self._microphone_track.source
        elif destination and destination in self._custom_audio_tracks:
            track = self._custom_audio_tracks[destination]
            audio_source = track.source

        if audio_source:
            audio_source.write_frames(frame.audio, completion=completion_callback(future))
        else:
            logger.warning(f"{self} unable to write audio frames to destination [{destination}]")
            future.set_result(0)

        num_frames = await future
        return num_frames > 0

    async def write_video_frame(self, frame: OutputImageRawFrame) -> bool:
        """Write a video frame to the camera device.

        Args:
            frame: The image frame to write.

        Returns:
            True if the video frame was written successfully, False otherwise.
        """
        if not frame.transport_destination and self._camera:
            self._camera.write_frame(frame.image)
            return True
        return False

    async def setup(self, setup: FrameProcessorSetup):
        """Setup the client with task manager and event queues.

        Args:
            setup: The frame processor setup configuration.
        """
        if self._task_manager:
            return

        self._task_manager = setup.task_manager

        self._event_queue = asyncio.Queue()
        self._event_task = self._task_manager.create_task(
            self._callback_task_handler(self._event_queue),
            f"{self}::event_callback_task",
        )

    async def cleanup(self):
        """Cleanup client resources and cancel tasks."""
        if self._event_task and self._task_manager:
            await self._task_manager.cancel_task(self._event_task)
            self._event_task = None
        if self._audio_task and self._task_manager:
            await self._task_manager.cancel_task(self._audio_task)
            self._audio_task = None
        if self._video_task and self._task_manager:
            await self._task_manager.cancel_task(self._video_task)
            self._video_task = None
        # Make sure we don't block the event loop in case `client.release()`
        # takes extra time.
        await self._get_event_loop().run_in_executor(self._executor, self._cleanup)

    async def start(self, frame: StartFrame):
        """Start the client and initialize audio/video components.

        Args:
            frame: The start frame containing initialization parameters.
        """
        self._in_sample_rate = self._params.audio_in_sample_rate or frame.audio_in_sample_rate
        self._out_sample_rate = self._params.audio_out_sample_rate or frame.audio_out_sample_rate

        if self._params.audio_in_enabled:
            if self._params.audio_in_user_tracks and not self._audio_task and self._task_manager:
                self._audio_queue = asyncio.Queue()
                self._audio_task = self._task_manager.create_task(
                    self._callback_task_handler(self._audio_queue),
                    f"{self}::audio_callback_task",
                )
            elif not self._speaker:
                self._speaker = Daily.create_speaker_device(
                    self._speaker_name(),
                    sample_rate=self._in_sample_rate,
                    channels=self._params.audio_in_channels,
                    non_blocking=True,
                )
                Daily.select_speaker_device(self._speaker_name())

        if self._params.video_in_enabled and not self._video_task and self._task_manager:
            self._video_queue = asyncio.Queue()
            self._video_task = self._task_manager.create_task(
                self._callback_task_handler(self._video_queue),
                f"{self}::video_callback_task",
            )
        if self._params.video_out_enabled and not self._camera:
            self._camera = Daily.create_camera_device(
                self._camera_name(),
                width=self._params.video_out_width,
                height=self._params.video_out_height,
                color_format=self._params.video_out_color_format,
            )

        if self._params.audio_out_enabled and not self._microphone_track:
            audio_source = CustomAudioSource(self._out_sample_rate, self._params.audio_out_channels)
            audio_track = CustomAudioTrack(audio_source)
            self._microphone_track = DailyAudioTrack(source=audio_source, track=audio_track)

    async def join(self):
        """Join the Daily room with configured settings."""
        # Transport already joined or joining, ignore.
        if self._joined or self._joining:
            # Increment leave counter if we already joined.
            self._leave_counter += 1
            return

        logger.info(f"Joining {self._room_url}")
        self._joining = True

        # For performance reasons, never subscribe to video streams (unless a
        # video renderer is registered).
        self._client.update_subscription_profiles(
            {"base": {"camera": "unsubscribed", "screenVideo": "unsubscribed"}}
        )

        self._client.set_user_name(self._bot_name)

        try:
            (data, error) = await self._join()

            if not error:
                self._joined = True
                self._joining = False
                # Increment leave counter if we successfully joined.
                self._leave_counter += 1

                logger.info(f"Joined {self._room_url}")

                if self._params.transcription_enabled:
                    await self.start_transcription(self._params.transcription_settings)

                await self._callbacks.on_joined(data)

                self._joined_event.set()
            else:
                error_msg = f"Error joining {self._room_url}: {error}"
                logger.error(error_msg)
                await self._callbacks.on_error(error_msg)
        except asyncio.TimeoutError:
            error_msg = f"Time out joining {self._room_url}"
            logger.error(error_msg)
            self._joining = False
            await self._callbacks.on_error(error_msg)

    async def _join(self):
        """Execute the actual room join operation."""
        if not self._client:
            return

        future = self._get_event_loop().create_future()

        camera_enabled = self._params.video_out_enabled and self._params.camera_out_enabled
        microphone_enabled = self._params.audio_out_enabled and self._params.microphone_out_enabled

        self._client.join(
            self._room_url,
            self._token,
            completion=completion_callback(future),
            client_settings={
                "inputs": {
                    "camera": {
                        "isEnabled": camera_enabled,
                        "settings": {
                            "deviceId": self._camera_name(),
                        },
                    },
                    "microphone": {
                        "isEnabled": microphone_enabled,
                        "settings": {
                            "customTrack": {
                                "id": self._microphone_track.track.id
                                if self._microphone_track
                                else "no-microphone-track"
                            }
                        },
                    },
                },
                "publishing": {
                    "camera": {
                        "sendSettings": {
                            "maxQuality": "low",
                            "encodings": {
                                "low": {
                                    "maxBitrate": self._params.video_out_bitrate,
                                    "maxFramerate": self._params.video_out_framerate,
                                }
                            },
                        }
                    },
                    "microphone": {
                        "sendSettings": {
                            "channelConfig": "stereo"
                            if self._params.audio_out_channels == 2
                            else "mono",
                            "bitrate": self._params.audio_out_bitrate,
                        }
                    },
                },
            },
        )

        return await asyncio.wait_for(future, timeout=10)

    async def leave(self):
        """Leave the Daily room and cleanup resources."""
        # Decrement leave counter when leaving.
        self._leave_counter -= 1

        # Transport not joined, ignore.
        if not self._joined or self._leave_counter > 0:
            return

        self._joined = False
        self._joined_event.clear()

        logger.info(f"Leaving {self._room_url}")

        # Call callback before leaving.
        await self._callbacks.on_before_leave()

        if self._params.transcription_enabled:
            await self.stop_transcription()

        # Remove any custom tracks, if any.
        for track_name, _ in self._custom_audio_tracks.items():
            await self.remove_custom_audio_track(track_name)

        try:
            error = await self._leave()
            if not error:
                logger.info(f"Left {self._room_url}")
                await self._callbacks.on_left()
            else:
                error_msg = f"Error leaving {self._room_url}: {error}"
                logger.error(error_msg)
                await self._callbacks.on_error(error_msg)
        except asyncio.TimeoutError:
            error_msg = f"Time out leaving {self._room_url}"
            logger.error(error_msg)
            await self._callbacks.on_error(error_msg)

    async def _leave(self):
        """Execute the actual room leave operation."""
        if not self._client:
            return

        future = self._get_event_loop().create_future()
        self._client.leave(completion=completion_callback(future))
        return await asyncio.wait_for(future, timeout=10)

    def _cleanup(self):
        """Cleanup the Daily client instance."""
        if self._client:
            self._client.release()
            self._client = None

    def participants(self):
        """Get current participants in the room.

        Returns:
            Dictionary of participants keyed by participant ID.
        """
        return self._client.participants()

    def participant_counts(self):
        """Get participant count information.

        Returns:
            Dictionary with participant count details.
        """
        return self._client.participant_counts()

    async def start_dialout(self, settings):
        """Start a dial-out call to a phone number.

        Args:
            settings: Dial-out configuration settings.
        """
        logger.debug(f"Starting dialout: settings={settings}")

        future = self._get_event_loop().create_future()
        self._client.start_dialout(settings, completion=completion_callback(future))
        error = await future
        if error:
            logger.error(f"Unable to start dialout: {error}")

    async def stop_dialout(self, participant_id):
        """Stop a dial-out call for a specific participant.

        Args:
            participant_id: ID of the participant to stop dial-out for.
        """
        logger.debug(f"Stopping dialout: participant_id={participant_id}")

        future = self._get_event_loop().create_future()
        self._client.stop_dialout(participant_id, completion=completion_callback(future))
        error = await future
        if error:
            logger.error(f"Unable to stop dialout: {error}")

    async def send_dtmf(self, settings):
        """Send DTMF tones during a call.

        Args:
            settings: DTMF settings including tones and target session.
        """
        session_id = settings.get("sessionId") or self._dial_out_session_id
        if not session_id:
            logger.error("Unable to send DTMF: 'sessionId' is not set")
            return

        # Update 'sessionId' field.
        settings["sessionId"] = session_id

        future = self._get_event_loop().create_future()
        self._client.send_dtmf(settings, completion=completion_callback(future))
        await future

    async def sip_call_transfer(self, settings):
        """Transfer a SIP call to another destination.

        Args:
            settings: SIP call transfer settings.
        """
        session_id = (
            settings.get("sessionId") or self._dial_out_session_id or self._dial_in_session_id
        )
        if not session_id:
            logger.error("Unable to transfer SIP call: 'sessionId' is not set")
            return

        # Update 'sessionId' field.
        settings["sessionId"] = session_id

        future = self._get_event_loop().create_future()
        self._client.sip_call_transfer(settings, completion=completion_callback(future))
        await future

    async def sip_refer(self, settings):
        """Send a SIP REFER request.

        Args:
            settings: SIP REFER settings.
        """
        future = self._get_event_loop().create_future()
        self._client.sip_refer(settings, completion=completion_callback(future))
        await future

    async def start_recording(self, streaming_settings, stream_id, force_new):
        """Start recording the call.

        Args:
            streaming_settings: Recording configuration settings.
            stream_id: Unique identifier for the recording stream.
            force_new: Whether to force a new recording session.
        """
        logger.debug(
            f"Starting recording: stream_id={stream_id} force_new={force_new} settings={streaming_settings}"
        )

        future = self._get_event_loop().create_future()
        self._client.start_recording(
            streaming_settings, stream_id, force_new, completion=completion_callback(future)
        )
        error = await future
        if error:
            logger.error(f"Unable to start recording: {error}")

    async def stop_recording(self, stream_id):
        """Stop recording the call.

        Args:
            stream_id: Unique identifier for the recording stream to stop.
        """
        logger.debug(f"Stopping recording: stream_id={stream_id}")

        future = self._get_event_loop().create_future()
        self._client.stop_recording(stream_id, completion=completion_callback(future))
        error = await future
        if error:
            logger.error(f"Unable to stop recording: {error}")

    async def start_transcription(self, settings):
        """Start transcription for the call.

        Args:
            settings: Transcription configuration settings.
        """
        if not self._token:
            logger.warning("Transcription can't be started without a room token")
            return

        logger.debug(f"Starting transcription: settings={settings}")

        future = self._get_event_loop().create_future()
        self._client.start_transcription(
            settings=self._params.transcription_settings.model_dump(exclude_none=True),
            completion=completion_callback(future),
        )
        error = await future
        if error:
            logger.error(f"Unable to start transcription: {error}")

    async def stop_transcription(self):
        """Stop transcription for the call."""
        if not self._token:
            return

        logger.debug(f"Stopping transcription")

        future = self._get_event_loop().create_future()
        self._client.stop_transcription(completion=completion_callback(future))
        error = await future
        if error:
            logger.error(f"Unable to stop transcription: {error}")

    async def send_prebuilt_chat_message(self, message: str, user_name: Optional[str] = None):
        """Send a chat message to Daily's Prebuilt main room.

        Args:
            message: The chat message to send.
            user_name: Optional user name that will appear as sender of the message.
        """
        if not self._joined:
            return

        future = self._get_event_loop().create_future()
        self._client.send_prebuilt_chat_message(
            message, user_name=user_name, completion=completion_callback(future)
        )
        await future

    async def capture_participant_transcription(self, participant_id: str):
        """Enable transcription capture for a specific participant.

        Args:
            participant_id: ID of the participant to capture transcription for.
        """
        if not self._params.transcription_enabled:
            return

        self._transcription_ids.append(participant_id)
        if self._joined and self._transcription_status:
            await self.update_transcription(self._transcription_ids)

    async def capture_participant_audio(
        self,
        participant_id: str,
        callback: Callable,
        audio_source: str = "microphone",
        sample_rate: int = 16000,
        callback_interval_ms: int = 20,
    ):
        """Capture audio from a specific participant.

        Args:
            participant_id: ID of the participant to capture audio from.
            callback: Callback function to handle audio data.
            audio_source: Audio source to capture (microphone, screenAudio, or custom).
            sample_rate: Desired sample rate for audio capture.
            callback_interval_ms: Interval between audio callbacks in milliseconds.
        """
        # Only enable the desired audio source subscription on this participant.
        if audio_source in ("microphone", "screenAudio"):
            media = {"media": {audio_source: "subscribed"}}
        else:
            media = {"media": {"customAudio": {audio_source: "subscribed"}}}

        await self.update_subscriptions(participant_settings={participant_id: media})

        self._audio_renderers.setdefault(participant_id, {})[audio_source] = callback

        logger.debug(
            f"Starting to capture [{audio_source}] audio from participant {participant_id}"
        )

        self._client.set_audio_renderer(
            participant_id,
            self._audio_data_received,
            audio_source=audio_source,
            sample_rate=sample_rate,
            callback_interval_ms=callback_interval_ms,
        )

    async def capture_participant_video(
        self,
        participant_id: str,
        callback: Callable,
        framerate: int = 30,
        video_source: str = "camera",
        color_format: str = "RGB",
    ):
        """Capture video from a specific participant.

        Args:
            participant_id: ID of the participant to capture video from.
            callback: Callback function to handle video frames.
            framerate: Desired framerate for video capture.
            video_source: Video source to capture (camera, screenVideo, or custom).
            color_format: Color format for video frames.
        """
        # Only enable the desired audio source subscription on this participant.
        if video_source in ("camera", "screenVideo"):
            media = {"media": {video_source: "subscribed"}}
        else:
            media = {"media": {"customVideo": {video_source: "subscribed"}}}

        await self.update_subscriptions(participant_settings={participant_id: media})

        self._video_renderers.setdefault(participant_id, {})[video_source] = callback

        logger.debug(
            f"Starting to capture [{video_source}] video from participant {participant_id}"
        )

        self._client.set_video_renderer(
            participant_id,
            self._video_frame_received,
            video_source=video_source,
            color_format=color_format,
        )

    async def add_custom_audio_track(self, track_name: str) -> DailyAudioTrack:
        """Add a custom audio track for multi-stream output.

        Args:
            track_name: Name for the custom audio track.

        Returns:
            The created DailyAudioTrack instance.
        """
        future = self._get_event_loop().create_future()

        audio_source = CustomAudioSource(self._out_sample_rate, 1)

        audio_track = CustomAudioTrack(audio_source)

        self._client.add_custom_audio_track(
            track_name=track_name,
            audio_track=audio_track,
            ignore_audio_level=True,
            completion=completion_callback(future),
        )

        await future

        track = DailyAudioTrack(source=audio_source, track=audio_track)

        return track

    async def remove_custom_audio_track(self, track_name: str):
        """Remove a custom audio track.

        Args:
            track_name: Name of the custom audio track to remove.
        """
        future = self._get_event_loop().create_future()
        self._client.remove_custom_audio_track(
            track_name=track_name,
            completion=completion_callback(future),
        )
        await future

    async def update_transcription(self, participants=None, instance_id=None):
        """Update transcription settings for specific participants.

        Args:
            participants: List of participant IDs to enable transcription for.
            instance_id: Optional transcription instance ID.
        """
        future = self._get_event_loop().create_future()
        self._client.update_transcription(
            participants, instance_id, completion=completion_callback(future)
        )
        await future

    async def update_subscriptions(self, participant_settings=None, profile_settings=None):
        """Update media subscription settings.

        Args:
            participant_settings: Per-participant subscription settings.
            profile_settings: Global subscription profile settings.
        """
        future = self._get_event_loop().create_future()
        self._client.update_subscriptions(
            participant_settings=participant_settings,
            profile_settings=profile_settings,
            completion=completion_callback(future),
        )
        await future

    async def update_publishing(self, publishing_settings: Mapping[str, Any]):
        """Update media publishing settings.

        Args:
            publishing_settings: Publishing configuration settings.
        """
        future = self._get_event_loop().create_future()
        self._client.update_publishing(
            publishing_settings=publishing_settings,
            completion=completion_callback(future),
        )
        await future

    async def update_remote_participants(self, remote_participants: Mapping[str, Any]):
        """Update settings for remote participants.

        Args:
            remote_participants: Remote participant configuration settings.
        """
        future = self._get_event_loop().create_future()
        self._client.update_remote_participants(
            remote_participants=remote_participants, completion=completion_callback(future)
        )
        await future

    #
    #
    # Daily (EventHandler)
    #

    def on_active_speaker_changed(self, participant):
        """Handle active speaker change events.

        Args:
            participant: The new active speaker participant info.
        """
        self._call_event_callback(self._callbacks.on_active_speaker_changed, participant)

    def on_app_message(self, message: Any, sender: str):
        """Handle application message events.

        Args:
            message: The received message data.
            sender: ID of the message sender.
        """
        self._call_event_callback(self._callbacks.on_app_message, message, sender)

    def on_call_state_updated(self, state: str):
        """Handle call state update events.

        Args:
            state: The new call state.
        """
        self._call_event_callback(self._callbacks.on_call_state_updated, state)

    def on_dialin_connected(self, data: Any):
        """Handle dial-in connected events.

        Args:
            data: Dial-in connection data.
        """
        self._dial_in_session_id = data["sessionId"] if "sessionId" in data else ""
        self._call_event_callback(self._callbacks.on_dialin_connected, data)

    def on_dialin_ready(self, sip_endpoint: str):
        """Handle dial-in ready events.

        Args:
            sip_endpoint: The SIP endpoint for dial-in.
        """
        self._call_event_callback(self._callbacks.on_dialin_ready, sip_endpoint)

    def on_dialin_stopped(self, data: Any):
        """Handle dial-in stopped events.

        Args:
            data: Dial-in stop data.
        """
        # Cleanup only if our session stopped.
        if data.get("sessionId") == self._dial_in_session_id:
            self._dial_in_session_id = ""
        self._call_event_callback(self._callbacks.on_dialin_stopped, data)

    def on_dialin_error(self, data: Any):
        """Handle dial-in error events.

        Args:
            data: Dial-in error data.
        """
        # Cleanup only if our session errored out.
        if data.get("sessionId") == self._dial_in_session_id:
            self._dial_in_session_id = ""
        self._call_event_callback(self._callbacks.on_dialin_error, data)

    def on_dialin_warning(self, data: Any):
        """Handle dial-in warning events.

        Args:
            data: Dial-in warning data.
        """
        self._call_event_callback(self._callbacks.on_dialin_warning, data)

    def on_dialout_answered(self, data: Any):
        """Handle dial-out answered events.

        Args:
            data: Dial-out answered data.
        """
        self._call_event_callback(self._callbacks.on_dialout_answered, data)

    def on_dialout_connected(self, data: Any):
        """Handle dial-out connected events.

        Args:
            data: Dial-out connection data.
        """
        self._dial_out_session_id = data["sessionId"] if "sessionId" in data else ""
        self._call_event_callback(self._callbacks.on_dialout_connected, data)

    def on_dialout_stopped(self, data: Any):
        """Handle dial-out stopped events.

        Args:
            data: Dial-out stop data.
        """
        # Cleanup only if our session stopped.
        if data.get("sessionId") == self._dial_out_session_id:
            self._dial_out_session_id = ""
        self._call_event_callback(self._callbacks.on_dialout_stopped, data)

    def on_dialout_error(self, data: Any):
        """Handle dial-out error events.

        Args:
            data: Dial-out error data.
        """
        # Cleanup only if our session errored out.
        if data.get("sessionId") == self._dial_out_session_id:
            self._dial_out_session_id = ""
        self._call_event_callback(self._callbacks.on_dialout_error, data)

    def on_dialout_warning(self, data: Any):
        """Handle dial-out warning events.

        Args:
            data: Dial-out warning data.
        """
        self._call_event_callback(self._callbacks.on_dialout_warning, data)

    def on_participant_joined(self, participant):
        """Handle participant joined events.

        Args:
            participant: The participant that joined.
        """
        self._call_event_callback(self._callbacks.on_participant_joined, participant)

    def on_participant_left(self, participant, reason):
        """Handle participant left events.

        Args:
            participant: The participant that left.
            reason: Reason for leaving.
        """
        self._call_event_callback(self._callbacks.on_participant_left, participant, reason)

    def on_participant_updated(self, participant):
        """Handle participant updated events.

        Args:
            participant: The updated participant info.
        """
        self._call_event_callback(self._callbacks.on_participant_updated, participant)

    def on_transcription_started(self, status):
        """Handle transcription started events.

        Args:
            status: Transcription start status.
        """
        logger.debug(f"Transcription started: {status}")
        self._transcription_status = status
        self._call_event_callback(self.update_transcription, self._transcription_ids)

    def on_transcription_stopped(self, stopped_by, stopped_by_error):
        """Handle transcription stopped events.

        Args:
            stopped_by: Who stopped the transcription.
            stopped_by_error: Whether stopped due to error.
        """
        logger.debug("Transcription stopped")
        self._call_event_callback(
            self._callbacks.on_transcription_stopped, stopped_by, stopped_by_error
        )

    def on_transcription_error(self, message):
        """Handle transcription error events.

        Args:
            message: Error message.
        """
        logger.error(f"Transcription error: {message}")
        self._call_event_callback(self._callbacks.on_transcription_error, message)

    def on_transcription_message(self, message):
        """Handle transcription message events.

        Args:
            message: The transcription message data.
        """
        self._call_event_callback(self._callbacks.on_transcription_message, message)

    def on_recording_started(self, status):
        """Handle recording started events.

        Args:
            status: Recording start status.
        """
        logger.debug(f"Recording started: {status}")
        self._call_event_callback(self._callbacks.on_recording_started, status)

    def on_recording_stopped(self, stream_id):
        """Handle recording stopped events.

        Args:
            stream_id: ID of the stopped recording stream.
        """
        logger.debug(f"Recording stopped: {stream_id}")
        self._call_event_callback(self._callbacks.on_recording_stopped, stream_id)

    def on_recording_error(self, stream_id, message):
        """Handle recording error events.

        Args:
            stream_id: ID of the recording stream with error.
            message: Error message.
        """
        logger.error(f"Recording error for {stream_id}: {message}")
        self._call_event_callback(self._callbacks.on_recording_error, stream_id, message)

    #
    # Daily (CallClient callbacks)
    #

    def _audio_data_received(self, participant_id: str, audio_data: AudioData, audio_source: str):
        """Handle received audio data from participants."""
        callback = self._audio_renderers[participant_id][audio_source]
        self._call_audio_callback(callback, participant_id, audio_data, audio_source)

    def _video_frame_received(
        self, participant_id: str, video_frame: VideoFrame, video_source: str
    ):
        """Handle received video frames from participants."""
        callback = self._video_renderers[participant_id][video_source]
        self._call_video_callback(callback, participant_id, video_frame, video_source)

    #
    # Queue callbacks handling
    #

    def _call_audio_callback(self, callback, *args):
        """Queue an audio callback for async execution."""
        self._call_async_callback(self._audio_queue, callback, *args)

    def _call_video_callback(self, callback, *args):
        """Queue a video callback for async execution."""
        self._call_async_callback(self._video_queue, callback, *args)

    def _call_event_callback(self, callback, *args):
        """Queue an event callback for async execution."""
        self._call_async_callback(self._event_queue, callback, *args)

    def _call_async_callback(self, queue: asyncio.Queue, callback, *args):
        """Queue a callback for async execution on the event loop."""
        try:
            future = asyncio.run_coroutine_threadsafe(
                queue.put((callback, *args)), self._get_event_loop()
            )
            future.result()
        except FuturesCancelledError:
            pass

    async def _callback_task_handler(self, queue: asyncio.Queue):
        """Handle queued callbacks from the specified queue."""
        while True:
            # Wait to process any callback until we are joined.
            await self._joined_event.wait()
            (callback, *args) = await queue.get()
            await callback(*args)
            queue.task_done()

    def _get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get the event loop from the task manager."""
        if not self._task_manager:
            raise Exception(f"{self}: missing task manager (pipeline not started?)")
        return self._task_manager.get_event_loop()

    def __str__(self):
        """String representation of the DailyTransportClient."""
        return f"{self._transport_name}::DailyTransportClient"


class DailyInputTransport(BaseInputTransport):
    """Handles incoming media streams and events from Daily calls.

    Processes incoming audio, video, transcriptions and other events from Daily
    room participants, including participant media capture and event forwarding.
    """

    def __init__(
        self,
        transport: BaseTransport,
        client: DailyTransportClient,
        params: DailyParams,
        **kwargs,
    ):
        """Initialize the Daily input transport.

        Args:
            transport: The parent transport instance.
            client: DailyTransportClient instance.
            params: Configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)

        self._transport = transport
        self._client = client
        self._params = params

        self._video_renderers = {}

        # Whether we have seen a StartFrame already.
        self._initialized = False

        # Whether we have started audio streaming.
        self._streaming_started = False

        # Store the list of participants we should stream. This is necessary in
        # case we don't start streaming right away.
        self._capture_participant_audio = []

        # Audio task when using a virtual speaker (i.e. no user tracks).
        self._audio_in_task: Optional[asyncio.Task] = None

        self._vad_analyzer: Optional[VADAnalyzer] = params.vad_analyzer

    @property
    def vad_analyzer(self) -> Optional[VADAnalyzer]:
        """Get the Voice Activity Detection analyzer.

        Returns:
            The VAD analyzer instance if configured.
        """
        return self._vad_analyzer

    async def start_audio_in_streaming(self):
        """Start receiving audio from participants."""
        if not self._params.audio_in_enabled:
            return

        logger.debug(f"Start receiving audio")

        if self._params.audio_in_enabled:
            if self._params.audio_in_user_tracks:
                # Capture invididual participant tracks.
                for participant_id, audio_source, sample_rate in self._capture_participant_audio:
                    await self._client.capture_participant_audio(
                        participant_id, self._on_participant_audio_data, audio_source, sample_rate
                    )
            elif not self._audio_in_task:
                # Create audio task. It reads audio frames from a single room
                # track and pushes them internally for VAD processing.
                self._audio_in_task = self.create_task(self._audio_in_task_handler())

        self._streaming_started = True

    async def setup(self, setup: FrameProcessorSetup):
        """Setup the input transport with shared client setup.

        Args:
            setup: The frame processor setup configuration.
        """
        await super().setup(setup)
        await self._client.setup(setup)

    async def cleanup(self):
        """Cleanup input transport and shared resources."""
        await super().cleanup()
        await self._client.cleanup()
        await self._transport.cleanup()

    async def start(self, frame: StartFrame):
        """Start the input transport and join the Daily room.

        Args:
            frame: The start frame containing initialization parameters.
        """
        # Parent start.
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        # Setup client.
        await self._client.start(frame)

        # Join the room.
        await self._client.join()

        # Indicate the transport that we are connected.
        await self.set_transport_ready(frame)

        if self._params.audio_in_stream_on_start:
            await self.start_audio_in_streaming()

    async def stop(self, frame: EndFrame):
        """Stop the input transport and leave the Daily room.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        # Parent stop.
        await super().stop(frame)
        # Leave the room.
        await self._client.leave()
        # Stop audio thread.
        if self._audio_in_task:
            await self.cancel_task(self._audio_in_task)
            self._audio_in_task = None

    async def cancel(self, frame: CancelFrame):
        """Cancel the input transport and leave the Daily room.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        # Parent stop.
        await super().cancel(frame)
        # Leave the room.
        await self._client.leave()
        # Stop audio thread.
        if self._audio_in_task:
            await self.cancel_task(self._audio_in_task)
            self._audio_in_task = None

    #
    # FrameProcessor
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames, including user image requests.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, UserImageRequestFrame):
            await self.request_participant_image(frame)

    #
    # Frames
    #

    async def push_transcription_frame(self, frame: TranscriptionFrame | InterimTranscriptionFrame):
        """Push a transcription frame downstream.

        Args:
            frame: The transcription frame to push.
        """
        await self.push_frame(frame)

    async def push_app_message(self, message: Any, sender: str):
        """Push an application message as an urgent transport frame.

        Args:
            message: The message data to send.
            sender: ID of the message sender.
        """
        frame = DailyInputTransportMessageFrame(message=message, participant_id=sender)
        await self.push_frame(frame)

    #
    # Audio in
    #

    async def capture_participant_audio(
        self,
        participant_id: str,
        audio_source: str = "microphone",
        sample_rate: int = 16000,
    ):
        """Capture audio from a specific participant.

        Args:
            participant_id: ID of the participant to capture audio from.
            audio_source: Audio source to capture from.
            sample_rate: Desired sample rate for audio capture.
        """
        if self._streaming_started:
            await self._client.capture_participant_audio(
                participant_id, self._on_participant_audio_data, audio_source, sample_rate
            )
        else:
            self._capture_participant_audio.append((participant_id, audio_source, sample_rate))

    async def _on_participant_audio_data(
        self, participant_id: str, audio: AudioData, audio_source: str
    ):
        """Handle received participant audio data."""
        frame = UserAudioRawFrame(
            user_id=participant_id,
            audio=audio.audio_frames,
            sample_rate=audio.sample_rate,
            num_channels=audio.num_channels,
        )
        frame.transport_source = audio_source
        await self.push_audio_frame(frame)

    async def _audio_in_task_handler(self):
        while True:
            frame = await self._client.read_next_audio_frame()
            if frame:
                await self.push_audio_frame(frame)

    #
    # Camera in
    #

    async def capture_participant_video(
        self,
        participant_id: str,
        framerate: int = 30,
        video_source: str = "camera",
        color_format: str = "RGB",
    ):
        """Capture video from a specific participant.

        Args:
            participant_id: ID of the participant to capture video from.
            framerate: Desired framerate for video capture.
            video_source: Video source to capture from.
            color_format: Color format for video frames.
        """
        if participant_id not in self._video_renderers:
            self._video_renderers[participant_id] = {}

        self._video_renderers[participant_id][video_source] = {
            "framerate": framerate,
            "timestamp": 0,
            "render_next_frame": [],
        }

        await self._client.capture_participant_video(
            participant_id, self._on_participant_video_frame, framerate, video_source, color_format
        )

    async def request_participant_image(self, frame: UserImageRequestFrame):
        """Request a video frame from a specific participant.

        Args:
            frame: The user image request frame.
        """
        if frame.user_id in self._video_renderers:
            video_source = frame.video_source if frame.video_source else "camera"
            self._video_renderers[frame.user_id][video_source]["render_next_frame"].append(frame)

    async def _on_participant_video_frame(
        self, participant_id: str, video_frame: VideoFrame, video_source: str
    ):
        """Handle received participant video frames."""
        render_frame = False

        curr_time = time.time()
        prev_time = self._video_renderers[participant_id][video_source]["timestamp"]
        framerate = self._video_renderers[participant_id][video_source]["framerate"]

        # Some times we render frames because of a request.
        request_frame = None

        if framerate > 0:
            next_time = prev_time + 1 / framerate
            render_frame = (next_time - curr_time) < 0.1

        if self._video_renderers[participant_id][video_source]["render_next_frame"]:
            request_frame = self._video_renderers[participant_id][video_source][
                "render_next_frame"
            ].pop(0)
            render_frame = True

        if render_frame:
            frame = UserImageRawFrame(
                user_id=participant_id,
                request=request_frame,
                image=video_frame.buffer,
                size=(video_frame.width, video_frame.height),
                format=video_frame.color_format,
            )
            frame.transport_source = video_source
            await self.push_video_frame(frame)
            self._video_renderers[participant_id][video_source]["timestamp"] = curr_time


class DailyOutputTransport(BaseOutputTransport):
    """Handles outgoing media streams and events to Daily calls.

    Manages sending audio, video and other data to Daily calls,
    including audio destination registration and message transmission.
    """

    def __init__(
        self, transport: BaseTransport, client: DailyTransportClient, params: DailyParams, **kwargs
    ):
        """Initialize the Daily output transport.

        Args:
            transport: The parent transport instance.
            client: DailyTransportClient instance.
            params: Configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)

        self._transport = transport
        self._client = client

        # Whether we have seen a StartFrame already.
        self._initialized = False

    async def setup(self, setup: FrameProcessorSetup):
        """Setup the output transport with shared client setup.

        Args:
            setup: The frame processor setup configuration.
        """
        await super().setup(setup)
        await self._client.setup(setup)

    async def cleanup(self):
        """Cleanup output transport and shared resources."""
        await super().cleanup()
        await self._client.cleanup()
        await self._transport.cleanup()

    async def start(self, frame: StartFrame):
        """Start the output transport and join the Daily room.

        Args:
            frame: The start frame containing initialization parameters.
        """
        # Parent start.
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        # Setup client.
        await self._client.start(frame)

        # Join the room.
        await self._client.join()

        # Indicate the transport that we are connected.
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the output transport and leave the Daily room.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        # Parent stop.
        await super().stop(frame)
        # Leave the room.
        await self._client.leave()

    async def cancel(self, frame: CancelFrame):
        """Cancel the output transport and leave the Daily room.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        # Parent stop.
        await super().cancel(frame)
        # Leave the room.
        await self._client.leave()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process outgoing frames, including transport messages.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, DailyUpdateRemoteParticipantsFrame):
            await self._client.update_remote_participants(frame.remote_participants)

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send a transport message to participants.

        Args:
            frame: The transport message frame to send.
        """
        await self._client.send_message(frame)

    async def register_video_destination(self, destination: str):
        """Register a video output destination.

        Args:
            destination: The destination identifier to register.
        """
        logger.warning(f"{self} registering video destinations is not supported yet")

    async def register_audio_destination(self, destination: str):
        """Register an audio output destination.

        Args:
            destination: The destination identifier to register.

        Returns:
            True if the audio frame was written successfully, False otherwise.
        """
        await self._client.register_audio_destination(destination)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the Daily call.

        Args:
            frame: The audio frame to write.

        Returns:
            True if the audio frame was written successfully, False otherwise.
        """
        return await self._client.write_audio_frame(frame)

    async def write_video_frame(self, frame: OutputImageRawFrame) -> bool:
        """Write a video frame to the Daily call.

        Args:
            frame: The video frame to write.

        Returns:
            True if the video frame was written successfully, False otherwise.
        """
        return await self._client.write_video_frame(frame)

    def _supports_native_dtmf(self) -> bool:
        """Daily supports native DTMF via telephone events.

        Returns:
            True, as Daily supports native DTMF transmission.
        """
        return True

    async def _write_dtmf_native(self, frame):
        """Use Daily's native send_dtmf method for telephone events.

        Args:
            frame: The DTMF frame to write.
        """
        await self._client.send_dtmf(
            {
                "sessionId": frame.transport_destination,
                "tones": frame.button.value,
            }
        )


class DailyTransport(BaseTransport):
    """Transport implementation for Daily audio and video calls.

    Provides comprehensive Daily integration including audio/video streaming,
    transcription, recording, dial-in/out functionality, and real-time communication
    features for conversational AI applications.
    """

    def __init__(
        self,
        room_url: str,
        token: Optional[str],
        bot_name: str,
        params: Optional[DailyParams] = None,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        """Initialize the Daily transport.

        Args:
            room_url: URL of the Daily room to connect to.
            token: Optional authentication token for the room.
            bot_name: Display name for the bot in the call.
            params: Configuration parameters for the transport.
            input_name: Optional name for the input transport.
            output_name: Optional name for the output transport.
        """
        super().__init__(input_name=input_name, output_name=output_name)

        callbacks = DailyCallbacks(
            on_active_speaker_changed=self._on_active_speaker_changed,
            on_joined=self._on_joined,
            on_left=self._on_left,
            on_before_leave=self._on_before_leave,
            on_error=self._on_error,
            on_app_message=self._on_app_message,
            on_call_state_updated=self._on_call_state_updated,
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_dialin_connected=self._on_dialin_connected,
            on_dialin_ready=self._on_dialin_ready,
            on_dialin_stopped=self._on_dialin_stopped,
            on_dialin_error=self._on_dialin_error,
            on_dialin_warning=self._on_dialin_warning,
            on_dialout_answered=self._on_dialout_answered,
            on_dialout_connected=self._on_dialout_connected,
            on_dialout_stopped=self._on_dialout_stopped,
            on_dialout_error=self._on_dialout_error,
            on_dialout_warning=self._on_dialout_warning,
            on_participant_joined=self._on_participant_joined,
            on_participant_left=self._on_participant_left,
            on_participant_updated=self._on_participant_updated,
            on_transcription_message=self._on_transcription_message,
            on_transcription_stopped=self._on_transcription_stopped,
            on_transcription_error=self._on_transcription_error,
            on_recording_started=self._on_recording_started,
            on_recording_stopped=self._on_recording_stopped,
            on_recording_error=self._on_recording_error,
        )
        self._params = params or DailyParams()

        self._client = DailyTransportClient(
            room_url, token, bot_name, self._params, callbacks, self.name
        )
        self._input: Optional[DailyInputTransport] = None
        self._output: Optional[DailyOutputTransport] = None

        self._other_participant_has_joined = False

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_active_speaker_changed")
        self._register_event_handler("on_joined")
        self._register_event_handler("on_left")
        self._register_event_handler("on_error")
        self._register_event_handler("on_app_message")
        self._register_event_handler("on_call_state_updated")
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_dialin_connected")
        self._register_event_handler("on_dialin_ready")
        self._register_event_handler("on_dialin_stopped")
        self._register_event_handler("on_dialin_error")
        self._register_event_handler("on_dialin_warning")
        self._register_event_handler("on_dialout_answered")
        self._register_event_handler("on_dialout_connected")
        self._register_event_handler("on_dialout_stopped")
        self._register_event_handler("on_dialout_error")
        self._register_event_handler("on_dialout_warning")
        self._register_event_handler("on_first_participant_joined")
        self._register_event_handler("on_participant_joined")
        self._register_event_handler("on_participant_left")
        self._register_event_handler("on_participant_updated")
        self._register_event_handler("on_transcription_message")
        self._register_event_handler("on_recording_started")
        self._register_event_handler("on_recording_stopped")
        self._register_event_handler("on_recording_error")
        self._register_event_handler("on_before_leave", sync=True)

    #
    # BaseTransport
    #

    def input(self) -> DailyInputTransport:
        """Get the input transport for receiving media and events.

        Returns:
            The Daily input transport instance.
        """
        if not self._input:
            self._input = DailyInputTransport(
                self, self._client, self._params, name=self._input_name
            )
        return self._input

    def output(self) -> DailyOutputTransport:
        """Get the output transport for sending media and events.

        Returns:
            The Daily output transport instance.
        """
        if not self._output:
            self._output = DailyOutputTransport(
                self, self._client, self._params, name=self._output_name
            )
        return self._output

    #
    # DailyTransport
    #

    @property
    def room_url(self) -> str:
        """Get the Daily room URL.

        Returns:
            The room URL this transport is connected to.
        """
        return self._client.room_url

    @property
    def participant_id(self) -> str:
        """Get the participant ID for this transport.

        Returns:
            The participant ID assigned by Daily.
        """
        return self._client.participant_id

    def set_log_level(self, level: DailyLogLevel):
        """Set the logging level for Daily's internal logging system.

        Args:
            level: The log level to set. Should be a member of the DailyLogLevel enum,
                  such as DailyLogLevel.Info, DailyLogLevel.Debug, etc.

        Example:
            transport.set_log_level(DailyLogLevel.Info)
        """
        Daily.set_log_level(level)

    async def send_image(self, frame: OutputImageRawFrame | SpriteFrame):
        """Send an image frame to the Daily call.

        Args:
            frame: The image frame to send.
        """
        if self._output:
            await self._output.queue_frame(frame, FrameDirection.DOWNSTREAM)

    async def send_audio(self, frame: OutputAudioRawFrame):
        """Send an audio frame to the Daily call.

        Args:
            frame: The audio frame to send.
        """
        if self._output:
            await self._output.queue_frame(frame, FrameDirection.DOWNSTREAM)

    def participants(self):
        """Get current participants in the room.

        Returns:
            Dictionary of participants keyed by participant ID.
        """
        return self._client.participants()

    def participant_counts(self):
        """Get participant count information.

        Returns:
            Dictionary with participant count details.
        """
        return self._client.participant_counts()

    async def start_dialout(self, settings=None):
        """Start a dial-out call to a phone number.

        Args:
            settings: Dial-out configuration settings.
        """
        await self._client.start_dialout(settings)

    async def stop_dialout(self, participant_id):
        """Stop a dial-out call for a specific participant.

        Args:
            participant_id: ID of the participant to stop dial-out for.
        """
        await self._client.stop_dialout(participant_id)

    async def sip_call_transfer(self, settings):
        """Transfer a SIP call to another destination.

        Args:
            settings: SIP call transfer settings.
        """
        await self._client.sip_call_transfer(settings)

    async def sip_refer(self, settings):
        """Send a SIP REFER request.

        Args:
            settings: SIP REFER settings.
        """
        await self._client.sip_refer(settings)

    async def start_recording(self, streaming_settings=None, stream_id=None, force_new=None):
        """Start recording the call.

        Args:
            streaming_settings: Recording configuration settings.
            stream_id: Unique identifier for the recording stream.
            force_new: Whether to force a new recording session.
        """
        await self._client.start_recording(streaming_settings, stream_id, force_new)

    async def stop_recording(self, stream_id=None):
        """Stop recording the call.

        Args:
            stream_id: Unique identifier for the recording stream to stop.
        """
        await self._client.stop_recording(stream_id)

    async def start_transcription(self, settings=None):
        """Start transcription for the call.

        Args:
            settings: Transcription configuration settings.
        """
        await self._client.start_transcription(settings)

    async def stop_transcription(self):
        """Stop transcription for the call."""
        await self._client.stop_transcription()

    async def send_prebuilt_chat_message(self, message: str, user_name: Optional[str] = None):
        """Send a chat message to Daily's Prebuilt main room.

        Args:
            message: The chat message to send.
            user_name: Optional user name that will appear as sender of the message.
        """
        await self._client.send_prebuilt_chat_message(message, user_name)

    async def capture_participant_transcription(self, participant_id: str):
        """Enable transcription capture for a specific participant.

        Args:
            participant_id: ID of the participant to capture transcription for.
        """
        await self._client.capture_participant_transcription(participant_id)

    async def capture_participant_audio(
        self,
        participant_id: str,
        audio_source: str = "microphone",
        sample_rate: int = 16000,
    ):
        """Capture audio from a specific participant.

        Args:
            participant_id: ID of the participant to capture audio from.
            audio_source: Audio source to capture from.
            sample_rate: Desired sample rate for audio capture.
        """
        if self._input:
            await self._input.capture_participant_audio(participant_id, audio_source, sample_rate)

    async def capture_participant_video(
        self,
        participant_id: str,
        framerate: int = 30,
        video_source: str = "camera",
        color_format: str = "RGB",
    ):
        """Capture video from a specific participant.

        Args:
            participant_id: ID of the participant to capture video from.
            framerate: Desired framerate for video capture.
            video_source: Video source to capture from.
            color_format: Color format for video frames.
        """
        if self._input:
            await self._input.capture_participant_video(
                participant_id, framerate, video_source, color_format
            )

    async def update_publishing(self, publishing_settings: Mapping[str, Any]):
        """Update media publishing settings.

        Args:
            publishing_settings: Publishing configuration settings.
        """
        await self._client.update_publishing(publishing_settings=publishing_settings)

    async def update_subscriptions(self, participant_settings=None, profile_settings=None):
        """Update media subscription settings.

        Args:
            participant_settings: Per-participant subscription settings.
            profile_settings: Global subscription profile settings.
        """
        await self._client.update_subscriptions(
            participant_settings=participant_settings, profile_settings=profile_settings
        )

    async def update_remote_participants(self, remote_participants: Mapping[str, Any]):
        """Update settings for remote participants.

        Args:
            remote_participants: Remote participant configuration settings.
        """
        await self._client.update_remote_participants(remote_participants=remote_participants)

    async def _on_active_speaker_changed(self, participant: Any):
        """Handle active speaker change events."""
        await self._call_event_handler("on_active_speaker_changed", participant)

    async def _on_joined(self, data):
        """Handle room joined events."""
        await self._call_event_handler("on_joined", data)

    async def _on_left(self):
        """Handle room left events."""
        await self._call_event_handler("on_left")

    async def _on_before_leave(self):
        """Handle before leave room events."""
        await self._call_event_handler("on_before_leave")

    async def _on_error(self, error):
        """Handle error events and push error frames."""
        await self._call_event_handler("on_error", error)
        # Push error frame to notify the pipeline
        error_frame = ErrorFrame(error)

        if self._input:
            await self._input.push_error(error_frame)
        elif self._output:
            await self._output.push_error(error_frame)
        else:
            logger.error("Both input and output are None while trying to push error")
            raise Exception("No valid input or output channel to push error")

    async def _on_app_message(self, message: Any, sender: str):
        """Handle application message events."""
        if self._input:
            await self._input.push_app_message(message, sender)
        await self._call_event_handler("on_app_message", message, sender)

    async def _on_call_state_updated(self, state: str):
        """Handle call state update events."""
        await self._call_event_handler("on_call_state_updated", state)

    async def _on_client_connected(self, participant: Any):
        """Handle client connected events."""
        await self._call_event_handler("on_client_connected", participant)

    async def _on_client_disconnected(self, participant: Any):
        """Handle client disconnected events."""
        await self._call_event_handler("on_client_disconnected", participant)

    async def _handle_dialin_ready(self, sip_endpoint: str):
        """Handle dial-in ready events by updating SIP configuration."""
        if not self._params.dialin_settings:
            return

        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self._params.api_key}",
                "Content-Type": "application/json",
            }
            data = {
                "callId": self._params.dialin_settings.call_id,
                "callDomain": self._params.dialin_settings.call_domain,
                "sipUri": sip_endpoint,
            }

            url = f"{self._params.api_url}/dialin/pinlessCallUpdate"

            try:
                async with session.post(
                    url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=10)
                ) as r:
                    if r.status != 200:
                        text = await r.text()
                        logger.error(
                            f"Unable to handle dialin-ready event (status: {r.status}, error: {text})"
                        )
                        return

                    logger.debug("Event dialin-ready was handled successfully")
            except asyncio.TimeoutError:
                logger.error(f"Timeout handling dialin-ready event ({url})")
            except Exception as e:
                logger.exception(f"Error handling dialin-ready event ({url}): {e}")

    async def _on_dialin_connected(self, data):
        """Handle dial-in connected events."""
        await self._call_event_handler("on_dialin_connected", data)

    async def _on_dialin_ready(self, sip_endpoint):
        """Handle dial-in ready events."""
        if self._params.dialin_settings:
            await self._handle_dialin_ready(sip_endpoint)
        await self._call_event_handler("on_dialin_ready", sip_endpoint)

    async def _on_dialin_stopped(self, data):
        """Handle dial-in stopped events."""
        await self._call_event_handler("on_dialin_stopped", data)

    async def _on_dialin_error(self, data):
        """Handle dial-in error events."""
        await self._call_event_handler("on_dialin_error", data)

    async def _on_dialin_warning(self, data):
        """Handle dial-in warning events."""
        await self._call_event_handler("on_dialin_warning", data)

    async def _on_dialout_answered(self, data):
        """Handle dial-out answered events."""
        await self._call_event_handler("on_dialout_answered", data)

    async def _on_dialout_connected(self, data):
        """Handle dial-out connected events."""
        await self._call_event_handler("on_dialout_connected", data)

    async def _on_dialout_stopped(self, data):
        """Handle dial-out stopped events."""
        await self._call_event_handler("on_dialout_stopped", data)

    async def _on_dialout_error(self, data):
        """Handle dial-out error events."""
        await self._call_event_handler("on_dialout_error", data)

    async def _on_dialout_warning(self, data):
        """Handle dial-out warning events."""
        await self._call_event_handler("on_dialout_warning", data)

    async def _on_participant_joined(self, participant):
        """Handle participant joined events."""
        id = participant["id"]
        logger.info(f"Participant joined {id}")

        if self._input and self._params.audio_in_enabled and self._params.audio_in_user_tracks:
            await self._input.capture_participant_audio(
                id, "microphone", self._client.in_sample_rate
            )

        if not self._other_participant_has_joined:
            self._other_participant_has_joined = True
            await self._call_event_handler("on_first_participant_joined", participant)

        await self._call_event_handler("on_participant_joined", participant)
        # Also call on_client_connected for compatibility with other transports
        await self._call_event_handler("on_client_connected", participant)

    async def _on_participant_left(self, participant, reason):
        """Handle participant left events."""
        id = participant["id"]
        logger.info(f"Participant left {id}")
        await self._call_event_handler("on_participant_left", participant, reason)
        # Also call on_client_disconnected for compatibility with other transports
        await self._call_event_handler("on_client_disconnected", participant)

    async def _on_participant_updated(self, participant):
        """Handle participant updated events."""
        await self._call_event_handler("on_participant_updated", participant)

    async def _on_transcription_message(self, message: Mapping[str, Any]) -> None:
        """Handle transcription message events."""
        await self._call_event_handler("on_transcription_message", message)

        participant_id = ""
        if "participantId" in message:
            participant_id = message["participantId"]
        if not participant_id:
            return

        text = message["text"]
        timestamp = message["timestamp"]
        raw_response = message.get("rawResponse", {})
        is_final = raw_response.get("is_final", False)
        try:
            language = raw_response["channel"]["alternatives"][0]["languages"][0]
            language = Language(language)
        except KeyError:
            language = None
        if is_final:
            frame = TranscriptionFrame(text, participant_id, timestamp, language, result=message)
            logger.debug(f"Transcription (from: {participant_id}): [{text}]")
        else:
            frame = InterimTranscriptionFrame(
                text,
                participant_id,
                timestamp,
                language,
                result=message,
            )

        if self._input:
            await self._input.push_transcription_frame(frame)

    async def _on_transcription_stopped(self, stopped_by, stopped_by_error):
        """Handle transcription stopped events."""
        await self._call_event_handler("on_transcription_stopped", stopped_by, stopped_by_error)

    async def _on_transcription_error(self, message):
        """Handle transcription error events."""
        await self._call_event_handler("on_transcription_error", message)

    async def _on_recording_started(self, status):
        """Handle recording started events."""
        await self._call_event_handler("on_recording_started", status)

    async def _on_recording_stopped(self, stream_id):
        """Handle recording stopped events."""
        await self._call_event_handler("on_recording_stopped", stream_id)

    async def _on_recording_error(self, stream_id, message):
        """Handle recording error events."""
        await self._call_event_handler("on_recording_error", stream_id, message)
