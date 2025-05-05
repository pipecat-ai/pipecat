#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_default_resampler
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    SpriteFrame,
    StartFrame,
    TranscriptionFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
    UserAudioRawFrame,
    UserImageRawFrame,
    UserImageRequestFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transcriptions.language import Language
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.utils.asyncio import BaseTaskManager

try:
    from daily import (
        AudioData,
        CallClient,
        CustomAudioSource,
        Daily,
        EventHandler,
        VideoFrame,
        VirtualCameraDevice,
        VirtualMicrophoneDevice,
        VirtualSpeakerDevice,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use the Daily transport, you need to `pip install pipecat-ai[daily]`."
    )
    raise Exception(f"Missing module: {e}")

VAD_RESET_PERIOD_MS = 2000


@dataclass
class DailyTransportMessageFrame(TransportMessageFrame):
    """Frame for transport messages in Daily calls.

    Attributes:
        participant_id: Optional ID of the participant this message is for/from.
    """

    participant_id: Optional[str] = None


@dataclass
class DailyTransportMessageUrgentFrame(TransportMessageUrgentFrame):
    """Frame for urgent transport messages in Daily calls.

    Attributes:
        participant_id: Optional ID of the participant this message is for/from.
    """

    participant_id: Optional[str] = None


class WebRTCVADAnalyzer(VADAnalyzer):
    """Voice Activity Detection analyzer using WebRTC.

    Implements voice activity detection using Daily's native WebRTC VAD.

    Args:
        sample_rate: Audio sample rate in Hz.
        params: VAD configuration parameters (VADParams).
    """

    def __init__(self, *, sample_rate: Optional[int] = None, params: VADParams = VADParams()):
        super().__init__(sample_rate=sample_rate, params=params)

        self._webrtc_vad = Daily.create_native_vad(
            reset_period_ms=VAD_RESET_PERIOD_MS, sample_rate=self.sample_rate, channels=1
        )
        logger.debug("Loaded native WebRTC VAD")

    def num_frames_required(self) -> int:
        return int(self.sample_rate / 100.0)

    def voice_confidence(self, buffer) -> float:
        confidence = 0
        if len(buffer) > 0:
            confidence = self._webrtc_vad.analyze_frames(buffer)
        return confidence


class DailyDialinSettings(BaseModel):
    """Settings for Daily's dial-in functionality.

    Attributes:
        call_id: CallId is represented by UUID and represents the sessionId in the SIP Network.
        call_domain: Call Domain is represented by UUID and represents your Daily Domain on the SIP Network.
    """

    call_id: str = ""
    call_domain: str = ""


class DailyTranscriptionSettings(BaseModel):
    """Configuration settings for Daily's transcription service.

    Attributes:
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

    Args:
        api_url: Daily API base URL
        api_key: Daily API authentication key
        dialin_settings: Optional settings for dial-in functionality
        camera_out_enabled: Whether to enable the main camera output track. If enabled, it still needs `video_out_enabled=True`
        microphone_out_enabled: Whether to enable the main microphone track. If enabled, it still needs `audio_out_enabled=True`
        transcription_enabled: Whether to enable speech transcription
        transcription_settings: Configuration for transcription service
    """

    api_url: str = "https://api.daily.co/v1"
    api_key: str = ""
    dialin_settings: Optional[DailyDialinSettings] = None
    camera_out_enabled: bool = True
    microphone_out_enabled: bool = True
    transcription_enabled: bool = False
    transcription_settings: DailyTranscriptionSettings = DailyTranscriptionSettings()


class DailyCallbacks(BaseModel):
    """Callback handlers for Daily events.

    Attributes:
        on_joined: Called when bot successfully joined a room.
        on_left: Called when bot left a room.
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
        on_recording_started: Called when recording starts.
        on_recording_stopped: Called when recording stops.
        on_recording_error: Called when recording encounters an error.
    """

    on_joined: Callable[[Mapping[str, Any]], Awaitable[None]]
    on_left: Callable[[], Awaitable[None]]
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
    on_recording_started: Callable[[Mapping[str, Any]], Awaitable[None]]
    on_recording_stopped: Callable[[str], Awaitable[None]]
    on_recording_error: Callable[[str, str], Awaitable[None]]


def completion_callback(future):
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


class DailyTransportClient(EventHandler):
    """Core client for interacting with Daily's API.

    Manages the connection to Daily rooms and handles all low-level API interactions.

    Args:
        room_url: URL of the Daily room to connect to.
        token: Optional authentication token for the room.
        bot_name: Display name for the bot in the call.
        params: Configuration parameters (DailyParams).
        callbacks: Event callback handlers (DailyCallbacks).
        transport_name: Name identifier for the transport.
    """

    _daily_initialized: bool = False

    # This is necessary to override EventHandler's __new__ method.
    def __new__(cls, *args, **kwargs):
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

        self._joining = False
        self._joined = False
        self._joined_event = asyncio.Event()
        self._leave_counter = 0

        self._task_manager: Optional[BaseTaskManager] = None

        # We use the executor to cleanup the client. We just do it from one
        # place, so only one thread is really needed.
        self._executor = ThreadPoolExecutor(max_workers=1)

        self._client: CallClient = CallClient(event_handler=self)

        # We use a separate task to execute the callbacks, otherwise if we call
        # a `CallClient` function and wait for its completion this will
        # currently result in a deadlock. This is because `_call_async_callback`
        # can be used inside `CallClient` event handlers which are holding the
        # GIL in `daily-python`. So if the `callback` passed here makes a
        # `CallClient` call and waits for it to finish using completions (and a
        # future) we will deadlock because completions use event handlers (which
        # are holding the GIL).
        self._callback_queue = asyncio.Queue()
        self._callback_task = None

        # Input and ouput sample rates. They will be initialize on setup().
        self._in_sample_rate = 0
        self._out_sample_rate = 0

        self._camera: Optional[VirtualCameraDevice] = None
        self._mic: Optional[VirtualMicrophoneDevice] = None
        self._speaker: Optional[VirtualSpeakerDevice] = None
        self._audio_sources: Dict[str, CustomAudioSource] = {}

    def _camera_name(self):
        return f"camera-{self}"

    def _mic_name(self):
        return f"mic-{self}"

    def _speaker_name(self):
        return f"speaker-{self}"

    @property
    def room_url(self) -> str:
        return self._room_url

    @property
    def participant_id(self) -> str:
        return self._participant_id

    @property
    def in_sample_rate(self) -> int:
        return self._in_sample_rate

    @property
    def out_sample_rate(self) -> int:
        return self._out_sample_rate

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        if not self._joined:
            return

        participant_id = None
        if isinstance(frame, (DailyTransportMessageFrame, DailyTransportMessageUrgentFrame)):
            participant_id = frame.participant_id

        future = self._get_event_loop().create_future()
        self._client.send_app_message(
            frame.message, participant_id, completion=completion_callback(future)
        )
        await future

    async def read_next_audio_frame(self) -> Optional[InputAudioRawFrame]:
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
        self._audio_sources[destination] = await self.add_custom_audio_track(destination)
        self._client.update_publishing({"customAudio": {destination: True}})

    async def write_raw_audio_frames(self, frames: bytes, destination: Optional[str] = None):
        future = self._get_event_loop().create_future()
        if not destination and self._mic:
            self._mic.write_frames(frames, completion=completion_callback(future))
        elif destination and destination in self._audio_sources:
            source = self._audio_sources[destination]
            source.write_frames(frames, completion=completion_callback(future))
        else:
            logger.warning(f"{self} unable to write audio frames to destination [{destination}]")
            future.set_result(None)
        await future

    async def write_raw_video_frame(
        self, frame: OutputImageRawFrame, destination: Optional[str] = None
    ):
        if not destination and self._camera:
            self._camera.write_frame(frame.image)

    async def setup(self, frame: StartFrame):
        self._in_sample_rate = self._params.audio_in_sample_rate or frame.audio_in_sample_rate
        self._out_sample_rate = self._params.audio_out_sample_rate or frame.audio_out_sample_rate

        if self._params.video_out_enabled and not self._camera:
            self._camera = Daily.create_camera_device(
                self._camera_name(),
                width=self._params.video_out_width,
                height=self._params.video_out_height,
                color_format=self._params.video_out_color_format,
            )

        if self._params.audio_out_enabled and not self._mic:
            self._mic = Daily.create_microphone_device(
                self._mic_name(),
                sample_rate=self._out_sample_rate,
                channels=self._params.audio_out_channels,
                non_blocking=True,
            )

        if self._params.audio_in_enabled and not self._speaker:
            self._speaker = Daily.create_speaker_device(
                self._speaker_name(),
                sample_rate=self._in_sample_rate,
                channels=self._params.audio_in_channels,
                non_blocking=True,
            )
            Daily.select_speaker_device(self._speaker_name())

        if not self._task_manager:
            self._task_manager = frame.task_manager
            self._callback_task = self._task_manager.create_task(
                self._callback_task_handler(),
                f"{self}::callback_task",
            )

    async def join(self):
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
                    await self._start_transcription()

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

    async def _start_transcription(self):
        if not self._token:
            logger.warning("Transcription can't be started without a room token")
            return

        logger.info(f"Enabling transcription with settings {self._params.transcription_settings}")

        future = self._get_event_loop().create_future()
        self._client.start_transcription(
            settings=self._params.transcription_settings.model_dump(exclude_none=True),
            completion=completion_callback(future),
        )
        error = await future
        if error:
            logger.error(f"Unable to start transcription: {error}")
            return

    async def _join(self):
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
                            "deviceId": self._mic_name(),
                            "customConstraints": {
                                "autoGainControl": {"exact": False},
                                "echoCancellation": {"exact": False},
                                "noiseSuppression": {"exact": False},
                            },
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
        # Decrement leave counter when leaving.
        self._leave_counter -= 1

        # Transport not joined, ignore.
        if not self._joined or self._leave_counter > 0:
            return

        self._joined = False
        self._joined_event.clear()

        logger.info(f"Leaving {self._room_url}")

        if self._params.transcription_enabled:
            await self._stop_transcription()

        # Remove any custom tracks, if any.
        for track_name, _ in self._audio_sources.items():
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

    async def _stop_transcription(self):
        if not self._token:
            return
        future = self._get_event_loop().create_future()
        self._client.stop_transcription(completion=completion_callback(future))
        error = await future
        if error:
            logger.error(f"Unable to stop transcription: {error}")

    async def _leave(self):
        future = self._get_event_loop().create_future()
        self._client.leave(completion=completion_callback(future))
        return await asyncio.wait_for(future, timeout=10)

    async def cleanup(self):
        if self._callback_task and self._task_manager:
            await self._task_manager.cancel_task(self._callback_task)
            self._callback_task = None
        # Make sure we don't block the event loop in case `client.release()`
        # takes extra time.
        await self._get_event_loop().run_in_executor(self._executor, self._cleanup)

    def _cleanup(self):
        if self._client:
            self._client.release()
            self._client = None

    def participants(self):
        return self._client.participants()

    def participant_counts(self):
        return self._client.participant_counts()

    async def start_dialout(self, settings):
        future = self._get_event_loop().create_future()
        self._client.start_dialout(settings, completion=completion_callback(future))
        await future

    async def stop_dialout(self, participant_id):
        future = self._get_event_loop().create_future()
        self._client.stop_dialout(participant_id, completion=completion_callback(future))
        await future

    async def send_dtmf(self, settings):
        future = self._get_event_loop().create_future()
        self._client.send_dtmf(settings, completion=completion_callback(future))
        await future

    async def sip_call_transfer(self, settings):
        future = self._get_event_loop().create_future()
        self._client.sip_call_transfer(settings, completion=completion_callback(future))
        await future

    async def sip_refer(self, settings):
        future = self._get_event_loop().create_future()
        self._client.sip_refer(settings, completion=completion_callback(future))
        await future

    async def start_recording(self, streaming_settings, stream_id, force_new):
        future = self._get_event_loop().create_future()
        self._client.start_recording(
            streaming_settings, stream_id, force_new, completion=completion_callback(future)
        )
        await future

    async def stop_recording(self, stream_id):
        future = self._get_event_loop().create_future()
        self._client.stop_recording(stream_id, completion=completion_callback(future))
        await future

    async def send_prebuilt_chat_message(self, message: str, user_name: Optional[str] = None):
        if not self._joined:
            return

        future = self._get_event_loop().create_future()
        self._client.send_prebuilt_chat_message(
            message, user_name=user_name, completion=completion_callback(future)
        )
        await future

    async def capture_participant_transcription(self, participant_id: str):
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
    ):
        # Only enable the desired audio source subscription on this participant.
        if audio_source in ("microphone", "screenAudio"):
            media = {"media": {audio_source: "subscribed"}}
        else:
            media = {"media": {"customAudio": {audio_source: "subscribed"}}}

        await self.update_subscriptions(participant_settings={participant_id: media})

        self._audio_renderers[participant_id] = {audio_source: callback}

        self._client.set_audio_renderer(
            participant_id,
            self._audio_data_received,
            audio_source=audio_source,
        )

    async def capture_participant_video(
        self,
        participant_id: str,
        callback: Callable,
        framerate: int = 30,
        video_source: str = "camera",
        color_format: str = "RGB",
    ):
        # Only enable the desired audio source subscription on this participant.
        if video_source in ("camera", "screenVideo"):
            media = {"media": {video_source: "subscribed"}}
        else:
            media = {"media": {"customVideo": {video_source: "subscribed"}}}

        await self.update_subscriptions(participant_settings={participant_id: media})

        self._video_renderers[participant_id] = {video_source: callback}

        self._client.set_video_renderer(
            participant_id,
            self._video_frame_received,
            video_source=video_source,
            color_format=color_format,
        )

    async def add_custom_audio_track(self, track_name: str) -> CustomAudioSource:
        future = self._get_event_loop().create_future()

        audio_source = CustomAudioSource(self._out_sample_rate, 1)
        self._client.add_custom_audio_track(
            track_name=track_name,
            audio_source=audio_source,
            completion=completion_callback(future),
        )

        await future

        return audio_source

    async def remove_custom_audio_track(self, track_name: str):
        future = self._get_event_loop().create_future()
        self._client.remove_custom_audio_track(
            track_name=track_name,
            completion=completion_callback(future),
        )
        await future

    async def update_transcription(self, participants=None, instance_id=None):
        future = self._get_event_loop().create_future()
        self._client.update_transcription(
            participants, instance_id, completion=completion_callback(future)
        )
        await future

    async def update_subscriptions(self, participant_settings=None, profile_settings=None):
        future = self._get_event_loop().create_future()
        self._client.update_subscriptions(
            participant_settings=participant_settings,
            profile_settings=profile_settings,
            completion=completion_callback(future),
        )
        await future

    async def update_publishing(self, publishing_settings: Mapping[str, Any]):
        future = self._get_event_loop().create_future()
        self._client.update_publishing(
            publishing_settings=publishing_settings,
            completion=completion_callback(future),
        )
        await future

    async def update_remote_participants(self, remote_participants: Mapping[str, Any]):
        future = self._get_event_loop().create_future()
        self._client.update_remote_participants(
            remote_participants=remote_participants, completion=completion_callback(future)
        )
        await future

    #
    #
    # Daily (EventHandler)
    #

    def on_app_message(self, message: Any, sender: str):
        self._call_async_callback(self._callbacks.on_app_message, message, sender)

    def on_call_state_updated(self, state: str):
        self._call_async_callback(self._callbacks.on_call_state_updated, state)

    def on_dialin_connected(self, data: Any):
        self._call_async_callback(self._callbacks.on_dialin_connected, data)

    def on_dialin_ready(self, sip_endpoint: str):
        self._call_async_callback(self._callbacks.on_dialin_ready, sip_endpoint)

    def on_dialin_stopped(self, data: Any):
        self._call_async_callback(self._callbacks.on_dialin_stopped, data)

    def on_dialin_error(self, data: Any):
        self._call_async_callback(self._callbacks.on_dialin_error, data)

    def on_dialin_warning(self, data: Any):
        self._call_async_callback(self._callbacks.on_dialin_warning, data)

    def on_dialout_answered(self, data: Any):
        self._call_async_callback(self._callbacks.on_dialout_answered, data)

    def on_dialout_connected(self, data: Any):
        self._call_async_callback(self._callbacks.on_dialout_connected, data)

    def on_dialout_stopped(self, data: Any):
        self._call_async_callback(self._callbacks.on_dialout_stopped, data)

    def on_dialout_error(self, data: Any):
        self._call_async_callback(self._callbacks.on_dialout_error, data)

    def on_dialout_warning(self, data: Any):
        self._call_async_callback(self._callbacks.on_dialout_warning, data)

    def on_participant_joined(self, participant):
        self._call_async_callback(self._callbacks.on_participant_joined, participant)

    def on_participant_left(self, participant, reason):
        self._call_async_callback(self._callbacks.on_participant_left, participant, reason)

    def on_participant_updated(self, participant):
        self._call_async_callback(self._callbacks.on_participant_updated, participant)

    def on_transcription_started(self, status):
        logger.debug(f"Transcription started: {status}")
        self._transcription_status = status
        self._call_async_callback(self.update_transcription, self._transcription_ids)

    def on_transcription_stopped(self, stopped_by, stopped_by_error):
        logger.debug("Transcription stopped")

    def on_transcription_error(self, message):
        logger.error(f"Transcription error: {message}")

    def on_transcription_message(self, message):
        self._call_async_callback(self._callbacks.on_transcription_message, message)

    def on_recording_started(self, status):
        logger.debug(f"Recording started: {status}")
        self._call_async_callback(self._callbacks.on_recording_started, status)

    def on_recording_stopped(self, stream_id):
        logger.debug(f"Recording stopped: {stream_id}")
        self._call_async_callback(self._callbacks.on_recording_stopped, stream_id)

    def on_recording_error(self, stream_id, message):
        logger.error(f"Recording error for {stream_id}: {message}")
        self._call_async_callback(self._callbacks.on_recording_error, stream_id, message)

    #
    # Daily (CallClient callbacks)
    #

    def _audio_data_received(self, participant_id: str, audio_data: AudioData, audio_source: str):
        callback = self._audio_renderers[participant_id][audio_source]
        self._call_async_callback(callback, participant_id, audio_data, audio_source)

    def _video_frame_received(
        self, participant_id: str, video_frame: VideoFrame, video_source: str
    ):
        callback = self._video_renderers[participant_id][video_source]
        self._call_async_callback(callback, participant_id, video_frame, video_source)

    def _call_async_callback(self, callback, *args):
        future = asyncio.run_coroutine_threadsafe(
            self._callback_queue.put((callback, *args)), self._get_event_loop()
        )
        future.result()

    async def _callback_task_handler(self):
        while True:
            # Wait to process any callback until we are joined.
            await self._joined_event.wait()
            (callback, *args) = await self._callback_queue.get()
            await callback(*args)

    def _get_event_loop(self) -> asyncio.AbstractEventLoop:
        if not self._task_manager:
            raise Exception(f"{self}: missing task manager (pipeline not started?)")
        return self._task_manager.get_event_loop()

    def __str__(self):
        return f"{self._transport_name}::DailyTransportClient"


class DailyInputTransport(BaseInputTransport):
    """Handles incoming media streams and events from Daily calls.

    Processes incoming audio, video, transcriptions and other events from Daily.

    Args:
        client: DailyTransportClient instance.
        params: Configuration parameters.
    """

    def __init__(
        self,
        transport: BaseTransport,
        client: DailyTransportClient,
        params: DailyParams,
        **kwargs,
    ):
        super().__init__(params, **kwargs)

        self._transport = transport
        self._client = client
        self._params = params

        self._video_renderers = {}

        # Whether we have seen a StartFrame already.
        self._initialized = False

        # Task that gets audio data from a device or the network and queues it
        # internally to be processed.
        self._audio_in_task = None

        self._resampler = create_default_resampler()

        self._vad_analyzer: Optional[VADAnalyzer] = params.vad_analyzer

    @property
    def vad_analyzer(self) -> Optional[VADAnalyzer]:
        return self._vad_analyzer

    def start_audio_in_streaming(self):
        # Create audio task. It reads audio frames from Daily and push them
        # internally for VAD processing.
        if not self._audio_in_task and self._params.audio_in_enabled:
            logger.debug(f"Start receiving audio")
            self._audio_in_task = self.create_task(self._audio_in_task_handler())

    async def start(self, frame: StartFrame):
        if self._initialized:
            return

        self._initialized = True

        # Parent start.
        await super().start(frame)

        # Setup client.
        await self._client.setup(frame)

        # Join the room.
        await self._client.join()

        # Indicate the transport that we are connected.
        await self.set_transport_ready(frame)

        if self._params.audio_in_stream_on_start:
            self.start_audio_in_streaming()

    async def stop(self, frame: EndFrame):
        # Parent stop.
        await super().stop(frame)
        # Leave the room.
        await self._client.leave()
        # Stop audio thread.
        if self._audio_in_task and self._params.audio_in_enabled:
            await self.cancel_task(self._audio_in_task)
            self._audio_in_task = None

    async def cancel(self, frame: CancelFrame):
        # Parent stop.
        await super().cancel(frame)
        # Leave the room.
        await self._client.leave()
        # Stop audio thread.
        if self._audio_in_task and self._params.audio_in_enabled:
            await self.cancel_task(self._audio_in_task)
            self._audio_in_task = None

    async def cleanup(self):
        await super().cleanup()
        await self._client.cleanup()
        await self._transport.cleanup()

    #
    # FrameProcessor
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserImageRequestFrame):
            await self.request_participant_image(frame)

    #
    # Frames
    #

    async def push_transcription_frame(self, frame: TranscriptionFrame | InterimTranscriptionFrame):
        await self.push_frame(frame)

    async def push_app_message(self, message: Any, sender: str):
        frame = DailyTransportMessageUrgentFrame(message=message, participant_id=sender)
        await self.push_frame(frame)

    #
    # Audio in
    #

    async def capture_participant_audio(
        self,
        participant_id: str,
        audio_source: str = "camera",
    ):
        await self._client.capture_participant_audio(
            participant_id, self._on_participant_audio_data, audio_source
        )

    async def _on_participant_audio_data(
        self, participant_id: str, audio: AudioData, audio_source: str
    ):
        resampled = await self._resampler.resample(
            audio.audio_frames, audio.sample_rate, self._client.out_sample_rate
        )

        frame = UserAudioRawFrame(
            user_id=participant_id,
            audio=resampled,
            sample_rate=self._client.out_sample_rate,
            num_channels=audio.num_channels,
        )
        frame.transport_source = audio_source
        await self.push_frame(frame)

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
        self._video_renderers[participant_id] = {
            video_source: {
                "framerate": framerate,
                "timestamp": 0,
                "render_next_frame": [],
            }
        }

        await self._client.capture_participant_video(
            participant_id, self._on_participant_video_frame, framerate, video_source, color_format
        )

    async def request_participant_image(self, frame: UserImageRequestFrame):
        if frame.user_id in self._video_renderers:
            self._video_renderers[frame.user_id]["render_next_frame"].append(frame)

    async def _on_participant_video_frame(
        self, participant_id: str, video_frame: VideoFrame, video_source: str
    ):
        render_frame = False

        curr_time = time.time()
        prev_time = self._video_renderers[participant_id][video_source]["timestamp"]
        framerate = self._video_renderers[participant_id][video_source]["framerate"]

        # Some times we render frames because of a request.
        request_frame = None

        if framerate > 0:
            next_time = prev_time + 1 / framerate
            render_frame = (next_time - curr_time) < 0.1

        elif self._video_renderers[participant_id][video_source]["render_next_frame"]:
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
            await self.push_frame(frame)
            self._video_renderers[participant_id][video_source]["timestamp"] = curr_time


class DailyOutputTransport(BaseOutputTransport):
    """Handles outgoing media streams and events to Daily calls.

    Manages sending audio, video and other data to Daily calls.

    Args:
        client: DailyTransportClient instance.
        params: Configuration parameters.
    """

    def __init__(
        self, transport: BaseTransport, client: DailyTransportClient, params: DailyParams, **kwargs
    ):
        super().__init__(params, **kwargs)

        self._transport = transport
        self._client = client

        # Whether we have seen a StartFrame already.
        self._initialized = False

    async def start(self, frame: StartFrame):
        if self._initialized:
            return

        self._initialized = True

        # Parent start.
        await super().start(frame)

        # Setup client.
        await self._client.setup(frame)

        # Join the room.
        await self._client.join()

        # Indicate the transport that we are connected.
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        # Parent stop.
        await super().stop(frame)
        # Leave the room.
        await self._client.leave()

    async def cancel(self, frame: CancelFrame):
        # Parent stop.
        await super().cancel(frame)
        # Leave the room.
        await self._client.leave()

    async def cleanup(self):
        await super().cleanup()
        await self._client.cleanup()
        await self._transport.cleanup()

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        await self._client.send_message(frame)

    async def register_video_destination(self, destination: str):
        logger.warning(f"{self} registering video destinations is not supported yet")

    async def register_audio_destination(self, destination: str):
        await self._client.register_audio_destination(destination)

    async def write_raw_audio_frames(self, frames: bytes, destination: Optional[str] = None):
        await self._client.write_raw_audio_frames(frames, destination)

    async def write_raw_video_frame(
        self, frame: OutputImageRawFrame, destination: Optional[str] = None
    ):
        await self._client.write_raw_video_frame(frame, destination)


class DailyTransport(BaseTransport):
    """Transport implementation for Daily audio and video calls.

    Handles audio/video streaming, transcription, recordings, dial-in,
        dial-out, and call management through Daily's API.

    Args:
        room_url: URL of the Daily room to connect to.
        token: Optional authentication token for the room.
        bot_name: Display name for the bot in the call.
        params: Configuration parameters (DailyParams) for the transport.
        input_name: Optional name for the input transport.
        output_name: Optional name for the output transport.
    """

    def __init__(
        self,
        room_url: str,
        token: Optional[str],
        bot_name: str,
        params: DailyParams = DailyParams(),
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name)

        callbacks = DailyCallbacks(
            on_joined=self._on_joined,
            on_left=self._on_left,
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
            on_recording_started=self._on_recording_started,
            on_recording_stopped=self._on_recording_stopped,
            on_recording_error=self._on_recording_error,
        )
        self._params = params

        self._client = DailyTransportClient(room_url, token, bot_name, params, callbacks, self.name)
        self._input: Optional[DailyInputTransport] = None
        self._output: Optional[DailyOutputTransport] = None

        self._other_participant_has_joined = False

        # Register supported handlers. The user will only be able to register
        # these handlers.
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

    #
    # BaseTransport
    #

    def input(self) -> DailyInputTransport:
        if not self._input:
            self._input = DailyInputTransport(
                self, self._client, self._params, name=self._input_name
            )
        return self._input

    def output(self) -> DailyOutputTransport:
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
        return self._client.room_url

    @property
    def participant_id(self) -> str:
        return self._client.participant_id

    async def send_image(self, frame: OutputImageRawFrame | SpriteFrame):
        if self._output:
            await self._output.queue_frame(frame, FrameDirection.DOWNSTREAM)

    async def send_audio(self, frame: OutputAudioRawFrame):
        if self._output:
            await self._output.queue_frame(frame, FrameDirection.DOWNSTREAM)

    def participants(self):
        return self._client.participants()

    def participant_counts(self):
        return self._client.participant_counts()

    async def start_dialout(self, settings=None):
        await self._client.start_dialout(settings)

    async def stop_dialout(self, participant_id):
        await self._client.stop_dialout(participant_id)

    async def send_dtmf(self, settings):
        await self._client.send_dtmf(settings)

    async def sip_call_transfer(self, settings):
        await self._client.sip_call_transfer(settings)

    async def sip_refer(self, settings):
        await self._client.sip_refer(settings)

    async def start_recording(self, streaming_settings=None, stream_id=None, force_new=None):
        await self._client.start_recording(streaming_settings, stream_id, force_new)

    async def stop_recording(self, stream_id=None):
        await self._client.stop_recording(stream_id)

    async def send_prebuilt_chat_message(self, message: str, user_name: Optional[str] = None):
        """Sends a chat message to Daily's Prebuilt main room.

        Args:
        message: The chat message to send
        user_name: Optional user name that will appear as sender of the message
        """
        await self._client.send_prebuilt_chat_message(message, user_name)

    async def capture_participant_transcription(self, participant_id: str):
        await self._client.capture_participant_transcription(participant_id)

    async def capture_participant_audio(
        self,
        participant_id: str,
        audio_source: str = "microphone",
    ):
        if self._input:
            await self._input.capture_participant_audio(participant_id, audio_source)

    async def capture_participant_video(
        self,
        participant_id: str,
        framerate: int = 30,
        video_source: str = "camera",
        color_format: str = "RGB",
    ):
        if self._input:
            await self._input.capture_participant_video(
                participant_id, framerate, video_source, color_format
            )

    async def update_publishing(self, publishing_settings: Mapping[str, Any]):
        await self._client.update_publishing(publishing_settings=publishing_settings)

    async def update_subscriptions(self, participant_settings=None, profile_settings=None):
        await self._client.update_subscriptions(
            participant_settings=participant_settings, profile_settings=profile_settings
        )

    async def update_remote_participants(self, remote_participants: Mapping[str, Any]):
        await self._client.update_remote_participants(remote_participants=remote_participants)

    async def _on_joined(self, data):
        await self._call_event_handler("on_joined", data)

    async def _on_left(self):
        await self._call_event_handler("on_left")

    async def _on_error(self, error):
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
        if self._input:
            await self._input.push_app_message(message, sender)
        await self._call_event_handler("on_app_message", message, sender)

    async def _on_call_state_updated(self, state: str):
        await self._call_event_handler("on_call_state_updated", state)

    async def _on_client_connected(self, participant: Any):
        await self._call_event_handler("on_client_connected", participant)

    async def _on_client_disconnected(self, participant: Any):
        await self._call_event_handler("on_client_disconnected", participant)

    async def _handle_dialin_ready(self, sip_endpoint: str):
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
        await self._call_event_handler("on_dialin_connected", data)

    async def _on_dialin_ready(self, sip_endpoint):
        if self._params.dialin_settings:
            await self._handle_dialin_ready(sip_endpoint)
        await self._call_event_handler("on_dialin_ready", sip_endpoint)

    async def _on_dialin_stopped(self, data):
        await self._call_event_handler("on_dialin_stopped", data)

    async def _on_dialin_error(self, data):
        await self._call_event_handler("on_dialin_error", data)

    async def _on_dialin_warning(self, data):
        await self._call_event_handler("on_dialin_warning", data)

    async def _on_dialout_answered(self, data):
        await self._call_event_handler("on_dialout_answered", data)

    async def _on_dialout_connected(self, data):
        await self._call_event_handler("on_dialout_connected", data)

    async def _on_dialout_stopped(self, data):
        await self._call_event_handler("on_dialout_stopped", data)

    async def _on_dialout_error(self, data):
        await self._call_event_handler("on_dialout_error", data)

    async def _on_dialout_warning(self, data):
        await self._call_event_handler("on_dialout_warning", data)

    async def _on_participant_joined(self, participant):
        id = participant["id"]
        logger.info(f"Participant joined {id}")

        if not self._other_participant_has_joined:
            self._other_participant_has_joined = True
            await self._call_event_handler("on_first_participant_joined", participant)

        await self._call_event_handler("on_participant_joined", participant)
        # Also call on_client_connected for compatibility with other transports
        await self._call_event_handler("on_client_connected", participant)

    async def _on_participant_left(self, participant, reason):
        id = participant["id"]
        logger.info(f"Participant left {id}")
        await self._call_event_handler("on_participant_left", participant, reason)
        # Also call on_client_disconnected for compatibility with other transports
        await self._call_event_handler("on_client_disconnected", participant)

    async def _on_participant_updated(self, participant):
        await self._call_event_handler("on_participant_updated", participant)

    async def _on_transcription_message(self, message):
        await self._call_event_handler("on_transcription_message", message)

        participant_id = ""
        if "participantId" in message:
            participant_id = message["participantId"]
        if not participant_id:
            return

        text = message["text"]
        timestamp = message["timestamp"]
        is_final = message["rawResponse"]["is_final"]
        try:
            language = message["rawResponse"]["channel"]["alternatives"][0]["languages"][0]
            language = Language(language)
        except KeyError:
            language = None
        if is_final:
            frame = TranscriptionFrame(text, participant_id, timestamp, language)
            logger.debug(f"Transcription (from: {participant_id}): [{text}]")
        else:
            frame = InterimTranscriptionFrame(text, participant_id, timestamp, language)

        if self._input:
            await self._input.push_transcription_frame(frame)

    async def _on_recording_started(self, status):
        await self._call_event_handler("on_recording_started", status)

    async def _on_recording_stopped(self, stream_id):
        await self._call_event_handler("on_recording_stopped", stream_id)

    async def _on_recording_error(self, stream_id, message):
        await self._call_event_handler("on_recording_error", stream_id, message)
