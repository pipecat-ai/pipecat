#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Optional

import aiohttp
from daily import (
    CallClient,
    Daily,
    EventHandler,
    VirtualCameraDevice,
    VirtualMicrophoneDevice,
    VirtualSpeakerDevice,
)
from loguru import logger
from pydantic import BaseModel, model_validator

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
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
    UserImageRawFrame,
    UserImageRequestFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transcriptions.language import Language
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

try:
    from daily import CallClient, Daily, EventHandler
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use the Daily transport, you need to `pip install pipecat-ai[daily]`."
    )
    raise Exception(f"Missing module: {e}")

VAD_RESET_PERIOD_MS = 2000


@dataclass
class DailyTransportMessageFrame(TransportMessageFrame):
    participant_id: str | None = None


@dataclass
class DailyTransportMessageUrgentFrame(TransportMessageUrgentFrame):
    participant_id: str | None = None


class WebRTCVADAnalyzer(VADAnalyzer):
    def __init__(self, *, sample_rate=16000, num_channels=1, params: VADParams = VADParams()):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, params=params)

        self._webrtc_vad = Daily.create_native_vad(
            reset_period_ms=VAD_RESET_PERIOD_MS, sample_rate=sample_rate, channels=num_channels
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
    call_id: str = ""
    call_domain: str = ""


class DailyTranscriptionSettings(BaseModel):
    language: str = "en"
    tier: Optional[str] = None
    model: str = "nova-2-general"
    profanity_filter: bool = True
    redact: bool = False
    endpointing: bool = True
    punctuate: bool = True
    includeRawResponse: bool = True
    extra: Mapping[str, Any] = {"interim_results": True}

    @model_validator(mode="before")
    def check_deprecated_fields(cls, values):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            if "tier" in values:
                warnings.warn(
                    "Field 'tier' is deprecated, use 'model' instead.", DeprecationWarning
                )
        return values


class DailyParams(TransportParams):
    api_url: str = "https://api.daily.co/v1"
    api_key: str = ""
    dialin_settings: Optional[DailyDialinSettings] = None
    transcription_enabled: bool = False
    transcription_settings: DailyTranscriptionSettings = DailyTranscriptionSettings()


class DailyCallbacks(BaseModel):
    on_joined: Callable[[Mapping[str, Any]], Awaitable[None]]
    on_left: Callable[[], Awaitable[None]]
    on_error: Callable[[str], Awaitable[None]]
    on_app_message: Callable[[Any, str], Awaitable[None]]
    on_call_state_updated: Callable[[str], Awaitable[None]]
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
    on_first_participant_joined: Callable[[Mapping[str, Any]], Awaitable[None]]
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
    _daily_initialized: bool = False

    # This is necessary to override EventHandler's __new__ method.
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(
        self,
        room_url: str,
        token: str | None,
        bot_name: str,
        params: DailyParams,
        callbacks: DailyCallbacks,
        loop: asyncio.AbstractEventLoop,
    ):
        super().__init__()

        if not DailyTransportClient._daily_initialized:
            DailyTransportClient._daily_initialized = True
            Daily.init()

        self._room_url: str = room_url
        self._token: str | None = token
        self._bot_name: str = bot_name
        self._params: DailyParams = params
        self._callbacks = callbacks
        self._loop = loop

        self._participant_id: str = ""
        self._video_renderers = {}
        self._transcription_ids = []
        self._transcription_status = None
        self._other_participant_has_joined = False

        self._joined = False
        self._leave_counter = 0

        self._executor = ThreadPoolExecutor(max_workers=5)

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
        self._callback_task = self._loop.create_task(self._callback_task_handler())

        self._camera: VirtualCameraDevice | None = None
        if self._params.camera_out_enabled:
            self._camera = Daily.create_camera_device(
                self._camera_name(),
                width=self._params.camera_out_width,
                height=self._params.camera_out_height,
                color_format=self._params.camera_out_color_format,
            )

        self._mic: VirtualMicrophoneDevice | None = None
        if self._params.audio_out_enabled:
            self._mic = Daily.create_microphone_device(
                self._mic_name(),
                sample_rate=self._params.audio_out_sample_rate,
                channels=self._params.audio_out_channels,
                non_blocking=True,
            )

        self._speaker: VirtualSpeakerDevice | None = None
        if self._params.audio_in_enabled or self._params.vad_enabled:
            self._speaker = Daily.create_speaker_device(
                self._speaker_name(),
                sample_rate=self._params.audio_in_sample_rate,
                channels=self._params.audio_in_channels,
                non_blocking=True,
            )
            Daily.select_speaker_device(self._speaker_name())

    def _camera_name(self):
        return f"camera-{self}"

    def _mic_name(self):
        return f"mic-{self}"

    def _speaker_name(self):
        return f"speaker-{self}"

    @property
    def participant_id(self) -> str:
        return self._participant_id

    def set_callbacks(self, callbacks: DailyCallbacks):
        self._callbacks = callbacks

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        if not self._joined:
            return

        participant_id = None
        if isinstance(frame, (DailyTransportMessageFrame, DailyTransportMessageUrgentFrame)):
            participant_id = frame.participant_id

        future = self._loop.create_future()
        self._client.send_app_message(
            frame.message, participant_id, completion=completion_callback(future)
        )
        await future

    async def read_next_audio_frame(self) -> InputAudioRawFrame | None:
        if not self._speaker:
            return None

        sample_rate = self._params.audio_in_sample_rate
        num_channels = self._params.audio_in_channels
        num_frames = int(sample_rate / 100) * 2  # 20ms of audio

        future = self._loop.create_future()
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

    async def write_raw_audio_frames(self, frames: bytes):
        if not self._mic:
            return None

        future = self._loop.create_future()
        self._mic.write_frames(frames, completion=completion_callback(future))
        await future

    async def write_frame_to_camera(self, frame: OutputImageRawFrame):
        if not self._camera:
            return None

        self._camera.write_frame(frame.image)

    async def join(self):
        # Transport already joined, ignore.
        if self._joined:
            # Increment leave counter if we already joined.
            self._leave_counter += 1
            return

        logger.info(f"Joining {self._room_url}")

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
                # Increment leave counter if we successfully joined.
                self._leave_counter += 1

                logger.info(f"Joined {self._room_url}")

                if self._params.transcription_enabled:
                    await self._start_transcription()

                await self._callbacks.on_joined(data)
            else:
                error_msg = f"Error joining {self._room_url}: {error}"
                logger.error(error_msg)
                await self._callbacks.on_error(error_msg)
        except asyncio.TimeoutError:
            error_msg = f"Time out joining {self._room_url}"
            logger.error(error_msg)
            await self._callbacks.on_error(error_msg)

    async def _start_transcription(self):
        if not self._token:
            logger.warning("Transcription can't be started without a room token")
            return

        logger.info(f"Enabling transcription with settings {self._params.transcription_settings}")

        future = self._loop.create_future()
        self._client.start_transcription(
            settings=self._params.transcription_settings.model_dump(exclude_none=True),
            completion=completion_callback(future),
        )
        error = await future
        if error:
            logger.error(f"Unable to start transcription: {error}")
            return

    async def _join(self):
        future = self._loop.create_future()

        self._client.join(
            self._room_url,
            self._token,
            completion=completion_callback(future),
            client_settings={
                "inputs": {
                    "camera": {
                        "isEnabled": self._params.camera_out_enabled,
                        "settings": {
                            "deviceId": self._camera_name(),
                        },
                    },
                    "microphone": {
                        "isEnabled": self._params.audio_out_enabled,
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
                                    "maxBitrate": self._params.camera_out_bitrate,
                                    "maxFramerate": self._params.camera_out_framerate,
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

        logger.info(f"Leaving {self._room_url}")

        if self._params.transcription_enabled:
            await self._stop_transcription()

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
        future = self._loop.create_future()
        self._client.stop_transcription(completion=completion_callback(future))
        error = await future
        if error:
            logger.error(f"Unable to stop transcription: {error}")

    async def _leave(self):
        future = self._loop.create_future()
        self._client.leave(completion=completion_callback(future))
        return await asyncio.wait_for(future, timeout=10)

    async def cleanup(self):
        await self._loop.run_in_executor(self._executor, self._cleanup)
        self._callback_task.cancel()
        await self._callback_task

    def _cleanup(self):
        if self._client:
            self._client.release()
            self._client = None

    def participants(self):
        return self._client.participants()

    def participant_counts(self):
        return self._client.participant_counts()

    async def start_dialout(self, settings):
        future = self._loop.create_future()
        self._client.start_dialout(settings, completion=completion_callback(future))
        await future

    async def stop_dialout(self, participant_id):
        future = self._loop.create_future()
        self._client.stop_dialout(participant_id, completion=completion_callback(future))
        await future

    async def start_recording(self, streaming_settings, stream_id, force_new):
        future = self._loop.create_future()
        self._client.start_recording(
            streaming_settings, stream_id, force_new, completion=completion_callback(future)
        )
        await future

    async def stop_recording(self, stream_id):
        future = self._loop.create_future()
        self._client.stop_recording(stream_id, completion=completion_callback(future))
        await future

    async def capture_participant_transcription(self, participant_id: str):
        if not self._params.transcription_enabled:
            return

        self._transcription_ids.append(participant_id)
        if self._joined and self._transcription_status:
            await self.update_transcription(self._transcription_ids)

    async def capture_participant_video(
        self,
        participant_id: str,
        callback: Callable,
        framerate: int = 30,
        video_source: str = "camera",
        color_format: str = "RGB",
    ):
        # Only enable the desired video source subscription on this participant.
        await self.update_subscriptions(
            participant_settings={participant_id: {"media": {video_source: "subscribed"}}}
        )

        self._video_renderers[participant_id] = callback

        self._client.set_video_renderer(
            participant_id,
            self._video_frame_received,
            video_source=video_source,
            color_format=color_format,
        )

    async def update_transcription(self, participants=None, instance_id=None):
        future = self._loop.create_future()
        self._client.update_transcription(
            participants, instance_id, completion=completion_callback(future)
        )
        await future

    async def update_subscriptions(self, participant_settings=None, profile_settings=None):
        future = self._loop.create_future()
        self._client.update_subscriptions(
            participant_settings=participant_settings,
            profile_settings=profile_settings,
            completion=completion_callback(future),
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
        id = participant["id"]
        logger.info(f"Participant joined {id}")

        if not self._other_participant_has_joined:
            self._other_participant_has_joined = True
            self._call_async_callback(self._callbacks.on_first_participant_joined, participant)

        self._call_async_callback(self._callbacks.on_participant_joined, participant)

    def on_participant_left(self, participant, reason):
        id = participant["id"]
        logger.info(f"Participant left {id}")

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

    def _video_frame_received(self, participant_id, video_frame):
        callback = self._video_renderers[participant_id]
        self._call_async_callback(
            callback,
            participant_id,
            video_frame.buffer,
            (video_frame.width, video_frame.height),
            video_frame.color_format,
        )

    def _call_async_callback(self, callback, *args):
        future = asyncio.run_coroutine_threadsafe(
            self._callback_queue.put((callback, *args)), self._loop
        )
        future.result()

    async def _callback_task_handler(self):
        while True:
            try:
                (callback, *args) = await self._callback_queue.get()
                await callback(*args)
            except asyncio.CancelledError:
                break


class DailyInputTransport(BaseInputTransport):
    def __init__(self, client: DailyTransportClient, params: DailyParams, **kwargs):
        super().__init__(params, **kwargs)

        self._client = client

        self._video_renderers = {}

        # Task that gets audio data from a device or the network and queues it
        # internally to be processed.
        self._audio_in_task = None

        self._vad_analyzer: VADAnalyzer | None = params.vad_analyzer
        if params.vad_enabled and not params.vad_analyzer:
            self._vad_analyzer = WebRTCVADAnalyzer(
                sample_rate=self._params.audio_in_sample_rate,
                num_channels=self._params.audio_in_channels,
            )

    async def start(self, frame: StartFrame):
        # Parent start.
        await super().start(frame)
        # Join the room.
        await self._client.join()
        # Create audio task. It reads audio frames from Daily and push them
        # internally for VAD processing.
        if self._params.audio_in_enabled or self._params.vad_enabled:
            self._audio_in_task = self.get_event_loop().create_task(self._audio_in_task_handler())

    async def stop(self, frame: EndFrame):
        # Parent stop.
        await super().stop(frame)
        # Leave the room.
        await self._client.leave()
        # Stop audio thread.
        if self._audio_in_task and (self._params.audio_in_enabled or self._params.vad_enabled):
            self._audio_in_task.cancel()
            await self._audio_in_task
            self._audio_in_task = None

    async def cancel(self, frame: CancelFrame):
        # Parent stop.
        await super().cancel(frame)
        # Leave the room.
        await self._client.leave()
        # Stop audio thread.
        if self._audio_in_task and (self._params.audio_in_enabled or self._params.vad_enabled):
            self._audio_in_task.cancel()
            await self._audio_in_task
            self._audio_in_task = None

    async def cleanup(self):
        await super().cleanup()
        await self._client.cleanup()

    def vad_analyzer(self) -> VADAnalyzer | None:
        return self._vad_analyzer

    #
    # FrameProcessor
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserImageRequestFrame):
            await self.request_participant_image(frame.user_id)

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

    async def _audio_in_task_handler(self):
        while True:
            try:
                frame = await self._client.read_next_audio_frame()
                if frame:
                    await self.push_audio_frame(frame)
            except asyncio.CancelledError:
                break

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
            "framerate": framerate,
            "timestamp": 0,
            "render_next_frame": False,
        }

        await self._client.capture_participant_video(
            participant_id, self._on_participant_video_frame, framerate, video_source, color_format
        )

    async def request_participant_image(self, participant_id: str):
        if participant_id in self._video_renderers:
            self._video_renderers[participant_id]["render_next_frame"] = True

    async def _on_participant_video_frame(self, participant_id: str, buffer, size, format):
        render_frame = False

        curr_time = time.time()
        prev_time = self._video_renderers[participant_id]["timestamp"]
        framerate = self._video_renderers[participant_id]["framerate"]

        if framerate > 0:
            next_time = prev_time + 1 / framerate
            render_frame = (next_time - curr_time) < 0.1

        elif self._video_renderers[participant_id]["render_next_frame"]:
            self._video_renderers[participant_id]["render_next_frame"] = False
            render_frame = True

        if render_frame:
            frame = UserImageRawFrame(
                user_id=participant_id, image=buffer, size=size, format=format
            )
            await self.push_frame(frame)
            self._video_renderers[participant_id]["timestamp"] = curr_time


class DailyOutputTransport(BaseOutputTransport):
    def __init__(self, client: DailyTransportClient, params: DailyParams, **kwargs):
        super().__init__(params, **kwargs)

        self._client = client

    async def start(self, frame: StartFrame):
        # Parent start.
        await super().start(frame)
        # Join the room.
        await self._client.join()

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

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        await self._client.send_message(frame)

    async def write_raw_audio_frames(self, frames: bytes):
        await self._client.write_raw_audio_frames(frames)

    async def write_frame_to_camera(self, frame: OutputImageRawFrame):
        await self._client.write_frame_to_camera(frame)


class DailyTransport(BaseTransport):
    def __init__(
        self,
        room_url: str,
        token: str | None,
        bot_name: str,
        params: DailyParams = DailyParams(),
        input_name: str | None = None,
        output_name: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name, loop=loop)

        callbacks = DailyCallbacks(
            on_joined=self._on_joined,
            on_left=self._on_left,
            on_error=self._on_error,
            on_app_message=self._on_app_message,
            on_call_state_updated=self._on_call_state_updated,
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
            on_first_participant_joined=self._on_first_participant_joined,
            on_participant_joined=self._on_participant_joined,
            on_participant_left=self._on_participant_left,
            on_participant_updated=self._on_participant_updated,
            on_transcription_message=self._on_transcription_message,
            on_recording_started=self._on_recording_started,
            on_recording_stopped=self._on_recording_stopped,
            on_recording_error=self._on_recording_error,
        )
        self._params = params

        self._client = DailyTransportClient(
            room_url, token, bot_name, params, callbacks, self._loop
        )
        self._input: DailyInputTransport | None = None
        self._output: DailyOutputTransport | None = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_joined")
        self._register_event_handler("on_left")
        self._register_event_handler("on_app_message")
        self._register_event_handler("on_call_state_updated")
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
            self._input = DailyInputTransport(self._client, self._params, name=self._input_name)
        return self._input

    def output(self) -> DailyOutputTransport:
        if not self._output:
            self._output = DailyOutputTransport(self._client, self._params, name=self._output_name)
        return self._output

    #
    # DailyTransport
    #

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

    async def start_recording(self, streaming_settings=None, stream_id=None, force_new=None):
        await self._client.start_recording(streaming_settings, stream_id, force_new)

    async def stop_recording(self, stream_id=None):
        await self._client.stop_recording(stream_id)

    async def capture_participant_transcription(self, participant_id: str):
        await self._client.capture_participant_transcription(participant_id)

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

    async def update_subscriptions(self, participant_settings=None, profile_settings=None):
        await self._client.update_subscriptions(
            participant_settings=participant_settings, profile_settings=profile_settings
        )

    async def _on_joined(self, data):
        await self._call_event_handler("on_joined", data)

    async def _on_left(self):
        await self._call_event_handler("on_left")

    async def _on_error(self, error):
        # TODO(aleix): Report error to input/output transports. The one managing
        # the client should report the error.
        pass

    async def _on_app_message(self, message: Any, sender: str):
        if self._input:
            await self._input.push_app_message(message, sender)
        await self._call_event_handler("on_app_message", message, sender)

    async def _on_call_state_updated(self, state: str):
        await self._call_event_handler("on_call_state_updated", state)

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
        await self._call_event_handler("on_participant_joined", participant)

    async def _on_participant_left(self, participant, reason):
        await self._call_event_handler("on_participant_left", participant, reason)

    async def _on_participant_updated(self, participant):
        await self._call_event_handler("on_participant_updated", participant)

    async def _on_first_participant_joined(self, participant):
        await self._call_event_handler("on_first_participant_joined", participant)

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
