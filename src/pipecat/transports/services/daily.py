#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp
import asyncio
import queue
import time

from dataclasses import dataclass
from typing import Any, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor

from daily import (
    CallClient,
    Daily,
    EventHandler,
    VirtualCameraDevice,
    VirtualMicrophoneDevice,
    VirtualSpeakerDevice)
from pydantic.main import BaseModel

from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    ImageRawFrame,
    InterimTranscriptionFrame,
    SpriteFrame,
    StartFrame,
    TranscriptionFrame,
    TransportMessageFrame,
    UserImageRawFrame,
    UserImageRequestFrame)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.vad.vad_analyzer import VADAnalyzer, VADParams

from loguru import logger

try:
    from daily import (EventHandler, CallClient, Daily)
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use the Daily transport, you need to `pip install pipecat-ai[daily]`.")
    raise Exception(f"Missing module: {e}")

VAD_RESET_PERIOD_MS = 2000


@dataclass
class DailyTransportMessageFrame(TransportMessageFrame):
    participant_id: str | None = None


class WebRTCVADAnalyzer(VADAnalyzer):

    def __init__(self, sample_rate=16000, num_channels=1, params: VADParams = VADParams()):
        super().__init__(sample_rate, num_channels, params)

        self._webrtc_vad = Daily.create_native_vad(
            reset_period_ms=VAD_RESET_PERIOD_MS,
            sample_rate=sample_rate,
            channels=num_channels
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
    tier: str = "nova"
    model: str = "2-conversationalai"
    profanity_filter: bool = True
    redact: bool = False
    endpointing: bool = True
    punctuate: bool = True
    includeRawResponse: bool = True
    extra: Mapping[str, Any] = {
        "interim_results": True
    }


class DailyParams(TransportParams):
    api_url: str = "https://api.daily.co/v1"
    api_key: str = ""
    dialin_settings: DailyDialinSettings | None = None
    transcription_enabled: bool = False
    transcription_settings: DailyTranscriptionSettings = DailyTranscriptionSettings()


class DailyCallbacks(BaseModel):
    on_joined: Callable[[Mapping[str, Any]], None]
    on_left: Callable[[], None]
    on_error: Callable[[str], None]
    on_app_message: Callable[[Any, str], None]
    on_call_state_updated: Callable[[str], None]
    on_dialin_ready: Callable[[str], None]
    on_dialout_connected: Callable[[Any], None]
    on_dialout_stopped: Callable[[Any], None]
    on_dialout_error: Callable[[Any], None]
    on_dialout_warning: Callable[[Any], None]
    on_first_participant_joined: Callable[[Mapping[str, Any]], None]
    on_participant_joined: Callable[[Mapping[str, Any]], None]
    on_participant_left: Callable[[Mapping[str, Any], str], None]


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
            loop: asyncio.AbstractEventLoop):
        super().__init__()

        if not self._daily_initialized:
            self._daily_initialized = True
            Daily.init()

        self._room_url: str = room_url
        self._token: str | None = token
        self._bot_name: str = bot_name
        self._params: DailyParams = params
        self._callbacks = callbacks
        self._loop = loop

        self._participant_id: str = ""
        self._video_renderers = {}
        self._transcription_renderers = {}
        self._other_participant_has_joined = False

        self._joined = False
        self._joining = False
        self._leaving = False
        self._sync_response = {k: queue.Queue() for k in ["join", "leave"]}

        self._executor = ThreadPoolExecutor(max_workers=5)

        self._client: CallClient = CallClient(event_handler=self)

        self._camera: VirtualCameraDevice = Daily.create_camera_device(
            "camera",
            width=self._params.camera_out_width,
            height=self._params.camera_out_height,
            color_format=self._params.camera_out_color_format)

        self._mic: VirtualMicrophoneDevice = Daily.create_microphone_device(
            "mic", sample_rate=self._params.audio_out_sample_rate, channels=self._params.audio_out_channels)

        self._speaker: VirtualSpeakerDevice = Daily.create_speaker_device(
            "speaker", sample_rate=self._params.audio_in_sample_rate, channels=self._params.audio_in_channels)
        Daily.select_speaker_device("speaker")

    @property
    def participant_id(self) -> str:
        return self._participant_id

    def set_callbacks(self, callbacks: DailyCallbacks):
        self._callbacks = callbacks

    def send_message(self, frame: DailyTransportMessageFrame):
        self._client.send_app_message(frame.message, frame.participant_id)

    def read_next_audio_frame(self) -> AudioRawFrame | None:
        sample_rate = self._params.audio_in_sample_rate
        num_channels = self._params.audio_in_channels

        if self._other_participant_has_joined:
            num_frames = int(sample_rate / 100) * 2  # 20ms of audio

            audio = self._speaker.read_frames(num_frames)

            return AudioRawFrame(audio=audio, sample_rate=sample_rate, num_channels=num_channels)
        else:
            # If no one has ever joined the meeting `read_frames()` would block,
            # instead we just wait a bit. daily-python should probably return
            # silence instead.
            time.sleep(0.01)
            return None

    def write_raw_audio_frames(self, frames: bytes):
        self._mic.write_frames(frames)

    def write_frame_to_camera(self, frame: ImageRawFrame):
        self._camera.write_frame(frame.image)

    async def join(self):
        # Transport already joined, ignore.
        if self._joined or self._joining:
            return

        self._joining = True

        await self._loop.run_in_executor(self._executor, self._join)

    def _join(self):
        logger.info(f"Joining {self._room_url}")

        # For performance reasons, never subscribe to video streams (unless a
        # video renderer is registered).
        self._client.update_subscription_profiles({
            "base": {
                "camera": "unsubscribed",
                "screenVideo": "unsubscribed"
            }
        })

        self._client.set_user_name(self._bot_name)

        self._client.join(
            self._room_url,
            self._token,
            completion=self._call_joined,
            client_settings={
                "inputs": {
                    "camera": {
                        "isEnabled": self._params.camera_out_enabled,
                        "settings": {
                            "deviceId": "camera",
                        },
                    },
                    "microphone": {
                        "isEnabled": self._params.audio_out_enabled,
                        "settings": {
                            "deviceId": "mic",
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
                    }
                },
            })

        self._handle_join_response()

    def _handle_join_response(self):
        try:
            (data, error) = self._sync_response["join"].get(timeout=10)
            if not error:
                self._joined = True
                self._joining = False

                logger.info(f"Joined {self._room_url}")

                if self._token and self._params.transcription_enabled:
                    logger.info(
                        f"Enabling transcription with settings {self._params.transcription_settings}")
                    self._client.start_transcription(
                        self._params.transcription_settings.model_dump())

                self._callbacks.on_joined(data["participants"]["local"])
            else:
                error_msg = f"Error joining {self._room_url}: {error}"
                logger.error(error_msg)
                self._callbacks.on_error(error_msg)
            self._sync_response["join"].task_done()
        except queue.Empty:
            error_msg = f"Time out joining {self._room_url}"
            logger.error(error_msg)
            self._callbacks.on_error(error_msg)

    async def leave(self):
        # Transport not joined, ignore.
        if not self._joined or self._leaving:
            return

        self._joined = False
        self._leaving = True

        await self._loop.run_in_executor(self._executor, self._leave)

    def _leave(self):
        logger.info(f"Leaving {self._room_url}")

        if self._params.transcription_enabled:
            self._client.stop_transcription()

        self._client.leave(completion=self._call_left)

        self._handle_leave_response()

    def _handle_leave_response(self):
        try:
            error = self._sync_response["leave"].get(timeout=10)
            if not error:
                self._leaving = False
                logger.info(f"Left {self._room_url}")
                self._callbacks.on_left()
            else:
                error_msg = f"Error leaving {self._room_url}: {error}"
                logger.error(error_msg)
                self._callbacks.on_error(error_msg)
            self._sync_response["leave"].task_done()
        except queue.Empty:
            error_msg = f"Time out leaving {self._room_url}"
            logger.error(error_msg)
            self._callbacks.on_error(error_msg)

    async def cleanup(self):
        await self._loop.run_in_executor(self._executor, self._cleanup)

    def _cleanup(self):
        if self._client:
            self._client.release()
            self._client = None

    def start_dialout(self, settings):
        self._client.start_dialout(settings)

    def stop_dialout(self, participant_id):
        self._client.stop_dialout(participant_id)

    def start_recording(self, streaming_settings, stream_id, force_new):
        self._client.start_recording(streaming_settings, stream_id, force_new)

    def stop_recording(self, stream_id):
        self._client.stop_recording(stream_id)

    def capture_participant_transcription(self, participant_id: str, callback: Callable):
        if not self._params.transcription_enabled:
            return

        self._transcription_renderers[participant_id] = callback

    def capture_participant_video(
            self,
            participant_id: str,
            callback: Callable,
            framerate: int = 30,
            video_source: str = "camera",
            color_format: str = "RGB"):
        # Only enable camera subscription on this participant
        self._client.update_subscriptions(participant_settings={
            participant_id: {
                "media": "subscribed"
            }
        })

        self._video_renderers[participant_id] = callback

        self._client.set_video_renderer(
            participant_id,
            self._video_frame_received,
            video_source=video_source,
            color_format=color_format)

    #
    #
    # Daily (EventHandler)
    #

    def on_app_message(self, message: Any, sender: str):
        self._callbacks.on_app_message(message, sender)

    def on_call_state_updated(self, state: str):
        self._callbacks.on_call_state_updated(state)

    def on_dialin_ready(self, sip_endpoint: str):
        self._callbacks.on_dialin_ready(sip_endpoint)

    def on_dialout_connected(self, data: Any):
        self._callbacks.on_dialout_connected(data)

    def on_dialout_stopped(self, data: Any):
        self._callbacks.on_dialout_stopped(data)

    def on_dialout_error(self, data: Any):
        self._callbacks.on_dialout_error(data)

    def on_dialout_warning(self, data: Any):
        self._callbacks.on_dialout_warning(data)

    def on_participant_joined(self, participant):
        id = participant["id"]
        logger.info(f"Participant joined {id}")

        if not self._other_participant_has_joined:
            self._other_participant_has_joined = True
            self._callbacks.on_first_participant_joined(participant)

        self._callbacks.on_participant_joined(participant)

    def on_participant_left(self, participant, reason):
        id = participant["id"]
        logger.info(f"Participant left {id}")

        self._callbacks.on_participant_left(participant, reason)

    def on_transcription_message(self, message: Mapping[str, Any]):
        participant_id = ""
        if "participantId" in message:
            participant_id = message["participantId"]

        if participant_id in self._transcription_renderers:
            callback = self._transcription_renderers[participant_id]
            callback(participant_id, message)

    def on_transcription_error(self, message):
        logger.error(f"Transcription error: {message}")

    def on_transcription_started(self, status):
        logger.debug(f"Transcription started: {status}")

    def on_transcription_stopped(self, stopped_by, stopped_by_error):
        logger.debug("Transcription stopped")

    #
    # Daily (CallClient callbacks)
    #

    def _call_joined(self, data, error):
        self._sync_response["join"].put((data, error))

    def _call_left(self, error):
        self._sync_response["leave"].put(error)

    def _video_frame_received(self, participant_id, video_frame):
        callback = self._video_renderers[participant_id]
        callback(participant_id,
                 video_frame.buffer,
                 (video_frame.width, video_frame.height),
                 video_frame.color_format)


class DailyInputTransport(BaseInputTransport):

    def __init__(self, client: DailyTransportClient, params: DailyParams):
        super().__init__(params)

        self._client = client

        self._video_renderers = {}

        self._vad_analyzer: VADAnalyzer | None = params.vad_analyzer
        if params.vad_enabled and not params.vad_analyzer:
            self._vad_analyzer = WebRTCVADAnalyzer(
                sample_rate=self._params.audio_in_sample_rate,
                num_channels=self._params.audio_in_channels)

    async def start(self, frame: StartFrame):
        if self._running:
            return
        # Parent start.
        await super().start(frame)
        # Join the room.
        await self._client.join()
        # Create audio task. It reads audio frames from Daily and push them
        # internally for VAD processing.
        if self._params.audio_in_enabled or self._params.vad_enabled:
            self._audio_in_thread = self._loop.run_in_executor(
                self._executor, self._audio_in_thread_handler)

    async def stop(self):
        if not self._running:
            return
        # Parent stop. This will set _running to False.
        await super().stop()
        # Leave the room.
        await self._client.leave()
        # Stop audio thread.
        if self._params.audio_in_enabled or self._params.vad_enabled:
            await self._audio_in_thread

    async def cleanup(self):
        await super().cleanup()
        await self._client.cleanup()

    def vad_analyzer(self) -> VADAnalyzer | None:
        return self._vad_analyzer

    #
    # FrameProcessor
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, UserImageRequestFrame):
            self.request_participant_image(frame.user_id)

        await super().process_frame(frame, direction)

    #
    # Frames
    #

    def push_transcription_frame(self, frame: TranscriptionFrame | InterimTranscriptionFrame):
        future = asyncio.run_coroutine_threadsafe(
            self._internal_push_frame(frame), self.get_event_loop())
        future.result()

    def push_app_message(self, message: Any, sender: str):
        frame = DailyTransportMessageFrame(message=message, participant_id=sender)
        future = asyncio.run_coroutine_threadsafe(
            self._internal_push_frame(frame), self.get_event_loop())
        future.result()

    #
    # Audio in
    #

    def _audio_in_thread_handler(self):
        while self._running:
            frame = self._client.read_next_audio_frame()
            if frame:
                self.push_audio_frame(frame)

    #
    # Camera in
    #

    def capture_participant_video(
            self,
            participant_id: str,
            framerate: int = 30,
            video_source: str = "camera",
            color_format: str = "RGB"):
        self._video_renderers[participant_id] = {
            "framerate": framerate,
            "timestamp": 0,
            "render_next_frame": False,
        }

        self._client.capture_participant_video(
            participant_id,
            self._on_participant_video_frame,
            framerate,
            video_source,
            color_format
        )

    def request_participant_image(self, participant_id: str):
        if participant_id in self._video_renderers:
            self._video_renderers[participant_id]["render_next_frame"] = True

    def _on_participant_video_frame(self, participant_id: str, buffer, size, format):
        render_frame = False

        curr_time = time.time()
        prev_time = self._video_renderers[participant_id]["timestamp"] or curr_time
        framerate = self._video_renderers[participant_id]["framerate"]

        if framerate > 0:
            next_time = prev_time + 1 / framerate
            render_frame = (curr_time - next_time) < 0.1
        elif self._video_renderers[participant_id]["render_next_frame"]:
            self._video_renderers[participant_id]["render_next_frame"] = False
            render_frame = True

        if render_frame:
            frame = UserImageRawFrame(
                user_id=participant_id,
                image=buffer,
                size=size,
                format=format)
            future = asyncio.run_coroutine_threadsafe(
                self._internal_push_frame(frame), self.get_event_loop())
            future.result()

        self._video_renderers[participant_id]["timestamp"] = curr_time


class DailyOutputTransport(BaseOutputTransport):

    def __init__(self, client: DailyTransportClient, params: DailyParams):
        super().__init__(params)

        self._client = client

    async def start(self, frame: StartFrame):
        if self._running:
            return
        # Parent start.
        await super().start(frame)
        # Join the room.
        await self._client.join()

    async def stop(self):
        if not self._running:
            return
        # Parent stop. This will set _running to False.
        await super().stop()
        # Leave the room.
        await self._client.leave()

    async def cleanup(self):
        await super().cleanup()
        await self._client.cleanup()

    def send_message(self, frame: DailyTransportMessageFrame):
        self._client.send_message(frame)

    def write_raw_audio_frames(self, frames: bytes):
        self._client.write_raw_audio_frames(frames)

    def write_frame_to_camera(self, frame: ImageRawFrame):
        self._client.write_frame_to_camera(frame)


class DailyTransport(BaseTransport):

    def __init__(
            self,
            room_url: str,
            token: str | None,
            bot_name: str,
            params: DailyParams,
            loop: asyncio.AbstractEventLoop | None = None):
        super().__init__(loop)

        callbacks = DailyCallbacks(
            on_joined=self._on_joined,
            on_left=self._on_left,
            on_error=self._on_error,
            on_app_message=self._on_app_message,
            on_call_state_updated=self._on_call_state_updated,
            on_dialin_ready=self._on_dialin_ready,
            on_dialout_connected=self._on_dialout_connected,
            on_dialout_stopped=self._on_dialout_stopped,
            on_dialout_error=self._on_dialout_error,
            on_dialout_warning=self._on_dialout_warning,
            on_first_participant_joined=self._on_first_participant_joined,
            on_participant_joined=self._on_participant_joined,
            on_participant_left=self._on_participant_left,
        )
        self._params = params

        self._client = DailyTransportClient(
            room_url, token, bot_name, params, callbacks, self._loop)
        self._input: DailyInputTransport | None = None
        self._output: DailyOutputTransport | None = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_joined")
        self._register_event_handler("on_left")
        self._register_event_handler("on_app_message")
        self._register_event_handler("on_call_state_updated")
        self._register_event_handler("on_dialin_ready")
        self._register_event_handler("on_dialout_connected")
        self._register_event_handler("on_dialout_stopped")
        self._register_event_handler("on_dialout_error")
        self._register_event_handler("on_dialout_warning")
        self._register_event_handler("on_first_participant_joined")
        self._register_event_handler("on_participant_joined")
        self._register_event_handler("on_participant_left")

    #
    # BaseTransport
    #

    def input(self) -> FrameProcessor:
        if not self._input:
            self._input = DailyInputTransport(self._client, self._params)
        return self._input

    def output(self) -> FrameProcessor:
        if not self._output:
            self._output = DailyOutputTransport(self._client, self._params)
        return self._output

    #
    # DailyTransport
    #

    @property
    def participant_id(self) -> str:
        return self._client.participant_id

    async def send_image(self, frame: ImageRawFrame | SpriteFrame):
        if self._output:
            await self._output.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def send_audio(self, frame: AudioRawFrame):
        if self._output:
            await self._output.process_frame(frame, FrameDirection.DOWNSTREAM)

    def start_dialout(self, settings=None):
        self._client.start_dialout(settings)

    def stop_dialout(self, participant_id):
        self._client.stop_dialout(participant_id)

    def start_recording(self, streaming_settings=None, stream_id=None, force_new=None):
        self._client.start_recording(streaming_settings, stream_id, force_new)

    def stop_recording(self, stream_id=None):
        self._client.stop_recording(stream_id)

    def capture_participant_transcription(self, participant_id: str):
        self._client.capture_participant_transcription(
            participant_id,
            self._on_transcription_message
        )

    def capture_participant_video(
            self,
            participant_id: str,
            framerate: int = 30,
            video_source: str = "camera",
            color_format: str = "RGB"):
        if self._input:
            self._input.capture_participant_video(
                participant_id, framerate, video_source, color_format)

    def _on_joined(self, participant):
        self._call_async_event_handler("on_joined", participant)

    def _on_left(self):
        self._call_async_event_handler("on_left")

    def _on_error(self, error):
        # TODO(aleix): Report error to input/output transports. The one managing
        # the client should report the error.
        pass

    def _on_app_message(self, message: Any, sender: str):
        if self._input:
            self._input.push_app_message(message, sender)
        self._call_async_event_handler("on_app_message", message, sender)

    def _on_call_state_updated(self, state: str):
        self._call_async_event_handler("on_call_state_updated", state)

    async def _handle_dialin_ready(self, sip_endpoint: str):
        if not self._params.dialin_settings:
            return

        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self._params.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "callId": self._params.dialin_settings.call_id,
                "callDomain": self._params.dialin_settings.call_domain,
                "sipUri": sip_endpoint
            }

            url = f"{self._params.api_url}/dialin/pinlessCallUpdate"

            try:
                async with session.post(url, headers=headers, json=data, timeout=10) as r:
                    if r.status != 200:
                        text = await r.text()
                        logger.error(
                            f"Unable to handle dialin-ready event (status: {r.status}, error: {text})")
                        return

                    logger.debug("Event dialin-ready was handled successfully")
            except asyncio.TimeoutError:
                logger.error(f"Timeout handling dialin-ready event ({url})")
            except BaseException as e:
                logger.error(f"Error handling dialin-ready event ({url}): {e}")

    def _on_dialin_ready(self, sip_endpoint):
        if self._params.dialin_settings:
            asyncio.run_coroutine_threadsafe(self._handle_dialin_ready(sip_endpoint), self._loop)
        self._call_async_event_handler("on_dialin_ready", sip_endpoint)

    def _on_dialout_connected(self, data):
        self._call_async_event_handler("on_dialout_connected", data)

    def _on_dialout_stopped(self, data):
        self._call_async_event_handler("on_dialout_stopped", data)

    def _on_dialout_error(self, data):
        self._call_async_event_handler("on_dialout_error", data)

    def _on_dialout_warning(self, data):
        self._call_async_event_handler("on_dialout_warning", data)

    def _on_participant_joined(self, participant):
        self._call_async_event_handler("on_participant_joined", participant)

    def _on_participant_left(self, participant, reason):
        self._call_async_event_handler("on_participant_left", participant, reason)

    def _on_first_participant_joined(self, participant):
        self._call_async_event_handler("on_first_participant_joined", participant)

    def _on_transcription_message(self, participant_id, message):
        text = message["text"]
        timestamp = message["timestamp"]
        is_final = message["rawResponse"]["is_final"]
        if is_final:
            frame = TranscriptionFrame(text, participant_id, timestamp)
            logger.debug(f"Transcription (from: {participant_id}): [{text}]")
        else:
            frame = InterimTranscriptionFrame(text, participant_id, timestamp)

        if self._input:
            self._input.push_transcription_frame(frame)

    def _call_async_event_handler(self, event_name: str, *args, **kwargs):
        future = asyncio.run_coroutine_threadsafe(
            self._call_event_handler(event_name, *args, **kwargs), self._loop)
        future.result()
