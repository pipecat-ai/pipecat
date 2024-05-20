#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import inspect
import queue
import time
import types

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Mapping

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
from pipecat.vad.vad_analyzer import VADAnalyzer, VADParams, VADState

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
    transcription_enabled: bool = False
    transcription_settings: DailyTranscriptionSettings = DailyTranscriptionSettings()


class DailyCallbacks(BaseModel):
    on_joined: Callable[[Mapping[str, Any]], None]
    on_left: Callable[[], None]
    on_participant_joined: Callable[[Mapping[str, Any]], None]
    on_participant_left: Callable[[Mapping[str, Any], str], None]
    on_first_participant_joined: Callable[[Mapping[str, Any]], None]
    on_app_message: Callable[[Any, str], None]
    on_error: Callable[[str], None]


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
            callbacks: DailyCallbacks):
        super().__init__()

        if not self._daily_initialized:
            self._daily_initialized = True
            Daily.init()

        self._room_url: str = room_url
        self._token: str | None = token
        self._bot_name: str = bot_name
        self._params: DailyParams = params
        self._callbacks = callbacks

        self._participant_id: str = ""
        self._video_renderers = {}
        self._transcription_renderers = {}
        self._other_participant_has_joined = False

        self._joined = False
        self._joining = False
        self._leaving = False
        self._sync_response = {k: queue.Queue() for k in ["join", "leave"]}

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

    def read_raw_audio_frames(self, frame_count: int) -> bytes:
        if self._other_participant_has_joined:
            return self._speaker.read_frames(frame_count)
        else:
            # If no one has ever joined the meeting `read_frames()` would block,
            # instead we just wait a bit. daily-python should probably return
            # silence instead.
            time.sleep(0.01)
            return b''

    def write_raw_audio_frames(self, frames: bytes):
        self._mic.write_frames(frames)

    def write_frame_to_camera(self, frame: ImageRawFrame):
        self._camera.write_frame(frame.image)

    async def join(self):
        # Transport already joined, ignore.
        if self._joined or self._joining:
            return

        self._joining = True

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._join)

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

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._leave)

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
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._cleanup)

    def _cleanup(self):
        if self._client:
            self._client.release()
            self._client = None

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

    def on_app_message(self, message: Any, sender: str):
        self._callbacks.on_app_message(message, sender)

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
        self._camera_in_queue = queue.Queue()

        self._vad_analyzer = params.vad_analyzer
        if params.vad_enabled and not params.vad_analyzer:
            self._vad_analyzer = WebRTCVADAnalyzer(
                sample_rate=self._params.audio_in_sample_rate,
                num_channels=self._params.audio_in_channels)

    async def start(self, frame: StartFrame):
        if self._running:
            return
        # Join the room.
        await self._client.join()
        # This will set _running=True
        await super().start(frame)
        # Create camera in thread (runs if _running is true).
        loop = asyncio.get_running_loop()
        self._camera_in_thread = loop.run_in_executor(None, self._camera_in_thread_handler)

    async def stop(self):
        if not self._running:
            return
        # Leave the room.
        await self._client.leave()
        # This will set _running=False
        await super().stop()
        # The thread will stop.
        await self._camera_in_thread

    async def cleanup(self):
        await super().cleanup()
        await self._client.cleanup()

    def vad_analyze(self, audio_frames: bytes) -> VADState:
        state = VADState.QUIET
        if self._vad_analyzer:
            state = self._vad_analyzer.analyze_audio(audio_frames)
        return state

    def read_raw_audio_frames(self, frame_count: int) -> bytes:
        return self._client.read_raw_audio_frames(frame_count)

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
            self._camera_in_queue.put(frame)

        self._video_renderers[participant_id]["timestamp"] = curr_time

    def _camera_in_thread_handler(self):
        while self._running:
            try:
                frame = self._camera_in_queue.get(timeout=1)
                future = asyncio.run_coroutine_threadsafe(
                    self._internal_push_frame(frame), self.get_event_loop())
                future.result()
                self._camera_in_queue.task_done()
            except queue.Empty:
                pass
            except BaseException as e:
                logger.error(f"Error capturing video: {e}")


class DailyOutputTransport(BaseOutputTransport):

    def __init__(self, client: DailyTransportClient, params: DailyParams):
        super().__init__(params)

        self._client = client

    async def start(self, frame: StartFrame):
        if self._running:
            return
        # This will set _running=True
        await super().start(frame)
        # Join the room.
        await self._client.join()

    async def stop(self):
        if not self._running:
            return
        # This will set _running=False
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

    def __init__(self, room_url: str, token: str | None, bot_name: str, params: DailyParams):
        callbacks = DailyCallbacks(
            on_joined=self._on_joined,
            on_left=self._on_left,
            on_first_participant_joined=self._on_first_participant_joined,
            on_participant_joined=self._on_participant_joined,
            on_participant_left=self._on_participant_left,
            on_app_message=self._on_app_message,
            on_error=self._on_error,
        )
        self._params = params

        self._client = DailyTransportClient(room_url, token, bot_name, params, callbacks)
        self._input: DailyInputTransport | None = None
        self._output: DailyOutputTransport | None = None
        self._loop = asyncio.get_running_loop()

        self._event_handlers: dict = {}

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_joined")
        self._register_event_handler("on_left")
        self._register_event_handler("on_participant_joined")
        self._register_event_handler("on_participant_left")
        self._register_event_handler("on_first_participant_joined")

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
        self.on_joined(participant)

    def _on_left(self):
        self.on_left()

    def _on_error(self, error):
        # TODO(aleix): Report error to input/output transports. The one managing
        # the client should report the error.
        pass

    def _on_participant_joined(self, participant):
        self.on_participant_joined(participant)

    def _on_participant_left(self, participant, reason):
        self.on_participant_left(participant, reason)

    def _on_first_participant_joined(self, participant):
        self.on_first_participant_joined(participant)

    def _on_app_message(self, message: Any, sender: str):
        if self._input:
            self._input.push_app_message(message, sender)

    def _on_transcription_message(self, participant_id, message):
        text = message["text"]
        timestamp = message["timestamp"]
        is_final = message["rawResponse"]["is_final"]
        if is_final:
            frame = TranscriptionFrame(text, participant_id, timestamp)
        else:
            frame = InterimTranscriptionFrame(text, participant_id, timestamp)

        if self._input:
            self._input.push_transcription_frame(frame)

    #
    # Decorators (event handlers)
    #

    def on_joined(self, participant):
        pass

    def on_left(self):
        pass

    def on_participant_joined(self, participant):
        pass

    def on_participant_left(self, participant, reason):
        pass

    def on_first_participant_joined(self, participant):
        pass

    def event_handler(self, event_name: str):
        def decorator(handler):
            self._add_event_handler(event_name, handler)
            return handler
        return decorator

    def _register_event_handler(self, event_name: str):
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        if event_name not in [method[0] for method in methods]:
            raise Exception(f"Event handler {event_name} not found")

        self._event_handlers[event_name] = [getattr(self, event_name)]

        patch_method = types.MethodType(partial(self._patch_method, event_name), self)
        setattr(self, event_name, patch_method)

    def _add_event_handler(self, event_name: str, handler):
        if event_name not in self._event_handlers:
            raise Exception(f"Event handler {event_name} not registered")
        self._event_handlers[event_name].append(types.MethodType(handler, self))

    def _patch_method(self, event_name, *args, **kwargs):
        try:
            for handler in self._event_handlers[event_name]:
                if inspect.iscoroutinefunction(handler):
                    # Beware, if handler() calls another event handler it
                    # will deadlock. You shouldn't do that anyways.
                    future = asyncio.run_coroutine_threadsafe(
                        handler(*args[1:], **kwargs), self._loop)

                    # wait for the coroutine to finish. This will also
                    # raise any exceptions raised by the coroutine.
                    future.result()
                else:
                    handler(*args[1:], **kwargs)
        except Exception as e:
            logger.error(f"Exception in event handler {event_name}: {e}")
            raise e

    #     def dialout(self, number):
    #         self.client.start_dialout({"phoneNumber": number})

    #     def start_recording(self):
    #         self.client.start_recording()
