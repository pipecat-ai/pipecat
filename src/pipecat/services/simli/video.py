#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Simli video service for real-time avatar generation."""

import asyncio
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    OutputImageRawFrame,
    SpeechOutputAudioRawFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.ai_service import AIService
from pipecat.services.settings import ServiceSettings
from pipecat.utils.deprecation import deprecated

try:
    from av.audio.frame import AudioFrame
    from av.audio.resampler import AudioResampler
    from simli import SimliClient, SimliConfig
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Simli, you need to `uv add "pipecat-ai[simli]"`.')
    raise ImportError(f"Missing module: {e}") from e


@dataclass
class SimliVideoSettings(ServiceSettings):
    """Settings for the Simli video service."""

    pass


class SimliCamera(BaseModel):
    """The camera a Simli Trinity-2 session starts with.

    Offsets on top of the avatar's own framing. A field left unset stays
    neutral: 0, or 1 for ``zoom``. Simli still applies a value outside its
    supported range. See https://docs.simli.com/api-reference/trinity-2/camera.

    Parameters:
        pos_x_off: Moves the avatar right, in half canvas widths. Supported
            range -10 to 10.
        pos_y_off: Moves the avatar down, in half canvas heights. Supported
            range -10 to 10.
        pos_z_off: Metres the camera backs away from the avatar (negative moves
            in). Supported range -5 to 5.
        rot_x_off: Degrees the camera tilts up (pitch). Supported range -180 to 180.
        rot_y_off: Degrees the camera pans right (yaw). Supported range -180 to 180.
        rot_z_off: Degrees the camera rolls clockwise. Supported range -180 to 180.
        zoom: Multiplies the avatar's zoom; 1 is no change. Supported range
            above 0, up to 4.
        orbit_x_off: Degrees the camera orbits up over the avatar's head
            (elevation). Supported range -80 to 80.
        orbit_y_off: Degrees the camera orbits round to its right (azimuth).
            Supported range -180 to 180.
    """

    pos_x_off: float | None = None
    pos_y_off: float | None = None
    pos_z_off: float | None = None
    rot_x_off: float | None = None
    rot_y_off: float | None = None
    rot_z_off: float | None = None
    zoom: float | None = None
    orbit_x_off: float | None = None
    orbit_y_off: float | None = None


class SimliFraming(BaseModel):
    """How a Simli Trinity-2 session frames its video.

    The video's size, where the avatar sits in it, the camera, and an uploaded
    background or foreground. Simli applies them when the session starts. Only
    Trinity-2 faces use them; other faces keep their own frame size. A field
    left unset keeps Simli's default. See
    https://docs.simli.com/api-reference/trinity-2#framing-the-video.

    Set the output transport's ``video_out_width`` and ``video_out_height`` to
    the same size as ``streaming_width`` and ``streaming_height``.

    Example::

        # 16:9 video with the avatar a little left of center.
        SimliVideoService.Framing(
            streaming_width=1366,
            streaming_height=768,
            avatar_position_x=0.4,
        )

    Parameters:
        streaming_width: Width of the video in pixels, at least 512. With
            ``streaming_height`` it must make a 1:1, 16:9 or 9:16 frame, such
            as 1366x768; Simli rejects other shapes. Simli's default is 768.
        streaming_height: Height of the video in pixels, at least 512. See
            ``streaming_width``. Simli's default is 768.
        avatar_position_x: Where the avatar sits across the video, from 0
            (left) to 1 (right). Simli's default is 0.5.
        avatar_position_y: Where the avatar sits down the video, from 0 (top)
            to 1 (bottom). Simli's default is 1.
        camera: The camera the session starts with. Unset keeps the avatar's
            own framing.
        background_id: The id of a background uploaded to your Simli account
            (``POST /assets/backgrounds``) to show behind the avatar. Unset
            keeps the avatar's own.
        foreground_id: The id of a foreground uploaded to your Simli account
            (``POST /assets/foregrounds``) to lay over the avatar. Unset keeps
            the avatar's own.
    """

    streaming_width: int | None = Field(default=None, ge=512)
    streaming_height: int | None = Field(default=None, ge=512)
    avatar_position_x: float | None = Field(default=None, ge=0.0, le=1.0)
    avatar_position_y: float | None = Field(default=None, ge=0.0, le=1.0)
    camera: SimliCamera | None = None
    background_id: str | None = None
    foreground_id: str | None = None


class SimliVideoService(AIService):
    """Simli video service for real-time avatar generation.

    Provides real-time avatar video generation by processing audio frames
    and producing synchronized video output using the Simli API. Handles
    audio resampling, video frame processing, and connection management.

    The avatar speaks the bot's voice: ``TTSAudioRawFrame`` from a TTS service,
    or ``SpeechOutputAudioRawFrame`` from a speech-to-speech service that
    streams its voice in real time, such as ``OpenAILiveLLMService``. The
    service sends that audio to Simli and pushes the avatar's video and its
    voice, in sync, in place of it.
    """

    Settings = SimliVideoSettings
    Framing = SimliFraming
    Camera = SimliCamera
    _settings: Settings

    @deprecated(
        "`SimliVideoService.InputParams` is deprecated since 0.0.106 and will be removed in "
        "2.0.0. Use `SimliVideoService.Settings` instead."
    )
    class InputParams(BaseModel):
        """Input parameters for Simli video configuration.

        .. deprecated:: 0.0.106
            Use ``SimliVideoService.Settings(...)`` instead.
            Will be removed in 2.0.0.

        Parameters:
            enable_logging: Whether to enable Simli logging.
            max_session_length: Absolute maximum session duration in seconds.
                Avatar will disconnect after this time even if it's speaking.
            max_idle_time: Maximum duration in seconds the avatar is not speaking
                before the avatar disconnects.
        """

        enable_logging: bool | None = None
        max_session_length: int | None = None
        max_idle_time: int | None = None

    def __init__(
        self,
        *,
        api_key: str,
        face_id: str,
        simli_url: str = "https://api.simli.ai",
        is_trinity_avatar: bool = False,
        params: InputParams | None = None,
        max_session_length: int | None = None,
        max_idle_time: int | None = None,
        enable_logging: bool | None = None,
        framing: SimliFraming | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Simli video service.

        Args:
            api_key: Simli API key for authentication.
            face_id: Simli Face ID. For Trinity avatars, specify "faceId/emotionId"
                to use a different emotion than the default.
            simli_url: URL of the simli servers. Can be changed for custom deployments
                of enterprise users.
            is_trinity_avatar: Boolean to tell simli client that this is a Trinity avatar
                which reduces latency when using Trinity.
            params: Additional input parameters for session configuration.

                .. deprecated:: 0.0.106
                    Use ``settings=SimliVideoService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            max_session_length: Absolute maximum session duration in seconds.
                Avatar will disconnect after this time even if it's speaking.
            max_idle_time: Maximum duration in seconds the avatar is not speaking
                before the avatar disconnects.
            enable_logging: Whether to enable Simli logging.
            framing: Trinity-2 framing for the session: the video's size, where
                the avatar sits in it, the camera, and an uploaded background or
                foreground. Unset keeps Simli's defaults (768x768, avatar at
                the bottom center).
            settings: Service settings.
            **kwargs: Additional arguments passed to the parent AIService.
        """
        # 1. Default settings
        default_settings = ServiceSettings(model=None)

        # 2. Apply deprecated params overrides
        if params is not None:
            self._warn_init_param_moved_to_settings("params", deprecated_since="0.0.106")
            if max_session_length is None and hasattr(params, "max_session_length"):
                max_session_length = params.max_session_length
            if max_idle_time is None and hasattr(params, "max_idle_time"):
                max_idle_time = params.max_idle_time
            if enable_logging is None and hasattr(params, "enable_logging"):
                enable_logging = params.enable_logging

        # 3. Apply settings delta
        if settings is not None:
            default_settings.apply_update(settings)

        # 4. Call super
        super().__init__(settings=default_settings, **kwargs)

        # Build SimliConfig from parameters
        config_kwargs: dict[str, Any] = {
            "faceId": face_id,
        }
        if max_session_length is not None:
            config_kwargs["maxSessionLength"] = max_session_length
        if max_idle_time is not None:
            config_kwargs["maxIdleTime"] = max_idle_time

        config = SimliConfig(**config_kwargs)
        if framing is not None:
            # simli-ai 2.0.x's SimliConfig has no fields for these, but the SDK sends
            # every attribute of the config as the session request. Later releases
            # declare them as fields, which setting them also fills.
            for key, value in framing.model_dump(exclude_none=True).items():
                setattr(config, key, value)

        self._initialized = False
        # Add buffer time to session limits
        config.maxIdleTime += 5
        config.maxSessionLength += 5
        self._simli_client = SimliClient(
            api_key=api_key,
            config=config,
            simliURL=simli_url,
            enableSFU=True,
        )

        self._pipecat_resampler: AudioResampler | None = None
        self._pipecat_resampler_event = asyncio.Event()
        self._simli_resampler = AudioResampler("s16", "mono", 16000)

        self._audio_task: asyncio.Task | None = None
        self._video_task: asyncio.Task | None = None
        self._is_trinity_avatar = is_trinity_avatar
        self._previously_interrupted = is_trinity_avatar
        self._audio_buffer = bytearray()
        # Whether the bot's voice last came as a speech stream
        # (SpeechOutputAudioRawFrame) rather than as TTS audio.
        self._speech_stream = False

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._start_connection()

    async def stop(self, frame: EndFrame):
        """Stop the Simli video service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._stop_connection()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Simli video service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._stop_connection()

    async def cleanup(self):
        """Clean up the Simli video service."""
        await super().cleanup()
        await self._stop_connection()

    async def _start_connection(self):
        """Start the connection to Simli service and begin processing tasks."""
        try:
            if not self._initialized:
                await self._simli_client.start()
                self._initialized = True

            # Create task to consume and process audio and video
            await self._simli_client.sendSilence()
            self._audio_task = self.create_task(self._consume_and_process_audio())
            self._video_task = self.create_task(self._consume_and_process_video())
        except Exception as e:
            await self.push_error(error_msg=f"Unable to start connection: {e}", exception=e)

    async def _consume_and_process_audio(self):
        """Consume audio frames from Simli and push them downstream."""
        await self._pipecat_resampler_event.wait()
        # The _pipecat_resampler_event waits for _pipecat_resampler to be built
        assert self._pipecat_resampler is not None

        audio_iterator = self._simli_client.getAudioStreamIterator()
        async for audio_frame in audio_iterator:
            resampled_frames = self._pipecat_resampler.resample(audio_frame)
            for resampled_frame in resampled_frames:
                audio_array = resampled_frame.to_ndarray()
                # Only push frame is there is audio (e.g. not silence)
                if audio_array.any():
                    # The avatar's voice goes on as the same kind of audio as the
                    # bot's came in. For a speech stream the output transport tells
                    # speech from silence itself, as it would without Simli.
                    frame_type = (
                        SpeechOutputAudioRawFrame if self._speech_stream else TTSAudioRawFrame
                    )
                    await self.push_frame(
                        frame_type(
                            audio=audio_array.tobytes(),
                            sample_rate=self._pipecat_resampler.rate,
                            num_channels=1,
                        ),
                    )

    async def _consume_and_process_video(self):
        """Consume video frames from Simli and convert them to output frames."""
        await self._pipecat_resampler_event.wait()
        video_iterator = self._simli_client.getVideoStreamIterator(targetFormat="rgb24")
        async for video_frame in video_iterator:
            # Process the video frame
            convertedFrame: OutputImageRawFrame = OutputImageRawFrame(
                image=video_frame.to_rgb().to_image().tobytes(),
                size=(video_frame.width, video_frame.height),
                format="RGB",
            )
            convertedFrame.pts = video_frame.pts
            await self.push_frame(convertedFrame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle Simli video generation.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, (TTSAudioRawFrame, SpeechOutputAudioRawFrame)):
            # Send the bot's voice to Simli, which sends back the avatar's in sync
            # with its video, so the frame doesn't go on. TTS services send
            # TTSAudioRawFrame; speech-to-speech services that stream their voice
            # in real time, such as OpenAILiveLLMService, send
            # SpeechOutputAudioRawFrame.
            self._speech_stream = isinstance(frame, SpeechOutputAudioRawFrame)
            try:
                old_frame = AudioFrame.from_ndarray(
                    np.frombuffer(frame.audio, dtype=np.int16)[None, :],
                    layout="mono" if frame.num_channels == 1 else "stereo",
                )
                old_frame.sample_rate = frame.sample_rate

                if self._pipecat_resampler is None:
                    self._pipecat_resampler = AudioResampler(
                        "s16", old_frame.layout, old_frame.sample_rate
                    )
                    self._pipecat_resampler_event.set()

                resampled_frames = self._simli_resampler.resample(old_frame)
                for resampled_frame in resampled_frames:
                    audioBytes = resampled_frame.to_ndarray().astype(np.int16).tobytes()
                    if self._previously_interrupted and self._speech_stream:
                        # A speech stream comes at the pace it plays at, so holding
                        # it back to fill the buffer below would only delay the
                        # avatar. Send what's buffered and go on.
                        if self._audio_buffer:
                            await self._simli_client.sendImmediate(bytes(self._audio_buffer))
                            self._audio_buffer = bytearray()
                        self._previously_interrupted = False
                    if self._previously_interrupted:
                        self._audio_buffer.extend(audioBytes)
                        if len(self._audio_buffer) >= 128000:
                            try:
                                for flushFrame in self._simli_resampler.resample(None):
                                    self._audio_buffer.extend(
                                        flushFrame.to_ndarray().astype(np.int16).tobytes()
                                    )
                            finally:
                                await self._simli_client.sendImmediate(bytes(self._audio_buffer))
                                self._previously_interrupted = False
                                self._audio_buffer = bytearray()
                    else:
                        await self._simli_client.send(audioBytes)
                return
            except Exception as e:
                await self.push_error(error_msg=f"Error sending audio: {e}", exception=e)
        elif isinstance(frame, TTSStoppedFrame):
            try:
                if self._previously_interrupted and len(self._audio_buffer) > 0:
                    await self._simli_client.sendImmediate(bytes(self._audio_buffer))
                    self._previously_interrupted = False
                    self._audio_buffer = bytearray()
            except Exception as e:
                await self.push_error(error_msg=f"Error stopping TTS: {e}", exception=e)
            return
        elif isinstance(frame, (InterruptionFrame, UserStartedSpeakingFrame)):
            if not self._previously_interrupted:
                await self._simli_client.clearBuffer()
            self._previously_interrupted = self._is_trinity_avatar

        await self.push_frame(frame, direction)

    async def _stop_connection(self):
        """Stop the Simli client and cancel processing tasks.

        Idempotent so it can run from stop(), cancel(), and cleanup().
        """
        if self._initialized:
            await self._simli_client.stop()
            self._initialized = False
        if self._audio_task:
            await self.cancel_task(self._audio_task)
            self._audio_task = None
        if self._video_task:
            await self.cancel_task(self._video_task)
            self._video_task = None
