#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Haru local LLM service: full-duplex speech-to-speech run in this process."""

import asyncio
import os
import platform
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from loguru import logger

from pipecat.services.haru.base import (
    DEFAULT_MODEL,
    HARU_FRAME_SAMPLES,
    HARU_SAMPLE_RATE,
    HaruBaseLLMService,
    HaruLLMSettings,
    float_to_pcm16,
)
from pipecat.utils.types import assert_given

HaruEngine = Literal["auto", "mlx", "torch"]

# Seconds of audio in one model frame. A frame that takes longer than this to
# compute can't keep up with a live conversation.
_FRAME_SECS = HARU_FRAME_SAMPLES / HARU_SAMPLE_RATE

# A recurring warning is logged once per this many frames, about every 10 seconds.
_WARN_EVERY_FRAMES = 75


@dataclass(frozen=True)
class _Engine:
    """An implementation of Haru with a checkpoint loaded.

    Parameters:
        name: ``mlx`` or ``torch``.
        model: The loaded model.
        DuplexSession: The class that runs one conversation on the model.
        DuplexConfig: The class holding a conversation's options.
        load_reference: Reads a voice clip from a path, ready for a session.
    """

    name: str
    model: Any
    DuplexSession: Callable[..., Any]
    DuplexConfig: Callable[..., Any]
    load_reference: Callable[[Any], np.ndarray]


def resolve_engine(engine: str, *, system: str | None = None, machine: str | None = None) -> str:
    """Choose the implementation of Haru to run.

    Args:
        engine: ``mlx``, ``torch``, or ``auto`` to pick by platform.
        system: Operating system name, as ``platform.system()`` reports it.
            Defaults to the current one.
        machine: Machine type, as ``platform.machine()`` reports it. Defaults
            to the current one.

    Returns:
        ``mlx`` or ``torch``. ``auto`` is MLX on Apple silicon and PyTorch
        everywhere else.

    Raises:
        ValueError: If ``engine`` is not one of the known names.
    """
    if engine in ("mlx", "torch"):
        return engine
    if engine != "auto":
        raise ValueError(f"Unknown Haru engine {engine!r}: expected 'auto', 'mlx' or 'torch'")
    system = system or platform.system()
    machine = machine or platform.machine()
    return "mlx" if system == "Darwin" and machine == "arm64" else "torch"


def load_engine(
    engine: str, model: str, *, device: str | None = None, quantize: int | None = None
) -> _Engine:
    """Import an implementation of Haru and load a checkpoint with it.

    MLX keeps its GPU streams per thread, so this must run on the thread that
    will drive the model's sessions.

    Args:
        engine: ``mlx`` or ``torch``.
        model: A checkpoint directory or a Hugging Face repository id.
        device: PyTorch device to load onto. Defaults to CUDA when available.
        quantize: Bits per weight of the MLX backbone (``8``), or ``None``
            for the checkpoint's own precision.

    Returns:
        The loaded ``model`` with the ``DuplexSession`` and ``DuplexConfig``
        classes that drive it and a ``load_reference`` function that reads a
        voice clip.

    Raises:
        ImportError: If the engine's package is not installed.
    """
    if engine == "mlx":
        try:
            import haru_mlx as package
            from haru_mlx.audio import load_audio, prepare_reference
        except ImportError as e:
            logger.error(f"Exception: {e}")
            logger.error(
                'In order to run Haru on Apple silicon, you need to `uv add "pipecat-ai[haru-mlx]"`.'
            )
            raise ImportError(f'Missing module: {e}. Install "pipecat-ai[haru-mlx]".') from e
        loaded = package.HaruModel.from_pretrained(model, quantize=quantize)
    else:
        try:
            import haru as package
            from haru.audio import load_audio, prepare_reference
        except ImportError as e:
            logger.error(f"Exception: {e}")
            logger.error(
                'In order to run Haru with PyTorch, you need to `uv add "pipecat-ai[haru-torch]"`.'
            )
            raise ImportError(f'Missing module: {e}. Install "pipecat-ai[haru-torch]".') from e
        loaded = package.HaruModel.from_pretrained(model, device=device)

    return _Engine(
        name=engine,
        model=loaded,
        DuplexSession=package.DuplexSession,
        DuplexConfig=package.DuplexConfig,
        load_reference=lambda path: prepare_reference(load_audio(path)),
    )


class HaruLocalLLMService(HaruBaseLLMService):
    """Haru full-duplex speech-to-speech, run in this process.

    Loads the Haru checkpoint (about 10 GB, downloaded from Hugging Face on
    first use) and runs it on this machine: with MLX on Apple silicon
    (``pipecat-ai[haru-mlx]``) or with PyTorch on an NVIDIA GPU with about
    12 GB of memory (``pipecat-ai[haru-torch]``). See
    :class:`~pipecat.services.haru.base.HaruBaseLLMService` for how the model
    behaves in a pipeline, and
    :class:`~pipecat.services.haru.llm.HaruLLMService` to use a server
    instead.

    The model starts loading when the service is created. All model work
    runs on one dedicated thread, so the event loop keeps moving frames while
    a model frame is computed. The model has to compute each 133 ms frame in
    less than 133 ms to keep up; when it can't, the oldest waiting user audio
    is dropped rather than letting the delay grow.

    Event handlers available:

    - on_session_started: Called once the model has run its prompt and is
      listening.

    Example::

        llm = HaruLocalLLMService(
            voice_reference="voice.wav",
            settings=HaruLocalLLMService.Settings(system_instruction="..."),
        )
    """

    Settings = HaruLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        engine: HaruEngine = "auto",
        device: str | None = None,
        quantize: int | None = None,
        max_backlog_frames: int = 4,
        voice_reference: str | os.PathLike | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Haru local LLM service.

        Args:
            engine: The implementation to run: ``mlx``, ``torch``, or
                ``auto`` for MLX on Apple silicon and PyTorch elsewhere.
            device: PyTorch device to load the model onto. Defaults to CUDA
                when available. Not used by MLX.
            quantize: Bits per weight of the MLX backbone. ``8`` is smaller
                and reads fewer bytes per frame; ``None`` keeps the
                checkpoint's precision. Not used by PyTorch.
            max_backlog_frames: User frames that may wait for the model
                before the oldest is dropped. Each is 133 ms of delay.
            voice_reference: Path of an audio clip of the voice the model
                should speak with: one speaker, 3 seconds to 3 minutes of
                clean speech. Haru ships without voices.
            settings: Service settings. ``model`` is a Hugging Face
                repository id or a checkpoint directory. All settings are
                fixed once the conversation has started.
            **kwargs: Additional arguments passed to the parent service.
        """
        super().__init__(voice_reference=voice_reference, settings=settings, **kwargs)

        self._engine_name = resolve_engine(engine)
        self._max_backlog_frames = max_backlog_frames

        # One thread loads the model and runs every frame: MLX requires a
        # single thread, and PyTorch sessions take turns on the model anyway.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="haru-model")
        self._engine_future = self._executor.submit(
            load_engine,
            self._engine_name,
            assert_given(self._settings.model) or DEFAULT_MODEL,
            device=device,
            quantize=quantize,
        )

        self._session: Any = None
        self._session_task: asyncio.Task | None = None
        self._user_frames: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._dropped_frames = 0
        self._slow_frames = 0

    async def cleanup(self):
        """Release the model's thread at teardown."""
        await super().cleanup()
        await self._close_session()
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def _run(self, fn, *args):
        """Call ``fn(*args)`` on the model's thread."""
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    #
    # session
    #

    async def _open_session(self, instructions: str):
        # Loading the model and running its prompt take seconds, so the
        # conversation runs in a task of its own and frames keep flowing.
        self._session_task = self.create_task(self._run_session(instructions), "session")

    async def _close_session(self):
        if self._session_task:
            await self.cancel_task(self._session_task)
            self._session_task = None
        await super()._close_session()
        self._session = None  # drops the conversation's cache
        while not self._user_frames.empty():
            self._user_frames.get_nowait()

    async def _send_user_frame(self, audio: np.ndarray):
        self._user_frames.put_nowait(audio)
        while self._user_frames.qsize() > self._max_backlog_frames:
            self._user_frames.get_nowait()
            self._dropped_frames += 1
            if self._dropped_frames % _WARN_EVERY_FRAMES == 1:
                logger.warning(
                    f"{self}: the model is not keeping up with real time; "
                    f"{self._dropped_frames} frames of user audio dropped so far"
                )

    async def _run_session(self, instructions: str):
        try:
            engine = await asyncio.wrap_future(self._engine_future)
            reference = None
            if self._voice_reference is not None:
                reference = await self._run(engine.load_reference, self._voice_reference)
            config = engine.DuplexConfig(**self._session_options())
            self._session = await self._run(
                engine.DuplexSession, engine.model, instructions, reference, config
            )
            frame = await self._run(self._session.start)
            logger.debug(f"{self}: prompt ran in {frame.seconds:.1f}s on the {engine.name} engine")
        except Exception as e:
            await self.push_error(
                error_msg=f"Conversation failed to start: {e}",
                exception=e,
                force_treat_as_permanent=True,
            )
            return

        await self._push_model_frame(frame)
        await self._session_ready()

        while not frame.done:
            user_audio = await self._user_frames.get()
            try:
                frame = await self._run(self._session.step, user_audio)
            except Exception as e:
                await self.push_error(error_msg=f"Model frame failed: {e}", exception=e)
                continue
            if frame.seconds > _FRAME_SECS:
                self._slow_frames += 1
                if self._slow_frames % _WARN_EVERY_FRAMES == 1:
                    logger.warning(
                        f"{self}: a model frame took {frame.seconds * 1000:.0f} ms to compute "
                        f"({_FRAME_SECS * 1000:.0f} ms is real time)"
                    )
            await self._push_model_frame(frame)

        logger.warning(f"{self}: the conversation reached the model's maximum length")
        self._session_started = False

    async def _push_model_frame(self, frame: Any):
        await self._push_model_audio(float_to_pcm16(frame.audio))
        await self._push_model_text(frame.text)
