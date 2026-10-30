#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Capture a bot file's text path, so a trace runs the bot's own processors.

:func:`capture_bot` imports a bot file, calls its ``bot(runner_args)`` entry
point and intercepts the :class:`~pipecat.pipeline.worker.PipelineWorker` it
builds, before anything runs: the transport is a stub and the runner never
starts. What comes back are the bot's real ``LLMTextProcessor``, ``TTSService``
and ``RTVIObserver`` parameters, whose aggregators, transforms and filters the
trace then exercises as they are, callables included.
"""

import asyncio
import contextlib
import dataclasses
import importlib.util
import sys
from collections.abc import Callable, Coroutine, Generator, Iterator
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field

from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver, RTVIObserverParams
from pipecat.runner.types import RunnerArguments
from pipecat.services.llm_service import LLMService
from pipecat.services.tts_service import TTSService, WebsocketTTSService
from pipecat.transports.base_transport import BaseTransport
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator
from pipecat.utils.text.pattern_pair_aggregator import PatternPairAggregator
from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator
from pipecat.workers.runner import WorkerRunner

from .setup import (
    AggregatorSpec,
    LLMTextProcessorSpec,
    PatternSpec,
    RTVISpec,
    TextFlowSetup,
    TransformSpec,
    TTSSpec,
)

CAPTURE_TIMEOUT_S = 30.0


class TextFlowCaptureError(Exception):
    """The bot file could not be captured; the message says why."""


class BotSummary(BaseModel):
    """What was found in a bot file.

    Parameters:
        path: The bot file.
        pipeline: The pipeline's processors, in order, by class name.
        traced: Positions in ``pipeline`` of the processors the trace runs as the
            bot's own: its ``LLMTextProcessor`` and its TTS service, run live or
            with its parts taken over by the stand-in.
        tts_service: The class name of the bot's TTS service, or None without one.
        notes: What the trace models differently from the bot, such as a processor
            between the LLM and the TTS it does not run.
    """

    path: str
    pipeline: list[str] = Field(default_factory=list)
    traced: list[int] = Field(default_factory=list)
    tts_service: str | None = None
    notes: list[str] = Field(default_factory=list)


@dataclasses.dataclass
class CapturedBot:
    """A bot's text path, as live objects.

    The bot's ``bot()`` coroutine is held at its runner's ``run()`` until
    :meth:`close` releases it, so resources it opened around the run, such as an
    HTTP session its services share, stay open while its objects are used.

    Parameters:
        summary: What was found, for display.
        llm_text_processor: The bot's ``LLMTextProcessor``, or None when the TTS
            service aggregates the LLM text itself.
        tts: The bot's TTS service.
        rtvi_params: The parameters of the bot's ``RTVIObserver``, or None when it
            has none.
    """

    summary: BotSummary
    llm_text_processor: LLMTextProcessor | None
    tts: TTSService | None
    rtvi_params: RTVIObserverParams | None
    _hold: "_Hold | None" = dataclasses.field(default=None, repr=False)

    async def close(self, *, timeout_s: float = CAPTURE_TIMEOUT_S) -> str | None:
        """Let ``bot()`` return and wait for it.

        Args:
            timeout_s: How long the bot may take to finish once released.

        Returns:
            What went wrong while it finished, or None.
        """
        hold, self._hold = self._hold, None
        if hold is None or hold.task is None:
            return None
        hold.release.set()
        try:
            await asyncio.wait_for(hold.task, timeout_s)
        except TimeoutError:
            return f"bot() did not finish within {timeout_s:.0f}s of being released"
        except Exception as e:
            return f"bot() failed after the run: {e!r}"
        return None

    async def __aenter__(self) -> "CapturedBot":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


class _Hold:
    """Where ``bot()`` waits while its objects are in use."""

    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.task: asyncio.Task[None] | None = None


def callable_label(func: Any) -> str:
    """A short name for a transform or filter, for display.

    A closure made by a factory such as ``replace_text`` is named after the
    factory.
    """
    qualname = getattr(func, "__qualname__", None)
    if isinstance(qualname, str):
        if ".<locals>." in qualname:
            return qualname.split(".<locals>.")[0]
        return qualname.rsplit(".", 1)[-1]
    return type(func).__name__


class _StubTransport(BaseTransport):
    """A transport that connects to nothing and accepts any event handler."""

    def __init__(self):
        super().__init__(name="textflow-transport")
        self._in = FrameProcessor(name="textflow-transport-input")
        self._out = FrameProcessor(name="textflow-transport-output")

    def input(self) -> FrameProcessor:
        return self._in

    def output(self) -> FrameProcessor:
        return self._out

    def add_event_handler(self, event_name: str, handler):
        self._register_event_handler(event_name)
        super().add_event_handler(event_name, handler)


@contextlib.contextmanager
def _intercept(
    workers: list[PipelineWorker], observers: list[RTVIObserver], hold: _Hold
) -> Generator[None]:
    """Keep the bot from connecting or running while it builds its pipeline.

    Patched before the bot module is imported, so a ``from ... import
    create_transport`` in the bot binds the stub. The runner's ``run()`` parks
    the bot until the hold is released.
    """
    import pipecat.runner.utils as runner_utils

    async def create_transport(runner_args: Any, transport_params: Any) -> BaseTransport:
        return _StubTransport()

    async def run(self: WorkerRunner, *args: Any, **kwargs: Any) -> None:
        hold.started.set()
        await hold.release.wait()

    worker_init = PipelineWorker.__init__
    observer_init = RTVIObserver.__init__

    def init_worker(self: PipelineWorker, *args: Any, **kwargs: Any) -> None:
        worker_init(self, *args, **kwargs)
        workers.append(self)

    def init_observer(self: RTVIObserver, *args: Any, **kwargs: Any) -> None:
        observer_init(self, *args, **kwargs)
        observers.append(self)

    originals = (runner_utils.create_transport, WorkerRunner.run)
    runner_utils.create_transport = create_transport  # type: ignore[assignment]
    WorkerRunner.run = run  # type: ignore[method-assign]
    PipelineWorker.__init__ = init_worker  # type: ignore[method-assign]
    RTVIObserver.__init__ = init_observer  # type: ignore[method-assign]
    try:
        yield
    finally:
        runner_utils.create_transport, WorkerRunner.run = originals  # type: ignore[assignment]
        PipelineWorker.__init__ = worker_init  # type: ignore[method-assign]
        RTVIObserver.__init__ = observer_init  # type: ignore[method-assign]


_imports = 0


def _import_bot(path: Path):
    """Import the bot file as a fresh module, with its folder importable."""
    global _imports
    _imports += 1
    spec = importlib.util.spec_from_file_location(f"textflow_bot_{path.stem}_{_imports}", path)
    if spec is None or spec.loader is None:
        raise TextFlowCaptureError(f"{path} is not a Python module")
    module = importlib.util.module_from_spec(spec)
    folder = str(path.parent)
    added = folder not in sys.path
    if added:
        sys.path.insert(0, folder)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise TextFlowCaptureError(f"importing {path.name} failed: {e!r}") from e
    finally:
        if added:
            sys.path.remove(folder)
    return module


def _flatten(processor: FrameProcessor) -> Iterator[FrameProcessor]:
    children = list(processor.processors)
    if children:
        for child in children:
            yield from _flatten(child)
    else:
        yield processor


async def capture_bot(path: str | Path, *, timeout_s: float = CAPTURE_TIMEOUT_S) -> CapturedBot:
    """Import a bot file and capture its text path.

    The bot's ``bot(runner_args)`` runs with a stub transport and a runner whose
    ``run()`` waits instead of running, so its services are constructed as in the
    bot but nothing connects. The bot's own imports and environment apply: a
    service that needs an API key at construction needs it here too. The bot is
    held there until the result is closed (``async with`` does it), so what it
    set up around the run stays usable in the meantime.

    Args:
        path: The bot file, with a ``bot(runner_args)`` coroutine.
        timeout_s: How long ``bot()`` may take to build its pipeline.

    Returns:
        The bot's text path. Close it once its objects are no longer needed.

    Raises:
        TextFlowCaptureError: If the file cannot be imported, has no ``bot``
            function, or builds no pipeline.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise TextFlowCaptureError(f"{path} does not exist")

    workers: list[PipelineWorker] = []
    observers: list[RTVIObserver] = []
    hold = _Hold()
    with _intercept(workers, observers, hold):
        module = _import_bot(path)
        entry = getattr(module, "bot", None)
        if not callable(entry):
            raise TextFlowCaptureError(f"{path.name} has no bot(runner_args) function")
        bot = cast(Callable[[RunnerArguments], Coroutine[Any, Any, None]], entry)
        task = asyncio.create_task(bot(RunnerArguments()))
        started = asyncio.ensure_future(hold.started.wait())
        done, _ = await asyncio.wait(
            {task, started}, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED
        )
        if not done:
            task.cancel()
            started.cancel()
            raise TextFlowCaptureError(
                f"bot() in {path.name} did not build its pipeline within {timeout_s:.0f}s"
            )
        if task in done:
            # bot() returned or failed without reaching run().
            started.cancel()
            try:
                task.result()
            except TextFlowCaptureError:
                raise
            except KeyError as e:
                raise TextFlowCaptureError(
                    f"bot() in {path.name} failed: {e!r}. The bot runs its own setup code, "
                    f"so the variables it reads must be set, as they are when it runs for real."
                ) from e
            except Exception as e:
                raise TextFlowCaptureError(f"bot() in {path.name} failed: {e!r}") from e
        else:
            hold.task = task
    if not workers:
        hold.release.set()
        raise TextFlowCaptureError(f"bot() in {path.name} created no PipelineWorker")

    processors = [
        p
        for p in _flatten(workers[0].pipeline)
        if not type(p).__name__.startswith(("PipelineSource", "PipelineSink"))
    ]
    summary = BotSummary(path=str(path), pipeline=[type(p).__name__ for p in processors])
    if len(workers) > 1:
        summary.notes.append(f"The bot builds {len(workers)} workers; the first one is traced.")

    llm = next((p for p in processors if isinstance(p, LLMService)), None)
    tts = next((p for p in processors if isinstance(p, TTSService)), None)
    ltp = next((p for p in processors if isinstance(p, LLMTextProcessor)), None)
    summary.traced = [processors.index(p) for p in (ltp, tts) if p is not None]
    summary.tts_service = type(tts).__name__ if tts is not None else None
    if tts is None:
        summary.notes.append("No TTSService in the pipeline; the stand-in's defaults apply.")
    if llm is not None and tts is not None:
        between = processors[processors.index(llm) + 1 : processors.index(tts)]
        skipped = [type(p).__name__ for p in between if p is not ltp]
        if skipped:
            summary.notes.append(
                "Between the LLM and the TTS but not run by the trace: " + ", ".join(skipped)
            )
    if ltp is not None and tts is not None and processors.index(ltp) > processors.index(tts):
        summary.notes.append("The LLMTextProcessor sits after the TTS; the trace runs it before.")

    rtvi = next((o for o in observers if o is not None), None)
    if rtvi is None:
        summary.notes.append("No RTVIObserver; the client is left out.")

    return CapturedBot(
        summary=summary,
        llm_text_processor=ltp,
        tts=tts,
        rtvi_params=rtvi._params if rtvi is not None else None,
        _hold=hold,
    )


def _aggregator_spec(aggregator: BaseTextAggregator, notes: list[str]) -> AggregatorSpec:
    text_type = str(aggregator.aggregation_type)
    aggregation_type = text_type if text_type in ("sentence", "word", "token") else "sentence"
    if isinstance(aggregator, PatternPairAggregator):
        return AggregatorSpec(
            kind="pattern_pair",
            aggregation_type=aggregation_type,  # type: ignore[arg-type]
            patterns=[
                PatternSpec(
                    type=p["type"],
                    start=p["start"],
                    end=p["end"],
                    action=str(p["action"].value),  # type: ignore[arg-type]
                )
                for p in aggregator._patterns.values()
            ],
        )
    if isinstance(aggregator, SkipTagsAggregator):
        return AggregatorSpec(
            kind="skip_tags",
            aggregation_type=aggregation_type,  # type: ignore[arg-type]
            skip_tags=[[t[0], t[1]] for t in aggregator._tags],
        )
    if type(aggregator).__name__ != "SimpleTextAggregator":
        notes.append(f"Custom text aggregator {type(aggregator).__name__} shown as simple.")
    return AggregatorSpec(kind="simple", aggregation_type=aggregation_type)  # type: ignore[arg-type]


def _transform_specs(pairs: list[tuple[Any, Any]]) -> list[TransformSpec]:
    return [TransformSpec(text_type=str(tt), name=callable_label(fn)) for tt, fn in pairs]


def mirror_setup(captured: CapturedBot, base: TextFlowSetup | None = None) -> TextFlowSetup:
    """A setup that describes the captured bot, for display and for the stand-in knobs.

    The processing parts (aggregators, transforms, filters, RTVI parameters) are
    descriptions of the bot's own objects, which a run with ``bot`` set uses
    directly; the stand-in knobs (word timestamps, delivery, tokenization,
    chunking) come from ``base`` when given, or from what the TTS service's class
    suggests.

    Args:
        captured: The captured bot.
        base: A setup whose stand-in knobs and sample text to keep.

    Returns:
        The mirrored setup, with ``bot`` set to the bot file.
    """
    notes = list(captured.summary.notes)
    path = Path(captured.summary.path)
    tts = captured.tts
    ltp = captured.llm_text_processor

    base_tts = base.tts if base is not None else None
    if tts is not None:
        word_timestamps = (
            not tts._push_text_frames if base_tts is None else base_tts.word_timestamps
        )
        delivery = (
            ("websocket" if isinstance(tts, WebsocketTTSService) else "http")
            if base_tts is None
            else base_tts.delivery
        )
        filters = list(tts._text_filters)
        tts_spec = TTSSpec(
            word_timestamps=word_timestamps,
            delivery=delivery,  # type: ignore[arg-type]
            aggregation_mode=str(tts._text_aggregation_mode),  # type: ignore[arg-type]
            aggregator=_aggregator_spec(tts._text_aggregator, notes) if ltp is None else None,
            skip_text_types=list(tts._skip_text_types),
            transforms=_transform_specs(tts._text_transforms),
            markdown_filter=any(type(f).__name__ == "MarkdownTextFilter" for f in filters),
            append_trailing_space=tts._append_trailing_space,
            tokenization=base_tts.tokenization if base_tts is not None else TTSSpec().tokenization,
            live=base_tts.live if base_tts is not None else None,
        )
        other_filters = [
            type(f).__name__ for f in filters if type(f).__name__ != "MarkdownTextFilter"
        ]
        if other_filters:
            notes.append("TTS text filters run but not shown: " + ", ".join(other_filters))
    else:
        tts_spec = base_tts.model_copy() if base_tts is not None else TTSSpec()

    ltp_spec = None
    if ltp is not None:
        ltp_spec = LLMTextProcessorSpec(
            aggregator=_aggregator_spec(ltp._text_aggregator, notes),
            transforms=_transform_specs(ltp._text_transforms),
        )

    params = captured.rtvi_params
    if params is not None:
        rtvi_spec = RTVISpec(
            enabled=base.rtvi.enabled if base is not None else True,
            skip_text_types=list(params.skip_text_types or []),
            transforms=_transform_specs(list(params.bot_output_transforms or [])),
            bot_llm=params.bot_llm_enabled,
            bot_tts=params.bot_tts_enabled,
        )
    else:
        rtvi_spec = RTVISpec(enabled=False)

    captured.summary.notes = notes
    return TextFlowSetup(
        name=base.name if base is not None else path.stem,
        description=base.description if base is not None else f"Captured from {path}",
        bot=str(path),
        llm_text_processor=ltp_spec,
        tts=tts_spec,
        rtvi=rtvi_spec,
        chunking=base.chunking if base is not None else TextFlowSetup().chunking,
        sample_text=base.sample_text if base is not None else TextFlowSetup().sample_text,
    )


async def setup_for_bot(path: str | Path) -> tuple[TextFlowSetup, BotSummary]:
    """Capture a bot file and describe it as a setup.

    Args:
        path: The bot file.

    Returns:
        The mirrored setup, with ``bot`` set, and the summary of what was found.
    """
    async with await capture_bot(path) as captured:
        return mirror_setup(captured), captured.summary
