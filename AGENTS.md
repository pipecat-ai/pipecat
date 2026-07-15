# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Project Overview

Pipecat is an open-source Python framework for building real-time voice and multimodal conversational AI agents. It orchestrates audio/video, AI services, transports, and conversation pipelines using a frame-based architecture.

## Common Commands

```bash
# Setup development environment
uv sync --group dev --all-extras --no-extra gstreamer --no-extra local

# Install pre-commit hooks
uv run pre-commit install

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_name.py

# Run a specific test
uv run pytest tests/test_name.py::test_function_name

# Preview changelog
uv run towncrier build --draft --version Unreleased

# Run a behavioral eval scenario against a running bot (bot started with `-t eval`)
pipecat eval run scenarios/<name>.yaml --bot-url ws://localhost:7860

# Run the full release-eval suite (spawns bots from a manifest, runs scenarios in parallel)
pipecat eval suite scripts/release-evals/manifest.yaml -p <bot-pattern> -s <scenario>

# Lint and format check
uv run ruff check
uv run ruff format --check

# Update dependencies (after editing pyproject.toml)
uv lock && uv sync
```

## Architecture

### Frame-Based Pipeline Processing

All data flows as **Frame** objects through a pipeline of **FrameProcessors**:

```
[Processor1] → [Processor2] → ... → [ProcessorN]
```

**Key components:**

- **Frames** (`src/pipecat/frames/frames.py`): Data units (audio, text, video) and control signals. Flow DOWNSTREAM (input→output) or UPSTREAM (acknowledgments/errors).

- **FrameProcessor** (`src/pipecat/processors/frame_processor.py`): Base processing unit. Each processor receives frames, processes them, and pushes results downstream.

- **Pipeline** (`src/pipecat/pipeline/pipeline.py`): Chains processors together.

- **ParallelPipeline** (`src/pipecat/pipeline/parallel_pipeline.py`): Runs multiple pipelines in parallel.

- **Transports** (`src/pipecat/transports/`): Transports are frame processors used for external I/O layer (Daily WebRTC, LiveKit WebRTC, WebSocket, Local). Abstract interface via `BaseTransport`, `BaseInputTransport` and `BaseOutputTransport`.

- **Workers (`src/pipecat/workers/`)**: A **worker** is the top-level runnable unit (it replaces the old "pipeline task"). `BaseWorker` (`src/pipecat/workers/base_worker.py`) is the abstract base: it owns activation, end/cancel, bus subscription, job RPC, and a default `run()` for bus-only workers. `WorkerParams` configures it. Specialized workers live under `src/pipecat/workers/llm/` (`LLMWorker` adds `@tool` collection; `LLMContextWorker` adds an `LLMContext` plus aggregator pair) and `src/pipecat/workers/ui/` (`UIWorker`).

- **Pipeline Worker (`src/pipecat/pipeline/worker.py`)**: `PipelineWorker` is the `BaseWorker` subclass that wraps a user-defined pipeline — the common single-bot case. It sends the first frame, `StartFrame`, so processors know they can start processing and pushing frames, and internally wraps the pipeline with a source processor (before the pipeline) and a sink processor (at the end) used for error handling, worker-level events, heartbeat monitoring, etc. Optional `bridged=` adds bus edge processors so the pipeline can exchange frames with other workers. `PipelineTask` is a deprecated (1.3.0) alias — construct `PipelineWorker` instead.

- **Worker Runner (`src/pipecat/workers/runner.py`)**: `WorkerRunner` is the high-level entry point for executing workers. It owns the shared `WorkerBus` and `WorkerRegistry` and handles signal management (SIGINT/SIGTERM) for graceful shutdown. Register workers with `await runner.add_workers(*workers)`, then `await runner.run()`. By default (`auto_end=True`) the runner ends once every root worker finishes — so a single-pipeline bot ends when its pipeline does; pass `auto_end=False` for long-lived hosts (e.g. a FastAPI server) that add/remove workers across sessions. `PipelineRunner`, and passing a worker directly to `run(worker)`, are deprecated (1.3.0).

- **Services** (`src/pipecat/services/`): 60+ AI provider integrations (STT, TTS, LLM, etc.). Extend base classes: `AIService`, `LLMService`, `STTService`, `TTSService`, `VisionService`.

- **Serializers** (`src/pipecat/serializers/`): Convert frames to/from wire formats for WebSocket transports. `FrameSerializer` base class defines `serialize()` and `deserialize()`. Telephony serializers (Twilio, Plivo, Vonage, Telnyx, Exotel, Genesys) handle provider-specific protocols and audio encoding (e.g., μ-law).

- **RTVI** (`src/pipecat/processors/frameworks/rtvi.py`): Real-Time Voice Interface protocol bridging clients and the pipeline. `RTVIProcessor` handles incoming client messages (text input, audio, DTMF keypresses, function call results). `RTVIObserver` converts pipeline frames to outgoing messages: user/bot speaking events, transcriptions, LLM/TTS lifecycle, function calls, metrics, and audio levels.

- **Observers** (`src/pipecat/observers/`): Monitor frame flow without modifying the pipeline. Passed to `PipelineWorker` via the `observers` parameter. Implement `on_process_frame()` and `on_push_frame()` callbacks.

### Workers, Bus, and Jobs

Beyond the single-pipeline case, Pipecat supports multiple cooperating **workers** coordinated by a shared bus (folded in from the former `pipecat-subagents` package). Terminology note: a "worker" is a runnable unit, "task" now refers only to asyncio tasks, and cross-worker RPC uses "jobs" and "job groups".

- **Bus** (`src/pipecat/bus/`): `WorkerBus` is the abstract pub/sub message bus; `AsyncQueueBus` (`bus/local/async_queue.py`) is the default in-process implementation, with `PgmqBus` / `RedisBus` (`bus/network/`) for distributed workers. Typed messages live in `bus/messages.py` (a `BusMessage` hierarchy split into normal-priority data messages and high-priority system messages, covering frame transport, worker lifecycle, and job RPC). `BusBridgeProcessor` (`bus/bridge_processor.py`) bridges a pipeline to the bus; `BusSubscriber` is the receive-side mixin.

- **Registry** (`src/pipecat/registry/`): `WorkerRegistry` tracks local and remote workers. The runner manages registration; code uses `watch(name, handler)` (or the `@worker_ready(name=...)` decorator) to be notified when a named worker becomes ready.

- **Jobs** (`src/pipecat/pipeline/job_context.py`, `job_decorator.py`, `worker_ready_decorator.py`): A worker exposes handlers with `@job(name=..., sequential=...)`. A caller opens `async with self.job(worker_name, ...)` (single worker) or `self.job_group(*worker_names, ...)` (fan-out) to send a request and await `JobStatus` results / streamed updates over the bus.

- **LLM tools** (`src/pipecat/workers/llm/tool_decorator.py`): On an `LLMWorker`, methods marked `@tool` are auto-collected and registered with the LLM service (supports `cancel_on_interruption` and `timeout`).

Runnable examples live in `examples/multi-worker/` (local handoff, distributed handoff via pgmq/redis, parallel debate, remote proxy, UI worker).

### Important Patterns

- **Context Aggregation**: `LLMContext` accumulates messages for LLM calls; the aggregators created by `LLMContextAggregatorPair` keep it updated with user and assistant turns

- **Turn Management**: Turn management is done through `LLMUserAggregator` and
  `LLMAssistantAggregator`, created with `LLMContextAggregatorPair`

- **User turn strategies**: Detection of when the user starts and stops speaking is done via user turn start/stop strategies. They push `UserStartedSpeakingFrame` and `UserStoppedSpeakingFrame` respectively.

- **Interruptions**: Interruptions are usually triggered by a user turn start strategy (e.g. `VADUserTurnStartStrategy`), but any processor can trigger one by calling `await self.broadcast_interruption()`, which broadcasts an `InterruptionFrame` both upstream and downstream. The old `push_interruption_task_frame_and_wait()` is deprecated and delegates to `broadcast_interruption()`.

- **Uninterruptible Frames**: These are frames that will not be removed from internal queues even if there's an interruption. For example, `EndFrame` and `StopFrame`.

- **Events**: Most classes in Pipecat have `BaseObject` as the very base class. `BaseObject` has support for events. Events can run in the background in an async task (default) or synchronously (`sync=True`) if we want immediate action. Synchronous event handlers need to execute fast.

- **Async Task Management**: Always use `self.create_task(coroutine, name)` instead of raw `asyncio.create_task()`. The `TaskManager` automatically tracks tasks and cleans them up on processor shutdown. Use `await self.cancel_task(task, timeout)` for cancellation.

- **Error Handling**: Use `await self.push_error(msg, exception, fatal)` to push errors upstream. Services should use `fatal=False` (the default) so application code can handle errors and take action (e.g. switch to another service).

- **Accessing the worker**: Reach the running worker via the `pipeline_worker` property on `FrameProcessor` and the `pipeline_worker` field on `FunctionCallParams`. Both are required once the processor is set up (the property raises if accessed before setup). The old `pipeline_task` accessor is deprecated (1.3.0).

### Key Directories

| Directory                  | Purpose                                            |
| -------------------------- | -------------------------------------------------- |
| `src/pipecat/frames/`      | Frame definitions (100+ types)                     |
| `src/pipecat/processors/`  | FrameProcessor base + aggregators, filters, audio  |
| `src/pipecat/pipeline/`    | Pipeline orchestration; PipelineWorker + job RPC   |
| `src/pipecat/workers/`     | Worker model: BaseWorker, runner, LLM/UI workers   |
| `src/pipecat/bus/`         | Inter-worker message bus (local + pgmq/redis)      |
| `src/pipecat/registry/`    | Worker registry (local + remote tracking)          |
| `src/pipecat/services/`    | AI service integrations (60+ providers)            |
| `src/pipecat/transports/`  | Transport layer (Daily, LiveKit, WebSocket, Local) |
| `src/pipecat/serializers/` | Frame serialization for WebSocket protocols        |
| `src/pipecat/observers/`   | Pipeline observers for monitoring frame flow       |
| `src/pipecat/audio/`       | VAD, filters, mixers, turn detection, DTMF         |
| `src/pipecat/turns/`       | User turn management                               |
| `src/pipecat/adapters/`    | LLM provider adapters (context/tools conversion)   |
| `src/pipecat/runner/`      | Development runner (multi-transport bot hosting)   |
| `src/pipecat/cli/`         | `pipecat` CLI (`init`, `eval`)                     |
| `src/pipecat/evals/`       | Behavioral eval framework (run via `pipecat eval`) |
| `src/pipecat/metrics/`     | Metrics data models                                |

## Code Style

- **Docstrings**: Google-style. Classes describe purpose; `__init__` has `Args:` section; dataclasses use `Parameters:` section.
- **Deprecations**: Every deprecation needs a `.. deprecated:: <version>` directive in the docstring (never inline `[DEPRECATED]` tags) — it's the registry's source of truth. Its body must **lead with the replacement as the first reference** — `Use :class:`X` instead.` / `Moved to :mod:`X`.` / `Merged into :class:`X`.` — or state `No replacement.` explicitly; **never lead with a contextual reference** (the deprecated thing itself, a `DeprecationWarning`, or a related-but-not-replacement API), and don't rely on incidental words like "no longer" to signal no-replacement. Prefer Sphinx roles (`:class:`/`:meth:`/`:func:`/`:attr:`/`:mod:`) over plain backticks, but use a backtick when a role wouldn't resolve (aliases like `Service.Settings`, usage idioms, parameters). For the runtime warning: **classes, functions, methods, and properties** use the PEP 702 `@deprecated` decorator from `pipecat.utils.deprecation` with a string-literal message matching the canonical template — `` `Subject` is deprecated since X.Y.Z and will be removed in A.B.C. Use `Replacement` instead. `` — where the removal is a concrete version (e.g. `2.0.0`, never "a future release") and the tail is `No replacement.` when nothing replaces it. Parameters, module moves, and behavior/value changes can't use the decorator — call `warnings.warn(..., DeprecationWarning)` by hand. Enforced by `tests/test_deprecation_markers.py`; full conventions in `CONTRIBUTING.md`.
- **Linting**: Ruff (line length 100). Pre-commit hooks enforce formatting.
- **Type hints**: Required for complex async code.
- **Dataclass vs Pydantic**: Use `@dataclass` for frames and internal pipeline data (high-frequency, no validation needed). Use Pydantic `BaseModel` for configuration, parameters, metrics, and external API data (benefits from validation and serialization). Specifically:
  - `@dataclass`: Frame types, context aggregator pairs, internal data containers
  - `BaseModel`: Service `InputParams`, transport/VAD/turn params, metrics data, API request/response models, serializer params

### Docstring Example

```python
class MyService(LLMService):
    """Description of what the service does.

    More detailed description.

    Event handlers available:

    - on_connected: Called when we are connected

    Example::

        @service.event_handler("on_connected")
        async def on_connected(service, frame):
            ...
    """

    def __init__(self, param1: str, **kwargs):
        """Initialize the service.

        Args:
            param1: Description of param1.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(**kwargs)


# Pydantic params class with a deprecated field
class MyParams(BaseModel):
    """Configuration parameters for MyService.

    Parameters:
        new_setting: Replacement for ``old_setting``.
        old_setting: Legacy setting, no longer used.

            .. deprecated:: 1.2.0
                Use ``new_setting`` instead. Will be removed in 2.0.0.
    """

    new_setting: str = "default"
    old_setting: str | None = None
```

## Writing for Future Readers

This applies to everything that documents the code — comments, docstrings, commit messages, changelog entries, PR descriptions. Write for a future reader of the codebase, NOT for whoever is reviewing and collaborating on the work right now.

- **Leave the current moment out of it.** Detail that feels important while making a change — alternatives considered and not taken, what the code used to do, shorthand that only made sense while the work was in progress — usually isn't worth a future reader's time, and may not even make sense to them. Include it only when they genuinely need it to understand the code as it stands.
- **Match the weight of the prose to the code.** Keep it general, high-level, and concise. Reserve long comments for architecturally salient pieces, genuinely tricky sections, or decisions non-obvious enough that a reader would otherwise be puzzled. Routine code needs a short note or none at all.
- **Litmus test for commits, changelogs, and comments:** every sentence must answer "what does the code do now?" — not "what happened while we worked on it." Write comments as if the code had always existed in its current form: a comment earns its place only by stating a constraint or non-obvious decision the code can't show. If a sentence describes a superseded implementation, a rejected alternative, verification performed, or argues the change is correct, delete it. Rationale is allowed only as a trailing note when its absence would puzzle a future reader.

## Service Implementation

When adding a new service:

1. Extend the appropriate base class (`STTService`, `TTSService`, `LLMService`, etc.)
2. Implement required abstract methods
3. Handle necessary frames
4. By default, all frames should be pushed in the direction they came
5. Push `ErrorFrame` on failures
6. Add metrics tracking via `MetricsData` if relevant
7. Follow the pattern of existing services in `src/pipecat/services/`

## Testing

**Unit tests.** Test utilities live in `src/pipecat/tests/utils.py`. Use `run_test()` to send frames through a pipeline and assert expected output frames in each direction. Use `SleepFrame(sleep=N)` to add delays between frames.

**Behavioral evals.** `pipecat.evals` (`src/pipecat/evals/`) is a behavioral eval framework that drives a *real bot* end-to-end and asserts on its behavior — use it to confirm a feature works (interruptions, function calls, vision, multi-turn, transcription, DTMF) rather than only checking frame plumbing. The harness connects to a bot's **eval transport** as an RTVI client, plays scripted user turns (synthesizing audio in audio mode), and checks each expectation (latency, `text_contains`, an expected `function_call`, or an LLM judge of the bot's reply).

A scenario is a YAML file of `turns` with `expect:` assertions; scenarios are reusable across bots. To confirm a behavior while developing:

1. Run the bot with its eval transport: `python bot.py -t eval --port 7860`
2. Run a scenario against it: `pipecat eval run scenarios/<name>.yaml --bot-url ws://localhost:7860 -v`

For many bots at once, `pipecat eval suite <manifest.yaml>` spawns each bot and runs its scenarios in parallel. Reusable scenarios and the pre-release validation manifest live in `scripts/release-evals/` — see its `README.md` for the full workflow (prerequisites: a local Ollama judge `gemma2:9b`, plus Kokoro/Moonshine for audio mode) and the `pipecat.evals.scenario` module docstring for the complete scenario file format.
