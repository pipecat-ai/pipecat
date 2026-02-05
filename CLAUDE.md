# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pipecat is an open-source Python framework for building real-time voice and multimodal conversational AI agents. It orchestrates audio/video, AI services, transports, and conversation pipelines using a frame-based architecture.

## Common Commands

```bash
# Setup development environment
uv sync --group dev --all-extras --no-extra gstreamer --no-extra krisp

# Install pre-commit hooks
uv run pre-commit install

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_name.py

# Run a specific test
uv run pytest tests/test_name.py::test_function_name

# Preview changelog
towncrier build --draft --version Unreleased

# Update dependencies (after editing pyproject.toml)
uv lock && uv sync
```

## Architecture

### Frame-Based Pipeline Processing

All data flows as **Frame** objects through a pipeline of **FrameProcessors**:

```
Transport Input → Pipeline Source → [Processor1] → [Processor2] → ... → Pipeline Sink → Transport Output
```

**Key components:**

- **Frames** (`src/pipecat/frames/frames.py`): Data units (audio, text, video) and control signals. Flow DOWNSTREAM (input→output) or UPSTREAM (acknowledgments/errors).

- **FrameProcessor** (`src/pipecat/processors/frame_processor.py`): Base processing unit. Each processor receives frames, processes them, and pushes results downstream.

- **Pipeline** (`src/pipecat/pipeline/pipeline.py`): Chains processors together.

- **ParallelPipeline** (`src/pipecat/pipeline/parallel_pipeline.py`): Runs multiple pipelines in parallel.

- **Transports** (`src/pipecat/transports/`): External I/O layer (Daily WebRTC, LiveKit WebRTC, WebSocket, Local). Abstract interface via `BaseTransport`.

- **Services** (`src/pipecat/services/`): 60+ AI provider integrations (STT, TTS, LLM, etc.). Extend base classes: `AIService`, `LLMService`, `STTService`, `TTSService`, `VisionService`.

- **Serializers** (`src/pipecat/serializers/`): Convert frames to/from wire formats for WebSocket transports. `FrameSerializer` base class defines `serialize()` and `deserialize()`. Telephony serializers (Twilio, Plivo, Vonage, Telnyx, Exotel, Genesys) handle provider-specific protocols and audio encoding (e.g., μ-law).

- **RTVI** (`src/pipecat/processors/frameworks/rtvi.py`): Real-Time Voice Interface protocol bridging clients and the pipeline. `RTVIProcessor` handles incoming client messages (text input, audio, function call results). `RTVIObserver` converts pipeline frames to outgoing messages: user/bot speaking events, transcriptions, LLM/TTS lifecycle, function calls, metrics, and audio levels.

### Important Patterns

- **Context Aggregation**: `LLMContext` accumulates messages for LLM calls; `UserResponse` aggregates user input

- **Turn Management**: Turn management is done through `LLMUserAggregator` and
`LLMAssistantAggregator`, created with `LLMContextAggregatorPair`

- **User turn strategies**: Detection of when the user starts and stops speaking is done via user turn start/stop strategies. They push `UserStartedSpeakingFrame` and `UserStoppedSpeakingFrame` respectively.

- **Interruptions**: Interruptions are usually triggered by a user turn start strategy (e.g. `VADUserTurnStartStrategy`) but they can be triggered by other processors as well, in which case the user turn start strategies don't need to.

- **Uninterruptible Frames**: These are frames that will not be removed from internal queues even if there's an interruption. For example, `EndFrame` and `StopFrame`.

- **Events**: Most classes in Pipecat have `BaseObject` as the very base class. `BaseObject` has support for events. Events can run in the background in an async task (default) or synchronously (`sync=True`) if we want immediate action. Synchronous event handlers need to exectue fast.

### Key Directories

| Directory                 | Purpose                                            |
|---------------------------|----------------------------------------------------|
| `src/pipecat/frames/`     | Frame definitions (100+ types)                     |
| `src/pipecat/processors/` | FrameProcessor base + aggregators, filters, audio  |
| `src/pipecat/pipeline/`   | Pipeline orchestration                             |
| `src/pipecat/services/`   | AI service integrations (60+ providers)            |
| `src/pipecat/transports/` | Transport layer (Daily, LiveKit, WebSocket, Local) |
| `src/pipecat/serializers/`| Frame serialization for WebSocket protocols        |
| `src/pipecat/audio/`      | VAD, filters, mixers, turn detection, DTMF         |
| `src/pipecat/turns/`      | User turn management                               |

## Code Style

- **Docstrings**: Google-style. Classes describe purpose; `__init__` has `Args:` section; dataclasses use `Parameters:` section.
- **Linting**: Ruff (line length 100). Pre-commit hooks enforce formatting.
- **Type hints**: Required for complex async code.

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
```

## Changelog

Every user-facing PR needs a changelog fragment in `changelog/`:

```
changelog/<PR_number>.<type>.md
```

Types: `added`, `changed`, `deprecated`, `removed`, `fixed`, `security`, `other`

Content format (include the `-`):

```markdown
- Added support for new feature X.
```

Skip changelog for: documentation-only, internal refactoring, test-only, CI changes.

## Service Implementation

When adding a new service:

1. Extend the appropriate base class (`STTService`, `TTSService`, `LLMService`, etc.)
2. Implement required abstract methods
3. Handle necessary frames
4. By default, all frames should be pushed in the direction they came
5. Push `ErrorFrame` on failures
6. Add metrics tracking via `MetricsData` if relevant
7. Follow the pattern of existing services in `src/pipecat/services/`
