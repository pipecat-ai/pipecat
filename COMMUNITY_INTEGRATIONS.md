# Community Integrations Guide

Pipecat welcomes community-maintained integrations! As our ecosystem grows, we've established a process for any developer to create and maintain their own service integrations while ensuring discoverability for the Pipecat community.

## Overview

**What we support:** Community-maintained integrations that live in separate repositories and are maintained by their authors.

**What we don't do:** The Pipecat team does not code review, test, or maintain community integrations. We provide guidance and list approved integrations for discoverability.

**Why this approach:** This allows the community to move quickly while keeping the Pipecat core team focused on maintaining the framework itself.

## Submitting your Integration

To be listed as an official community integration, follow these steps:

### Step 1: Build Your Integration

Create your integration following the patterns and examples shown in the "Integration Patterns and Examples" section below.

### Step 2: Set Up Your Repository

Your repository must contain these components:

- **Source code** - Complete implementation following Pipecat patterns
- **Foundational example** - Single file example showing basic usage (see [Pipecat examples](https://github.com/pipecat-ai/pipecat/tree/main/examples))
- **README.md** - Must include:
  - Introduction and explanation of your integration
  - Installation instructions
  - Usage instructions with Pipecat Pipeline
  - How to run your example
  - Pipecat version compatibility (e.g., "Tested with Pipecat v0.0.86")
  - Company attribution: If you work for the company providing the service, please mention this in your README. This helps build confidence that the integration will be actively maintained.

- **LICENSE** - Permissive license (BSD-2 like Pipecat, or equivalent open source terms)
- **Code documentation** - Source code with docstrings (we recommend following [Pipecat's docstring conventions](https://github.com/pipecat-ai/pipecat/blob/main/CONTRIBUTING.md#docstring-conventions))
- **Changelog** - Maintain a changelog for version updates

### Step 3: Join Discord

Join our Discord: https://discord.gg/pipecat

### Step 4: Submit Your Documentation

Community integrations are documented alongside the core services on the [Supported Services page](https://docs.pipecat.ai/api-reference/server/services/supported-services), with a `Community` maintainer badge and their own service page. Submitting your docs means two things: a row on the Supported Services page and a dedicated service page.

**To submit:**

1. Fork the [Pipecat docs repository](https://github.com/pipecat-ai/docs)
2. **Add a row to the Supported Services page.** Edit `api-reference/server/services/supported-services.mdx` and add your integration to the appropriate category table:
   - **Service** — your service name, linked to your new service page (see next step)
   - **Setup** — the install command (e.g. `uv add pipecat-yourservice`)
   - **Maintainer** — `Community`
3. **Add a service page.** Create `api-reference/server/services/<category>/<your-service>.mdx`. The easiest path is to copy an existing community page (e.g. `image-generation/replicate.mdx`) and adapt it. Each page should include:
   - The **community-maintained badge** at the top, via the shared snippet:
     ```mdx
     import { CommunityMaintained } from "/snippets/community-maintained.mdx";

     <CommunityMaintained
       maintainer="your-github-username"
       maintainerUrl="https://github.com/your-github-username"
       repo="https://github.com/your-org/pipecat-yourservice"
     />
     ```
   - A short **overview** describing what the integration does (you can adapt the intro from your README)
   - **Installation** — your install command
   - **Prerequisites** — required accounts, API keys, and environment variables
   - **Configuration** — constructor parameters and runtime `Settings`
   - A minimal **usage** example showing the service in a pipeline
   - A **compatibility** note with the last tested Pipecat version
4. **Register the page in navigation.** Add the page path to `docs.json` under the matching `navigation` group, and add a redirect entry following the existing pattern.
5. Include a link to your demo video (approx 30-60 seconds) in your PR description showing:
   - Core functionality of your integration
   - Handling of an interruption (if applicable to service type)
6. Submit your pull request

Keep your service page lightweight: it should point readers to your repository as the source of truth, not duplicate your full README. Once your PR is submitted, post in the `#community-integrations` Discord channel to let us know.

## Integration Patterns and Examples

### STT (Speech-to-Text) Services

#### Websocket-based Services

**Base class:** `WebsocketSTTService`

**Use for:** Services where you manage the websocket connection directly. Combines `STTService` with `WebsocketService` for automatic reconnection and keepalive support.

**Examples:**

- [CartesiaSTTService](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/cartesia/stt.py)
- [ElevenLabsRealtimeSTTService](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/elevenlabs/stt.py)

#### SDK-based Streaming Services

**Base class:** `STTService`

**Use for:** Streaming services where the provider's Python SDK manages the connection internally.

**Examples:**

- [DeepgramSTTService](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/deepgram/stt.py)
- [GoogleSTTService](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/google/stt.py)

#### File-based Services

**Base class:** `SegmentedSTTService`

**Examples:**

- [NvidiaSTTService](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/nvidia/stt.py)
- [FalSTTService](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/fal/stt.py)

#### Key requirements:

- STT services should push `InterimTranscriptionFrames` and `TranscriptionFrames`
- If confidence values are available, filter for values >50% confidence

### LLM (Large Language Model) Services

#### OpenAI-Compatible Services

**Base class:** `OpenAILLMService`

**Examples:**

- [AzureLLMService](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/azure/llm.py)
- [GrokLLMService](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/grok/llm.py) - Shows overriding the base class where needed

#### Non-OpenAI Compatible Services

**Requires:** Full implementation

**Examples:**

- [AnthropicLLMService](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/anthropic/llm.py)
- [GoogleLLMService](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/google/llm.py)

#### Key requirements:

- **`_process_context(self, context: LLMContext)`** — The main method that processes an LLM context and generates a response. Each LLM service overrides `process_frame` to extract context from `LLMContextFrame` and calls `_process_context`.

- **`adapter_class`** — Class attribute pointing to a `BaseLLMAdapter` subclass. Defaults to `OpenAILLMAdapter`. Non-OpenAI services must implement their own adapter (see `src/pipecat/adapters/base_llm_adapter.py`) with methods:
  - `get_llm_invocation_params(context)` — Extract provider-specific params from universal context
  - `to_provider_tools_format(tools_schema)` — Convert standard tools to provider format
  - `get_messages_for_logging(context)` — Format messages for logging
  - Reference adapters: `src/pipecat/adapters/services/` (anthropic, gemini, bedrock, etc.)

- **Frame sequence:** Output must follow this frame sequence pattern:
  - `LLMFullResponseStartFrame` — Signals the start of an LLM response
  - `LLMTextFrame` — Contains LLM content, typically streamed as tokens
  - `LLMFullResponseEndFrame` — Signals the end of an LLM response

- **Thought frames (reasoning models):** If the model supports extended thinking / chain-of-thought, emit thought frames alongside the response:
  - `LLMThoughtStartFrame` — Signals the start of a thought
  - `LLMThoughtTextFrame` — Contains thought content, streamed as tokens
  - `LLMThoughtEndFrame` — Signals the end of a thought

- **Context aggregation** is handled by the framework via `LLMContext` + `LLMContextAggregatorPair`. The LLM service just processes context it receives — no need to implement aggregators.

### TTS (Text-to-Speech) Services

#### WebsocketTTSService

**Use for:** Websocket-based streaming services (with or without word timestamps)

**Examples:**

- [CartesiaTTSService](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/cartesia/tts.py)
- [ElevenLabsTTSService](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/elevenlabs/tts.py)

#### InterruptibleTTSService

**Use for:** Websocket-based services without word timestamps that reconnect on interruption (e.g. don't support a context ID or interruption message)

**Example:**

- [SarvamTTSService](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/sarvam/tts.py)

#### TTSService

**Use for:** HTTP-based services (word timestamps are supported in the base class)

**Examples:**

- [GoogleHttpTTSService](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/google/tts.py)
- [OpenAITTSService](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/openai/tts.py)

#### Key requirements:

- For websocket services, use asyncio WebSocket implementation
- Handle idle service timeouts with keepalives
- TTS services push both audio (`TTSAudioRawFrame`) and text (`TTSTextFrame`) frames

### Telephony Serializers

Pipecat supports telephony provider integration using websocket connections to exchange MediaStreams. These services use a FrameSerializer to serialize and deserialize inputs from the FastAPIWebsocketTransport.

**Examples:**

- [Twilio](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/serializers/twilio.py)
- [Telnyx](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/serializers/telnyx.py)

#### Key requirements:

- Include hang-up functionality using the provider's native API, ideally using `aiohttp`
- Support DTMF (dual-tone multi-frequency) events if the provider supports them:
  - Deserialize DTMF events from the provider's protocol to `InputDTMFFrame`
  - Use `KeypadEntry` enum for valid keypad entries (0-9, \*, #, A-D)
  - Handle invalid DTMF digits gracefully by returning `None`

### Image Generation Services

**Base class:** `ImageGenService`

**Examples:**

- [FalImageGenService](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/fal/image.py)
- [GoogleImageGenService](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/google/image.py)

#### Key requirements:

- Must implement `run_image_gen` method returning an `AsyncGenerator`

### Vision Services

Vision services process images and provide analysis such as descriptions, object detection, or visual question answering.

**Base class:** `VisionService`

**Example:**

- [MoondreamVisionService](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/moondream/vision.py)

#### Key requirements:

- Must implement `run_vision` method that takes a `UserImageRawFrame` and returns an `AsyncGenerator[Frame, None]`
- The method processes the image frame and yields frames with analysis results
- Must yield the frame sequence: `VisionFullResponseStartFrame`, `VisionTextFrame`, `VisionFullResponseEndFrame`

## Implementation Guidelines

### Naming Conventions

#### Package and Repository Naming

Use the `pipecat-{vendor}` naming convention for your PyPI package and repository:

- `pipecat-{vendor}` — for single-service integrations (e.g., `pipecat-deepdub`)
- `pipecat-{vendor}-{type}` — when a vendor offers multiple service types (e.g., `pipecat-upliftai-stt`, `pipecat-upliftai-tts`)

This convention makes community packages easily discoverable via PyPI search and clearly identifies them as part of the Pipecat ecosystem.

#### Class Naming

- **STT:** `VendorSTTService`
- **LLM:** `VendorLLMService`
- **TTS:**
  - Websocket: `VendorTTSService`
  - HTTP: `VendorHttpTTSService`
- **Image:** `VendorImageGenService`
- **Vision:** `VendorVisionService`
- **Telephony:** `VendorFrameSerializer`

### Metrics Support

Enable metrics in your service:

```python
def can_generate_metrics(self) -> bool:
    """Check if this service can generate processing metrics.

    Returns:
        True, as this service supports metrics.
    """
    return True
```

### Service Settings

Every AI service (STT, LLM, TTS, image generation, etc.) exposes a **Settings dataclass** that serves two roles:

1. **Store mode** — the service's `self._settings` holds the current value of every runtime-updatable field.
2. **Delta mode** — an update frame (e.g. `TTSUpdateSettingsFrame`) specifies only the fields that should change; unspecified fields remain `NOT_GIVEN`.

#### Defining your Settings class

Extend `STTSettings`, `TTSSettings`, `LLMSettings`, or `ImageGenSettings` (or, if your service directly subclasses `AIService`, `ServiceSettings`). The base classes already provide common fields (e.g. `model`, `voice`, `language`). You only need to add **service-specific knobs that should be runtime-updatable**:

```python
from dataclasses import dataclass, field

from pipecat.services.settings import TTSSettings, NOT_GIVEN

@dataclass
class MyTTSSettings(TTSSettings):
    """Settings for MyTTS service.

    Parameters:
        speaking_rate: Speed multiplier (0.5–2.0).
    """

    speaking_rate: float | None = field(default_factory=lambda: NOT_GIVEN)
```

**What goes in Settings vs. `__init__` params:**

| Belongs in Settings                                      | Stays as `__init__` params                |
| -------------------------------------------------------- | ----------------------------------------- |
| Model name, voice, language                              | API keys, auth tokens                     |
| Service-specific tuning knobs (rate, pitch, temperature) | Base URLs, endpoint overrides             |
| Anything users may want to change mid-session            | Audio encoding, sample format             |
|                                                          | Connection parameters (timeouts, retries) |

The rule of thumb: if a caller might send an update frame to change it at runtime, it belongs in Settings. Everything else is init-only config stored as `self._xxx`.

#### Wiring settings into `__init__`

Accept an **optional** `settings` parameter. Build a `default_settings` object with all fields set to real values, then merge any caller overrides with `apply_update`.

Add a `Settings` **class attribute** that points to your settings dataclass. This lets callers access the settings class through the service itself (e.g. `MyTTSService.Settings(...)`) without a separate import:

```python
from typing import Optional

class MyTTSService(TTSService):
    Settings = MyTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        # 1. Defaults — every field has a real value (store mode).
        default_settings = self.Settings(
            model="my-model-v1",
            voice="default-voice",
            language="en",
            speaking_rate=1.0,
        )

        # 2. Merge caller overrides (only given fields win).
        if settings is not None:
            default_settings.apply_update(settings)

        # 3. Pass the fully-populated settings to the base class.
        super().__init__(settings=default_settings, **kwargs)

        # 4. Init-only config stored separately.
        self._api_key = api_key
```

This pattern lets callers override only what they care about:

```python
# Uses all defaults
svc = MyTTSService(api_key="sk-xxx")

# Overrides just the voice — access Settings through the service class
svc = MyTTSService(
    api_key="sk-xxx",
    settings=MyTTSService.Settings(voice="custom-voice"),
)
```

#### Reacting to runtime changes

AI services support runtime configuration changes via `*UpdateSettingsFrame`s (e.g. `STTUpdateSettingsFrame`, `TTSUpdateSettingsFrame`, `LLMUpdateSettingsFrame`).

To react to runtime setting changes, override `_update_settings`. The base implementation applies the delta to `self._settings` and returns a `dict` mapping each changed field name to its **pre-update** value. Your override should call `super()` first, then act on the changed fields. A common implementation might look like:

```python
async def _update_settings(self, update: TTSSettings) -> dict[str, Any]:
    """Apply a settings update, reconfiguring the connection if needed."""
    changed = await super()._update_settings(update)

    if not changed:
        return changed

    await self._disconnect()
    await self._connect()

    return changed
```

The dict keys work like a set for membership tests (`"language" in changed`) and truthiness (`if changed`). Use `changed.keys() - {"language"}` for set difference, or `changed["language"]` to inspect the previous value of a field.

Note that, in this example, the service requires a reconnect to apply the new language. Consider, for each setting, whether your service requires reconnection or can apply changes in-place.

If your service can't yet apply certain settings at runtime, call `self._warn_unhandled_updated_settings(changed)` with any unhandled field names so users get a clear log message:

```python
async def _update_settings(self, update: TTSSettings) -> dict[str, Any]:
    changed = await super()._update_settings(update)

    if not changed:
        return changed

    if "language" in changed:
        await self._update_language()
    else:
        # TODO: this should be temporary - handle changes to other settings soon!
        self._warn_unhandled_updated_settings(changed.keys() - {"language"})

    return changed
```

### Service Metadata

A service can describe itself to the rest of the pipeline **at start** by overriding `service_metadata_frame()`. The service broadcasts the returned frame (a `ServiceMetadataFrame` or subtype) right after the `StartFrame`, and processors that care read it to **auto-configure themselves** — so users get correct behavior without wiring it by hand.

Service metadata is a small, growing surface: only a few fields exist today, but the frame types are meant to gain more as more of the framework becomes self-configuring.

The base `ServiceMetadataFrame` carries:

- `service_name` — the broadcasting service's name.
- `user_turn_strategies` — turn strategies the service recommends. A service that does its own **server-side end-of-turn detection** returns `ExternalUserTurnStrategies()` here; the user aggregator then defers to the service's turn frames instead of running local VAD/smart-turn. The recommendation applies **unless the user passed their own `user_turn_strategies`**, which always wins. `None` leaves the defaults in place.

Subtypes add fields for their service kind:

- **`STTMetadataFrame`** — `ttfs_p99_latency`, the 99th-percentile time from end-of-speech to final transcript, which turn strategies use to tune their timing.
- **`LLMServiceMetadataFrame`** — `is_realtime_service`, flagging a realtime (speech-to-speech) LLM so the context aggregator auto-enables realtime mode.

The base `STTService` and `LLMService` already return a sensible frame, so most services need nothing here. Override `service_metadata_frame()` — calling `super()` and adjusting the frame — only when your service has something extra to declare, e.g. an STT that detects turns server-side:

```python
from pipecat.frames.frames import STTMetadataFrame
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies

def service_metadata_frame(self) -> STTMetadataFrame:
    # This service defines turn boundaries server-side and emits
    # UserStarted/StoppedSpeakingFrame, so recommend external turn
    # strategies; the user aggregator defers to those over local VAD.
    frame = super().service_metadata_frame()
    frame.user_turn_strategies = ExternalUserTurnStrategies()
    return frame
```

### Sample Rate Handling

Sample rates are set via PipelineParams and passed to each frame processor at initialization. The pattern is to _not_ set the sample rate value in the constructor of a given service. Instead, use the `start()` method to initialize sample rates from the frame:

```python
async def start(self, frame: StartFrame):
    """Start the service."""
    await super().start(frame)
    self._settings.output_sample_rate = self.sample_rate
    await self._connect()
```

Note that `self.sample_rate` is a `@property` set in the TTSService base class, which provides access to the private sample rate value obtained from the StartFrame.

### Tracing Decorators

Use Pipecat's tracing decorators:

- **STT:** `@traced_stt` - decorate `_handle_transcription(self, transcript, is_final, language)` (the standard method name convention)
- **LLM:** `@traced_llm` - decorate the `_process_context()` method
- **TTS:** `@traced_tts` - decorate the `run_tts()` method

## Processor Lifecycle: setup, cleanup, start, stop, cancel

Frame processors have two kinds of lifecycle hooks: a framework-driven pair that
always runs, and a frame-driven set that may be skipped. Putting setup or
teardown logic in the wrong one is a common source of resource leaks, so follow
this rule.

### The hooks

**Framework-driven, guaranteed: `setup()` and `cleanup()`.** Both are defined on
`FrameProcessor`, so **every** processor has them. The pipeline calls them
directly on each processor — `setup()` once before the pipeline runs
(`Pipeline._setup_processors()`) and `cleanup()` once at teardown
(`Pipeline._cleanup_processors()`) — independent of frame flow. They are the only
hooks **guaranteed** to run, no matter how the pipeline started or ended.
`setup()` acquires, `cleanup()` releases: they are counterparts, so whatever
`setup()` allocates, `cleanup()` must be able to release.

**Frame-driven, skippable: `start(StartFrame)`, `stop(EndFrame)`, and
`cancel(CancelFrame)`.** These are **not** part of `FrameProcessor`. They are
conventions provided by some base classes (`AIService` and its subclasses,
`BaseInputTransport`, `BaseOutputTransport`), each of which dispatches to them
from its own `process_frame`. A plain `FrameProcessor` (an aggregator, a filter)
has none of them; it has only `process_frame`, `setup()`, and `cleanup()`.
`start()` initializes when the `StartFrame` arrives; `stop()` and `cancel()` tear
down. Because they run only when the corresponding frame reaches the processor, a
processor that never receives the frame never runs them.

Some processors handle these frames inline instead, with
`isinstance(frame, CancelFrame)` (or `(EndFrame, CancelFrame)`) branches in
`process_frame`. For this rule, that is equivalent to a `cancel()`/`stop()`
override.

`stop` (EndFrame) and `cancel` (CancelFrame) differ in urgency: `EndFrame` is a
control frame processed in order, so `stop()` runs after pending frames drain
(graceful, "finish then stop"); `CancelFrame` is a system frame processed
immediately ahead of the queue, so `cancel()` runs at once and discards pending
work ("stop now").

### The rule

**Initialization.** `setup()` runs before any frames, so it cannot see runtime
pipeline configuration such as the sample rate carried by the `StartFrame`.
Acquire config-independent resources there. Initialization that needs the
negotiated configuration — opening a connection sized to the pipeline sample
rate, for example — belongs in `start()`, which receives the `StartFrame`.

**Teardown.** Decide where each teardown action goes with two questions:

1. **Must it happen on every exit path?** Releasing resources (closing sockets,
   releasing clients, cancelling tasks you created with `self.create_task()`,
   deleting temp files) must. Put it in `cleanup()` and make it idempotent.
   `cleanup()` is guaranteed; `stop()`/`cancel()`/inline branches are not (the
   frame can be filtered, swallowed, or never reach the processor), so they must
   never be the *only* place a resource is released.

2. **Must it happen promptly, before the queue drains?** Stopping active output
   (cancelling the task still generating audio, telling the transport to stop
   sending) must, or the bot keeps talking until teardown. Do that in `cancel()`
   (and, for graceful shutdown, `stop()`), *in addition to* `cleanup()`. Because
   the release/cancel is idempotent, doing it in both places is safe.

   This includes shutting down an **independent producer**. A websocket or gRPC
   receive loop runs on its own task and keeps delivering data until you
   disconnect it, so stopping only the consumer side (for example the task that
   drains decoded audio) is not enough: the producer keeps reading until
   teardown. Disconnect the producer in `cancel()`/`stop()`, then repeat it in
   `cleanup()`.

In short: **`cleanup()` owns the complete, guaranteed teardown; `cancel()` does
the time-sensitive subset early.** Do not call `cleanup()` from `cancel()` or
`stop()`: the framework already calls it separately, so doing both runs the logic
twice. (A plain helper object owned by a processor, not itself a
`FrameProcessor`, has no framework-driven `cleanup()`, so it is fine for its
`stop()`/`cancel()` to delegate to its own idempotent `cleanup()`.)

`FrameProcessor.cleanup()` cancels only the processor's *internal* input/process
tasks. Tasks you create with `self.create_task()` are yours to cancel, so they
belong in your `cleanup()` override.

**Centralize shared teardown.** When more than one hook needs the same work (for
example `cancel()`, `stop()`, and `cleanup()` all closing the same connection),
put that work in a single idempotent private method and have each hook call it.
Never copy the body into more than one hook: if a later change updates one copy
and misses another, the processor leaks. Each hook stays a thin wrapper that
calls `super().<hook>()` (which differs per hook and is required) and then the
shared helper. Reuse an existing helper (`_disconnect()`, `_stop_tasks()`) when
there is one; if the hooks share a superset of it, extract that superset into its
own method and leave the lower-level helper for the paths that reuse it (such as
a reconnect).

### Example

```python
class MyService(AIService):
    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._teardown()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._teardown()

    async def cleanup(self):
        await super().cleanup()
        await self._teardown()

    async def _teardown(self):
        # One idempotent teardown body shared by all three hooks. _disconnect()
        # also cancels the receive-loop task (an independent producer), which is
        # why it must run on the prompt cancel()/stop() paths, not only here.
        await self._disconnect()
```

A plain `FrameProcessor` has only `setup()` and `cleanup()`, so its custom tasks
go there:

```python
class MyAggregator(FrameProcessor):
    async def cleanup(self):
        await super().cleanup()
        await self._cancel_my_task()  # the only teardown hook it has
```

### Exception: serializers

`FrameSerializer` is not a `FrameProcessor` and has no `setup()`/`cleanup()`.
Serializers that act on `EndFrame`/`CancelFrame` (for example telephony
serializers sending a provider disconnect message) can only do so on the frame
path. That is expected: there is no guaranteed hook to move them to.

## Best Practices

### Packaging and Distribution

- Name your package `pipecat-{vendor}` (see [Naming Conventions](#naming-conventions))
- Use [uv](https://docs.astral.sh/uv/) for packaging (encouraged)
- Publish to PyPI for easier installation
- Follow semantic versioning principles
- Maintain a changelog

### HTTP Communication

For REST-based communication, use aiohttp. Pipecat includes this as a required dependency, so using it prevents adding an additional dependency to your integration.

### Error Handling

- Wrap API calls in appropriate try/catch blocks
- Handle rate limits and network failures gracefully
- Provide meaningful error messages
- When errors occur, raise exceptions AND push errors to notify the pipeline:

```python
try:
    # Your API call
    result = await self._make_api_call()
except Exception as e:
    # Push error upstream to notify the pipeline
    await self.push_error(f"{self} error: {e}", exception=e)
    # Raise or handle as appropriate
    raise
```

### Testing

- Your foundational example serves as a valuable integration-level test
- Use the behavioral eval harness (`pipecat eval run`) to test your foundational example end-to-end; [see the docs](https://docs.pipecat.ai/pipecat/evals/overview) for more details
- Unit tests are nice to have

## Disclaimer

Community integrations are community-maintained and not officially supported by the Pipecat team. Users should evaluate these integrations independently. The Pipecat team reserves the right to remove listings that become unmaintained or problematic.

## Staying Up to Date

Pipecat evolves rapidly to support the latest AI technologies and patterns. While we strive to minimize breaking changes, they do occur as the framework matures.

**We strongly recommend:**

- Join our Discord at https://discord.gg/pipecat and monitor the `#announcements` channel for release notifications
- Follow our changelog: https://github.com/pipecat-ai/pipecat/blob/main/CHANGELOG.md
- Test your integration against new Pipecat releases promptly
- Update your README with the last tested Pipecat version

This helps ensure your integration remains compatible and your users have clear expectations about version support.

## Questions?

Join our Discord community at https://discord.gg/pipecat and post in the `#community-integrations` channel for guidance and support.

For additional questions, you can also reach out to us at pipecat-ai@daily.co.
