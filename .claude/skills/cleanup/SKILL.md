# Code Cleanup Skill

The **Code Cleanup Skill** reviews, refactors, and documents code changes in your current branch, ensuring alignment with **Pipecat’s architecture, coding standards, and example patterns**.
It focuses on **readability, correctness, performance, and consistency**, while avoiding breaking changes.

---

## Skill Overview

This skill analyzes all changes introduced in your branch and performs the following actions:

1. **Analyze Branch Changes**
   - Review uncommitted changes and outgoing commits
2. **Refactor for Readability**
   - Improve clarity, naming, structure, and modern Python usage
3. **Enhance Performance**
   - Identify safe, conservative optimization opportunities
4. **Add Documentation**
   - Apply Pipecat-style, Google-format docstrings
5. **Ensure Pattern Consistency**
   - Match existing Pipecat services, pipelines, and examples
6. **Validate Examples**
   - Ensure examples follow foundational patterns (e.g. `07-interruptible.py`)

---

## Usage

Invoke the skill using any of the following commands:

- “Clean up my branch code”
- “Refactor the changes in my branch”
- “Review and improve my branch code”
- `/cleanup`

---

## What This Skill Does

### 1. Analyze Branch Changes

The skill retrieves all uncommitted changes and outgoing commits to understand:

- New files added
- Modified files
- Code additions and deletions
- Overall scope and intent of changes

---

### 2. Code Refactoring

#### Readability Improvements

- Replace tuples with named classes or dataclasses
- Improve variable, method, and class naming
- Extract complex logic into well-named helper methods
- Add missing type hints
- Simplify nested or complex conditionals
- Replace deprecated methods and features
- Normalize formatting to match Pipecat style

#### Performance Enhancements

- Identify inefficient loops or repeated work
- Suggest appropriate data structures
- Optimize async workflows and I/O
- Remove redundant operations

> Performance changes are conservative and non-breaking.

---

### 3. Documentation

Documentation follows **Google-style docstrings**, consistent with Pipecat conventions.

#### Class Documentation

```python
class ExampleService:
    """Brief one-line description.

    Detailed explanation of the class purpose, responsibilities,
    and important behaviors.

    Supported features:

    - Feature 1
    - Feature 2
    - Feature 3
    """
```

#### Method Documentation

```python
def process_data(self, data: str, options: Optional[dict] = None) -> bool:
    """Process incoming data with optional configuration.

    Args:
        data: The input data to process.
        options: Optional configuration dictionary.

    Returns:
        True if processing succeeded, False otherwise.

    Raises:
        ValueError: If data is empty or invalid.
    """
```

#### Pydantic Model Parameters

```python
class InputParams(BaseModel):
    """Configuration parameters for the service.

    Parameters:
        timeout: Request timeout in seconds.
        retry_count: Number of retry attempts.
        enable_logging: Whether to enable debug logging.
    """

    timeout: Optional[float] = None
    retry_count: int = 3
    enable_logging: bool = False
```

---

### 4. Pattern Consistency Checks

#### Service Classes

- Correct inheritance (`TTSService`, `STTService`, `LLMService`)
- Consistent constructor signatures
- Frame emission patterns
- Metrics support:
  - `can_generate_metrics()`
  - TTFB metrics
  - Usage metrics
- Alignment with similar existing services

#### Examples

Validated against `examples/foundational/07-interruptible.py`:

- Proper `create_transport()` usage
- Correct pipeline structure
- Task setup and observers
- Event handler registration
- Runner and bot entrypoint consistency

---

### 5. Specific Implementation Patterns

#### Service Implementation

```python
class ExampleTTSService(TTSService):

    def __init__(self, *, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self._api_key = api_key or os.getenv("SERVICE_API_KEY")

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        try:
            await self.start_ttfb_metrics()
            yield TTSStartedFrame()
            # ... processing ...
            yield TTSAudioRawFrame(...)
        finally:
            await self.stop_ttfb_metrics()
```

---

#### Example Structure Pattern

```python
transport_params = {
    "daily": lambda: DailyParams(...),
    "twilio": lambda: FastAPIWebsocketParams(...),
    "webrtc": lambda: TransportParams(...),
}

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    stt = DeepgramSTTService(...)
    tts = SomeTTSService(...)
    llm = OpenAILLMService(...)

    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(...)

    pipeline = Pipeline([...])
    task = PipelineTask(pipeline, params=..., observers=[...])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)

async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)
```

---

## Execution Flow

1. Fetch uncommitted and outgoing changes
2. Categorize files (services, examples, tests, utilities)
3. Analyze each file:
   - Readability
   - Performance
   - Documentation
   - Pattern consistency
4. Generate actionable recommendations
5. Apply Pipecat standards

---

## Examples

### Before: Tuple Usage

```python
def get_audio_info(self) -> Tuple[int, int]:
    return (48000, 1)
```

### After: Named Class

```python
class AudioInfo:
    """Audio configuration information.

    Parameters:
        sample_rate: Sample rate in Hz.
        num_channels: Number of audio channels.
    """

    sample_rate: int
    num_channels: int

def get_audio_info(self) -> AudioInfo:
    return AudioInfo(sample_rate=48000, num_channels=1)
```

---

### Before: Missing Documentation

```python
class NewTTSService(TTSService):
    def __init__(self, api_key: str, voice: str):
        self._api_key = api_key
        self._voice = voice
```

### After: Fully Documented

```python
class NewTTSService(TTSService):
    """Text-to-speech service using NewProvider API.

    Streams PCM audio and emits TTSAudioRawFrame frames compatible
    with Pipecat transports.

    Supported features:
    - Text-to-speech synthesis
    - Streaming PCM audio
    - Voice customization
    - TTFB metrics
    """

    def __init__(self, *, api_key: str, voice: str, **kwargs):
        """Initialize the NewTTSService.

        Args:
            api_key: API key for authentication.
            voice: Voice identifier to use.
            **kwargs: Additional arguments passed to the parent service.
        """
        super().__init__(**kwargs)
        self._api_key = api_key
        self.set_voice(voice)
```

---

## Notes

- Non-breaking improvements only
- Backward compatibility preserved
- Conservative performance changes
- Google-style docstrings
- Pattern checks follow recent Pipecat code
