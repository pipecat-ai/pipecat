---
name: docstring
description: Document a Python module and its classes using Google style
---

Document a Python module and its classes using Google-style docstrings following project conventions. The class name is provided as an argument.

## Instructions

1. First, find the class in the codebase:
   ```
   Search for "class ClassName" in src/pipecat/
   ```

2. If multiple files contain that class name:
   - List all matches with their file paths
   - Ask the user which one they want to document
   - Wait for confirmation before proceeding

3. Once the file is identified, read the module to understand its structure:
   - Identify all classes, functions, and important type aliases
   - Understand the purpose of each component

4. Apply documentation in this order:
   - Module docstring (at top, after imports)
   - Class docstrings
   - `__init__` methods (always document constructor parameters)
   - Public methods (not starting with `_`)
   - Dataclass/config classes with field descriptions

5. Skip documentation for:
   - Private methods (starting with `_`)
   - Simple dunder methods (`__str__`, `__repr__`, `__post_init__`)
   - Very simple pass-through properties
   - **Already documented code** - If a class, method, or function already has a complete docstring that follows the project style, do not modify it. A docstring is complete if it has:
     - A one-line summary
     - Args section (if it has parameters)
     - Returns section (if it returns something meaningful)
   - Only add or improve documentation where it is missing or incomplete

## Module Docstring Format

```python
"""[One-line description of module purpose].

[Optional: Longer explanation of functionality, key classes, or use cases.]
"""
```

Example:
```python
"""Neuphonic text-to-speech service implementations.

This module provides WebSocket and HTTP-based integrations with Neuphonic's
text-to-speech API for real-time audio synthesis.
"""
```

## Class Docstring Format

```python
class ClassName:
    """One-line summary describing what the class does.

    [Longer description explaining purpose, behavior, and key features.
    Use action-oriented language.]

    [Optional: Event handlers, usage notes, or important caveats.]
    """
```

Example:
```python
class FrameProcessor(BaseObject):
    """Base class for all frame processors in the pipeline.

    Frame processors are the building blocks of Pipecat pipelines, they can be
    linked to form complex processing pipelines. They receive frames, process
    them, and pass them to the next or previous processor in the chain.

    Event handlers available:

    - on_before_process_frame: Called before a frame is processed
    - on_after_process_frame: Called after a frame is processed

    Example::

        @processor.event_handler("on_before_process_frame")
        async def on_before_process_frame(processor, frame):
            ...

        @processor.event_handler("on_after_process_frame")
        async def on_after_process_frame(processor, frame):
            ...
    """
```

Note: When listing event handlers, do NOT use backticks. Include an `Example::` section (with double colon for Sphinx) showing the decorator pattern and function signature for each event.

## Constructor (`__init__`) Format

```python
def __init__(self, *, param1: Type, param2: Type = default, **kwargs):
    """Initialize the [ClassName].

    Args:
        param1: Description of param1 and its purpose.
        param2: Description of param2. Defaults to [default].
        **kwargs: Additional arguments passed to parent class.
    """
```

Example:
```python
def __init__(
    self,
    *,
    api_key: str,
    voice_id: Optional[str] = None,
    sample_rate: Optional[int] = 22050,
    **kwargs,
):
    """Initialize the Neuphonic TTS service.

    Args:
        api_key: Neuphonic API key for authentication.
        voice_id: ID of the voice to use for synthesis.
        sample_rate: Audio sample rate in Hz. Defaults to 22050.
        **kwargs: Additional arguments passed to parent InterruptibleTTSService.
    """
```

## Method Docstring Format

```python
async def method_name(self, param1: Type) -> ReturnType:
    """One-line summary of what method does.

    [Longer description if behavior isn't obvious.]

    Args:
        param1: Description of param1.

    Returns:
        Description of return value.

    Raises:
        ExceptionType: When this exception is raised.
    """
```

Example:
```python
async def put(self, item: Tuple[Frame, FrameDirection, FrameCallback]):
    """Put an item into the priority queue.

    System frames (`SystemFrame`) have higher priority than any other
    frames. If a non-frame item is provided it will have the highest priority.

    Args:
        item: The item to enqueue.
    """
```

## Dataclass/Config Format

```python
@dataclass
class ConfigName:
    """One-line description of configuration.

    [Explanation of when/how to use this config.]

    Parameters:
        field1: Description of field1.
        field2: Description of field2. Defaults to [default].
    """

    field1: Type
    field2: Type = default_value
```

Example:
```python
@dataclass
class FrameProcessorSetup:
    """Configuration parameters for frame processor initialization.

    Parameters:
        clock: The clock instance for timing operations.
        task_manager: The task manager for handling async operations.
        observer: Optional observer for monitoring frame processing events.
    """

    clock: BaseClock
    task_manager: BaseTaskManager
    observer: Optional[BaseObserver] = None
```

## Enum Documentation Format

```python
class EnumName(Enum):
    """One-line description of the enum purpose.

    [Longer description of how the enum is used.]

    Parameters:
        VALUE1: Description of VALUE1.
        VALUE2: Description of VALUE2.
    """

    VALUE1 = 1
    VALUE2 = 2
```

## Writing Style Guidelines

- **Concise and professional** - No casual language or filler words
- **Action-oriented** - Start with verbs: "Processes...", "Manages...", "Converts..."
- **Purpose before implementation** - Explain WHY before HOW
- **Clear parameter descriptions** - Include type hints, defaults, and purpose
- **No redundant type info** - Type hints are in the signature, don't repeat in description
- **Use backticks for code references** - Wrap class names, method names, event names, parameter names, and code snippets in backticks

Good: "Neuphonic API key for authentication."
Bad: "str: The API key (string) that is used for authenticating with Neuphonic."

Good: "Triggers `on_speech_started` when the `VADAnalyzer` detects speech."
Bad: "Triggers on_speech_started when the VADAnalyzer detects speech."

## Deprecation Notice Format

When documenting deprecated code:

```python
"""[Description].

.. deprecated:: X.X.X
    `ClassName` is deprecated and will be removed in a future version.
    Use `NewClassName` instead.
"""
```

## Checklist

Before finishing, verify:

- [ ] Module has a docstring at the top (after copyright header and imports)
- [ ] All public classes have docstrings
- [ ] All `__init__` methods document their parameters
- [ ] All public methods have docstrings with Args/Returns/Raises as needed
- [ ] Dataclasses use "Parameters:" section for field descriptions
- [ ] Enums document each value in "Parameters:" section
- [ ] Writing is concise and action-oriented
- [ ] No documentation added to private methods (starting with `_`)
- [ ] Existing complete docstrings were left unchanged
