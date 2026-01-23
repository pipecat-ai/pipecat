# Claude Code Setup for Pipecat

This directory contains configuration and custom skills for working with the Pipecat project using Claude Code.

## Project Overview

Pipecat is an open-source Python framework for building real-time voice and multimodal conversational agents. It provides a composable, frame-based architecture for orchestrating audio, video, AI services, and conversation pipelines.

## Architecture

### Core Concepts

1. **Frames** - The fundamental data units in Pipecat (audio, text, images, system messages, etc.)
   - Located in: `src/pipecat/frames/frames.py`
   - Different frame types for different data: `AudioRawFrame`, `TextFrame`, `ImageRawFrame`, etc.

2. **Processors** - Processing units that receive, transform, and emit frames
   - Base class: `src/pipecat/processors/frame_processor.py`
   - Can be chained to form pipelines
   - Examples: STT services, LLMs, TTS services, aggregators, etc.

3. **Pipelines** - Chains of processors that define data flow
   - Created using the `Pipeline` class
   - Processors linked using `link()` method or `|` operator

4. **Transports** - Handle input/output for audio/video streams
   - WebRTC (Daily), WebSocket, Local audio, etc.
   - Located in: `src/pipecat/transports/`

### Key Directories

- `src/pipecat/` - Main source code
  - `frames/` - Frame definitions and utilities
  - `processors/` - Base processors and common processors
  - `services/` - AI service integrations (STT, TTS, LLM, etc.)
  - `transports/` - Transport implementations
  - `audio/` - Audio processing utilities
- `examples/` - Example applications and foundational examples
- `tests/` - Test suite
- `docs/` - Documentation source

## Development Workflow

### Setup

```bash
# Install dependencies
uv sync --group dev --all-extras --no-extra gstreamer --no-extra krisp --no-extra local

# Install pre-commit hooks
uv run pre-commit install
```

### Running Tests

```bash
# All tests
uv run pytest

# Specific test file
uv run pytest tests/test_name.py

# With coverage
uv run coverage run --module pytest
uv run coverage report
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Fix linting issues
uv run ruff check --fix .

# Type checking
uv run pyright

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

### Building

```bash
# Build package
uv build
```

## Custom Skills

This project includes custom Claude Code skills:

### `/docstring`
Document Python modules and classes using Google-style docstrings.

Usage: `/docstring ClassName`

### `/changelog`
Generate changelog entries using towncrier.

### `/pr-description`
Generate comprehensive PR descriptions based on changes.

## Coding Standards

1. **Docstrings** - Use Google-style docstrings for all public APIs
   - Module docstrings required
   - Class docstrings with purpose and event handlers
   - Method docstrings with Args/Returns/Raises
   - Constructor (`__init__`) must document all parameters

2. **Type Hints** - Required for all function signatures
   - Use `from typing import ...` for complex types
   - Dataclasses should have field type annotations

3. **Async/Await** - Consistent use of async patterns
   - Most processors use async methods
   - Tests use pytest-asyncio

4. **Code Style**
   - Line length: 100 characters max
   - Ruff for linting and formatting
   - Follow existing patterns in the codebase

5. **Testing**
   - Write tests for new features
   - Use pytest fixtures for common setups
   - Mock external services when appropriate

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following coding standards
4. Add tests for new functionality
5. Run pre-commit hooks: `uv run pre-commit run --all-files`
6. Submit a pull request

## Common Tasks

### Adding a New Service Integration

1. Create service file in `src/pipecat/services/<category>/`
2. Inherit from appropriate base class (e.g., `TTSService`, `LLMService`)
3. Implement required abstract methods
4. Add service to `pyproject.toml` optional dependencies
5. Add documentation
6. Add tests in `tests/`

### Adding a New Processor

1. Create processor in `src/pipecat/processors/`
2. Inherit from `FrameProcessor` or appropriate subclass
3. Override `process_frame()` method
4. Handle relevant frame types
5. Emit frames using `await self.push_frame()`
6. Add tests

### Adding a New Frame Type

1. Add frame definition to `src/pipecat/frames/frames.py`
2. Inherit from appropriate base frame class
3. Use `@dataclass` decorator for data frames
4. Document the frame type and its fields
5. Update processors that should handle this frame type

## Resources

- [Documentation](https://docs.pipecat.ai)
- [GitHub Repository](https://github.com/pipecat-ai/pipecat)
- [Examples](https://github.com/pipecat-ai/pipecat-examples)
- [Discord Community](https://discord.gg/pipecat)
