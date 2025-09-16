# AGENTS.md

## Project Overview

Pipecat is an open-source Python framework for building real-time voice and multimodal conversational AI agents. The codebase is organized around a pipeline architecture where data flows through connected services (STT → LLM → TTS).

## Development Environment Setup

### Prerequisites
- **Minimum Python Version:** 3.10
- **Recommended Python Version:** 3.12
- **Package Manager:** uv (recommended) or pip

### Setup Commands

```bash
# Clone the repository
git clone https://github.com/pipecat-ai/pipecat.git
cd pipecat

# Install dependencies with uv (recommended)
uv sync --group dev --all-extras \
  --no-extra gstreamer \
  --no-extra krisp \
  --no-extra local \
  --no-extra ultravox

# Or with pip
pip install -e ".[dev]"

# Install pre-commit hooks
uv run pre-commit install

# Set up environment variables
cp env.example .env
```

## Build and Test Commands

### Running Tests
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_name.py

# Run tests with coverage
uv run pytest --cov=pipecat --cov-report=html
```

### Code Quality
```bash
# Format code (required before commits)
uv run ruff format

# Lint code
uv run ruff check

# Type checking
uv run mypy src/pipecat

# Run pre-commit checks manually
uv run pre-commit run --all-files
```

### Documentation
```bash
# Build API documentation
cd docs/api
./build-docs.sh

# Build docs manually
sphinx-build -b html . _build/html -W --keep-going
```

## Code Style Guidelines

### Python Standards
- **Formatting:** Strict PEP 8 via Ruff
- **Docstrings:** Google-style format
- **Type Hints:** Required for all public APIs
- **Import Organization:** Automated via Ruff

### Docstring Conventions
- **Classes:** Describe purpose + `__init__` with complete `Args:` section
- **Dataclasses:** Use `Parameters:` section, no `__init__` docstring
- **Methods:** Include `Args:` and `Returns:` sections
- **Properties:** Must have `Returns:` section
- **Examples:** Use `Examples:` section with `::` syntax

### File Organization
```
src/pipecat/           # Main package
├── processors/        # Frame processors
├── services/          # AI service integrations
├── transports/        # Communication layers
├── frames/            # Data frame definitions
└── pipeline/          # Pipeline orchestration

examples/foundational/ # Step-by-step tutorials
tests/                 # Test suite
```

## Testing Instructions

### Test Structure
- **Unit Tests:** Test individual components in isolation
- **Integration Tests:** Test service interactions
- **Example Tests:** Validate foundational examples work

### Adding Tests
```bash
# Test naming convention
test_<component>_<functionality>.py

# Run specific test pattern
uv run pytest -k "test_pipeline"

# Run with debugging
uv run pytest -s -vv tests/test_name.py::test_function
```

### Pre-commit Requirements
All commits must pass:
- Ruff formatting
- Ruff linting
- Type checking
- Basic test suite

## Dependency Management

### Using uv (Recommended)
```bash
# Add runtime dependency
uv add package-name

# Add optional dependency
uv add --optional service package-name

# Add development dependency
uv add --group dev package-name

# Update lockfile
uv lock

# Sync dependencies
uv sync
```

### Important Notes
- **Always commit both `pyproject.toml` and `uv.lock` together**
- **Never manually edit `uv.lock`** - it's auto-generated
- **Use extras for optional service dependencies** (e.g., `[openai]`, `[cartesia]`)

## Project Structure Guidelines

### Service Integration
When adding new AI services:
1. Create service class in `src/pipecat/services/<provider>/`
2. Follow existing patterns (e.g., STTService, LLMService)
3. Add to appropriate extras in `pyproject.toml`
4. Include tests in `tests/`
5. Add documentation examples

### Frame Processing
For custom processors:
1. Inherit from `FrameProcessor`
2. Implement `process_frame()` method. ALWAYS explicitly call `await super().process_frame(frame, direction)` at the top of this method.
3. Handle frame direction (FrameDirection.UPSTREAM/DOWNSTREAM)
4. Add proper type hints and docstrings

### Transport Implementation
For new transport layers:
1. Inherit from `BaseTransport`
2. Implement required abstract methods
3. Handle connection lifecycle
4. Support both input and output streams

## Security Considerations

### API Keys
- **Never commit API keys** to the repository
- **Use environment variables** for all secrets
- **Reference `env.example`** for required variables
- **Use `.env` files** for local development

### Input Validation
- **Validate all external inputs** (audio, text, API responses)
- **Sanitize user data** before processing
- **Handle rate limiting** for external services
- **Implement proper timeout handling**

## Performance Guidelines

### Memory Management
- **Clean up resources** in transport disconnection handlers
- **Use async context managers** for service connections
- **Implement proper frame lifecycle** management

### Latency Optimization
- **Choose appropriate STT services** for latency requirements
- **Use streaming TTS** when possible
- **Implement connection pooling** for HTTP services
- **Consider WebRTC** for real-time applications

## Common Patterns

### Error Handling
```python
@transport.event_handler("on_error")
async def on_error(transport, error):
    logger.error(f"Transport error: {error}")

    # Shutdown the pipeline
    await task.queue_frame(EndFrame())
 
```

### Service Configuration
```python
# Use environment variables for configuration
service = OpenAILLMService(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    model="gpt-4o",
    params={"temperature": 0.7}
)
```

### Pipeline Assembly
```python
pipeline = Pipeline([
    transport.input(),
    stt_service,
    context_aggregator.user(),
    llm_service,
    tts_service,
    transport.output(),
    context_aggregator.assistant(),
])
```

## Commit and PR Guidelines

### Commit Message Format
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### PR Requirements
- **All tests must pass**
- **Code must be properly formatted** (Ruff)
- **Include appropriate tests** for new functionality
- **Update documentation** if needed
- **Reference related issues** in description

### Review Process
1. Automated checks must pass
2. Manual code review by maintainers
3. Documentation review for user-facing changes
4. Integration testing for service additions

## Troubleshooting

### Common Issues
- **Import errors:** Run `uv sync` to ensure dependencies are installed
- **Test failures:** Check environment variables in `.env`
- **Format errors:** Run `uv run ruff format` before committing
- **Type errors:** Ensure all public methods have type hints

### Development Tips
- **Use foundational examples** as starting points for testing
- **Check existing services** for integration patterns
- **Run tests frequently** during development
- **Use IDE integration** for Ruff formatting

### Getting Help
- **Documentation:** [docs.pipecat.ai](https://docs.pipecat.ai)
- **Issues:** [GitHub Issues](https://github.com/pipecat-ai/pipecat/issues)
