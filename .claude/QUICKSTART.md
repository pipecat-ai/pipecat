# Claude Code Quick Start for Pipecat

This guide helps you get started using Claude Code with the Pipecat project.

## Initial Setup

1. **Install Claude Code** (if not already installed):
   ```bash
   # Follow instructions at https://claude.ai/claude-code
   ```

2. **Install project dependencies**:
   ```bash
   uv sync --group dev --all-extras --no-extra gstreamer --no-extra krisp --no-extra local
   ```

3. **Install pre-commit hooks**:
   ```bash
   uv run pre-commit install
   ```

## Common Commands

### Testing
- "Run all tests"
- "Run tests for [specific file]"
- "Run tests and show coverage"

### Code Quality
- "Format the code"
- "Fix linting issues"
- "Run type checking"
- "Run pre-commit hooks"

### Development
- "Add a new TTS service for [provider]"
- "Create a new processor that [does something]"
- "Add a new frame type for [purpose]"
- "Document the [ClassName] class" (uses `/docstring` skill)

### Documentation
- "Document this module using Google style"
- "Add docstrings to [file/class]"
- Use `/docstring ClassName` for comprehensive class documentation

### Git Operations
- "Create a commit for these changes"
- "Create a pull request"
- Use `/pr-description` skill for detailed PR descriptions
- Use `/changelog` skill for changelog entries

## Custom Skills

### `/docstring [ClassName]`
Automatically documents a Python class and its methods following Google-style conventions.

**Example:**
```
/docstring AudioProcessor
```

This will:
- Find the class in the codebase
- Add module docstring if missing
- Add class docstring with purpose and event handlers
- Document all public methods
- Document constructor parameters
- Skip private methods and already-documented code

### `/changelog`
Generates changelog entries using towncrier.

### `/pr-description`
Creates comprehensive pull request descriptions based on your changes.

## Project-Specific Tips

### Understanding Pipecat Architecture

When asking Claude Code to help with development:

1. **Frame-Based System**: All data flows through frames
   - Ask: "Explain how frames work in this pipeline"
   - Reference: `src/pipecat/frames/frames.py`

2. **Processor Pattern**: Everything is a processor
   - Ask: "Show me how to create a custom processor"
   - Reference: `src/pipecat/processors/frame_processor.py`

3. **Service Integrations**: Many AI service integrations
   - Ask: "How do I add a new TTS service?"
   - Reference: `src/pipecat/services/tts/`

### Working with Examples

- "Show me examples of [feature]"
- "Create a simple example that [does something]"
- Examples are in `examples/foundational/` (building blocks) and `examples/` (complete apps)

### Debugging

- "Help me debug this pipeline"
- "Why isn't my processor receiving frames?"
- "Trace the flow of this frame type through the pipeline"

## Best Practices

1. **Be Specific**: Instead of "fix this", say "fix the audio dropouts in the TTS processor"

2. **Context**: Provide context about what you're building
   - "I'm building a voice assistant that needs to interrupt TTS"
   - "I want to add vision capabilities to this chatbot"

3. **Reference Examples**: Point to existing patterns
   - "Similar to how DeepgramTTS works"
   - "Following the pattern in OpenAILLMService"

4. **Test-Driven**: Ask for tests
   - "Create tests for this processor"
   - "Add test coverage for the error handling"

5. **Documentation**: Keep docs updated
   - "Update the docstrings for these changes"
   - "Add a usage example to the class docstring"

## Example Conversations

### Adding a New Feature
```
You: "I need to add a processor that detects when the user says 'hello' and triggers an event"

Claude Code will:
1. Create the processor class
2. Implement frame processing logic
3. Add event emission
4. Create tests
5. Add documentation
```

### Debugging an Issue
```
You: "The audio is cutting out in my pipeline. Here's the code: [paste code]"

Claude Code will:
1. Analyze the pipeline structure
2. Check for common issues (buffer sizes, async handling, etc.)
3. Suggest fixes
4. Explain the root cause
```

### Refactoring
```
You: "Refactor the XYZ service to use the new WebSocket pattern from ABC service"

Claude Code will:
1. Analyze both services
2. Identify the pattern differences
3. Apply the refactoring
4. Update tests
5. Maintain backward compatibility if needed
```

## Useful Prompts

- "Explain how [feature] works in this codebase"
- "Add error handling for [scenario]"
- "Create an example that demonstrates [feature]"
- "Optimize this processor for [use case]"
- "Add logging to help debug [issue]"
- "Make this code more maintainable"
- "Add type hints to this file"
- "Create a comprehensive test suite for [component]"

## Configuration Reference

All Claude Code settings are in [.claude/settings.json](.claude/settings.json):
- Project commands (test, lint, format, etc.)
- Coding standards
- File patterns
- Important files and directories

For detailed architecture info, see [.claude/README.md](.claude/README.md).

## Getting Help

- **Project docs**: https://docs.pipecat.ai
- **Discord**: https://discord.gg/pipecat
- **GitHub Issues**: https://github.com/pipecat-ai/pipecat/issues
- **Examples**: https://github.com/pipecat-ai/pipecat-examples

## Tips for Success

1. Start with small, specific tasks
2. Use the custom skills (`/docstring`, `/pr-description`, etc.)
3. Reference existing code patterns
4. Ask for explanations when confused
5. Request tests and documentation
6. Run pre-commit hooks before committing

Happy coding with Claude! 🎙️🤖
