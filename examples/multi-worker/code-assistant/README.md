# code-assistant

Talk to your codebase hands-free. Ask questions about code, project structure, or file contents and get spoken answers based on actual files. The Claude Agent SDK worker navigates the filesystem using `Read`, `Bash`, `Glob`, and `Grep` tools.

See the [top-level multi-worker README](../README.md) for setup and shared environment variables.

## Additional environment variables

| Variable            | Required by                    |
| ------------------- | ------------------------------ |
| `ANTHROPIC_API_KEY` | Code worker (Claude Agent SDK) |
| `PROJECT_PATH`      | Optional, defaults to cwd      |

## Running

```bash
# Default: explores the current directory
uv run code-assistant/code-assistant.py

# Specify a project path
PROJECT_PATH=/path/to/your/project uv run code-assistant/code-assistant.py
```

Open <http://localhost:7860/client> in your browser to talk to your bot.

To use Daily transport:

```bash
uv run code-assistant/code-assistant.py --transport daily
```

## Example questions

- "What does the main module do?"
- "Find all TODO comments in the project"
- "How is error handling implemented?"
- "What dependencies does this project use?"
- "Explain the test structure"

## Architecture

```
Main worker (transport + LLM + `ask_code` tool)
  └── job → CodeWorker (Claude Agent SDK)
```

- **[`code-assistant.py`](code-assistant.py)** — Main worker: STT, LLM (with system prompt + `ask_code` direct function), TTS, and transport. The `ask_code` tool dispatches a job to the worker via `worker.job("code-worker", payload=...)`.
- **[`code_worker.py`](code_worker.py)** — `CodeWorker`: a bus-only `BaseWorker` spawned on the runner. It accepts `@job`-style requests through the bus and runs them sequentially through a persistent Claude SDK session so follow-up questions share context.
