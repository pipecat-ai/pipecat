# LLM with backend

A conversational **frontend** holds the conversation with a fast model and no tools of its own. Anything that needs tools, current information or careful reasoning it hands to a **backend**: a `BackendLLMWorker` running a heavier model with the tools. `LLMWithBackend` wraps the frontend so the pair drops into a pipeline where an LLM goes, installs the tools that join them, and runs the backend as a worker of its own.

```python
llm = LLMWithBackend(
    frontend=OpenAIResponsesLLMService(...),
    backend=BackendLLMWorker(
        llm=AnthropicLLMService(...),
        context=LLMContext(tools=[search_codebase, read_file, apply_patch, run_tests]),
    ),
)
```

The two exchange messages. The frontend's `delegate` tool puts a message to the backend and returns at once, so the frontend can acknowledge while the backend works. The backend's model decides what the user hears: a message it begins with `>>` comes back as a message in the frontend's conversation marked `Backend:`, which the frontend relays in its own words, whether or not the backend also called tools in that turn. Everything else it writes, does and thinks reaches the conversation silently, marked `Backend (note):`, `Backend (working):` and `Backend (thinking):`, so the frontend can say how the work is going if asked. Each model also gets a *pairing instruction* from the framework, appended to its system instruction: the frontend's (`FRONTEND_PAIRING_INSTRUCTION`) says when and how to delegate, with the request strategy's own guidance in place of its `{strategy}` placeholder, that to the user there is one assistant, what to do after delegating, and what to do with the backend's messages; the backend's (`BACKEND_PAIRING_INSTRUCTION`) that the user hears what it marks with `>>` and nothing else, and how to take requests that arrive while it works. An app's own prompts say what the assistant is and what the backend does, in plain words, as [`backend.py`](backend.py)'s do; an app that would rather word the pairing itself passes `BackendConnector(frontend_pairing_instruction=...)` and `BackendLLMWorker(pairing_instruction=...)`. A second request joins the work in progress, and a correction changes or stops it: the backend reads each message against the work it has under way and cancels the tools whose results are no longer wanted. The `BackendConnector` owns the session with the backend and how a request is worded: a text frontend hands over the conversation since the previous delegation (`TranscriptBackendRequestStrategy`), a speech-to-speech frontend words the request itself (`ExplicitBackendRequestStrategy`), since its context lags the audio.

## Examples

One engineering assistant, five frontends. [`backend.py`](backend.py) holds the backend, its tools and both prompts: a code change that loops through searching, reading, patching and running the tests until they pass, a two-step research task, and quick CI and pull-request lookups, each taking about as long as the real thing would.

| Example                                                        | Frontend                                              |
| -------------------------------------------------------------- | ----------------------------------------------------- |
| [`openai-responses-frontend.py`](openai-responses-frontend.py) | A cascade pipeline: STT, OpenAI's Responses API, TTS. |
| [`google-frontend.py`](google-frontend.py)                     | A cascade pipeline with a Gemini frontend.            |
| [`anthropic-frontend.py`](anthropic-frontend.py)               | A cascade pipeline with a Claude frontend.            |
| [`openai-realtime-frontend.py`](openai-realtime-frontend.py)   | OpenAI Realtime, speech to speech.                    |
| [`gemini-live-frontend.py`](gemini-live-frontend.py)           | Gemini Live, speech to speech.                        |

Run one the usual way, then connect a client and try "fix the flaky retry test in the HTTP client", then ask for something else while it works:

```bash
python openai-responses-frontend.py
```

Each frontend has a commented-out `connector=BackendConnector(client_trace=True)`: with it, every exchange with the backend is sent to the client as an RTVI server message and shows in the prebuilt UI's Events panel.

### A conversation to try

A longer session exercises more of what the pair does: delegating a quick lookup, delegating a judgment call, work that takes a while, and a change of course. One that works well, as a sequence of things to say:

1. Say you want to get through some code reviews this afternoon, and ask for the open pull requests.
2. Ask the bot to use its judgment and pick an impactful one to start with (it tends to pick the exponential-backoff one).
3. Ask what you should do to dig into the review.
4. If it finds a problem with the pull request, tell it to go ahead and make the change.

Whenever a step takes a while, either ask how it is going, formally or not ("what's the status?", "what are you working on now?"), or ask for a joke while you wait. A status question is answered from the backend's silent progress messages, without delegating again; a joke is the frontend's to tell, and the result arrives afterwards on whatever turn is under way.

## A backend in another process

The service addresses the backend by name over the bus, so it need not run in the same process. Give `LLMWithBackend` the worker's name instead of the worker, and run the backend under its own `WorkerRunner` on a shared network bus. See [`distributed-handoff`](../distributed-handoff/) for the bus setup.

Backend process:

```python
bus = RedisBus(redis=Redis.from_url(REDIS_URL), channel="pipecat:llm-with-backend")

backend = BackendLLMWorker(
    name="backend",
    llm=AnthropicLLMService(...),
    context=LLMContext(tools=[...]),
)

runner = WorkerRunner(bus=bus, handle_sigint=True)
await runner.add_workers(backend)
await runner.run()
```

Frontend process:

```python
bus = RedisBus(redis=Redis.from_url(REDIS_URL), channel="pipecat:llm-with-backend")

llm = LLMWithBackend(
    frontend=OpenAIResponsesLLMService(...),
    backend="backend",  # registered in the other process
)

runner = WorkerRunner(bus=bus, handle_sigint=runner_args.handle_sigint)
await runner.add_workers(worker)
await runner.run()
```

The frontend attaches to the backend when its pipeline starts and waits for the registry to report the backend ready, so a backend that starts a little late is fine. A `PgmqBus` works the same way.
