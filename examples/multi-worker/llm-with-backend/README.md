# LLM with backend

A conversational **frontend** holds the conversation with a fast model and no tools of its own. Anything that needs tools, current information or careful reasoning it hands to a **backend**: a `BackendLLMWorker` running a heavier model with the tools. `LLMWithBackend` wraps the frontend so the pair drops into a pipeline where an LLM goes, installs the `delegate` tool that joins them, and runs the backend as a worker of its own.

```python
llm = LLMWithBackend(
    frontend=OpenAIResponsesLLMService(...),
    backend=BackendLLMWorker(
        llm=AnthropicLLMService(...),
        context=LLMContext(tools=[get_current_weather, get_restaurant_recommendation]),
    ),
)
```

How a delegation crosses is the `BackendConnector`'s business, built from two strategies the service picks unless told otherwise. The request strategy goes by what the frontend is, the reply strategy by what it can do with a tool's intermediate results:

| frontend                                     | request (frontend → backend)                                                       |
| -------------------------------------------- | ---------------------------------------------------------------------------------- |
| text                                         | `TranscriptBackendRequestStrategy`: the conversation since the previous delegation |
| speech-to-speech (its context lags the audio) | `ExplicitBackendRequestStrategy`: a request the model words itself                 |

| frontend                        | reply (backend → frontend)                                                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| takes intermediate results      | `SpeakOnPrefersSpokenBackendReplyStrategy`: progress relayed as it comes, spoken as the backend's flag says  |
| doesn't                         | `OneShotBackendReplyStrategy`: every output at once, when done                                                |

## Examples

| Example                                                      | What it shows                                                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| [`openai-frontend.py`](openai-frontend.py)                   | A cascade pipeline (STT + GPT + TTS) as the frontend, all defaults.                               |
| [`openai-realtime-frontend.py`](openai-realtime-frontend.py) | OpenAI Realtime as the frontend, all defaults. Same backend, same prompts as the cascade example. |
| [`gemini-live-frontend.py`](gemini-live-frontend.py)         | Gemini Live (`gemini-3.8-live`) as the frontend, all defaults. Same backend, same prompts.        |

A speech-to-speech frontend hears the backend's progress if the service delivers a tool's intermediate results to its model, which OpenAI Realtime, Gemini Live, Grok, Nova Sonic and Ultravox do. One that doesn't gets every output at once instead.

Run any of them the usual way, then connect a client:

```bash
python openai-frontend.py
```

Or drive one with a behavioral eval:

```bash
python openai-frontend.py -t eval --port 7860
pipecat eval run ../../../scripts/release-evals/scenarios/scripted/weather_function_call_audio.yaml --bot-url ws://localhost:7860 -v
```

## A backend in another process

The service addresses the backend by name over the bus, so it need not run in the same process. Give `LLMWithBackend` the worker's name instead of the worker, and run the backend under its own `WorkerRunner` on a shared network bus. See [`distributed-handoff`](../distributed-handoff/) for the bus setup.

Backend process:

```python
bus = RedisBus(redis=Redis.from_url(REDIS_URL), channel="pipecat:llm-with-backend")

backend = BackendLLMWorker(
    name="backend",
    llm=AnthropicLLMService(...),
    context=LLMContext(tools=[get_current_weather, get_restaurant_recommendation]),
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

The first delegation waits for the registry to report the backend ready, so a backend that starts a little late is fine. A `PgmqBus` works the same way.
