# LLMWithBackend evals

Behavioral evals for the bots in [`examples/multi-worker/llm-with-backend/`](../../examples/multi-worker/llm-with-backend/): a fast frontend that holds the conversation (three cascade frontends on OpenAI, Gemini and Claude, and two speech-to-speech), and a backend it hands work to over messages. Each scenario is one behavior the pair has to show, driven against a real bot by [`pipecat eval`](../release/README.md). The prerequisites are the release suite's: a local Ollama judge (`gemma4:12b`), Kokoro and Moonshine for audio mode, and the bots' API keys in `.env`.

```bash
uv run python -m pipecat.evals suite -d evals/llm-with-backend/manifest.yaml            # everything
uv run python -m pipecat.evals suite -d evals/llm-with-backend/manifest.yaml -k script -p openai-responses -s delegate_ack/text
```

## Layout

The conversations live once, in `scenarios/turns/`, and are included by two scenario files each: `scenarios/text/<name>.yaml`, which judges the frontend's text, and `scenarios/audio/<name>.yaml`, which synthesizes the user's voice and judges a transcription of the bot's speech. Text mode drives the cascade bots only, since the speech-to-speech bots take audio, and is the fast loop while working on prompts or code; audio mode runs against every bot and is the judgment. The manifest lists which bots run which.

## Scenarios

The backend is an engineering assistant's: a code change that loops through searching, reading, patching and running the tests until they pass (twenty to thirty seconds), a two-step research task, and quick CI and pull-request lookups. Turns with no `user:` wait for what the bot says on its own, and `send_after:` sends a request while the backend is still working. `check_from: turn_start` marks the events whose order the frontend does not set: the backend's own calls, which land whenever it makes them, and an acknowledgement that some models write before the `delegate` call and others after. The acknowledgement and progress turns also check that the bot speaks as one assistant, with `text_excludes: "backend"` and a judge criterion to match, since the user is not to hear about the split.

| scenario                 | what it checks                                                                                                                               |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `frontend_only`          | A question the frontend answers itself, with no `delegate` call.                                                                             |
| `delegate_ack`           | A code change is handed off: the bot acknowledges at once, and reports the fix on its own when the backend is done.                          |
| `delegate_with_aside`    | The same, plus a joke the user asked for while waiting: the ack and the joke come in one reply.                                              |
| `concurrent_delegations` | A research request while the fix is in progress: a second `delegate`, the research reported first, the fix after.                            |
| `superseding_delegation` | A correction while the fix is in progress: the result reflects the latest instruction.                                                       |
| `cancel_delegation`      | "Never mind": delegated like any other message; the backend stops, the bot confirms, and nothing arrives later.                             |
| `cancel_and_replace`     | A long "stop that, do this instead" timed so a result lands mid-utterance: the stale result is not relayed as wanted, the new request is done. |
| `implicit_cancel`        | A change of mind while the integration tests run, with no "stop" said: the backend cancels `run_integration_tests`, the bot acknowledges once, and nothing arrives later. |
| `partial_cancel`         | Integration tests and a runbook lookup in flight, then "forget the tests": `run_integration_tests` is cancelled, the runbook answer still arrives, no test result does.    |
| `progress_inquiry`       | "How's it going?" mid-fix: answered from the backend's silent progress messages, without delegating again.                                   |
| `aside_while_waiting`    | Small talk while the fix is in progress: the chat is answered, and the fix is still reported when it lands.                                  |
| `result_during_interruption` | A lookup bundled with a request for a long story, so its result lands while the story is being spoken, which the user then cuts off: the result is reported all the same. Audio only. |
| `accepted_offer`         | "Yes, go ahead and check CI, and tell me a joke while we wait": the check is delegated, not merely promised, and "is it done?" is answered from the backend's report.       |

## Reading a verdict

The judge sees each reply as the harness attributes it to a turn, so a result that arrives inside the previous turn's reply can fail the turn that expected it, and audio the user talks over is discarded. When a verdict looks wrong, read the run's logs and recording under `test-runs/<timestamp>/` before changing anything: the bot log says what the backend sent and when, and whether the frontend spoke it. Budgets in the scenarios are generous on purpose, since the eval clock lags the bot by seconds and a `response` needs the bot to stop speaking and its speech to be transcribed.

## Predictive delegation's classifier

`predictive_delegation/` checks the classifier that `PredictiveDelegation` asks whether a user's turn needs the backend. `turns.jsonl` holds 107 turns recorded from the three text example bots in a ten-turn conversation that mixes requests for the backend with ones the frontend answers itself: each is the frontend's context when the user's turn ended, and whether the turn needs the backend. `replay.py` builds the classifier's input from each turn as a running bot would, with the prompts as they are in the code, and reports the turns it gets wrong at the threshold for delegating early. A false "yes" delegates a turn the frontend would have answered itself; a missed "yes" leaves a turn to the frontend that could have been delegated early.

```bash
uv run python evals/llm-with-backend/predictive_delegation/replay.py --out after.json
```

It needs `TYPESAFE_API_KEY` and takes about a minute. Run it, with the text evals, before and after any change to the prompts (see "Prompts" in `pipecat.pipeline.llm_with_backend`), and record the results in [`RESULTS.md`](RESULTS.md), where `results_table.py` turns a suite run into its table.
