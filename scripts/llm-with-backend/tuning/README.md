# Tuning LLMWithBackend

Tools for checking a change to `LLMWithBackend`'s behavior, and a log of what each change did ([`RESULTS.md`](RESULTS.md)). Reach for them when you change:

- a prompt: the frontend's in `pipecat.pipeline.llm_with_backend.prompts` (the pairing instruction, the `delegate` tool's description, a request strategy's guidance), or the backend's in `pipecat.workers.llm.backend_llm_worker`;
- predictive delegation's classifier: its question, threshold, or how much of the conversation it sees.

A small wording change can move a model in ways no unit test catches: gpt-4.1 stops delegating a "never mind", or the classifier starts delegating questions about how the work is going. Run the checks below before and after the change.

## The checks

They use the bots in [`examples/multi-worker/llm-with-backend/`](../../../examples/multi-worker/llm-with-backend/): three with a text frontend (OpenAI Responses on gpt-4.1, Google on gemini-3.6-flash, Anthropic on claude-sonnet-4-6) and two speech-to-speech. Fastest first:

1. **Frontend replay** (`frontend_replay/`, minutes). 75 recorded user turns, each labelled with whether it should be delegated. Runs one text frontend's model on each and lists the turns it gets wrong: a stop answered with no `delegate` call, or small talk delegated. A delegated turn runs on to the model's reply after the call, as in the bot, and the acknowledgement the user would hear is reported with its length, since a change to the pairing instruction moves that too. Run it for each of the three.

   ```bash
   uv run python scripts/llm-with-backend/tuning/frontend_replay/replay.py --frontend <openai|google|anthropic> --samples 5
   ```

2. **Classifier replay** (`classifier_replay/`, about a minute, needs `TYPESAFE_API_KEY`). 107 recorded user turns, each labelled with whether it needs the backend. Asks the classifier about each, as a running bot would, and reports false "yes" answers (delegated early, not needed) and missed ones. The classifier reads the frontend's prompts, so run it for a prompt change too.

   ```bash
   uv run python scripts/llm-with-backend/tuning/classifier_replay/replay.py
   ```

3. **Text evals** (about an hour): the [suite](../../../evals/llm-with-backend/README.md)'s scenarios on the three text frontends, five runs each.

   ```bash
   uv run python -m pipecat.evals suite evals/llm-with-backend/manifest.yaml -s text --repeat 5
   ```

4. **Audio evals**: every bot, speech-to-speech included.

   ```bash
   uv run python -m pipecat.evals suite evals/llm-with-backend/manifest.yaml -s audio
   ```

The replays check only whether a turn is delegated; the evals judge everything the bot says. A backend prompt doesn't reach either replay, so for one, run the evals only. In both replays, each turn's earlier conversation was recorded under the prompts of the time; a change moves only the decision on the last turn.

## Recording results

Update [`RESULTS.md`](RESULTS.md) in the same commit as the change, so `git log -p RESULTS.md` ties every number to its change and shows the numbers before. `results_table.py` turns a suite run into the table it uses:

```bash
uv run python scripts/llm-with-backend/tuning/results_table.py <runs-dir>/<run>/results.jsonl
```
