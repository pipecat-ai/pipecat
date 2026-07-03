# AGENTS.md — Building Pipecat Apps

> Guidance for AI coding agents building **applications** with Pipecat.
> This is **not** about contributing to the Pipecat framework itself — for that, see the framework's own AGENTS.md.
>
> Audience: any coding agent (Claude Code, Codex, …).

---

## 0. Golden rules (non-negotiable)

1. **Scaffold first. Never hand-write boilerplate.** Start every new app from the Pipecat CLI scaffold (§1) — it generates a *runnable* bot (services + pipeline already wired), not a bare skeleton. Modify what it generates; don't rebuild it or invent your own structure.
2. **Don't guess — learn, then verify.** Pipecat moves fast and your priors are stale. When a capability is unfamiliar, **learn how it works first** (§3 `search_docs`) instead of jumping into code; then confirm every class, import, and param against a **live source** (§3) as you write. Confidently-wrong old APIs are the #1 failure mode.
3. **Make it verifiable before you make it fancy.** Pipecat ships a behavioral **eval harness** (§6): scaffold with `--eval` and check every change by running scenarios headless — no live voice call. Stand up that loop before adding features.
4. **Ask for credentials; don't invent them.** Model names, voice IDs, and API keys are provider-specific and change often. If you don't have one, stop and ask — list exactly which keys are needed.

---

## 1. Start by scaffolding

Always begin from the deterministic CLI scaffold. It gives you a known-good structure, correct dependencies, a runnable entry point, and the deploy path for free.

```bash
# The CLI ships *with* pipecat-ai, behind the optional `cli` extra. Install it as a global tool:
uv tool install "pipecat-ai[cli]"   # provides the `pipecat` (alias `pc`) command
```

> ⚠️ **Agents: scaffold non-interactively** — bare `pipecat init` (no scaffold flags) opens an interactive wizard that **hangs an automated run**. Pass your choices as flags (or `--config`):

```bash
# Headless: any scaffold flag (--bot-type, a service, or --config) switches off the wizard.
# Run it in the project directory you're already in; `.` scaffolds in place, name from the dir.
# The service values below are EXAMPLES — map the user's actual choices, don't copy these.
pipecat init . \
  --transport smallwebrtc --mode cascade \
  --stt deepgram_stt --llm openai_llm --tts cartesia_tts \
  --eval                             # eval transport + starter scenarios, for verification (§6)
#   • Don't hand-write flags from memory — discover them:
#       pipecat init --help          # available flags
#       pipecat init --list-options  # valid service/transport VALUES
#   • --dry-run prints the resolved config as JSON; --config project.json drives it from a file.
#   • --transport is repeatable — pass each transport you want (production + a local-dev one, §2).
#   • --bot-type is inferred from --transport (telephony if any telephony transport, else web) — omit it.
```

**Choose *with* the user, not for them.** Map their requirements to the real options and confirm transport / services / mode / deployment (§7) before scaffolding — don't silently pick or guess. Mode affects testing speed — **cascade (STT→LLM→TTS)** gets the fast text-mode eval loop (§6); **realtime (speech-to-speech)** is tested in audio mode — but both run headless, so pick the mode the use case needs.

The scaffold pins a Pipecat version (treat it as the world you're building in — §3) and produces:

```
mybot/
├── server/
│   ├── bot.py            # entry point: an async `bot(runner_args)` function + `main()`
│   ├── evals/            # starter eval scenarios, with --eval (§6)
│   ├── pyproject.toml
│   ├── .env.example      # required env vars
│   ├── Dockerfile        # if cloud enabled
│   └── pcc-deploy.toml   # deploy config, if cloud enabled
├── client/               # web client, if generated
└── README.md
```

**`bot.py` is a runnable bot, not a blank skeleton.** For the services you chose, the scaffold emits the **constructed service objects** (keys read from env), the `transport_params`, the canonical pipeline + context aggregators, the runner wiring, and an on-connect handler that speaks the first turn. **Modify it in place** — system prompt, function-calling tools (§4 shows the pattern), business logic, the greeting — do **not** re-declare the service constructors, rebuild the pipeline, or write your own connect handler. Nearly every provider is a scaffold value; hand-wire a service only if `--list-options` genuinely doesn't list it.

Every bot exposes an async `bot(runner_args)` entry point that the dev runner discovers and executes per session. Keep that contract.

> If the user already has an app, do **not** re-scaffold — read their existing structure and follow it.

---

## 2. Choosing transport & services

Transport and services are scaffold inputs (the `--transport` / `--stt` / `--llm` / `--tts` flags) — map the use case to them before you scaffold, and confirm exact class names and params via §3.

**Transport — by where the bot runs:**
- **Web / mobile voice** — `DailyTransport` or `SmallWebRTCTransport`.
- **Telephony** — a WebSocket transport (`FastAPIWebsocketTransport`) + the provider's serializer (Twilio, Telnyx, …), or `DailyTransport` for Daily PSTN/SIP.

**Scaffold every transport you need at once** — repeat `--transport` (e.g. `--transport twilio --transport smallwebrtc`); the scaffold wires each one's params and dependencies. They coexist in `transport_params`; `-t <name>` picks one per run. `--bot-type` is inferred from your transports (telephony if any telephony transport, else web), so you can omit it. A **telephony** bot can include a WebRTC transport for local testing. A convenient dev setup is your production transport *plus*:
- **`SmallWebRTCTransport` + Pipecat Prebuilt** — scaffold it in (`--transport smallwebrtc`) for a built-in browser test UI at `http://localhost:7860`, so you can talk to the bot with no telephony/PSTN setup — how you dev/test a **telephony** bot locally.
- **the `eval` transport (§6)** — drive it headless with scripted scenarios to confirm behavior, tune prompts, and catch regressions.

**Services** — STT, TTS, LLM are independent choices. Confirm the exact service class, model name, and params via §3; do not assume. Services configure via the current `settings=` argument.

When the user describes a use case, find the closest example via `search_examples` (§3) and start there.

---

## 3. Find current truth (anti-staleness)

You have live sources for current truth — never substitute your memory. Use the **highest rung of this ladder that works right now** — every rung is something you can do yourself, so never fall back to guessing:

1. **The `pipecat-context-hub` MCP** (check your tool list) — the warm path; returns *primary source*. Two intents:
   - **Learn the concept** (before building something unfamiliar): `search_docs` — how a capability works (the Learn guides) and how to use an optional feature (the Fundamentals guides — recording, transcripts, metrics, idle detection, muting, IVR, voicemail, …); `get_doc` — read a page in full. `search_examples` / `get_example` — a working implementation to start from. When asked for a feature, search it rather than guess.
   - **Verify a specific API** (as you write): `check_deprecation` — **run on any symbol you're unsure about** (the stale-training antidote, e.g. `PipelineTask`→`PipelineWorker`); `search_api` / `get_code_snippet` — exact current signatures and usage. Examples can lag the framework — `check_deprecation` any symbol you copy from one.

   The index is **local** — check `get_hub_status` for `last_refresh_at`, and refresh (`uvx pipecat-ai-context-hub@latest refresh`) when it's stale or after a Pipecat version bump.
2. **No MCP? Query the same index from your shell** — zero setup beyond `uv`:
   ```bash
   uvx pipecat-ai-context-hub search-docs "turn detection"       # learn a concept
   uvx pipecat-ai-context-hub check-deprecation PipelineTask     # the reflex check; <1s
   uvx pipecat-ai-context-hub search-api "EvalTransportParams"
   uvx pipecat-ai-context-hub search-examples "twilio bot" --domain backend
   uvx pipecat-ai-context-hub status                             # index health / freshness
   ```
   Stdout is the tool's JSON. **Exit 2 means the local index isn't built yet** — run `uvx pipecat-ai-context-hub@latest refresh` once (downloads the package + local models and indexes the sources; allow several minutes), then re-run the query. Afterwards, **set up future sessions** with your agent's MCP command — `claude mcp add pipecat-context-hub -- uvx pipecat-ai-context-hub serve` (Codex: same args, `codex mcp add`). A newly added MCP server loads at the *next* session start, never mid-session — so keep using the CLI for the current one.
3. **Installed package source** — the pinned version is on disk; the code cannot be stale. Read it when the index is ambiguous:
   ```bash
   python -c "import pipecat, os; print(os.path.dirname(pipecat.__file__))"
   ```
4. **`llms.txt`** — machine-readable docs index at `https://docs.pipecat.ai/llms.txt` (full content: `llms-full.txt`). The last resort when nothing local works.

(Naming: the *package* is `pipecat-ai-context-hub`; the command and MCP server are `pipecat-context-hub`. Both spellings of the command work once installed.)

For browsing examples directly, the **`pipecat-examples` repo** groups demos by category in its README (telephony, vision, etc.); `scripts/demos.json` lists each example's run command.

**Rule of thumb:** if you're about to type a Pipecat class name, import path, or service parameter from memory — stop and run `check_deprecation` / `search_api` first (MCP, or the rung-2 CLI).

---

## 4. Mental model (concepts, not APIs)

These concepts are stable even as signatures change — internalize them so your code flows correctly.

**Terminology (current).** Get this vocabulary right — the framework renamed these and old terms are deprecated:
- **Worker** — the top-level runnable unit. A single bot is a `PipelineWorker` (it wraps your pipeline). *Replaces the old "pipeline task"; `PipelineTask` is a deprecated alias.*
- **Job / job group** — cross-worker RPC in multi-worker apps. Don't call these "agents." (The former `pipecat-subagents` package is folded into `pipecat.workers`.)
- **Task** — means *only* an asyncio task. Don't use "task" for the runnable unit.

Core concepts:

- **Everything is a Frame moving through a pipeline of FrameProcessors.** Frames flow **downstream** (input → output) or **upstream** (errors/acknowledgments). A processor receives a frame, does its work, and pushes it along — by default in the direction it came.
- **Change a running pipeline by pushing frames, not by calling methods.** Pipecat is real-time and ordered: push a frame (e.g. `LLMSetToolsFrame` to swap tools, a context-update frame to add a message) so the change stays in sequence with frames already in flight; reaching into an object directly (e.g. `context.set_tools()`) jumps the queue and causes subtle ordering bugs. Find the right frame via §3.
- **A worker runs the show.** For the common single-bot case, wrap your pipeline in a `PipelineWorker`, then run it with a `WorkerRunner`: `await runner.add_workers(worker)`, then `await runner.run()`. The runner owns the shared bus, handles SIGINT/SIGTERM, and by default ends once every root worker finishes; a long-lived host (e.g. a FastAPI server adding/removing sessions) passes `auto_end=False`. In a scaffolded app the dev runner drives all of this per session via your `bot(runner_args)` entry point — you don't wire it by hand.
- **Pipeline order matters.** The canonical **cascade** voice loop is:
  ```
  transport.input()
    → STT
    → user context aggregator
    → LLM
    → TTS
    → transport.output()
    → assistant context aggregator
  ```
  **Realtime (speech-to-speech)** is the same loop **without STT and TTS** — the S2S service does both internally:
  ```
  transport.input()
    → user context aggregator
    → LLM (realtime S2S service)
    → transport.output()
    → assistant context aggregator
  ```
  In **both**, the **assistant aggregator goes after `transport.output()`** so it records what was actually produced. Getting this order wrong is a common, subtle bug.
- **Context aggregation** accumulates conversation messages for the LLM. User and assistant aggregators are created together (a pair) and bracket the response leg (LLM→TTS in cascade; the S2S service in realtime) as shown above.
- **The LLM's output is spoken.** TTS reads exactly what the LLM writes — markdown, emojis, and bullet lists come out as noise. The scaffold's `system_instruction` already guards this: *"Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken."* When you rewrite the prompt for the use case, **carry that sentence over**. The bot's turns — including the on-connect greeting — are composed by the LLM by default; making it speak a fixed, verbatim line is a separate mechanism (confirm via §3).
- **Instructions live in two places — don't pour everything into one.** Durable identity and rules (who the bot is, the voice-safe guard above) live in the LLM's *system instruction*. Turn- or task-specific guidance (what to do next, a fact to use now) goes as a **`developer`** message in the context — Pipecat's standard role for "do this" instructions; its adapters translate it for LLMs that don't support the role natively — the scaffold's greeting already works this way. To seed or steer a turn yourself, add to the context and trigger a response; don't rewrite the system instruction mid-conversation. Confirm the exact roles and frames via §3.
- **Function calling (tools)** — how the bot *does* things, and where most of your app code goes. A tool is a plain async function: its **name, typed signature, and docstring become the schema automatically**, and it reports back through `params.result_callback` — that's what feeds the result into the conversation so the LLM can speak it:
  ```python
  async def check_stock(params: FunctionCallParams, item: str):
      """Check whether an item is in stock.

      Args:
          item: The item to look up, e.g. "tulips".
      """
      await params.result_callback({"in_stock": True})
  ```
  List tools on the context — `LLMContext(tools=[check_stock, ...])` — and they register with the LLM service by themselves; there is no separate registration step to write. To share backend handles or per-session state across handlers, pass `app_resources=` to `PipelineWorker` and read `params.app_resources` (don't use globals). An explicit-schema path (`FunctionSchema` + `llm.register_function`) exists for dynamic tool sets — confirm via §3 if you need it.
- **Ending the conversation from a tool** (the "goodbye" / hang-up feature nearly every phone bot needs) — push `EndWorkerFrame` from the handler, *after* reporting the result:
  ```python
  async def end_call(params: FunctionCallParams):
      """End the call once the user has said goodbye."""
      await params.result_callback({"success": True})
      await params.llm.push_frame(EndWorkerFrame())
  ```
  Downstream (the default) is correct: queued frames flush, so the bot can finish speaking before the pipeline ends. Don't invent shutdown paths (`sys.exit`, cancelling tasks) and don't use the deprecated `EndTaskFrame` your training data may suggest. For the prompt pattern that gets the bot to say a goodbye *around* the tool call, find a "graceful end" example via `search_examples` (§3). (Under the hood the pipeline ends with an `EndFrame` — graceful, finishes queued work — or a `CancelFrame` — abrupt; `EndWorkerFrame` is how a processor *requests* the graceful one in order, so the bot finishes its turn instead of being cut off. A bot is also one session per connection — it ends on client disconnect or idle timeout.)
- **A turn has two phases — start and stop — each driven by a strategy.** The defaults the scaffold wires: VAD + transcription to detect the user *starting*; a smart-turn analyzer + transcript to detect them *stopping*. These defaults are tuned — to change turn-taking, swap or configure the strategies via §3 rather than hand-tweaking timings blind (lowering the stop threshold to feel snappier truncates users).
- **Interruptions** — when the user barges in, an interruption propagates and clears in-flight work. Most apps get this for free from the turn strategy; to trigger one manually (rare), look up the call via §3.
- **Background work inside a processor/worker** — when you're *inside* a class that derives from the framework's `BaseObject` (a `FrameProcessor`, a custom worker, a service), use `self.create_task(coro, name)` rather than raw `asyncio.create_task` so the task is tracked and cleaned up on shutdown; cancel with `await self.cancel_task(task, timeout)`. This applies only to `BaseObject` subclasses — plain application code that isn't one of these can use `asyncio` directly.
- **Structured conversations (Pipecat Flows)** — when the conversation has distinct *stages* with different rules (intake → order → payment → confirmation; branching by intent), don't pile every instruction and tool into one giant system prompt. **Pipecat Flows** (a separate package, `pipecat-ai-flows`; docs on the same site) layers a state machine over the pipeline: each node gets its own prompt, tools, and actions, with explicit transitions. Routing heuristic: a single context + tools (above) is right for most bots; reach for Flows when the prompt/tool set must *change per stage* or the process must be enforced, not suggested. Its API churns — confirm everything via §3 before writing Flows code.
- **Multi-worker (advanced)** — multiple *concurrently running* workers cooperating over a shared **bus**, calling each other via **jobs** / **job groups** and discovering each other through a registry. Reach for this only when one pipeline genuinely isn't enough: live handoff between bots, parallel specialists, a separate UI worker. Stages *within* one conversation are Flows territory (above), not multi-worker. Find working examples via `search_examples` and confirm APIs via §3.

> Need more than the model here? Learn deeper via §3 — `search_docs` the Learn pages (turn detection, context, termination, transports) and the Fundamentals guides (optional features) *before* building them. For framework internals, the framework's own AGENTS.md "Architecture" and "Workers, Bus, and Jobs" sections help — but get *API specifics* from §3, not from prose.

---

## 5. Secrets & environment

- Keep a `.env.example` listing every key the app needs; never commit real keys.
- Before running anything, check which providers are wired and confirm the user has the corresponding keys. If a key is missing, **stop and ask** — name the provider and the env var.

---

## 6. Verify your work (the eval harness)

A voice app can't be eyeballed like a web page — but you don't need a live call to test it. Pipecat ships a **behavioral eval harness** (`pipecat.evals`): scripted YAML scenarios drive your *running bot* and assert on what it does — deterministic checks (substring, function name/args, latency) plus an optional LLM **judge** for natural-language criteria — exercising the whole pipeline (STT→LLM→TTS, turns, context, tools, interruptions) end to end.

> **Deep reference:** the **Pipecat Evals docs** are the authoritative spec — look them up via your Pipecat MCP (§3): **Overview**, **Writing Scenarios** (the schema + the two modality axes), **Using the Library** (the Python API), **Agent Self-Improvement** (the closed-loop workflow this section describes). For working examples, copy the **scaffolded starters in `server/evals/`** rather than writing YAML from scratch. The eval harness ships in the `pipecat-ai[evals]` extra (the `pipecat eval` command plus the local Kokoro/Moonshine speech models); scaffolding with `--eval` adds it, so run evals from the **bot's own environment**.

**Make your bot eval-able.** Scaffold with `pipecat init . --eval` (headless) — pass it whenever you scaffold a bot you intend to test. The generated bot has the `eval` transport entry, eval dependencies in its env, and **runnable starter scenarios in `server/evals/`**: `starter_text.yaml` (the fast inner loop; cascade only) and `starter_audio.yaml` (the full round trip). They pass against the freshly scaffolded bot, so run them *first* to prove the loop, then edit them to match the bot you're building and copy them to grow the suite. For an **existing** bot, add the transport entry by hand (a one-time change; RTVI is already on by default for `PipelineWorker`, so that's the only edit):
```python
from pipecat.evals.transport import EvalTransportParams

transport_params = {
    "eval": lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True),
    # ... your real transports (smallwebrtc, daily, …) stay as-is
}
```
The dev runner wraps this in the eval transport + serializer when you pass `-t eval`; you don't construct them. Two kinds of checks follow: quick smoke checks when you wire it up, then the eval loop you run as you work.

**Smoke checks (when you wire it up).** Two fast "did I break the plumbing" checks before any scenario:
- **Logs** — the bot logs to stdout (loguru). However you run it, keep the output durable and greppable — e.g. `uv run bot.py -t eval 2>&1 | tee /tmp/pipecat-output.txt` — so when a scenario fails you can grep the file for the traceback instead of re-running. If your harness captures background-process output natively, that works too.
- **Clean boot** — `uv run bot.py -t eval` boots the bot as a headless eval WebSocket server (default `ws://localhost:7860`). No exceptions + pipeline assembled is your fastest "did I wire it right" signal.

**The eval loop (where you live).** Boot the bot (`uv run bot.py -t eval`), then drive a scenario against it from a second terminal. The eval transport keeps the bot alive between runs, so boot it once and drive scenario after scenario as you iterate — no re-boot per run. Start in **text mode** (the default), the fast inner loop. A minimal scenario:
```yaml
name: capital_of_germany
turns:
  # Wait for the bot's on-connect greeting before speaking (avoids barging into it).
  - expect:
      - event: response
        eval: "the bot opens the conversation in some way (a greeting, an introduction, an offer to help, or a question to get the user started)"

  - user: "What is the capital of Germany?"
    expect:
      - event: response
        eval: "the response says the capital of Germany is Berlin"
```
```bash
uv run pipecat eval run my_scenario.yaml -v
```
Text mode **bypasses STT, VAD, and TTS** (the `user:` turn is sent as text), so assert on what the bot *produced* (`response`, `function_call`), not on `user_*_speaking` events. Prefer the modality-agnostic `response` event — it resolves to the LLM text in text mode and to the bot's transcribed audio in audio mode. Exit code is non-zero if any non-skipped scenario fails. When a run fails in a confusing way, re-run with `-d` to also write `<scenario>.debug.log` — the harness's full logs for the eval STT, TTS, and LLM judge. Route run output to `eval-runs/` with `--logs-dir eval-runs` (and `--record-dir eval-runs` for `-a` recordings); `pipecat eval suite` writes runs to a timestamped `eval-runs/<timestamp>/`.

**One bot serves many runs.** Because the bot stays up between runs, its context carries over — give a scenario a top-level `context:` (a list of LLM messages) to start that run from a known context, or boot fresh / use `pipecat eval suite` (a fresh bot per scenario from a manifest) when you want each run fully isolated. Its `on_client_disconnected` handler (which usually cancels the pipeline) also won't fire during a normal eval; to exercise that path — cleanup, or a goodbye on hang-up — pass `--trigger-disconnect` to `pipecat eval run`, or set `trigger_disconnect: true` on a single scenario. If the handler cancels the pipeline the bot exits, so treat that as a terminal run.

**Escalate to audio mode** for the full round trip. Set `user: {modality: audio, speech: {service: kokoro, voice: af_heart}}` and the harness synthesizes the user's speech, so the bot's **real VAD + STT** run; set `judge: {modality: audio}` (with a `transcription:` block) so the bot **speaks** and the harness transcribes that audio into the `response` event. Kokoro (user TTS) and Moonshine (bot transcription) run **locally with no API key** — audio mode is slower but free. Add `-a` to save the conversation to `<record-dir>/<scenario>.wav` for a human to listen back and confirm it sounds right.

**Which mode — iterate in text, confirm in audio.** Default to **text** for the inner loop and anything about the bot's *decisions* (LLM content, system-prompt adherence, function calls, multi-turn context, barge-in) — it's fast and cheap, so run it constantly. Escalate a scenario to **audio** when text mode would *lie*, i.e. when the thing under test is the audio path itself:

| Use **audio** when… | Why text mode is insufficient |
|---|---|
| correctness depends on how real speech transcribes (numbers, names, accents, homophones) | text skips STT — it never sees a transcript |
| turn-taking / end-of-turn / VAD timing matters | text skips VAD |
| you assert `user_started_speaking` / `user_stopped_speaking` / `tts_response` | these events **only fire in audio mode** (`user_transcription` fires in both modes — STT in audio, DTMF aggregator in text) |
| TTS intelligibility / pronunciation matters | needs `judge: {modality: audio}` so you judge the transcription of real speech (`response`) |
| final pre-ship / CI confidence pass over the real round trip | text never exercises the real STT/VAD/TTS round trip |

Heuristic: **text tests the brain; audio tests the ears and mouth.** The **judge is orthogonal to the mode** — `eval:` natural-language criteria work in either (a local Ollama with `gemma2:9b` by default, or `judge: {service: openai, model: gpt-4.1}`).

**Realtime (speech-to-speech) bots are audio-mode only.** An S2S model has no separate text LLM step to assert on, so text mode doesn't apply — eval it the same way a person would talk to it: `user: {modality: audio}` to synthesize the user's voice in, `judge: {modality: audio}` to transcribe its spoken output for the judge. Same Kokoro-in / Moonshine-out path as above, just **required** rather than an escalation; scenarios, the judge, and assertions are otherwise identical to a cascade bot.

> **Gotchas:** Only `eval:` natural-language criteria need a judge — deterministic checks (`text_contains`, `function_call`) need none. For the judge, use a free local model (Ollama) if one is already available; otherwise ask the user whether to pull it (`ollama pull gemma2:9b`, ~5 GB) or reuse the bot's provider key. Audio-mode scenarios need `audio_in_enabled=True` on the eval transport (above).

---

## 7. Deploying (optional — Pipecat Cloud)

[Pipecat Cloud](https://www.daily.co/products/pipecat-cloud/) is a **hosted** option — deploying there is optional (run locally or host elsewhere instead). It's a scaffold-time choice: `--deploy-to-cloud` (default on, §1) generates the `Dockerfile` + `pcc-deploy.toml`; with `--no-deploy-to-cloud` you'd add those by hand later. **Before deploying, confirm the user wants Pipecat Cloud and has an account** — you can't create one for them.

> `pipecat cloud` is an **optional plugin**, not bundled with `pipecat-ai[cli]`. Install it alongside: `uv tool install "pipecat-ai[cli]" --with pipecatcloud`. (Without it, `pipecat cloud` lists in `--help` but prints how to enable it when run.)

Auth is a one-time **user** action (a browser login — you can't do it for them); after it, deploy runs non-interactively:

```bash
# 1. one-time, by the user (browser login)
pipecat cloud auth login

# 2. Secret set — REQUIRED: a deployed bot has none of your local .env, so without it it
#    starts but every service call fails on missing keys. Name must match `secret_set` in pcc-deploy.toml.
pipecat cloud secrets set <secret_set-name> --file .env --skip

# 3. builds from the Dockerfile + pcc-deploy.toml; --yes skips confirms (agents/CI)
pipecat cloud deploy --yes
```

Re-set the secrets whenever keys change. Tail a live bot with `pipecat cloud agent logs <agent-name>`.

**Krisp noise cancellation** (`--enable-krisp`, requires cloud) is Cloud-provided, and **the scaffold already wires it correctly — keep that block, don't re-wire it**:
- The generated `if os.environ.get("ENV") != "local"` guard, plus the dev runner forcing `ENV=local` on every local run (including `-t eval`), means the bot boots locally with no Krisp model or license and turns Krisp on only once deployed to Pipecat Cloud. Don't add your own guard.
- Deploying to Pipecat Cloud → the scaffold default is already right; don't ask the user how to wire Krisp.
- Krisp with **no** cloud deploy is the only case to raise: the user self-hosts the SDK + model (`KRISP_VIVA_FILTER_MODEL_PATH`, `KRISP_VIVA_API_KEY`).
