# Rime Prism realtime conversations

`RimeRealtimeLLMService` uses Prism for speech input, generated speech, and tools through the direct Rime SDK session. It provides a context aggregator pair and uses Pipecat's standard output transports. Install the optional dependency:

```sh
uv pip install 'pipecat-ai[rime-realtime]'
```

The extra requires `rimelabs-sdk>=0.1.0a29,<0.2`.

Set `RIME_API_KEY`. The default endpoint is `wss://prism.api.rime.ai/v1/realtime`. Use `endpoint=` for a custom deployment. An injected `client=Rime(...)` remains caller-owned. Use one service instance per conversation.

```python
from rimelabs_sdk.realtime import PrismFiller

from pipecat.services.rime.realtime.llm import RimeRealtimeLLMService

llm = RimeRealtimeLLMService(
    settings=RimeRealtimeLLMService.Settings(
        system_instruction="Answer briefly. Use lookup_invoice for invoice status.",
        tools=[lookup_invoice],
        tool_fillers={"lookup_invoice": PrismFiller(type="text", text="Checking your invoice.")},
    ),
)
```

## Pipeline and turns

Create the pair with `user, assistant = llm.create_context_aggregator(context)` and use this pipeline:

```text
transport.input() → user aggregator → Rime → transport.output() → assistant aggregator
```

The service advertises realtime mode and external turn strategies through existing service metadata. Prism detects speech turns. Pipecat handles local interruption and output queues through its normal frames. Avoid adding another turn detector that also interrupts on the same speech.

The Rime pair reuses Pipecat's turn handling. It records speech transcripts without requesting another reply and preserves application history edits. Use this pair for Prism conversations.

Queue `LLMRunFrame()` to send initial context and request a reply. An opening reply with no useful history can fail on some deployments. The example includes system history that asks for a greeting.

Queue `RimeReplyFrame(instruction="Check in", response_metadata={"step": "3"})`
to request one reply with an optional instruction and labels. Import it from
`pipecat.services.rime.realtime.llm`. These labels replace matching session
labels for this reply only. Prism starts replies to caller speech without this
frame, so those replies use session labels.

Queue `RimeReplyFrame(text="Remember invoice 314")` through the Rime user aggregator for a native Prism typed turn that follows earlier history and settings frames. `text` cannot be combined with `instruction`. The user message enters local and provider history once. Its response carries the provider input ID in `LLMFullResponseStartFrame.metadata["rime"]["input_item_ids"]`. Empty input and input over 4000 characters are rejected. A refused turn does not enter local history or create an extra reply.

`InputTextRawFrame(text="...")` also starts a native turn, with system-frame priority. It can overtake queued history and settings frames. Use `RimeReplyFrame(text=...)` when that order matters.

Queued replies wait for preceding updates, but do not capture a fixed copy of session state. Later independent updates can take effect while a reply waits for provider admission. Tool results must remain able to update history during that wait.

The existing `LLMMessagesAppendFrame(messages=[...], run_llm=True)` path, including RTVI text input, imports history before requesting a reply. It remains suitable for explicit history edits. Imported history is not treated as fresh typed input.

## Settings, tools, and fillers

Use `LLMUpdateSettingsFrame(delta=RimeRealtimeLLMService.Settings(...))` for instructions, tools, voice, server interruption, fillers, truncation, or response labels. Omitted settings retain their values. An empty `system_instruction=""` clears the instruction. `LLMSetToolsFrame` also updates the provider's tools.

Instructions, tools, and fillers can share one settings delta. This supports a settings-only handoff in the same conversation. History changes are separate operations, so a handoff that also edits history is not atomic. Response labels use a separate SDK operation; their failure does not undo an accepted settings update.

Tool fillers use SDK `PrismFiller` values with `type="none"`, `"text"`, or `"auto"`. Only fillers for currently advertised tools are sent. The configured map survives tool removal and restoration. Supply `tool_fillers={}` to clear it.

Tools execute through Pipecat. Register a tool with `cancel_on_interruption=False` to send intermediate results using `FunctionCallResultProperties(is_final=False)`. The first result settles the provider call. Later progress and final results enter provider history as completed call/result pairs and never execute the tool again. The application tool remains active until its final result, cancellation, or deadline. Response references remain available while this work is pending.

`run_llm=False` records a result silently. False-like results such as `0`, `False`, `{}`, and `""` are preserved. `None` completes a call without a reply by default. The service owns continuation requests so the aggregator does not request a duplicate reply. If the original tool round has been superseded, a requested reply uses the completed result in the current conversation.

A queued call uses Pipecat's current registered handler. The service does not pin handlers to a response's settings revision.

`LLMSetToolChoiceFrame(tool_choice="none")` disables tools for subsequent requested replies; `"auto"` restores automatic selection. The service removes tools, waits for acknowledgment, creates the reply, then restores the configured tools. This operation is not atomic: an automatic speech reply can start while tools are removed. A tool call from a suppressed response is rejected before execution. Restoration failure ends the conversation. Required and named tool choices are unsupported.

The SDK acknowledgment controls accepted provider settings. Normal Pipecat aggregators can change local context or tools before provider acceptance. Rejected changes do not roll those local changes back.

## History and recovery

Append, update, and transform message frames change local context through the Rime pair. Appends import only the supplied messages. Updates and transforms replace supported provider history. An edit with `run_llm=False` remains local until another operation requests inference through either aggregator.

Ordinary transcript and assistant aggregation never replace provider history. A later `LLMRunFrame` applies pending application edits and requests a reply. After the initial import, direct `LLMContextFrame` input is a shared view: omitted provider records remain, and new trailing messages are appended. Use update or transform frames for explicit replacement. Change messages through these frames; changes made directly to an already-bound context are not synchronized by `LLMRunFrame`.

The private history ledger matches explicit replacement messages to provider records by content and order. Identical messages can retain their server identities.

History imports support user, assistant, and system text plus completed tool calls and results. Imported text messages have a 4000-character limit. Images, video, pending tool-call imports, and arbitrary developer messages are unsupported. Loading completed calls never executes them.

Replacing or reordering history can require multiple provider requests. A rebuild requires no active reply or pending tool call. Rejected edits can leave local and provider history different. Known rejection or partial application raises `PrismHistoryUpdateError` through an `ErrorFrame`. Its `status` is `"rejected"`, `"partial"`, or `"uncertain"`; `recoverable` says whether independent work can continue, and `confirmed_context` contains a detached provider projection when known. Import the exception from `pipecat.services.rime.realtime.llm`.

Each accepted write is recorded before the next write. A partial refusal can require one bounded history read to establish completed order. The dependent reply stops, and accepted writes are not replayed. Local application context is not rolled back. If the outcome or order cannot be established, the service becomes unusable. There is no automatic reconnect or replay. Start a new instance with known valid, completed history for recovery.

`await llm.get_history()` reads completed server history. Active items can be absent from that snapshot.
For a `RimeReplyFrame` with an instruction, the `LLMFullResponseStartFrame` has
`frame.metadata["rime"]["instruction_item_id"]` when Prism saved it. After
that reply ends, call `await llm.delete_instruction_item(item_id)` to remove
the instruction from later Prism history. The method accepts only IDs seen in
this conversation while they remain among the last 128 saved instruction IDs
that the adapter observed. It does not change Pipecat's local context.

Set `truncation="auto"`, `"disabled"`, or SDK `RetentionRatio(...)` at construction or through a settings frame. This controls future model context capacity and requires deployment support. It does not erase Pipecat's conversation record.

## Playback and interruption limits

The default `playback="estimated"` sends no playback completion receipts. Prism uses its playback estimate to admit later requests. Muted, disabled, or skipped output can delay the next request until that estimate expires.

Use `playback="discard"` at construction for a conversation that must produce no output audio. It preserves text and tools, drops audio from explicit and speech-triggered replies, and reports zero played after each response ends. Receipts follow response creation order and complete before dependent reply admission. This mode is fixed for the conversation. A muted downstream transport does not imply discard mode.

On interruption, the service cancels active generation and can send an estimated position for the latest audio output. The position uses elapsed time capped by emitted audio duration. It does not measure device playback, network buffering, or physical audibility. Application stops use the same path as user interruptions.

Early speech hints remain available through `on_rime_event` but do not change output volume. Local turn strategies control Pipecat interruptions independently of Prism's server interruption setting. There is no custom policy captured for each response.

## Events and limits

`on_session_updated` supplies acknowledged SDK session information. `on_rime_event` supplies SDK events with provider request IDs, response metadata, and diagnostics. Failures use normal Pipecat error frames.

SDK `ItemTruncated` events report the item ID, audio position, and retained server text when available. `kept_text=None` means unavailable; an empty string means no words remain. Retained text can differ from audible words. The private history ledger records retained text while keeping the generated variant for matching. The event does not change Pipecat's local context.

`ResponseEnded.truncation` reports conversation items excluded from that reply's prompt, with an optional first retained item ID. Those items remain in provider history. The summary is `None` when no exclusions are reported or the information is unavailable.
`ResponseEnded.usage` reports token counts when Prism provides them. The service emits one standard LLM usage metric per terminal response, including cancellation, when usage metrics are enabled. Missing usage is not reported as zero. Read raw diagnostics in `on_rime_event`. TTFB measures the first text delta, not device audio playback. An idle close raises SDK `RealtimeIdleTimeout`; Pipecat puts
that exception in its normal error frame.

Microphone input accepts PCM16 at 8–192 kHz with 1–32 channels. Unsupported SDK formats are converted before submission. Queue and input-byte limits bound pending work. Queue `RimeInputAudioClearFrame()` to discard queued microphone audio and converter tails, clear SDK input after any in-flight send, then accept new audio. Import it from `pipecat.services.rime.realtime.llm`.

Shutdown drains pending operations within a configurable deadline, then releases response holds and closes the connection. Injected clients remain caller-owned.

Opening a conversation uses `RimeStartupParams`, imported from the same module. Defaults allow three retries, a two-second minimum delay, a 30-second attempt limit, and a 60-second total limit. Only HTTP 408, 429, 500, 502, 503, and 504 refusals are retried. Valid `Retry-After` values can extend the delay within the total limit. Authentication failures and established-session disconnects are not retried.

## Example and validation

Run the [example](../../examples/realtime/realtime-rime.py), which includes invoice lookup and a billing settings update:

```sh
uv run --extra rime-realtime --extra runner \
  examples/realtime/realtime-rime.py --port 7860
```

Run the typed scenario against it:

```sh
uv run pipecat eval run evals/release/scenarios/scripted/rime_prism.yaml \
  --bot-url ws://localhost:7860/ws
```

The audio scenarios require Kokoro, Moonshine, and the Typesafe judge dependencies and credentials described in the [release eval guide](../../evals/release/README.md). See the [validation notes](rime-realtime-validation.md) for the scope of the checks.
