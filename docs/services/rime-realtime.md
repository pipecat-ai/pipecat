# Rime Prism realtime conversations

`RimeRealtimeLLMService` uses Prism for speech input, generated speech, and tools through the direct Rime SDK session. It provides a context aggregator pair and uses Pipecat's standard output transports. Install the optional dependency:

```sh
uv pip install 'pipecat-ai[rime-realtime]'
```

The extra requires `rimelabs-sdk>=0.1.0a25,<0.2`.

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

Typed input uses the existing `LLMMessagesAppendFrame(messages=[...], run_llm=True)` path, including RTVI text input. Rime imports the new history before requesting a reply. This does not use Prism's dedicated typed-turn operation, so the response does not identify that text as its current input item.

## Settings, tools, and fillers

Use `LLMUpdateSettingsFrame(delta=RimeRealtimeLLMService.Settings(...))` for instructions, tools, voice, server interruption, fillers, truncation, or response labels. Omitted settings retain their values. `LLMSetToolsFrame` also updates the provider's tools.

Instructions, tools, and fillers can share one settings delta. This supports a settings-only handoff in the same conversation. History changes are separate operations, so a handoff that also edits history is not atomic. Response labels use a separate SDK operation; their failure does not undo an accepted settings update.

Tool fillers use SDK `PrismFiller` values with `type="none"`, `"text"`, or `"auto"`. Keys must name advertised tools. Removing a tool also removes its filler unless the delta explicitly supplies a replacement filler map.

Tools execute through Pipecat. Each final result is submitted once before requesting continuation. A queued call uses Pipecat's current registered handler; the service does not pin handlers to a response's settings revision. Intermediate tool results and forced or disabled tool selection are unsupported.

The SDK acknowledgment controls accepted provider settings. Normal Pipecat aggregators can change local context or tools before provider acceptance. Rejected changes do not roll those local changes back.

## History and recovery

Append, update, and transform message frames change local context through the Rime pair. Appends import only the supplied messages. Updates and transforms replace supported provider history. An edit with `run_llm=False` remains local until another operation requests inference through either aggregator.

Ordinary transcript and assistant aggregation never replace provider history. A later `LLMRunFrame` applies pending application edits and requests a reply. Direct `LLMContextFrame` input is an explicit history snapshot. Change messages through these frames; changes made directly to an already-bound context are not synchronized by `LLMRunFrame`.

The private history ledger matches explicit replacement messages to provider records by content and order. Identical messages can retain their server identities.

History imports support user, assistant, and system text plus completed tool calls and results. Imported text messages have a 4000-character limit. Images, video, pending tool-call imports, and arbitrary developer messages are unsupported. Loading completed calls never executes them.

Replacing or reordering history can require multiple provider requests. A rebuild requires no active reply or pending tool call. Rejected edits can leave local and provider history different. Partial application or an unknown request outcome makes the service unusable. There is no automatic reconnect or replay. Start a new instance with known valid, completed history for recovery.

`await llm.get_history()` reads completed server history. Active items can be absent from that snapshot.
For a `RimeReplyFrame` with an instruction, the `LLMFullResponseStartFrame` has
`frame.metadata["rime"]["instruction_item_id"]` when Prism saved it. After
that reply ends, call `await llm.delete_instruction_item(item_id)` to remove
the instruction from later Prism history. The method accepts only IDs seen in
this conversation while they remain among the last 128 saved instruction IDs
that the adapter observed. It does not change Pipecat's local context.

Set `truncation="auto"`, `"disabled"`, or SDK `RetentionRatio(...)` at construction or through a settings frame. This controls future model context capacity and requires deployment support. It does not erase Pipecat's conversation record.

## Playback and interruption limits

The service sends no playback completion receipts. Prism uses its playback estimate to admit later requests. Muted, disabled, or skipped output can delay the next request until that estimate expires.

On interruption, the service cancels active generation and can send an estimated position for the latest audio output. The position uses elapsed time capped by emitted audio duration. It does not measure device playback, network buffering, or physical audibility. Application stops use the same path as user interruptions.

Early speech hints remain available through `on_rime_event` but do not change output volume. Local turn strategies control Pipecat interruptions independently of Prism's server interruption setting. There is no custom policy captured for each response.

## Events and limits

`on_session_updated` supplies acknowledged SDK session information. `on_rime_event` supplies SDK events with provider request IDs, response metadata, and diagnostics. Failures use normal Pipecat error frames.

SDK `ItemTruncated` events report the item ID, audio position, and retained server text when available. `kept_text=None` means unavailable; an empty string means no words remain. Retained text can differ from audible words. The event does not change Pipecat's local context.

`ResponseEnded.truncation` reports conversation items excluded from that reply's prompt, with an optional first retained item ID. Those items remain in provider history. The summary is `None` when no exclusions are reported or the information is unavailable.
`ResponseEnded.usage` reports token counts when Prism provides them. Read it in
`on_rime_event`. An idle close raises SDK `RealtimeIdleTimeout`; Pipecat puts
that exception in its normal error frame.

Microphone input accepts PCM16 at 8–192 kHz with 1–32 channels. Unsupported SDK formats are converted before submission. Queue and input-byte limits bound pending work. Shutdown has a configurable deadline and does not wait for playback receipts.

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
