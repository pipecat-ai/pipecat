# Turn Probability Logging — Sparsity Limitation & Solutions

## Current Limitation

The conversation logger currently produces **sparse turn probability events** — typically just **1–2 per user speaking turn**. This is not a logging bug; it reflects the architecture of the smart-turn analyzer.

### Root Cause

The pipeline uses `LocalSmartTurnAnalyzerV3`, a **batch analyzer** (`BaseSmartTurn`):

- **No streaming inference**: `append_audio()` only accumulates audio into a buffer.
- **Single model invocation per turn**: `analyze_end_of_turn()` is called only when `VADUserStoppedSpeakingFrame` arrives.
- **Result**: One `TurnMetricsData` emitted per turn (with exact duplicates due to `on_push_frame` firing ~6x per boundary crossing).

The batch design is intentional — ML models perform better with full-utterance context rather than partial audio.

---

## Concrete Solutions

### Option 1: Switch to Streaming Analyzer (**Easiest**)

Replace `LocalSmartTurnAnalyzerV3` with `KrispVivaTurn`:

- **Streaming analyzer** — emits `TurnMetricsData` per audio chunk (~50–100ms intervals).
- **Finer granularity**: ~10–20 probability samples per turn instead of 1–2.
- **Trade-off**: Requires Krisp SDK (commercial); may have different latency/accuracy characteristics.

**Implementation location**: `bot_dev.py`, in `run_bot()` where `LLMUserAggregatorParams(vad_analyzer=...)` is configured.

**Steps**:
1. Import `KrispVivaTurn` from `pipecat.audio.turn.krisp_viva_turn`
2. Replace `LocalSmartTurnAnalyzerV3()` with `KrispVivaTurn()` in the turn stop strategy
3. Test with live conversation to verify sampling rate

---

### Option 2: Custom Periodic Inference (**More Work**)

Subclass `LocalSmartTurnAnalyzerV3` to run inference on accumulated audio at fixed intervals (e.g., every 500ms):

- **Full control**: Choose inference frequency, model, and output format.
- **Self-contained**: No external SDK required.
- **Trade-off**: Requires implementing a custom analyzer; more complex debugging; potential for duplicate or overlapping inferences.

**Implementation sketch**:
```python
class PeriodicSmartTurnAnalyzer(LocalSmartTurnAnalyzerV3):
    """Runs inference periodically instead of only at turn end."""
    
    def __init__(self, inference_interval_ms=500, **kwargs):
        super().__init__(**kwargs)
        self._inference_interval_ms = inference_interval_ms
        self._last_inference_time = time.time()
        self._pending_metrics = []
    
    def append_audio(self, buffer: bytes, is_speech: bool):
        state = super().append_audio(buffer, is_speech)
        
        # Periodic inference while VAD is active
        if is_speech and time.time() - self._last_inference_time > self._inference_interval_ms / 1000:
            # Call _process_speech_segment directly for intermediate predictions
            self._last_inference_time = time.time()
            # Queue metrics for emission later
        
        return state
```

**Location**: Create `src/pipecat/audio/turn/smart_turn/periodic_smart_turn.py` or add as a new class in the observer itself.

---

## Recommendation

**Start with Option 1** (KrispVivaTurn) if Krisp SDK is available in your environment:
- Minimal code change
- Production-grade streaming analyzer
- Immediate 10–20x improvement in turn probability granularity

**Fall back to Option 2** if Krisp is unavailable or undesired.

**Apply the deduplication fix** in parallel — it's simple and reduces log noise regardless of which analyzer is used.


## Questions
Could we increase the firing rate of the smart turn analyzer by reducing the silence period needed to trigger it? i.e. triggering at "Transition-relevance-places"?


## Subprocess-Level State Logging

### Motivation

The current logger captures high-level pipeline events (bot speaking, user turn, etc.) but not the internal states of individual services. Logging per-service state transitions would allow:

- Reconstructing high-level states (`bot_speaking`, `user_interrupting`, `processing`) from first principles rather than relying on the coarse `BotStartedSpeakingFrame` / `BotStoppedSpeakingFrame` heuristic.
- Debugging latency (e.g. how much time is LLM TTFB vs. TTS startup vs. audio buffering).
- Detecting failure modes like TTS stall, LLM timeout, or STT silence.

### States to capture per service

| Service | Key frames / events |
|---|---|
| STT (Deepgram) | `TranscriptionFrame` (final), `InterimTranscriptionFrame`, VAD passthrough |
| LLM (OpenAI) | `LLMFullResponseStartFrame`, `LLMFullResponseEndFrame`, `LLMTextFrame` (token stream), `LLMRunFrame` (trigger) |
| TTS (Cartesia) | `TTSStartedFrame`, `TTSStoppedFrame`, `TTSTextFrame`, `TTSAudioRawFrame` (first chunk = startup latency) |
| Transport output | `BotStartedSpeakingFrame`, `BotStoppedSpeakingFrame`, `BotSpeakingFrame` |
| VAD / Turn | `VADUserStartedSpeakingFrame`, `VADUserStoppedSpeakingFrame`, `UserStoppedSpeakingFrame` |
| Interruption | `InterruptionFrame` (marks the moment the bot was cut off) |

### Translation to high-level states

A simple FSM reconstructed from subprocess events:

- `idle` → `listening`: after `BotStoppedSpeakingFrame` (or on session start)
- `listening` → `user_speaking`: `VADUserStartedSpeakingFrame`
- `user_speaking` → `processing`: `UserStoppedSpeakingFrame` + `LLMRunFrame`
- `processing` → `speaking`: `BotStartedSpeakingFrame` (with sub-phases: LLM streaming → TTS startup → audio playback)
- `speaking` → `user_interrupting`: `VADUserStartedSpeakingFrame` while `bot_speech = active`; confirm with `InterruptionFrame`
- `user_interrupting` → `user_speaking`: `InterruptionFrame` clears the bot queue; state fully transitions on `BotStoppedSpeakingFrame`

### Implementation approach

Extend `ConversationLogObserver.on_push_frame` to emit fine-grained `service_state_changed` events alongside the existing coarse events. Each event would carry `service` (e.g. `"tts"`, `"llm"`), `state`, and `ts_ns`. The high-level state reconstruction can then be done as a post-processing step on the JSONL log rather than in the hot path.