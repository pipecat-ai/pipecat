# Possible Improvements

Observations about the current implementation that are worth revisiting. Nothing here is
a bug — the behaviour is correct and covered by tests. These are places where the code
could be clearer or sturdier, recorded so the reasoning is not lost.

---

## 1. The `user_facing_*` naming

**Where:** `TextSegmentMap.user_facing_pos`, `WordCompletionTracker.user_facing_text`,
`get_accumulated_user_facing_text()`, `get_remaining_user_facing_text()`, plus internal
uses in `AggregatedFrameSequencer`.

**The name describes one consumer rather than the text itself.** The channel is simply
`AggregatedTextFrame.text` — the segment as the aggregator produced it — and the property
that makes it useful is that a progress frame always splits it exactly:

```
accumulated_text + remaining_text == AggregatedTextFrame.text
```

Any processor holding the segment frame can rely on that. A UI highlighting spoken words
is the common case, but a transcript writer, redaction filter, or logger consumes it on
identical terms — so naming the channel after the UI makes the API read as more
RTVI-specific than it is.

**Candidate replacement:** `segment_text` / `segment_pos`, matching the `segment_id`
already used on `AggregatedTextProgressFrame` and in the RTVI protocol.

**Cost:** three public members plus a constructor parameter, all in
`src/pipecat/utils/context/`. Nothing under `services/`, `processors/`, or `frames/`
references the name, so a bot author never types it — but the members are public, so a
rename needs the usual deprecation cycle.

