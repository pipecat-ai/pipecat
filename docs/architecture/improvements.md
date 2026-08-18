# Possible Improvements

Observations about the current implementation that are worth revisiting. Nothing here is
a bug — the behaviour is correct and covered by tests. These are places where the code
could be clearer or sturdier, recorded so the reasoning is not lost.

---

## 1. The `user_facing_*` naming

**Where:** `TextSegmentMap.user_facing_pos`, `WordCompletionTracker.user_facing_text`,
`get_accumulated_user_facing_text()`, `get_remaining_user_facing_text()`, plus internal
uses in `AggregatedFrameSequencer`.

The name describes **one consumer** rather than the text itself. The channel is simply
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

---

## 2. `_word_matches_remaining` duplicates the segment walk

**Where:** `TextSegmentMap._advance_raw` and `TextSegmentMap._word_matches_remaining`.

`word_belongs_current_segment()` needs to answer "would this token match?" without moving
any cursor, so `_word_matches_remaining` replays the same hop-by-hop walk as
`_advance_raw` against copies of the cursor state. The two loops classify hops in the same
order and must agree on every outcome, but they are separate code kept in sync by
convention — a new hop kind or a change in hop ordering has to be applied twice, and a
divergence would show up as a token that passes the dry run and then fails to advance.

**Candidate fix:** a single walk parameterised by whether it commits, so the dry run and
the real advance cannot drift apart.

**Cost:** contained to one file, but the committing path has side effects
(`_commit_raw_span` moves three cursors and completes segments) that would need care to
keep out of the dry run.
