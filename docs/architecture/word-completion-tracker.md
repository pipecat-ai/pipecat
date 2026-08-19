# WordCompletionTracker

`src/pipecat/utils/context/word_completion_tracker.py`

> **Job:** for each spoken word — how much of it is this frame's, which original text
> does it stand for, and is the frame finished?

## 1. The problem

[`TextSegmentMap`](./text-segment-map.md) will tell you where a spoken word lands. It
will not tell you what to *do* about it, and it assumes the provider behaves.

TTS providers do not behave. A tracker owns one `AggregatedTextFrame` from dispatch to
"fully spoken", and has to survive:

- **Dropped events** — the provider silently never reports a word it spoke. Waiting
  forever would stall the frame and everything queued behind it.
- **Straddling tokens** — one token spans two frames (`1111And` when one frame ends with
  `1111` and the next begins with `And`).
- **Desynced texts** — the TTS text and the LLM text can fall out of alignment, so the
  slice of LLM text credited to a word may not actually contain that word. Recording it
  anyway fills the conversation context with wrong text.

So the tracker is the *policy* layer: **the map says where things are, the tracker decides
what that means for this frame**.

## 2. What it produces per word

One call in, several answers out — each one read back by
[`AggregatedFrameSequencer`](./aggregated-frame-sequencer.md) to build the frames it
pushes downstream:

```python
tracker = WordCompletionTracker(
    tts_text="Your card is 4111 1111 thanks",
    llm_text="Your card is <card>4111 1111</card> thanks",
    user_facing_text="Your card is 4111 1111 thanks",
)
complete = tracker.add_word_and_check_complete("4111")
```

| Accessor | Read by | Used for |
| --- | --- | --- |
| `word_belongs_here()` | `process_word` | Decide whether the provider dropped an event, and this slot must be force-completed |
| `add_word_and_check_complete()` | `process_word` | Advance, and learn whether the slot is now done — which triggers `flush()` |
| `get_word_for_frame()` | `process_word` | The **text** of the emitted `TTSTextFrame` |
| `get_llm_consumed()` | `process_word` | The **`raw_text`** of that frame — what the conversation context records |
| `suppress_in_context()` | `process_word` | Sets `append_to_context=False` and skips the progress frame |
| `get_overflow_word()` | `process_word` | Re-entered as a new word against the *next* slot |
| `get_accumulated_user_facing_text()` | `_build_progress_frame` | `accumulated_text` → the spoken part of the segment text (what a UI highlights) |
| `get_remaining_user_facing_text(strip=False)` | `_build_progress_frame` | `remaining_text` → the unspoken part of the segment text (what a UI leaves plain) |
| `get_remaining_tts_text()` | `force_complete` | Text of the catch-up frame when a provider goes silent |
| `get_remaining_llm_text()` | `force_complete` | `raw_text` of that catch-up frame |

So a single `process_word` call reads six of these to build one `TTSTextFrame` plus one
`AggregatedTextProgressFrame` — see [RTVI integration](./rtvi-integration.md).

### What "span attribution" means

A **span** is a contiguous slice of `llm_text`, identified by where the cursor was before
the word and where it is after. **Attributing** it means declaring: *this slice is what
that spoken word stands for.*

It matters because **the conversation context is rebuilt by concatenating the spans, not
the spoken words**. The provider says `4111`; the context needs `<card>4111`:

```
llm_text   Your │ card │ is │ <card>4111 │ 1111 │ </card> thanks
spoken     Your   card   is         4111   1111           thanks
```

Each `│` is a span boundary. Note where the tags land: the opening `<card>` is attributed
to the word that *follows* it, and the closing `</card>` to the word that *precedes* it —
neither ever arrives as its own word-timestamp event, so each has to ride along with a
real word.

The spans cover `llm_text` in order and do not overlap, so **reassembling them reproduces
the LLM's output — tags included**. That is the mechanism
that stops the context from drifting away from what the LLM wrote.

### Recovering the LLM structure

The TTS only ever synthesized the segment text, so word events carry no trace of the
delimiters the aggregator split off into `raw_text`. The tracker maps each word back onto
that fuller string:

| word     | complete | `get_word_for_frame()` | `get_llm_consumed()` |
| -------- | -------- | ---------------------- | -------------------- |
| `Your`   | False    | `Your`                 | `Your`               |
| `card`   | False    | `card`                 | `card`               |
| `is`     | False    | `is`                   | `is`                 |
| `4111`   | False    | `4111`                 | **`<card>4111`**     |
| `1111`   | False    | `1111`                 | `1111`               |
| `thanks` | **True** | `thanks`               | **`</card> thanks`** |

The conversation context is rebuilt from the `raw_text` column, so it stores
`<card>4111 1111</card>` — not the bare digits the provider reported.

### Suppressing mid-transform words

For a transformed segment the individual spoken words are meaningless to the context.
`suppress_in_context()` is True while the map sits mid-segment, and attribution is
withheld until the completing word carries the whole original span:

| word      | `get_word_for_frame()` | `get_llm_consumed()` | `suppress_in_context()` |
| --------- | ---------------------- | -------------------- | ----------------------- |
| `is`      | `is`                   | `is`                 | False                   |
| `forty-two` | `forty-two`          | `None`               | **True**                |
| `dollars` | `dollars`              | `None`               | **True**                |
| `and`     | `and`                  | `None`               | **True**                |
| `fifty`   | `fifty`                | `None`               | **True**                |
| `cents`   | `cents`                | **`$42.50`**         | False                   |

The words are still emitted — the UI needs them to highlight progress — but **only the
last one writes to the context, and it writes `$42.50`**.

## 3. Recovery: dropped events

Before advancing, every word is checked with `word_belongs_here`. A word that does not
match means the provider skipped something, so the slot is **force-completed** rather
than left hanging:

```python
tracker = WordCompletionTracker("Hello there world")
tracker.add_word_and_check_complete("Hello")     # normal
tracker.add_word_and_check_complete("Goodbye")   # belongs to the *next* frame
```

| word      | `word_belongs_here` | complete | `get_word_for_frame()` | `get_overflow_word()` |
| --------- | ------------------- | -------- | ---------------------- | --------------------- |
| `Hello`   | True                | False    | `Hello`                | `None`                |
| `Goodbye` | **False**           | **True** | **`there world`**      | **`Goodbye`**         |

**The unspoken remainder `there world` is still emitted**, so the context keeps the full
sentence; the foreign word is handed back as overflow for the next slot. From this point
`_force_completed` — not the map — is the authoritative completion signal, since the map
was never advanced and its own `is_complete` stays stale.

## 4. Recovery: straddling tokens

```python
tracker = WordCompletionTracker("The code is 1111")
```

| word      | complete | `get_word_for_frame()` | `get_overflow_word()` |
| --------- | -------- | ---------------------- | --------------------- |
| `The`     | False    | `The`                  | `None`                |
| `code`    | False    | `code`                 | `None`                |
| `is`      | False    | `is`                   | `None`                |
| `1111And` | **True** | **`1111`**             | **`And`**             |

The token is split at the frame boundary. **Each half is attributed to the frame it
actually belongs to.**

## 5. The attribution safeguard

`_discard_llm_span_if_frame_word_missing` guards against that desync. It enforces one
rule: **the LLM span credited to a
word must contain that word.** If it does not, the two texts have fallen out of
alignment and the span is dropped with a warning rather than corrupting the context.

The comparison is **deliberately lenient** — casefolded, with hyphens and spaces
collapsed — so a legitimate replacement is not mistaken for a desync:

| Case                  | LLM text     | Spoken   | Verdict            |
| --------------------- | ------------ | -------- | ------------------ |
| Case-only replacement | `SQL`        | `sql`    | Match              |
| Joiner replacement    | `body-pump`  | `BODYPUMP` | Match            |
| Genuinely out of sync | `hello`      | `goodbye` | **Discard**       |

One special case gets repaired instead of discarded. Punctuation is normally swept into
the *preceding* word's span, so a provider that reports it with the *following* word
(`, I` rather than `Yeah,`) would emit it twice. The duplicate is trimmed from the frame
word and the attribution is kept.

Validation is skipped when the completing word finished a transformed segment — `dollars`
is never going to appear inside `$42.50`.

## 6. Public surface

| Member                                | Purpose                                            |
| ------------------------------------- | -------------------------------------------------- |
| `add_word_and_check_complete(word)`   | Record a word; returns True when the frame is done |
| `word_belongs_here(word)`             | Dry run, used to detect dropped events              |
| `get_word_for_frame()`                | This frame's share of the last word                 |
| `get_overflow_word()`                 | The next frame's share                              |
| `get_llm_consumed()`                  | LLM span for the last word                          |
| `suppress_in_context()`               | Mid-transform, keep out of the context              |
| `get_accumulated_*` / `get_remaining_*` | Spoken and unspoken text, per channel             |
| `is_complete`                         | Force-completed, or the segment map says done       |
| `reset()`                             | Rewind cursors, keep the texts                      |

The paired accessors are what drive RTVI progress. Mid-sentence:

```python
tracker = WordCompletionTracker("Hello there world")
tracker.add_word_and_check_complete("Hello")

tracker.get_accumulated_user_facing_text()   # 'Hello'
tracker.get_remaining_user_facing_text()     # 'there world'
```

`get_remaining_user_facing_text(strip=False)` preserves leading whitespace, so accumulated
+ remaining reconstructs the segment text exactly. That guarantee is what every consumer
of a progress frame relies on (see
[the guarantee](./README.md#33-the-guarantee-that-makes-it-useful)).

### `is_complete` is delegated, not tracked

**The tracker keeps no completion bookkeeping of its own.** It forwards the question to
the segment map, which owns the only cursor that can answer it:

```python
@property
def is_complete(self) -> bool:
    return self._force_completed or self._segment_map.is_complete
```

The one exception is the force-completed slot. There the map was deliberately **never
advanced** — the offending word was routed to the next slot instead — so its own
`is_complete` stays stale and would keep reporting `False` forever. `_force_completed` is
the override that makes the verdict stick, and from that point on it is the authoritative
answer.

### "Complete" and "no room left" are different questions

`is_complete` is alphanumeric-based: a frame whose remainder is only punctuation or markup
reports True before that remainder has been walked. That is the right question for
releasing whatever is queued behind the slot, and the wrong one for deciding whether to
accept another word — a frame ending in an emoji is already `is_complete` when the
emoji's own event arrives.

The guard that rejects a late word therefore asks the stricter question, whether every raw
character of `tts_text` has been consumed:

```python
if self._force_completed or self._segment_map.raw_pos >= len(self._tts_text):
```

The gap between the two answers is exactly the trailing non-alphanumeric content, and it
is the window in which a final emoji or a separated punctuation mark is still welcome.

## 7. Tests

`tests/test_word_completion_tracker.py` — 201 tests, the largest suite of the three.
Most of it is a regression corpus of real provider behaviour.

| Group                    | Classes                                                        |
| ------------------------ | -------------------------------------------------------------- |
| Mechanics                | `Basic`, `Reset`, `EdgeCases`, `RemainingText`, `AccumulatedText`, `UserFacingText` |
| Recovery                 | `MissingWord`, `Overflow`, `MultiFrameSimulation`               |
| TTS provider quirks      | `UnicodeSymbolSubstitution`, `AddedTerminalPunctuation`, `CaseFolding`, `AccentFolding`, `SpaceBeforePunctuation`, `MultiAttributeSsmlTag`, `EmojiInSentence`, `CJK`, `StrayAngleBracket` |
| Transform interaction    | `WithTransforms`, `TokenChangingReplacements`, `TransformAtEndOfUtterance`, `LLMText`, `Normalization` |

## Related

- [Architecture overview](./README.md)
- [TextSegmentMap](./text-segment-map.md) — the alignment this layer drives
- [AggregatedFrameSequencer](./aggregated-frame-sequencer.md) — the layer that owns these trackers
- [RTVI integration](./rtvi-integration.md) — where the accumulated/remaining text ends up
