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
  forever would stall the frame and everything queued behind it, so the frame ends early
  and emits its own remainder.
- **Text no word arrives for** — a closing `</card>`, or a tag between the last word and
  its punctuation, is never spoken. Whatever is left once everything speakable is done
  belongs to this frame, so the word that finishes it claims the rest.
- **Tokens that overhang the frame** — a token can spill into the next frame (`1111And`)
  or lead with punctuation the previous word already carried (`, I`). Only the middle is
  this frame's.

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
| `add_word_and_check_complete()` | `process_word` | Advance, and learn whether the slot is now done |
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

## 4. Trimming a token to this frame's share

Neither end of an incoming token necessarily belongs to this frame, and
[`TextSegmentMap`](./text-segment-map.md#trimming-a-token-to-what-it-actually-covers)
measures both:

| Map output | The part that is not this frame's | Example |
| --- | --- | --- |
| `last_leading_duplicate` | Head repeating punctuation the previous word already carried | `, I` → drop 2 |
| `last_overflow` | Tail spilling into the next frame | `1111And` → `And` |

The tracker keeps what is between them:

```python
self._frame_word = word[head:tail]
```

**A tail example.** One token closes this frame and opens the next:

```python
tracker = WordCompletionTracker("The code is 1111")
```

| word | complete | `get_word_for_frame()` | `get_overflow_word()` |
| --------- | -------- | ---------------------- | --------------------- |
| `The`     | False    | `The`                  | `None`                |
| `code`    | False    | `code`                 | `None`                |
| `is`      | False    | `is`                   | `None`                |
| `1111And` | **True** | **`1111`**             | **`And`**             |

The overflow is fed to the next frame's tracker, so each half is attributed to the frame
it actually belongs to.

**A head example.** The comma trailing `Yeah` is already part of its span, so a provider
reporting it again on the next token would emit it twice:

| after | token | `get_word_for_frame()` |
| --- | --- | --- |
| `Yeah` (span `Yeah,`) | `, I` | **`I`** |

That is the whole of the tracker's word-text handling — both decisions are the map's,
because both come from cursors it owns.

The attributed LLM span is used as given. The spans cover `llm_text` in order, so dropping
one removes text from the transcript rather than protecting it.

## 5. Public surface

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

## 6. Tests

`tests/test_word_completion_tracker.py` — 203 tests, the largest suite of the three.
Most of it is a regression corpus of real provider behaviour.

| Group                    | Classes                                                        |
| ------------------------ | -------------------------------------------------------------- |
| Mechanics                | `Basic`, `Reset`, `EdgeCases`, `RemainingText`, `AccumulatedText`, `UserFacingText` |
| Recovery                 | `MissingWord`, `Overflow`, `MultiFrameSimulation`, `ForceCompleteAttributesTaggedRemainder` |
| TTS provider quirks      | `UnicodeSymbolSubstitution`, `AddedTerminalPunctuation`, `CaseFolding`, `AccentFolding`, `SpaceBeforePunctuation`, `MultiAttributeSsmlTag`, `EmojiInSentence`, `CJK`, `StrayAngleBracket` |
| Transform interaction    | `WithTransforms`, `TokenChangingReplacements`, `TransformAtEndOfUtterance`, `LLMText`, `Normalization` |

## Related

- [Architecture overview](./README.md)
- [TextSegmentMap](./text-segment-map.md) — the alignment this layer drives
- [AggregatedFrameSequencer](./aggregated-frame-sequencer.md) — the layer that owns these trackers
- [RTVI integration](./rtvi-integration.md) — where the accumulated/remaining text ends up
