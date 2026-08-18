# WordCompletionTracker

`src/pipecat/utils/context/word_completion_tracker.py`

## The problem

[`TextSegmentMap`](./text-segment-map.md) will tell you where a spoken word lands. It
will not tell you what to *do* about it, and it assumes the provider behaves.

Providers do not behave. A tracker owns one `AggregatedTextFrame` from dispatch to
"fully spoken", and has to survive:

- **Dropped events** — the provider silently never reports a word it spoke. Waiting
  forever would stall the frame and everything queued behind it.
- **Straddling tokens** — one token spans two frames (`1111And` when one frame ends with
  `1111` and the next begins with `And`).
- **Attribution drift** — the span of LLM text credited to a word has to actually contain
  that word, or the conversation context silently fills with wrong text.

So the tracker is the *policy* layer: the map says where things are, the tracker decides
what that means for this frame.

## What it produces per word

One call in, four answers out:

```python
tracker = WordCompletionTracker(
    tts_text="Your card is 4111 1111 thanks",
    llm_text="Your card is <card>4111 1111</card> thanks",
    user_facing_text="Your card is 4111 1111 thanks",
)
complete = tracker.add_word_and_check_complete("4111")
```

| Accessor                     | Answers                                              |
| ---------------------------- | ---------------------------------------------------- |
| return value / `is_complete` | Is this frame fully spoken?                          |
| `get_word_for_frame()`       | What text belongs to *this* frame?                   |
| `get_llm_consumed()`         | Which LLM span does it represent?                    |
| `get_overflow_word()`        | What spilled into the next frame?                    |
| `suppress_in_context()`      | Should this word be kept out of the context?         |

### Recovering the LLM structure

Word events arrive stripped of the delimiters the LLM wrote. The tracker maps each one
back, so the closing tag rides along with the word that follows it:

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
| `forty`   | `forty`                | `None`               | **True**                |
| `two`     | `two`                  | `None`               | **True**                |
| `dollars` | `dollars`              | `None`               | **True**                |
| `and`     | `and`                  | `None`               | **True**                |
| `fifty`   | `fifty`                | `None`               | **True**                |
| `cents`   | `cents`                | **`$42.50`**         | False                   |

The words are still emitted — the UI needs them to highlight progress — but only the last
one writes to the context, and it writes `$42.50`.

## Recovery: dropped events

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

The unspoken remainder `there world` is still emitted, so the context keeps the full
sentence; the foreign word is handed back as overflow for the next slot. From this point
`_force_completed` — not the map — is the authoritative completion signal, since the map
was never advanced and its own `is_complete` stays stale.

## Recovery: straddling tokens

```python
tracker = WordCompletionTracker("The code is 1111")
```

| word      | complete | `get_word_for_frame()` | `get_overflow_word()` |
| --------- | -------- | ---------------------- | --------------------- |
| `The`     | False    | `The`                  | `None`                |
| `code`    | False    | `code`                 | `None`                |
| `is`      | False    | `is`                   | `None`                |
| `1111And` | **True** | **`1111`**             | **`And`**             |

The token is split at the frame boundary. Each half is attributed to the frame it
actually belongs to.

## The attribution safeguard

`_discard_llm_span_if_frame_word_missing` enforces one invariant: **the LLM span credited
to a word must contain that word.** If it does not, the two texts have drifted and the
span is dropped with a warning rather than corrupting the context.

The comparison is deliberately lenient — casefolded, with hyphens and spaces collapsed —
so legitimate replacements are not mistaken for drift:

| Case                  | LLM text     | Spoken   | Verdict            |
| --------------------- | ------------ | -------- | ------------------ |
| Case-only replacement | `SQL`        | `sql`    | Match              |
| Joiner replacement    | `body-pump`  | `BODYPUMP` | Match            |
| Genuine desync        | `hello`      | `goodbye` | **Discard**       |

One special case gets repaired instead of discarded. Punctuation is normally swept into
the *preceding* word's span, so a provider that reports it with the *following* word
(`, I` rather than `Yeah,`) would emit it twice. The duplicate is trimmed from the frame
word and the attribution is kept.

Validation is skipped when the completing word finished a transformed segment — `dollars`
is never going to appear inside `$5`.

## Public surface

| Member                                | Purpose                                            |
| ------------------------------------- | -------------------------------------------------- |
| `add_word_and_check_complete(word)`   | Record a word; returns True when the frame is done |
| `word_belongs_here(word)`             | Dry run, used to detect dropped events              |
| `get_word_for_frame()`                | This frame's share of the last word                 |
| `get_overflow_word()`                 | The next frame's share                              |
| `get_llm_consumed()`                  | LLM span for the last word                          |
| `suppress_in_context()`               | Mid-transform, keep out of the context              |
| `get_accumulated_*` / `get_remaining_*` | Spoken and unspoken text, per channel             |
| `is_complete`                         | Force-completed, or the map says done               |
| `reset()`                             | Rewind cursors, keep the texts                      |

The paired accessors are what drive RTVI progress. Mid-sentence:

```python
tracker = WordCompletionTracker("Hello there world")
tracker.add_word_and_check_complete("Hello")

tracker.get_accumulated_user_facing_text()   # 'Hello'
tracker.get_remaining_user_facing_text()     # 'there world'
```

`get_remaining_user_facing_text(strip=False)` preserves leading whitespace, so
accumulated + remaining reconstructs the original exactly.

## Tests

`tests/test_word_completion_tracker.py` — 199 tests, the largest suite of the three.
Most of it is a regression corpus of real provider behaviour.

| Group                    | Classes                                                        |
| ------------------------ | -------------------------------------------------------------- |
| Mechanics                | `Basic`, `Reset`, `EdgeCases`, `RemainingText`, `AccumulatedText`, `UserFacingText` |
| Recovery                 | `MissingWord`, `Overflow`, `MultiFrameSimulation`               |
| Provider quirks          | `UnicodeSymbolSubstitution`, `AddedTerminalPunctuation`, `CaseFolding`, `AccentFolding`, `SpaceBeforePunctuation`, `MultiAttributeSsmlTag`, `EmojiInSentence`, `CJK`, `StrayAngleBracket` |
| Transform interaction    | `WithTransforms`, `TokenChangingReplacements`, `TransformAtEndOfUtterance`, `LLMText`, `Normalization` |
