# TextSegmentMap

`src/pipecat/utils/context/text_segment_map.py`

> **Job:** given a word the TTS just spoke, say where we are in *all three* texts.

## 1. The problem

A provider tells you it just spoke `dollars`. Nothing in that event says which part of
`Your balance is $42.50` you have reached — the word does not appear in that text at all.

Every transform opens the same gap:

| Transform | Original | Sent to TTS |
| --- | --- | --- |
| Currency expansion | `$42.50` | `forty-two dollars and fifty cents` |
| Number expansion | `room 1994` | `room 1 9 9 4` |
| SSML markup | `1234-5678` | `<spell>1234-5678</spell>` |
| URL cleanup | `https://pipecat.ai` | `pipecat.ai` |
| Pattern delimiters | `<card>4111</card>` | `4111` |

Something has to translate a position in the spoken stream back into a position in the
original text. That is the entire job of `TextSegmentMap`.

## 2. Building the map: opcodes

At construction the map compares the TTS text against the original text using
`difflib.SequenceMatcher`, **at word level** (the text is tokenized on whitespace, keeping
the whitespace as tokens so offsets stay exact).

### What an opcode is

`SequenceMatcher` does not just say "these differ" — it returns a list of **opcodes**,
each one an instruction describing how a run of the original maps onto a run of the TTS
text. There are four kinds:

| Opcode | Meaning |
| --- | --- |
| `equal` | This run is identical on both sides |
| `replace` | This run of the original was rewritten into that run of the TTS text |
| `delete` | This run exists in the original but not in the TTS text |
| `insert` | This run exists in the TTS text but not in the original |

Real output for two examples:

```
original = 'Your balance is $42.50'
tts      = 'Your balance is forty-two dollars and fifty cents'

   equal    original='Your balance is '   tts='Your balance is '
   replace  original='$42.50'             tts='forty-two dollars and fifty cents'
```

```
original = 'Visit https://pipecat.ai now'
tts      = 'Visit pipecat.ai now'

   equal    original='Visit '             tts='Visit '
   replace  original='https://pipecat.ai' tts='pipecat.ai'
   equal    original=' now'               tts=' now'
```

Each opcode becomes one `TextSegment`, carrying both sides plus the span it occupies in
the original text:

```python
TextSegment(original='$42.50',
            tts='forty-two dollars and fifty cents',
            original_start=16, original_end=22)
```

### Transformed vs unchanged segments

A segment is **transformed** (`TextSegment.is_transformed`) when its two sides cannot be
walked character for character. Three things trigger it:

1. The alphanumeric content differs (`$42.50` vs `forty-two dollars…`)
2. The word count differs (a replacement changed tokenization)
3. The TTS side contains markup — even if the spoken content is identical

That third rule is why `<phoneme alphabet="ipa">Siobhan</phoneme>` counts as transformed:
the raw cursor must move through the tag characters while the original cursor stays put.

### Splitting markup into its own segment

Rule 3 has a cost. A segment holding *any* markup is atomic, so one tag in the middle of
otherwise identical text would freeze the cursors across the whole sentence. To contain
that, `_build` splits an `equal` opcode around the markup it carries:

```python
_split_markup_runs("I love to count <spell>1234</spell>.")
# -> ["I love to count ", "<spell>1234</spell>."]
```

```
"I love to count "       plain   — cursors advance word by word
"<spell>1234</spell>."   atomic  — commits when its last word lands
```

Only `equal` opcodes can be split this way: both sides hold the same text, so a single
offset cuts both. The other kinds differ side to side, leaving no shared offset to cut at.

## 3. Walking the map: one cursor drives three

There is exactly one real cursor — `raw_pos`, the position reached in the TTS text.
`user_facing_pos` and `llm_pos` are derived from it:

- **Unchanged segment** — they advance proportionally, word for word.
- **Transformed segment** — they are *held* until the segment's entire TTS text is
  consumed, then jump to the end of its original span in one step. There is no meaningful
  position halfway through `$42.50`, so the map refuses to invent one.

### Worked example

```python
TextSegmentMap(
    tts_text      = "Your balance is forty-two dollars and fifty cents",
    original_text = "Your balance is $42.50",     # user-facing
    llm_text      = "Your balance is $42.50",     # what the LLM wrote
)
```

Feeding the eight spoken words in one at a time:

| word | `raw_pos` | `user_facing_pos` | `llm_pos` | `in_transformed_segment` | accumulated user-facing |
| --- | ---: | ---: | ---: | --- | --- |
| `Your` | 4 | 4 | 4 | False | `Your` |
| `balance` | 12 | 12 | 12 | False | `Your balance` |
| `is` | 15 | 15 | 15 | False | `Your balance is` |
| `forty-two` | 25 | 15 | 15 | **True** | `Your balance is` |
| `dollars` | 33 | 15 | 15 | **True** | `Your balance is` |
| `and` | 37 | 15 | 15 | **True** | `Your balance is` |
| `fifty` | 43 | 15 | 15 | **True** | `Your balance is` |
| `cents` | 49 | **22** | **22** | False | `Your balance is $42.50` |

The raw cursor climbs steadily. The other two freeze at 15 for four words, then jump
straight to 22 when the segment completes. `last_completed_segment` then reports the
`$42.50` segment — the signal callers use to attribute the whole original span to the word
that finished it.

When the LLM text has extra delimiters the third cursor diverges from the second:
`llm_text="Your balance is <price>$42.50</price>"` moves `llm_pos` past the tags while
`user_facing_pos` never sees them.

## 4. Matching real provider tokens

The other half of the job: word-timestamp tokens are *messy*, and no two providers agree.
Rather than special-casing each one, `_classify_hop` matches the token against the
segment's remaining raw text using three strategies, in order, stopping at the first hit.

```mermaid
flowchart TD
    W(["incoming token"]) --> S1

    S1{"<b>1 · literal</b><br/>at 3 skip offsets<br/><i>as-is · past whitespace ·<br/>past all punctuation</i>"}
    S1 -->|hit| R
    S1 -->|miss| S2

    S2{"<b>2 · case + accent folded</b><br/><i>both sides folded,<br/>word boundary required</i>"}
    S2 -->|hit| R
    S2 -->|miss| S3

    S3{"<b>3 · markup stripped</b><br/><i>tags removed from<br/>both sides</i>"}
    S3 -->|hit| R
    S3 -->|miss| ST

    R(["<b>PLACED</b> — fits here<br/><b>CROSSES</b> — spills onward"])
    ST{"any alphanumeric<br/>left in this segment?"}
    ST -->|no| EX(["<b>EXHAUSTED</b><br/>drain, retry next segment"])
    ST -->|yes| NM(["<b>NO_MATCH</b><br/>nudge past punctuation, stop"])
```

Every strategy also retries with the token's own trailing punctuation removed. The whole
thing is stateless — recomputed fresh each call, no tag parsing, no cross-call
bookkeeping.

### Which strategy fires, for real token shapes

| Provider behaviour | Segment remaining | Token | Strategy | Result |
| --- | --- | --- | --- | --- |
| Plain word | `hello world` | `hello` | 1 literal | `PLACED` (5 chars) |
| Token carries its own leading space (Inworld) | `" world"` | `" world"` | 1 literal | `PLACED` (6 chars) |
| Provider skipped punctuation it did not speak | `", I can help"` | `I` | 1 literal *(skip offset)* | `PLACED` (3 chars) |
| Provider added a terminal period | `account and more` | `account.` | 1 literal *(trailing trim)* | `PLACED` (7 chars) |
| Provider lowercased the word | `SQL is great` | `sql` | 2 folded | `PLACED` (3 chars) |
| Provider stripped a diacritic | `café open` | `cafe` | 2 folded | `PLACED` (4 chars) |
| Token reported without its tags | `<spell>1234</spell> ok` | `1234` | 3 markup | `PLACED` (11 chars) |
| Token straddles the frame boundary | `1111` | `1111And` | 1 literal | `CROSSES` (4 chars used) |
| Foreign token (dropped event upstream) | `hello world` | `goodbye` | none | `NO_MATCH` |
| Nothing spoken left here | `<break/>` | `hello` | none | `EXHAUSTED` |

Note how strategy 1 alone covers four different provider quirks, because it tries three
skip offsets *and* a trailing-punctuation-trimmed variant of the token.

### The four outcomes

| Outcome | Meaning | Effect |
| --- | --- | --- |
| `PLACED` | Token fits inside this segment | Advance to the matched end, stop |
| `CROSSES` | Segment's remainder is only a prefix of the token | Drain segment, carry the remainder onward |
| `EXHAUSTED` | No spoken content left here (`<break/>`, whitespace) | Drain segment, retry the whole token |
| `NO_MATCH` | Token does not belong here | Nudge past leading punctuation, stop |

### Why folding needs a word boundary

Folding erases case, which can manufacture a false match: folded `account` is a prefix of
folded `Accountant`. Strategy 2 therefore only accepts a `PLACED` match that lands on a
word boundary. Strategy 1 does not need the guard — it is case-sensitive already.

## 5. Two definitions of "markup"

The file deliberately carries two strippers, because a streamed fragment and a complete
text need opposite treatment of a lone `<`:

| Function | Input | `5 < 10` becomes | Used for |
| --- | --- | --- | --- |
| `strip_markup` | A possibly-truncated token | `5 ` | Matching a token that may be mid-tag |
| `strip_complete_markup` | A whole, static text | `5 < 10` | `is_transformed`, default user-facing text |

In a truncated fragment an unclosed `<` really is the start of a tag, so swallowing the
rest is correct. In a complete text it is real content.

## 6. Public surface

| Member | Purpose |
| --- | --- |
| `advance_word(word)` | Consume one token, moving all cursors |
| `word_belongs_current_segment(word)` | Non-mutating dry run of the same matching |
| `user_facing_pos` / `llm_pos` / `raw_pos` | The three cursors |
| `is_complete` | All alphanumeric content accounted for |
| `in_transformed_segment` | Cursor sits mid-transform (callers suppress context writes) |
| `last_completed_segment` | Segment finished by the last `advance_word` |
| `last_overflow` | Raw suffix that ran past the end of the TTS text |

Two details in `is_complete` are worth knowing. A frame whose remainder is pure
punctuation or markup is already complete — a closing tag never arrives as its own token.
The exception is punctuation set off by a space (French `Comment ça va ?`), which *is*
emitted as its own token, so `_pending_separated_punctuation` holds completion open for it.

## 7. Tests

`tests/test_text_segment_map.py` — 52 tests.

| Class | Covers |
| --- | --- |
| `TestStripMarkupHelpers` | `strip_markup`, `_raw_len_for_clean_chars` round-trip |
| `TestStripCompleteMarkupHelper` | Lone `<` kept as content |
| `TestTextSegmentMapBuild` | Segment construction and spans |
| `TestTextSegmentMapAdvance` | Cursor hold/jump behaviour |
| `TestTextSegmentMapWithLlmText` | The third cursor |
| `TestTextSegmentMapTokenChangingReplacements` | Replacements that change word count |
| `TestTextSegmentMapSsmlPhonemeTag` | Markup segments |
| `TestClassifyHopSkipsLeadingPunctuation` | Strategy 1's skip offsets |
| `TestClassifyHopCaseFoldRequiresWordBoundary` | The `account` / `Accountant` guard |
| `TestProviderTokenShapes` | Leading-space tokens |
| `TestWordCarriesItsOwnPunctuation` | Provider-added terminal punctuation |
