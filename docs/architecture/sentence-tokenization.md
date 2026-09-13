# Sentence tokenization

Pipecat reads Punkt model parameters from its packaged `punkt_tab.zip` archive.
Importing the text utilities does not import NLTK. Pipeline warm-up loads the
English model in a background thread; TTS services also prepare their selected
model while services connect. Models are cached by language. Loading requires
neither network access nor an external NLTK data directory, and does not change
NLTK's data search paths. The archive stays compressed on disk.

## Selecting a language

TTS sentence aggregation follows the service's `Settings.language`. Pipecat
retains the input language before converting it to the provider's identifier.
Regional variants share one Punkt model: `de-DE` selects `german`, for example,
and `pt-BR` selects `portuguese`.

For services that infer language from the text or voice, or when the text needs
a different sentence model, pass `text_aggregation_language` to the service:

```python
tts = CartesiaTTSService(
    api_key=api_key,
    settings=CartesiaTTSService.Settings(voice=voice_id, language=Language.DE),
)

# An explicit override remains in effect when the TTS language changes.
tts = OpenAITTSService(api_key=api_key, text_aggregation_language=Language.DE)
```

The available models are Czech, Danish, Dutch, English, Estonian, Finnish,
French, German, Greek, Italian, Malayalam, Norwegian, Polish, Portuguese,
Russian, Slovene, Spanish, Swedish, and Turkish. Unspecified, automatic, and
unsupported languages use the English model plus Pipecat's additional
punctuation handling, including boundaries such as `。`, `؟`, and `।`.
This fallback does not detect the text's language automatically.

Standalone callers can select the same model:

```python
aggregator = SimpleTextAggregator(language=Language.DE)
boundary = match_endofsentence("Das ist z.B. wichtig. Weiter", language=Language.DE)
```

## Runtime updates

Send a settings update between LLM generations:

```python
await worker.queue_frame(
    TTSUpdateSettingsFrame(
        service=tts,
        delta=tts.Settings(language=Language.FR),
    )
)
```

The tokenizer language is captured at `LLMFullResponseStartFrame` and retained
through that generation's text. A settings update arriving during aggregation
selects the model for the next generation; it does not reinterpret or discard
buffered text. Interruptions discard the buffer. Text streams without a start
frame capture the language when their first text arrives, until an end frame
or interruption. `TTSSpeakFrame` uses the configured language for its independent
utterance.

This snapshot controls sentence detection, not the provider's synthesis
settings or the LLM's output language. Applications should coordinate the LLM
and TTS language and send synthesis-setting updates between generations.

In token aggregation mode, the TTS sentence tracker captures the model for each
audio context. An older context retains its model while newer contexts use a
different one. Newly selected models prepare in a background thread; the first
generation using a model waits for preparation if it is not yet complete.

## Observers

RTVI's legacy bot-transcription messages perform their own sentence detection.
Associate an observer explicitly with the relevant TTS service:

```python
observer = RTVIObserver(rtvi, tts_service=tts)
```

That observer aggregates LLM text when it reaches the selected TTS. Language
snapshots are stored on frames separately for each service, so delayed observer
callbacks do not read settings belonging to a newer generation. Other RTVI
messages keep their existing sources and timing. Input transcription language
is independent of the TTS selection.

An observer without a TTS association uses English by default, or a fixed
`text_aggregation_language` constructor argument. With multiple TTS services,
associate the observer with the intended service; it does not infer one from
pipeline topology.
