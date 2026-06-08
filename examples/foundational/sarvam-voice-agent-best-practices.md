# Building a Production-Ready Voice Agent for India with Pipecat and Sarvam AI

Voice AI in India presents unique challenges: diverse languages, heavy code-mixing between Hindi and English, noisy environments, and conversational patterns like short verbal acknowledgments ("haan", "okay", "achha") that can confuse naive turn-detection systems. This guide walks through building a voice agent that handles all of these gracefully using [Pipecat](https://github.com/pipecat-ai/pipecat) and Sarvam AI's full service stack — STT, TTS, and LLM.

By the end, you'll have a production-ready agent that:

- Understands Hindi, English, and Hinglish seamlessly
- Doesn't get interrupted by background noise or coughs
- Ignores short acknowledgments ("okay", "haan") while the bot is speaking
- Still lets the user interrupt with a real question when they need to
- Responds in the same language the user speaks

## Prerequisites

```bash
pip install "pipecat-ai[sarvam]"
```

You'll also need a Sarvam API key. Set it in your `.env` file:

```
SARVAM_API_KEY=your_key_here
```

## Architecture Overview

Pipecat uses a frame-based pipeline where data flows through a chain of processors:

```
Transport Input → STT → User Aggregator → LLM → TTS → Transport Output → Assistant Aggregator
```

Each component is a `FrameProcessor` that receives frames (audio, text, control signals), processes them, and pushes results to the next processor. The key insight is that **turn management** — deciding when the user has started and finished speaking — is separate from speech recognition and can be configured independently.


**Available STT models:**

| Model | Strengths | Language Detection |
|---|---|---|
| `saaras:v3` | Multi-mode (transcribe, translate, verbatim, translit, codemix) | Automatic or manual |
| `saaras:v3-realtime` | Streaming partials, server-side VAD, lower turn latency | Fixed language or `auto` with `stream_type="simulated"` |

The `saaras:v3` model supports five modes beyond basic transcription:

- **transcribe** (default): Standard speech-to-text
- **translate**: Transcribe and translate to English
- **verbatim**: Exact transcription including filler words
- **translit**: Transliterate speech into Roman script
- **codemix**: Optimized for mixed-language conversations (Hindi-English, etc.)

For a general-purpose voice agent, the default `transcribe` mode with auto-detection works well. If your users heavily code-mix, consider `codemix` mode.

**Supported languages:** Assamese, Bengali, English (India), Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, and Telugu.

For the realtime STT API, use `SarvamRealtimeSTTService` with Sarvam's server-side VAD driving turn detection. This API is currently beta-gated by Sarvam, so your subscription must have `enable_saaras_v3_realtime_streaming_users` enabled.

```python
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.services.sarvam.stt_realtime import SarvamRealtimeSTTService
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies

stt = SarvamRealtimeSTTService(
    api_key=os.getenv("SARVAM_API_KEY"),
    settings=SarvamRealtimeSTTService.Settings(
        language="hi-IN",
        stream_type="fast",
        endpointing="vad",
    ),
)

context_aggregator = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(user_turn_strategies=ExternalUserTurnStrategies()),
)
```

See `examples/foundational/agent-sarvam-realtime.py` for a complete STT-driven turn detection example with no external VAD analyzer.


**Configurable parameters for bulbul:v3:**

| Parameter | Range | Default | Description |
|---|---|---|---|
| `pace` | 0.5 – 2.0 | 1.0 | Speech speed multiplier |
| `temperature` | 0.01 – 1.0 | 0.6 | Output variation (lower = more consistent) |

**Available voices:** aditya (default), ritu, priya, neha, rahul, pooja, rohan, simran, kavya, amit, dev, ishita, shreya, ratan, varun, manan, sumit, roopa, kabir, aayan, shubh, ashutosh, advait, amelia, sophia.

You can change the voice at runtime using `TTSUpdateSettingsFrame`:

```python
from pipecat.frames.frames import TTSUpdateSettingsFrame

await task.queue_frame(TTSUpdateSettingsFrame(settings={"voice": "anushka"}))
```

## LLM: Sarvam sarvam-30b

```python
llm = SarvamLLMService(
    api_key=os.getenv("SARVAM_API_KEY"),
    settings=SarvamLLMService.Settings(
        system_instruction=(
            "You are a friendly and helpful voice assistant. "
            "You are having a real-time voice conversation, so always respond in natural, "
            "spoken language. Never use emojis, bullet points, markdown, or any formatting "
            "that cannot be spoken aloud. "
            "The user may speak in Hindi, English, or a mix of both (Hinglish). "
            "Always reply in the same language the user is speaking. "
            "Keep your responses concise and conversational, as if chatting with a friend."
        ),
    ),
)
```

Sarvam's LLM uses an OpenAI-compatible API, so integration is straightforward. The `sarvam-30b` model is the default and handles multilingual Indian conversations natively.

**Available models:**

| Model | Context Window |
|---|---|
| `sarvam-30b` | Default |
| `sarvam-30b-16k` | 16K tokens |
| `sarvam-105b` | Larger model |
| `sarvam-105b-32k` | 32K tokens |

**Sarvam-specific features:**

- **`wiki_grounding`** (bool): Ground responses in Wikipedia data for factual accuracy
- **`reasoning_effort`** ("low", "medium", "high"): Control how much reasoning the model applies

**System prompt best practices for voice:**

The system prompt is critical for natural voice output. Key guidelines:

1. **Explicitly ban formatting.** LLMs love to produce bullet points and markdown — these sound terrible when read aloud by TTS.
2. **Instruct language matching.** Indian users frequently switch between Hindi, English, and Hinglish mid-conversation. Telling the LLM to match the user's language produces a much more natural experience.
3. **Keep it concise.** Long responses create a poor voice UX. The user is waiting to respond and long monologues feel unnatural.

> **Note:** Sarvam's LLM does not support the `"developer"` message role (unlike OpenAI). Use `"system"` for system-level instructions.

## Turn Management: The Heart of a Natural Voice Agent

This is where a good voice agent becomes a great one. Turn management answers two questions:

1. **When has the user started speaking?** (turn start)
2. **When has the user finished speaking?** (turn stop)

Getting this wrong leads to the most common voice agent complaints: the bot talks over the user, or the bot interrupts itself when the user just says "okay".

### The Problem with Default VAD

Most voice frameworks use Voice Activity Detection (VAD) — an algorithm that detects whether audio contains speech. When speech is detected, the bot is interrupted. When speech stops, the bot responds.

This breaks in Indian conditions:

- **Background noise** (traffic, fans, TVs) can trigger false speech detection
- **Coughs and throat clearing** register as speech
- **Short acknowledgments** ("haan", "okay", "achha") while the bot is speaking cause unnecessary interruptions — the user is just listening, not trying to take the floor

### Our Approach: Transcription-Based Turn Detection

Instead of reacting to raw audio energy, we react to **what the user actually says**:

```python
user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(
        user_turn_strategies=UserTurnStrategies(
            start=[MinWordsUserTurnStartStrategy(min_words=3, use_interim=True)],
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.6)],
        ),
    ),
)
```

#### Turn Start: MinWordsUserTurnStartStrategy

This strategy counts words in the user's transcription before triggering a turn:

| Scenario | Threshold | Behavior |
|---|---|---|
| Bot IS speaking | 3 words | "okay" (1 word) → ignored. "I have a question" (4 words) → interrupts the bot |
| Bot is NOT speaking | 1 word | Any speech starts a turn immediately — the agent is responsive |

This single distinction solves the acknowledgment problem. When the bot is talking, the user saying "haan" or "okay" is treated as active listening, not an interruption. But saying "wait, I have a doubt" triggers a proper interruption.

**Coughs and non-speech sounds** produce no transcription at all, so they are completely invisible to this strategy.

The `use_interim=True` flag enables the strategy to process interim (partial) transcriptions, which means it can react faster — it doesn't need to wait for the final transcription to reach the 3-word threshold.

#### Turn Stop: SpeechTimeoutUserTurnStopStrategy

Once a turn has started, we need to know when the user has finished:

```python
stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.6)]
```

This waits for **0.6 seconds of silence** (no new transcriptions) before concluding the user has finished speaking. The strategy also accounts for STT latency — if Sarvam's STT has a known P99 latency, the timeout adjusts automatically to avoid cutting off the user.

This works purely from transcription timing, making it ideal for our setup where we don't use a local VAD analyzer.

### Why Not Sarvam's Built-in VAD?

Sarvam's STT offers a `vad_signals=True` option that sends speech start/stop events from the server. While this sounds appealing, it has a fundamental limitation: **when vad_signals is enabled, the STT broadcasts an interruption on every detected speech start, unconditionally**. There's no way to filter by content — a cough that passes the VAD would still interrupt the bot.

By not enabling `vad_signals` and instead using transcription-based turn management, we get content-aware interruption control. Sarvam's server-side audio processing still runs — the flag only controls whether VAD events are sent back to the client.

## Pipeline Configuration

```python
task = PipelineTask(
    pipeline,
    params=PipelineParams(
        enable_metrics=True,
        enable_usage_metrics=True,
    ),
    idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
)
```

- **`enable_metrics`**: Tracks processing latencies (Time-To-First-Byte for STT, TTS, LLM)
- **`enable_usage_metrics`**: Tracks token usage and API costs
- **`idle_timeout_secs`**: Automatically cleans up the pipeline after a period of inactivity — important for production deployments where connections may drop without a clean disconnect

## Running the Agent

The example supports multiple transports out of the box:

```bash
# With Daily (WebRTC)
python 07z-interruptible-sarvam.py --transport daily

# With Twilio (WebSocket)
python 07z-interruptible-sarvam.py --transport twilio

# With WebRTC directly
python 07z-interruptible-sarvam.py --transport webrtc
```

## Tuning Guide

### Adjusting Interruption Sensitivity

If users complain the bot is too hard to interrupt, lower the `min_words` threshold:

```python
# Easier to interrupt (2 words instead of 3)
MinWordsUserTurnStartStrategy(min_words=2, use_interim=True)
```

If the bot interrupts too easily in noisy environments, raise it:

```python
# Harder to interrupt (4+ words needed)
MinWordsUserTurnStartStrategy(min_words=4, use_interim=True)
```

### Adjusting Response Speed

If the bot responds before the user finishes (e.g., during a long pause mid-sentence), increase the speech timeout:

```python
# Wait longer before assuming the user is done
SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=1.0)
```

If the bot feels sluggish, reduce it:

```python
# Respond faster after the user stops
SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.4)
```

### Switching STT Mode for Code-Mixed Speech

If your users speak a lot of Hinglish:

```python
stt = SarvamSTTService(
    api_key=os.getenv("SARVAM_API_KEY"),
    settings=SarvamSTTService.Settings(
        model="saaras:v3",
    ),
    mode="codemix",
)
```


## Summary

Building a voice agent that works well in India comes down to getting turn management right. The combination of:

1. **MinWordsUserTurnStartStrategy** — content-aware interruption filtering that ignores coughs, noise, and short acknowledgments
2. **SpeechTimeoutUserTurnStopStrategy** — timeout-based turn completion that doesn't depend on local VAD
3. **Sarvam's multilingual stack** — STT, TTS, and LLM that natively understand Indian languages and code-mixing

...produces an agent that feels natural in real Indian conversational settings — noisy backgrounds, Hinglish, and all.
