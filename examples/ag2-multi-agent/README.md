# AG2 Multi-Agent Voice Bot

A voice bot that uses [AG2](https://docs.ag2.ai) multi-agent GroupChat as the reasoning engine behind a [pipecat](https://docs.pipecat.ai) voice pipeline. Instead of a single LLM call, user questions are processed by a coordinated team of AI agents — a Research Agent that gathers information and an Analyst Agent that synthesizes a concise, voice-friendly response.

## Architecture

```
                          Pipecat Pipeline
┌─────────┐   ┌─────┐   ┌──────────────────────┐   ┌─────┐   ┌──────────┐
│  Daily   │──▶│ STT │──▶│  AG2 Multi-Agent     │──▶│ TTS │──▶│  Daily   │
│Transport │   │     │   │  Processor           │   │     │   │Transport │
│ (input)  │   │     │   │                      │   │     │   │ (output) │
└─────────┘   └─────┘   │  ┌───────────────┐   │   └─────┘   └──────────┘
  WebRTC      Deepgram   │  │ UserProxy     │   │   Cartesia     WebRTC
  audio in               │  │   ├─Research  │   │   audio out
                         │  │   └─Analyst   │   │
                         │  │  (GroupChat)  │   │
                         │  └───────────────┘   │
                         └──────────────────────┘
```

**Flow:**
1. User speaks → Daily WebRTC captures audio
2. Deepgram STT transcribes speech to text
3. `AG2MultiAgentProcessor` receives the transcription and runs an AG2 GroupChat:
   - **Research Agent** gathers information (can call tools)
   - **Analyst Agent** synthesizes a short, conversational answer
   - **UserProxy** orchestrates the agents and detects termination
4. Final response is pushed as text to Cartesia TTS
5. Synthesized speech is sent back to the user via Daily WebRTC

## Prerequisites

- Python 3.11+
- API keys for the following services:

| Service | Env Variable | Get a Key |
|---------|-------------|-----------|
| OpenAI | `OPENAI_API_KEY` | https://platform.openai.com |
| Deepgram | `DEEPGRAM_API_KEY` | https://console.deepgram.com |
| Cartesia | `CARTESIA_API_KEY` | https://play.cartesia.ai |
| Daily | `DAILY_API_KEY` | https://dashboard.daily.co |

## Setup

1. Install dependencies:

```bash
pip install "pipecat-ai[daily,openai,deepgram,cartesia,silero]" "ag2[openai]>=0.11.4,<1.0"
```

2. Copy the environment template from the repo root and fill in your keys:

```bash
cp ../../env.example .env
# Edit .env with your API keys
```

## Running

```bash
python bot.py
```

This starts a local server. Open the displayed URL to connect via your browser.

You can also specify the transport type:

```bash
python bot.py -t daily    # Daily WebRTC (default)
python bot.py -t webrtc   # Small WebRTC
```

## How It Works

The core integration is `AG2MultiAgentProcessor`, a pipecat `FrameProcessor` that bridges the async pipecat pipeline with AG2's synchronous GroupChat:

- **Receives** `TranscriptionFrame` from STT (user speech as text)
- **Runs** AG2 GroupChat in a background thread via `asyncio.to_thread()` to avoid blocking the async event loop
- **Pushes** the final response as a `TextFrame` downstream to TTS
- **Passes through** all other frame types unchanged

The AG2 GroupChat is configured with:
- `max_round=10` to prevent runaway conversations
- `is_termination_msg` on both proxy and manager to stop when the Analyst says "TERMINATE"
- A fresh GroupChat per user turn so history doesn't accumulate

## Customization

- **Add real tools**: Replace the `search_knowledge` placeholder in `bot.py` with a real search API (Tavily, SerpAPI, etc.)
- **Add more agents**: Add specialist agents to the GroupChat for domain-specific reasoning
- **Change the LLM**: Update the `LLMConfig` dict to use a different model (e.g., `gpt-4o`, `gpt-4-turbo`)

## Resources

- [AG2 Documentation](https://docs.ag2.ai)
- [Pipecat Documentation](https://docs.pipecat.ai)
- [Pipecat Examples](https://github.com/pipecat-ai/pipecat/tree/main/examples)
