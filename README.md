# Onairos × Pipecat

**Build voice and multimodal conversational AI with unmatched personalization and onboarding.**

[![Pipecat](https://img.shields.io/badge/Powered%20by-Pipecat-blue)](https://github.com/pipecat-ai/pipecat)
[![Onairos](https://img.shields.io/badge/Enhanced%20by-Onairos-purple)](https://onairos.uk)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

Onairos integrates deeply with **Pipecat** — the open-source framework for real-time voice and multimodal agents — to deliver **state-of-the-art personalization, long-term memory, persona injection, and frictionless user onboarding**.

Your agents will:
* Remember every conversation across sessions
* Adapt to user preferences, history, and identity in real time
* Onboard new users faster with guided, context-aware interactions
* Inject rich personas and profiles for more human-like, engaging voice experiences

If you're building with Pipecat and want **the best onboarding and personalization**, use this Onairos integration instead of vanilla Pipecat.

## Why Onairos + Pipecat?

* **Persistent Memory** — Never lose context; users feel truly remembered
* **Seamless Onboarding** — Personalized greetings, adaptive flows, and quick value delivery
* **Persona & Profile Injection** — Bring user data/personas into every LLM call and voice response
* **Low Latency & Voice-First** — Built for real-time conversational AI
* **Easy Drop-In** — Works with existing Pipecat pipelines; just add Onairos as your memory/context provider

Vanilla Pipecat is great for basics. Onairos makes it production-ready for personalized, memory-rich agents.

## How It Works: Prompt Augmentation

Onairos augments your **base prompt** with rich user context:

**Without Onairos:**
```
You're onboarding a user. Have a genuine conversation to understand them.
```

**With Onairos:**
```
You're onboarding a user. Have a genuine conversation to understand them.

Personality Traits of User:
{"Stoic Wisdom Interest": 80, "AI Enthusiasm": 40, "Coffee Lover": 95}

Memory of User:
Reads Daily Stoic every morning. Prefers small coffee shop meetups.

MBTI (Personalities User Likes):
INFJ: 0.627, INTJ: 0.585, ENFJ: 0.580

Critical Instruction:
Always check context before asking. Complete onboarding efficiently.
```

The agent now **knows the user** before the conversation even starts.

## Quick Start

1. Clone this repo (or add as dependency once published):

   ```bash
   git clone https://github.com/onairos-dev/pipecat-onairos.git
   cd pipecat-onairos
   ```

2. Install dependencies (uses uv or pip; follows Pipecat conventions):

   ```bash
   uv sync  # or pip install -e ".[dev]" if using pip
   # If we publish to PyPI later: pip install onairos-pipecat
   ```

3. Set your Onairos API key:

   ```bash
   export ONAIROS_API_KEY=your_key_here
   ```

4. Run an example pipeline with Onairos enabled (see `/examples/` folder):

   ```bash
   python examples/onairos-basic-voice.py
   ```

   This demo shows a voice agent that remembers user details, personalizes responses, and onboards smoothly.

## Integration Details

Onairos provides services/processors for Pipecat:

* **OnairosPersonaInjector** — Fetches user persona from Onairos and injects preferences, interests, and communication style into LLM prompts
* **OnairosMemoryService** — Enhanced context with user profile data for richer personalization
* **OnairosContextAggregator** — Manages Onairos connection state and onboarding flows

### Backend (Python)

```python
from pipecat.pipeline.pipeline import Pipeline
from pipecat.services.onairos import OnairosPersonaInjector

# Initialize persona injector (credentials come from frontend)
persona = OnairosPersonaInjector(
    user_id="user_123",
    params=OnairosPersonaInjector.InputParams(
        include_personality_traits=True,
        include_memory=True,
        include_mbti=True,
    ),
)

# When frontend sends onComplete data:
@transport.event_handler("on_message")
async def on_message(transport, message):
    if message.get("type") == "onairos_credentials":
        # Frontend sends apiUrl + accessToken from onComplete
        persona.set_api_credentials(
            api_url=message["apiUrl"],
            access_token=message["accessToken"]
        )
        # Backend will call Onairos API to fetch actual user data

pipeline = Pipeline([
    transport.input(),
    stt,
    user_aggregator,
    persona,              # Fetches & injects user persona
    llm,
    tts,
    transport.output(),
    assistant_aggregator,
])
```

### Frontend (npm) - For Your Web/Mobile App

> **Note:** The npm packages are for your **separate frontend application**, not this Python repo.

In your React/React Native frontend project:

```bash
# Web
npm install @onairos/sdk @onairos/react

# React Native
npm install @onairos/react-native
```

```jsx
import { OnairosProvider, usePersona } from '@onairos/react';

function App() {
  return (
    <OnairosProvider apiKey={process.env.ONAIROS_PUBLISHABLE_KEY}>
      <VoiceAgent />
    </OnairosProvider>
  );
}
```

The frontend collects user data via `onComplete` and passes it to your Pipecat backend via RTVI config or WebSocket metadata.

📚 **Full integration guide:** [docs/ONAIROS_INTEGRATION.md](docs/ONAIROS_INTEGRATION.md)

Full examples in `/examples/foundational/38-onairos.py`.

## 🧠 Pipecat Framework

This integration is built on top of **Pipecat**, an open-source Python framework for building real-time voice and multimodal conversational agents.

### 🚀 What You Can Build

- **Voice Assistants** – natural, streaming conversations with AI
- **AI Companions** – coaches, meeting assistants, characters
- **Multimodal Interfaces** – voice, video, images, and more
- **Interactive Storytelling** – creative tools with generative media
- **Business Agents** – customer intake, support bots, guided flows
- **Complex Dialog Systems** – design logic with structured conversations

### 🌐 Pipecat Ecosystem

#### 📱 Client SDKs

Connect to Pipecat from any platform using official SDKs:

[JavaScript](https://docs.pipecat.ai/client/js/introduction) | [React](https://docs.pipecat.ai/client/react/introduction) | [React Native](https://docs.pipecat.ai/client/react-native/introduction) |
[Swift](https://docs.pipecat.ai/client/ios/introduction) | [Kotlin](https://docs.pipecat.ai/client/android/introduction) | [C++](https://docs.pipecat.ai/client/c++/introduction) | [ESP32](https://github.com/pipecat-ai/pipecat-esp32)

#### 🧭 Structured conversations

Looking to build structured conversations? Check out [Pipecat Flows](https://github.com/pipecat-ai/pipecat-flows) for managing complex conversational states and transitions.

#### 🪄 Beautiful UIs

Want to build beautiful and engaging experiences? Checkout the [Voice UI Kit](https://github.com/pipecat-ai/voice-ui-kit), a collection of components, hooks and templates for building voice AI applications quickly.

### 🧩 Available Services

| Category            | Services                                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------------------------ |
| Speech-to-Text      | AssemblyAI, AWS, Azure, Cartesia, Deepgram, ElevenLabs, Gladia, Google, Groq, OpenAI, Speechmatics, Whisper |
| LLMs                | Anthropic, AWS, Azure, Cerebras, DeepSeek, Gemini, Grok, Groq, Mistral, OpenAI, OpenRouter, Perplexity      |
| Text-to-Speech      | AWS, Azure, Cartesia, Deepgram, ElevenLabs, Google, Groq, LMNT, MiniMax, Neuphonic, OpenAI, Rime            |
| Speech-to-Speech    | AWS Nova Sonic, Gemini Multimodal Live, Grok Voice Agent, OpenAI Realtime, Ultravox                         |
| Transport           | Daily (WebRTC), FastAPI Websocket, SmallWebRTCTransport, WebSocket Server, Local                            |
| Memory              | **Onairos** (this integration), mem0                                                                         |
| Video               | HeyGen, Tavus, Simli                                                                                         |
| Analytics & Metrics | OpenTelemetry, Sentry                                                                                        |

📚 [View full services documentation →](https://docs.pipecat.ai/server/services/supported-services)

## Documentation & Resources

* **Onairos Docs**: [https://docs.onairos.uk](https://docs.onairos.uk)
* **Pipecat Original**: [https://github.com/pipecat-ai/pipecat](https://github.com/pipecat-ai/pipecat)
* **Onairos Website**: [https://onairos.uk](https://onairos.uk)
* **Join the conversation**: [Discord](https://discord.gg/pipecat) / [X @onairosapp](https://x.com/onairosapp)

## Getting Started with Development

### Prerequisites

**Minimum Python Version:** 3.10  
**Recommended Python Version:** 3.12

### Setup Steps

1. Clone the repository and navigate to it:

   ```bash
   git clone https://github.com/onairos-dev/pipecat-onairos.git
   cd pipecat-onairos
   ```

2. Install development and testing dependencies:

   ```bash
   uv sync --group dev --all-extras \
     --no-extra gstreamer \
     --no-extra krisp \
     --no-extra local
   ```

3. Install the git pre-commit hooks:

   ```bash
   uv run pre-commit install
   ```

### Running tests

```bash
uv run pytest
```

Run a specific test suite:

```bash
uv run pytest tests/test_name.py
```

## 🤝 Contributing

We welcome contributions from the community! Whether you're fixing bugs, improving documentation, or adding new features:

- **Found a bug?** Open an [issue](https://github.com/onairos-dev/pipecat-onairos/issues)
- **Have a feature idea?** Start a [discussion](https://discord.gg/pipecat)
- **Want to contribute code?** Check our [CONTRIBUTING.md](CONTRIBUTING.md) guide

## Attribution

This repo is a fork/extension of [pipecat-ai/pipecat](https://github.com/pipecat-ai/pipecat) (BSD-2-Clause license).  
Original copyright and license preserved in [./LICENSE](./LICENSE).  
All Onairos-specific code © Onairos contributors.

---

⭐ **Star this repo if you're building personalized voice AI!**
