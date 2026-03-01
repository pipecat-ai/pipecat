# Onairos Integration Guide

Complete guide to integrating Onairos personalization with Pipecat voice agents.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Backend Setup (Python/Pipecat)](#backend-setup-pythonpipecat)
4. [Frontend Setup (npm)](#frontend-setup-npm)
5. [API Reference](#api-reference)
6. [How Persona Data Augments Your Agent](#how-persona-data-augments-your-agent)
7. [Examples](#examples)

---

## Overview

Onairos is a privacy-first personalization platform that enables your voice agents to:

- **Personalize responses** based on personality traits, archetype, and MBTI alignment
- **Understand users deeply** with multi-paragraph user summaries generated from their data
- **Seamless onboarding** with user-controlled data sharing via frontend SDK
- **Respect privacy** with user-owned data architecture

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Your Application                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐  │
│  │   Frontend  │     │    Pipecat      │     │    Onairos API   │  │
│  │  (React/RN) │────▶│  Voice Agent    │────▶│ api2.onairos.uk  │  │
│  │             │     │                 │     │                  │  │
│  │ @onairos/sdk│     │ OnairosPersona  │     │ /inferenceNoProof│  │
│  │             │     │   Injector      │     │ /traits-only     │  │
│  │             │     │                 │     │ /combined-infer. │  │
│  └─────────────┘     └─────────────────┘     └──────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### 1. Onairos Developer Account

1. Go to [https://dashboard.onairos.uk/](https://dashboard.onairos.uk/)
2. Create a developer account
3. Create a new application
4. Get your credentials:
   - **Publishable Key** (`pk_*`) - for client-side SDK

### 2. Environment Variables

```bash
# For frontend SDK
ONAIROS_PUBLISHABLE_KEY=pk_your_publishable_key
```

---

## Backend Setup (Python/Pipecat)

The Pipecat backend calls the Onairos REST API directly using `aiohttp`. No npm/JavaScript needed on the backend.

### Installation

```bash
# Install pipecat with Onairos support
pip install "pipecat-ai[onairos]"

# Or with uv
uv add "pipecat-ai[onairos]"

# This installs aiohttp for REST API calls to api2.onairos.uk
```

### Basic Integration

```python
from pipecat.services.onairos import OnairosPersonaInjector

# Initialize without credentials - they arrive from frontend onComplete
persona = OnairosPersonaInjector(
    user_id="user_123",
    params=OnairosPersonaInjector.InputParams(
        include_personality_traits=True,
        include_traits_to_improve=True,
        include_user_summary=True,
        include_archetype=True,
        include_mbti=True,
        top_mbti_count=5,
    ),
)

# When frontend sends onComplete data:
persona.set_api_credentials(api_url=api_url, access_token=access_token)
```

### Pipeline Integration

```python
from pipecat.pipeline.pipeline import Pipeline
from pipecat.services.onairos import OnairosPersonaInjector

pipeline = Pipeline([
    transport.input(),           # Audio/video input
    stt,                         # Speech-to-text
    user_aggregator,
    persona,                     # ← Inject user persona
    llm,                         # Your LLM (OpenAI, Anthropic, etc.)
    tts,                         # Text-to-speech
    transport.output(),
    assistant_aggregator,
])
```

---

## Frontend Setup (npm) - Optional

> **Note:** This section is for your **separate frontend application** (React, React Native, etc.), not this Pipecat repo. The frontend handles user authentication with Onairos and passes credentials to your Pipecat backend.

For web and mobile applications, use the Onairos SDK to enable user connections.

**Platform SDKs:**
| Platform | Package |
|----------|---------|
| Web (React, Vue, JS) | `npm install onairos` |
| React Native | `npm install @onairos/react-native` |
| Swift (iOS) | See docs.onairos.uk |
| Flutter | See docs.onairos.uk |

### Web (React)

```bash
npm install onairos
```

```jsx
import Onairos from 'onairos';

function VoiceAgent({ websocket }) {
  return (
    <div>
      <h2>Connect for Personalized Experience</h2>

      {/* Onairos renders a connect button */}
      <Onairos
        requestData={{
          Traits: { type: "Personality", size: "Large" }
        }}
        onComplete={(apiUrl, accessToken) => {
          // Send credentials to your Pipecat backend
          websocket.send(JSON.stringify({
            type: "onairos_credentials",
            apiUrl: apiUrl,
            accessToken: accessToken
          }));
        }}
      />

      {/* Your Pipecat voice UI here */}
    </div>
  );
}
```

### React Native

```bash
npm install @onairos/react-native
```

```jsx
import Onairos from '@onairos/react-native';

function VoiceAgent({ websocket }) {
  return (
    <Onairos
      requestData={{
        Traits: { type: "Personality", size: "Large" }
      }}
      onComplete={(apiUrl, accessToken) => {
        websocket.send(JSON.stringify({
          type: "onairos_credentials",
          apiUrl,
          accessToken
        }));
      }}
    />
  );
}
```

### Vanilla JavaScript

```bash
npm install onairos
```

```javascript
import Onairos from 'onairos';

const onairosButton = new Onairos({
  requestData: {
    Traits: { type: "Personality", size: "Large" }
  },
  onComplete: (apiUrl, accessToken) => {
    websocket.send(JSON.stringify({
      type: "onairos_credentials",
      apiUrl: apiUrl,
      accessToken: accessToken
    }));
  }
});
```

---

## API Reference

### Onairos REST API

**Base URL:** `https://api2.onairos.uk`

#### Get Personality Traits Only

```
POST /traits-only
Authorization: Bearer {accessToken}
Content-Type: application/json

{
  "accessToken": "{accessToken}",
  "inputData": []
}

Response:
{
  "success": true,
  "traits": {
    "positive_traits": {
      "Stoic Wisdom Interest": 80,
      "AI Enthusiasm": 75
    },
    "traits_to_improve": {
      "Social Media Engagement": 35
    }
  }
}
```

#### Get MBTI Inference

MBTI scores come from running the user's trained FinalMLP model on 16 MBTI type descriptions. The output is preference scores (0-1) per type, indicating how much the user aligns with each personality type.

```
POST /inferenceNoProof
Authorization: Bearer {accessToken}
Content-Type: application/json

{
  "accessToken": "{accessToken}",
  "inputData": []
}

Response:
{
  "InferenceResult": {
    "output": [0.584, 0.500, 0.550, 0.520, 0.627, 0.511, 0.580, 0.490,
               0.460, 0.580, 0.470, 0.450, 0.430, 0.410, 0.400, 0.390]
  }
}
```

The output array maps to MBTI types in order:
`INTJ, INTP, ENTJ, ENTP, INFJ, INFP, ENFJ, ENFP, ISTJ, ISFJ, ESTJ, ESFJ, ISTP, ISFP, ESTP, ESFP`

#### Get Combined Inference (Traits + MBTI)

```
POST /combined-inference
Authorization: Bearer {accessToken}
Content-Type: application/json

{
  "accessToken": "{accessToken}",
  "inputData": []
}

Response:
{
  "InferenceResult": {
    "output": [0.584, 0.500, ...]
  },
  "traits": {
    "positive_traits": {"Stoic Wisdom Interest": 80, ...},
    "traits_to_improve": {"Social Media Engagement": 35, ...}
  },
  "connectedPlatforms": {
    "platforms": ["youtube", "reddit"]
  }
}
```

#### Full Personality Traits Structure

When available from the GenerateTraitsIterativeEnhanced pipeline, traits include richer detail:

```json
{
  "personality_traits": {
    "positive_traits": {
      "Stoic Wisdom Interest": {
        "score": 80,
        "emoji": "🏛️",
        "evidence": "You frequently engage with philosophy content"
      }
    },
    "traits_to_improve": {
      "Social Media Engagement": {
        "score": 35,
        "emoji": "📱",
        "evidence": "You show limited interest in social platforms"
      }
    },
    "user_summary": "2-3 paragraphs about the user in 2nd person",
    "top_traits_explanation": "1-2 paragraphs explaining reasoning",
    "archetype": "Strategic Explorer",
    "nudges": [
      {"text": "You're highly analytical — try journaling a decision you're mulling over"}
    ]
  }
}
```

---

## How Onairos Augments Your Agent Prompts

### The Augmentation Pattern

Onairos augments your **base prompt** with rich user context from the `onComplete` callback:

**Base Prompt (without Onairos):**
```
You're onboarding a user for dating matches. Have a genuine conversation
to understand them. Ask whatever feels relevant based on what they share.
Trust your judgment on what matters.
Once you genuinely feel you know enough to match them well, wrap up.

Critical Instruction:
Always check context before asking. Complete onboarding ASAP.
```

**Augmented Prompt (with Onairos):**
```
You're onboarding a user for dating matches. Have a genuine conversation
to understand them. Ask whatever feels relevant based on what they share.
Trust your judgment on what matters.
Once you genuinely feel you know enough to match them well, wrap up.

Positive Traits of User:
Stoic Wisdom Interest: 80, Daily Stoic Engagement: 90,
Coffee Lover: 95, Morning Person: 70

Areas to Improve:
Social Media Engagement: 35, Public Speaking Confidence: 40

User Summary:
You are drawn to deep philosophical thinking and have a strong interest
in Stoic philosophy. You enjoy morning routines and are a dedicated
coffee enthusiast.

Archetype: The Strategic Explorer

MBTI Alignment (Personalities User Likes):
INFJ: 0.627, INTJ: 0.585, ENFJ: 0.580, ISFJ: 0.580, INFP: 0.511

Critical Instruction:
Always check context before asking. Complete onboarding ASAP.
```

### How Onairos Data Flows

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   FRONTEND (React/JS)                    BACKEND (Pipecat/Python)        │
│                                                                          │
│   1. User clicks                                                         │
│      "Connect Onairos"                                                   │
│            │                                                             │
│            ▼                                                             │
│   2. Onairos popup                                                       │
│      User authorizes                                                     │
│            │                                                             │
│            ▼                                                             │
│   3. onComplete returns:        ──────▶  4. Backend receives             │
│      { apiUrl, accessToken }             { apiUrl, accessToken }         │
│      (NOT the actual data!)              via WebSocket/RTVI              │
│                                                │                         │
│                                                ▼                         │
│                                          5. Backend calls                │
│                                             POST apiUrl                  │
│                                             with accessToken             │
│                                                │                         │
│                                                ▼                         │
│                                          6. Onairos returns:             │
│                                             { traits, MBTI scores }      │
│                                                │                         │
│                                                ▼                         │
│                                          7. Augment LLM prompt           │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Key Insight:** The frontend only receives `apiUrl` and `accessToken`. The actual user data (traits, MBTI) stays secure on the backend.

### The onComplete Callback

When `onComplete` fires in the frontend, it receives two parameters:

```javascript
onComplete: (apiUrl, accessToken) => {
  // apiUrl: "https://api2.onairos.uk/combined-inference"
  // accessToken: "eyJhbGciOiJIUzI1NiIs..."  (JWT token)

  // Send these to your Pipecat backend
  websocket.send(JSON.stringify({
    type: "onairos_credentials",
    apiUrl,
    accessToken
  }));
}
```

### The Backend API Call

The backend uses these credentials to fetch the actual data:

```
POST https://api2.onairos.uk/combined-inference
Authorization: Bearer {accessToken}
Content-Type: application/json

{
  "accessToken": "{accessToken}",
  "inputData": []
}

Response:
{
  "InferenceResult": {
    "output": [0.584, 0.500, 0.550, 0.520, 0.627, ...]
  },
  "traits": {
    "positive_traits": {
      "Stoic Wisdom Interest": 80,
      "AI Enthusiasm": 40
    },
    "traits_to_improve": {
      "Social Media Engagement": 35
    }
  }
}
```

### Using OnairosPersonaInjector

```python
from pipecat.services.onairos import OnairosPersonaInjector, OnairosUserData

# Initialize without credentials (will be set when frontend sends them)
persona = OnairosPersonaInjector(
    user_id="user_123",
    params=OnairosPersonaInjector.InputParams(
        include_personality_traits=True,
        include_traits_to_improve=True,
        include_user_summary=True,
        include_archetype=True,
        include_mbti=True,
        top_mbti_count=5,
        critical_instruction="Always check context before asking.",
    ),
)

# When frontend sends onComplete data via WebSocket/RTVI:
@transport.event_handler("on_client_message")
async def on_client_message(transport, message):
    if message.get("type") == "onairos_credentials":
        persona.set_api_credentials(
            api_url=message["apiUrl"],
            access_token=message["accessToken"]
        )

# The persona injector will automatically call the Onairos API
# and augment the LLM prompt when processing frames
```

### Alternative: Pre-loaded Data

If you already have the data (e.g., from a webhook):

```python
persona = OnairosPersonaInjector(
    user_id="user_123",
    user_data=OnairosUserData(
        positive_traits={"Stoic Wisdom Interest": 80, "AI Enthusiasm": 40},
        traits_to_improve={"Social Media Engagement": 35},
        user_summary="You are drawn to deep philosophical thinking...",
        archetype="Strategic Explorer",
        mbti={"INFJ": 0.627, "INTJ": 0.585, "ENFJ": 0.580},
    ),
)
```

### Real Impact Examples

**Without Onairos (generic onboarding):**
> "Hi! Tell me about yourself. What are your hobbies?"

**With Onairos (knows user is a Strategic Explorer who loves Stoic philosophy):**
> "Hey! I see you're a Strategic Explorer with a real passion for Stoic philosophy.
> Do you have a favorite Stoic author, or is it more the daily practice that resonates?
> Also, I noticed you're a coffee person - any favorite spots?"

The agent **already knows** key information, so it can:
- Skip basic questions (saves time)
- Reference interests naturally (builds rapport)
- Adapt communication style (feels personal)

---

## Examples

### Full Voice Agent Example

See `examples/foundational/38-onairos.py` for a complete working example.

```python
from pipecat.services.onairos import OnairosPersonaInjector, OnairosUserData

persona = OnairosPersonaInjector(
    user_id=USER_ID,
    params=OnairosPersonaInjector.InputParams(
        include_personality_traits=True,
        include_traits_to_improve=True,
        include_user_summary=True,
        include_archetype=True,
        include_mbti=True,
        top_mbti_count=5,
    ),
)

# Event handlers
@persona.event_handler("on_user_data_loaded")
async def on_user_data_loaded(user_data):
    logger.info(f"Loaded persona with archetype: {user_data.archetype}")
    logger.info(f"Positive traits: {list(user_data.positive_traits.keys())}")
```

### With Voice UI Kit + RTVI

The recommended pattern is to pass Onairos credentials from frontend to backend via WebSocket message:

**Frontend (React):**
```jsx
import { RTVIClient } from '@pipecat-ai/client-js';
import Onairos from 'onairos';
import { useState, useRef } from 'react';

function VoiceAgent() {
  const [isConnected, setIsConnected] = useState(false);
  const clientRef = useRef(null);

  const startSession = async () => {
    const client = new RTVIClient({
      transport: new DailyTransport(),
      params: {
        baseUrl: 'https://your-server.com',
      }
    });

    clientRef.current = client;
    await client.connect();
  };

  return (
    <div>
      {/* Onairos button - sends credentials when user connects */}
      <Onairos
        requestData={{
          Traits: { type: "Personality", size: "Large" }
        }}
        onComplete={(apiUrl, accessToken) => {
          setIsConnected(true);
          if (clientRef.current) {
            clientRef.current.sendMessage({
              type: "onairos_credentials",
              apiUrl,
              accessToken
            });
          }
        }}
      />

      <button onClick={startSession} disabled={!isConnected}>
        {isConnected ? 'Start Personalized Call' : 'Connect Onairos First'}
      </button>
    </div>
  );
}
```

**Backend (Python) - Receiving Onairos Data:**
```python
from pipecat.services.onairos import OnairosPersonaInjector

async def run_bot(transport, runner_args):
    persona = OnairosPersonaInjector(user_id="user_123")

    @transport.event_handler("on_message")
    async def on_message(transport, message):
        if isinstance(message, dict) and message.get("type") == "onairos_credentials":
            persona.set_api_credentials(
                api_url=message["apiUrl"],
                access_token=message["accessToken"]
            )

    pipeline = Pipeline([...stt, user_aggregator, persona, llm, tts...])
```

---

## Troubleshooting

### No persona data in augmentation

Ensure the frontend `onComplete` callback is sending credentials to the backend, and that `set_api_credentials()` is being called before the first LLM frame is processed.

### Rate Limiting

| Plan       | Requests/min | Requests/day |
|------------|-------------|--------------|
| Free       | 60          | 1,000        |
| Startup    | 300         | 50,000       |
| Enterprise | Unlimited   | Unlimited    |

---

## Resources

- **Onairos Docs:** https://onairos.uk/docs
- **Onairos Dashboard:** https://dashboard.onairos.uk
- **JavaScript SDK:** https://onairos.uk/docs/javascript-sdk/
