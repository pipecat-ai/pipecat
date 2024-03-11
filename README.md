# dailyai — an open source framework for real-time, multi-modal, conversational AI applications

Build things like this:

[![AI-powered voice patient intake for healthcare](https://img.youtube.com/vi/lDevgsp9vn0/0.jpg)](https://www.youtube.com/watch?v=lDevgsp9vn0)




**`dailyai` started as a toolkit for implementing generative AI voice bots.** Things like personal coaches, meeting assistants, story-telling toys for kids, customer support bots, and snarky social companions. 


In 2023 a *lot* of us got excited about the possibility of having open-ended conversations with LLMs. It became clear pretty quickly that we were all solving the same [low-level problems](https://www.daily.co/blog/how-to-talk-to-an-llm-with-your-voice/):
- low-latency, reliable audio transport
- echo cancellation 
- phrase endpointing (knowing when the bot should respond to human speech),
- interruptibility
- writing clean code to stream data through "pipelines" of speech-to-text, LLM inference, and text-to-speech models

As our applications expanded to include additional things like image generation, function calling, and vision models, we started to think about what a complete framework for these kinds of apps could look like.

Today, `dailyai` is:

1. a set of code building blocks for interacting with generative AI services and creating low-latency, interruptible data pipelines that use multiple services
2. transport services that moves audio, video, and events across the Internet
3. implementations of specific generative AI services

Currently implemented services:
- Speech-to-text
  - Deepgram
  - Whisper
- LLMs
  - Azure
  - OpenAI
- Image generation
  - Azure
  - Fal
  - OpenAI
- Text-to-speech
  - Azure
  - Deepgram
  - ElevenLabs
- Transport
  - Daily
  - Local (in progress, intended as a quick start example service)

If you'd like to [implement a service]((https://github.com/daily-co/daily-ai-sdk/tree/main/src/dailyai/services)), we welcome PRs! Our goal is to support lots of services in all of the above categories, plus new categories (like real-time video) as they emerge.

## Step 1: Get started

Today, the easiest way to get started with `dailyai` is to use [Daily](https://www.daily.co/) as your transport service. This toolkit started life as an internal SDK at Daily and millions of minutes of AI conversation have been served using it and its earlier prototype incarnations. (The [transport base class](https://github.com/daily-co/daily-ai-sdk/blob/main/src/dailyai/services/base_transport_service.py) is easy to extend, though, so feel free to submit PRs if you'd like to implement another transport service.)

```
# install the module
pip install dailyai

# set up an .env file with API keys
# for example
OPENAI_API_KEY=...
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=...
DAILY_SAMPLE_ROOM_URL=https://...

# sign up for a free Daily account, if you don't already have one, and
# join the Daily room URL directly from a browser tab, then run one of the
# samples
python src/examples/foundational/02-llm-say-one-thing.py
```

## Code examples

There are two directories of examples:

- [foundational](https://github.com/daily-co/daily-ai-sdk/tree/main/src/examples/foundational) — demos that build on each other, introducing one or two concepts at a time
- [starter apps](https://github.com/daily-co/daily-ai-sdk/tree/main/src/examples/starter-apps) — complete applications that you can use as starting points for development



## Hacking on the framework itself

_Note that you may need to set up a virtual environment before following the instructions below. For instance, you might need to run the following from the root of the repo:_

```
python3 -m venv env
source env/bin/activate
```

From the root of this repo, run the following:

```
pip install -r requirements.txt
python -m build
```

This builds the package. To use the package locally (eg to run sample files), run

```
pip install --editable .
```

If you want to use this package from another directory, you can run:

```
pip install path_to_this_repo
```

