[![Try](https://img.shields.io/badge/try_it-here-blue)](https://storytelling-chatbot.fly.dev)

# Storytelling Chatbot

<img src="image.png" width="420px">

This example shows how to build a voice-driven interactive storytelling experience.
It periodically prompts the user for input for a 'choose your own adventure' style experience.

We use Gemini 2.0 for creating the story and image prompts, and we add visual elements to the story by generating images using Google's Imagen.


---

### It uses the following AI services:

**Deepgram - Speech-to-Text**

Transcribes inbound participant voice media to text.

**Google Gemini 2.0 - LLM**

Our creative writer LLM. You can see the context used to prompt it [here](src/prompts.py)

**ElevenLabs - Text-to-Speech**

Converts and streams the LLM response from text to audio

**Google Imagen - Image Generation**

Adds pictures to our story. Prompting is quite key for style consistency, so we task the LLM to turn each story page into a short image prompt.

---

## Setup

**Install requirements**

```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Create environment file and set variables:**

```shell
mv env.example .env
```

When deploying to production, to ensure only this app can spawn a new bot, set your `ENV` to `production`

**Build the frontend:**

This project uses a custom frontend, which needs to built. Note: this is done automatically as part of the Docker deployment.

```shell
cd frontend/
npm install
npm run build
```

The build UI files can be found in `frontend/out`

## Running it locally

Start the API / bot manager:

`python src/bot_runner.py --host localhost`

If you'd like to run a custom domain or port:

`python src/bot_runner.py --host somehost --p someport`

➡️ Open the host URL in your browser `http://localhost:7860`

If you've run previous versions of the demo, make sure to set `ENV=dev`, and remove the `RUN_AS_VM` line from the .env file.

---

## Improvements to make

- Wait for track_started event to avoid rushed intro
- Show 5 minute timer on the UI
