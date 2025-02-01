

# Pipecat &mdash; Examples

## Foundational snippets
Small snippets that build on each other, introducing one or two concepts at a time.

➡️ [Take a look](https://github.com/pipecat-ai/pipecat/tree/main/examples/foundational)

## Chatbot examples
Collection of self-contained real-time voice and video AI demo applications built with Pipecat.

### Quickstart

Each project has its own set of dependencies and configuration variables. They intentionally avoids shared code across projects &mdash; you can grab whichever demo folder you want to work with as a starting point.

We recommend you start with a virtual environment:

```shell
cd pipecat-ai/examples/simple-chatbot

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

Next, follow the steps in the README for each demo.

ℹ️ Make sure you `pip install -r requirements.txt` for each demo project, so you can be sure to have the necessary service dependencies that extend the functionality of Pipecat. You can read more about the framework architecture [here](https://github.com/pipecat-ai/pipecat/tree/main/docs).

## Projects:

| Project                                      | Description                                                                                                                                | Services                                                          |
|----------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| [Simple Chatbot](simple-chatbot)             | Basic voice-driven conversational bot. A good starting point for learning the flow of the framework.                                       | Deepgram, ElevenLabs, OpenAI, Daily, Daily Prebuilt UI            |
| [Storytelling Chatbot](storytelling-chatbot) | Stitches together multiple third-party services to create a collaborative storytime experience.                                            | Deepgram, ElevenLabs, OpenAI, Fal, Daily, Custom UI               |
| [Translation Chatbot](translation-chatbot)   | Listens for user speech, then translates that speech to Spanish and speaks the translation back. Demonstrates multi-participant use-cases. | Deepgram, Azure, OpenAI, Daily, Daily Prebuilt UI                 |
| [Moondream Chatbot](moondream-chatbot)       | Demonstrates how to add vision capabilities to GPT4. **Note: works best with a GPU**                                                       | Deepgram, ElevenLabs, OpenAI, Moondream, Daily, Daily Prebuilt UI |
| [Patient intake](patient-intake)             | A chatbot that can call functions in response to user input.                                                                               | Deepgram, ElevenLabs, OpenAI, Daily, Daily Prebuilt UI            |
| [Phone Chatbot](phone-chatbot)             | A chatbot that connects to PSTN/SIP phone calls, powered by Daily or Twilio.                                                                    | Deepgram, ElevenLabs, OpenAI, Daily, Twilio                       |
| [Twilio Chatbot](twilio-chatbot)             | A chatbot that connects to an incoming phone call from Twilio.                                                                             | Deepgram, ElevenLabs, OpenAI, Daily, Twilio                       |
| [studypal](studypal)                         | A chatbot to have a conversation about any article on the web                                                                              |                                                                   |
| [WebSocket Chatbot Server](websocket-server) | A real-time websocket server that handles audio streaming and bot interactions with speech-to-text and text-to-speech capabilities. | Cartesia, Deepgram, OpenAI, Websockets |

> [!IMPORTANT]
> These example projects use Daily as a WebRTC transport and can be joined using their hosted Prebuilt UI.
> It provides a quick way to join a real-time session with your bot and test your ideas without building any frontend code. If you'd like to see an example of a custom UI, try Storybot.


## FAQ

### Deployment

For each of these demos we've included a `Dockerfile`. Out of the box, this should provide everything needed to get the respective demo running on a VM:

```shell
docker build username/app:tag .

docker run -p 7860:7860 --env-file ./.env username/app:tag

docker push ...
```

### SSL

If you're working with a custom UI (such as with the Storytelling Chatbot), it's important to ensure your deployment platform supports HTTPS, as accessing user devices such as mics and webcams requires SSL.

If you try to run a custom UI without SSL, you may see an error in the console telling you that `navigator` is undefined, or no devices are available.

### Are these examples production ready?

Yes, kind of.

These demos attempt to keep things simple and are unopinionated regarding environment or scalability.

We're using FastAPI to spawn a subprocess for the bots / agents &mdash; useful for small tests, but not so great for production grade apps with many concurrent users. You can see how this works in each project's `start` endpoint in `server.py`.

Creating virtualized worker pools and on-demand instances is out of scope for these examples, but we hope to add some examples to this repo soon!

For projects that have CUDA as a requirement, such as Moondream Chatbot, be sure to deploy to a GPU-powered platform (such as [fly.io](https://fly.io) or [Runpod](https://runpod.io).)

## Getting help

➡️ [Join our Discord](https://discord.gg/pipecat)

➡️ [Reach us on Twitter](https://x.com/pipecat_ai)
