# Pipecat Ojin Chatbot Example 🚀🎤🎬

Welcome to the **Pipecat Ojin Chatbot Example**!

This example is meant to showcase the real-time speech-to-video generation capabilities of the Oris model available on the [Ojin platform](https://ojin.ai/).
Essentially, it will send a real-time stream of audio from a TTS generator to the Ojin Model API, receive a stream of lipsynced video frames and display them in real-time.

---

## 🎉 Features

- **OjinVideoService:**  
  Connects to Ojin platform through websockets and manages conversation state. It will generate idle frames on startup and cache them in memory so they don't need to be generated again. It will also manage transitions with speech so they feel smooth. You will need an OJIN_API_KEY and OJIN_CONFIG_ID setup in your .env file.

- **HumeSTSService**
  Connects to Hume platform through websockets which provides a Speech to Speech service covering STT/LLM/TTS in a single service. You will need an HUME_API_KEY and HUME_CONFIG_ID setup in your .env file.
- **Audio Input Processing:**  
  Capture and process audio input from the pipecat application.
- **Easy Setup:**  
  Everything you need is in the [`requirements.txt`](./requirements.txt).

---

## 🔧 Installation

Setup python environment:

```bash
python -m venv venv

source venv/bin/activate
```

Install Dependencies:

```bash
# Install remaining dependencies
pip install -r requirements.txt
```

For MacOS instead of pyaudio you can use portaudio

```bash
brew install portaudio
```

Make sure to fill out proper values for required environment variables in the `.env` file. An example of the needed variables is provided in `env.example`.

## 🚀 Usage

Run the main script:

```bash
python bot.py
```

## 📦 Dependencies

The project relies on:

- [pipecat](https://github.com/pipecat-ai/pipecat) – For building the audio processing pipeline.
- **Ojin** – For speech to video generation. [https://ojin.ai/](https://ojin.ai/)
- **Hume** – For Speech to Speech service covering STT/LLM/TTS in a single service. [https://hume.ai/](https://hume.ai/)
- **Tkinter** – For local audio input and output and video output.

## NOTE

This example currently requires headphones because the transports used lack echo cancellation. To use it without headphones, switch to a transport with built-in echo cancellation, such as `LiveKit` or `Daily`.


## In-Depth

If you want to check the API in more depth you can read through the `OJIN_WEBSOCKET_API.md` file, it's not required but gives a general idea about how the internal websocket protocol works.
