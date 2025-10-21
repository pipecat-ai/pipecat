
# Pipecat Ojin Chatbot Example 🚀🎤🎬

Welcome to the **Pipecat Ojin Chatbot Example**! 

This example is meant to showcase the realtime speech to video generation capabilities of https://dashboard.ojin.ai/ Oris model
Essentially it will be sending audio to ojin platform from a TTS generator and receiving video frames lipsynced in return

---

## 🎉 Features

- **OjinPersonaService:**  
  Connects to Ojin platform through websockets and manages conversation state. It will generate idle frames on startup and cache them in memory so they don't need to be generated again. It will also manage transitions with speech so they feel smooth. You will need an OJIN_API_KEY and OJIN_PERSONA_ID setup in your .env file.

- **HumeSTSService** 
  Connects to Hume platform through websockets which provides a Speech to Speech service covering STT/LLM/TTS in a single service. You will need an HUME_API_KEY and HUME_CONFIG_ID setup in your .env file.
  
- **Audio Input Processing:**  
  Capture and process audio input from the pipecat application.
  
- **Easy Setup:**  
  Everything you need is in the [`requirements.txt`](./requirements.txt).

---

## 🔧 Installation


Install Dependencies:

```bash
pip install -r requirements.txt
```

For MacOS instead of pyaudio you can use portaudio

```bash
brew install portaudio
```

Make sure to fill out proper values for required environment variables in the `.env` file.

## 🚀 Usage

Run the main script:

```bash
python bot.py
```

## 📦 Dependencies

The project relies on:

- [pipecat](https://github.com/pipecat-ai/pipecat) – For building the audio processing pipeline.
- **Ojin** – For avatar video generation from tts audio. https://dashboard.ojin.ai/
- **Hume** – For Speech to Speech service covering STT/LLM/TTS in a single service. https://hume.ai/
- **Tkinter** – For local audio input and output and video output.
