
# Pipecat Ojin Chatbot Example 🚀🎤🎬

Welcome to the **Pipecat Ojin Chatbot Example**! 

This example is meant to showcase the realtime avatar generation capabilities of https://dashboard.ojin.ai/
Essentially it will be sending audio to ojin platform from a TTS/LLM/STT pipeline and receiving video frames lipsynced in return

---

## 🎉 Features

- **OjinAvatarService Integration:**  
  Connect to Ojin platform to generate talking avatar video frames.
  
- **Audio Input Processing:**  
  Capture and process audio input from the pipecat application.
  
- **Real-time Avatar Video:**  
  Generate video frames of a talking avatar and synchronized audio output frames
  
- **Easy Setup:**  
  Everything you need is in the [`requirements.txt`](./requirements.txt).

---

## 🔧 Installation


Install Dependencies:

```bash
pip install -r requirements.txt
```

Make sure to fill out proper values for required environment variables in the `.env` file.

---

## 🚀 Usage

Run the main script:

```bash
python bot.py
```

When the app launches, it will connect to the OjinAvatarService and begin processing audio input to generate corresponding video frames from a talking avatar.

## 📦 Dependencies

The project relies on:

- [pipecat](https://github.com/pipecat-ai/pipecat) – For building the audio processing pipeline.
- **OjinAvatarService** – For connecting to the Ojin platform and generating avatar video. https://dashboard.ojin.ai/
- **OpenAI** – For LLM processing.
- **ElevenLabs** – For TTS processing.
- **Deepgram** – For STT processing.
- **Tkinter** – For local audio input and output and video output.