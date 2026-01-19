# Pipecat Ojin TTS Example 🚀🎙️🗣️

Welcome to the **Pipecat Ojin TTS Example**!

This example runs a speech-to-speech demo: your microphone audio is transcribed with Whisper, answered by OpenAI, and spoken back with Ojin TTS to your speakers.

---

## 🎉 Features

- **LocalAudioTransport + Silero VAD:** microphone input with voice-activity detection and speaker output at 24 kHz.
- **Whisper STT (local):** converts speech to text (tiny model on CPU for portability).
- **OpenAI LLM:** generates concise responses (defaults to `gpt-4o-mini`).
- **Ojin TTS:** converts the full LLM reply to speech at 24 kHz.
- **Buffering aggregator:** waits for the full LLM response before sending to TTS for better prosody.

---

## 🔧 Installation

### 1. Get Credentials

1. Go to [https://ojin.ai/](https://ojin.ai/)
2. Create an account
3. Create an **API key**
4. Open your TTS model configuration and copy its **config ID**
5. Get an **OpenAI API key** for the LLM (e.g., gpt-4o-mini)

### 2. Configure Environment

Copy the example environment file and fill in your credentials:

```bash
# macOS/Linux
cp env.example .env

# Windows PowerShell
copy env.example .env
```

Edit `.env`:
```
OJIN_API_KEY=your-ojin-api-key
OJIN_TTS_CONFIG_ID=your-ojin-tts-config-id
OPENAI_API_KEY=your-openai-api-key
```

### 3. Install Dependencies

Run from this folder so the editable installs resolve correctly:

```bash
pip install -r requirements.txt
```

If you are on macOS and run into pyaudio issues, install portaudio first:

```bash
brew install portaudio
```

---

## 🚀 Usage

Start the speech-to-speech pipeline (mic → Whisper STT → OpenAI → Ojin TTS → speakers):

```bash
python example.py
```

What happens:
- Logs show pipeline stages starting.
- Speak into your mic; Whisper transcribes locally (CPU by default).
- The transcription goes to OpenAI (gpt-4o-mini by default) for a reply.
- The full LLM reply is sent to Ojin TTS and played through your speakers.
- Press `Ctrl+C` to stop.

---

## 📦 Dependencies

The project relies on:

- [pipecat](https://github.com/pipecat-ai/pipecat) – audio pipeline with local transports, Whisper STT, and Ojin TTS.
- **Ojin** – for text-to-speech streaming. [https://ojin.ai/](https://ojin.ai/)
- **OpenAI** – for LLM responses. [https://platform.openai.com/](https://platform.openai.com/)
- **pyaudio** – for microphone/speaker IO (portaudio on macOS if needed).

## NOTE

- Use headphones if you encounter echo; the local transport here has no AEC. For echo cancellation, switch to a transport with built-in AEC (e.g., LiveKit or Daily).
