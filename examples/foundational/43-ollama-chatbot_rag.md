# 43-Ollama-Chatbot-RAG (Voice Edition)

End-to-end recipe for running the RAG chatbot with:

* Ollama LLM (already running via `ollama serve`)
* Kokoro-FastAPI Text-to-Speech (Docker)
* Lightning-Whisper-MLX Speech-to-Text (Apple Silicon)
* Pipecat WebRTC transport + browser UI

---

## 1. Prepare the Python environment

```bash
python3 -m venv venv2
source venv2/bin/activate
pip install -r requirements.txt  # or pip install -e .[all_needed_extras]
```

> The example needs the Pipecat extras `webrtc,whisper,openai,ollama,silero`.  If you built from source:
>
> ```bash
> pip install -e ".[webrtc,whisper,openai,ollama,silero]"
> ```

## 2. Build or pull Kokoro-FastAPI (first time only)

```bash
cd Kokoro-FastAPI/docker/cpu/
docker compose build  # or `docker compose pull` if you have a registry image
cd -
```

## 3. Launch Kokoro and export env-vars

```bash
# Start TTS service on localhost:8880
docker compose -f Kokoro-FastAPI/docker/cpu/docker-compose.yml up -d

# Tell Pipecat to use it
export OPENAI_API_BASE=http://localhost:8880/v1
export OPENAI_API_KEY=dummy  # any non-empty string works
```

## 4. Run the voice chatbot

```bash
python examples/foundational/43-ollama-chatbot_rag.py --transport webrtc
```

Uvicorn will advertise a URL such as `http://0.0.0.0:27880/client/` – open it, allow microphone, and start talking.

## 5. Make it public (optional)

```bash
cloudflared tunnel run pipecat-tunnel
```

Visit `https://ai.alexcovo.com/client/` from any device.

## 6. Shutdown

```bash
docker compose -f Kokoro-FastAPI/docker/cpu/docker-compose.yml down
```

---

### Networking modes (quick reference)
* **Local-only:** skip coturn & tunnel → peers on same LAN.
* **Cloud Tunnel (default):** use Cloudflared (`pipecat-tunnel`), minimal coturn.
* **Public IP:** full coturn with `--external-ip`, ports forwarded.

See `local-alex-covo-readme.md` for the detailed table and commands.

## 🏗️ Full Tech-Stack Overview

| Layer | Technology (default) | Purpose |
|-------|----------------------|---------|
| STT | **Lightning-Whisper-MLX** | Ultra-fast on Apple Neural Engine |
| LLM | **Ollama – gemma3n** | Local, switchable models |
| RAG | Pipecat FS vector store | Adds custom docs knowledge |
| TTS | **Kokoro-FastAPI** | Streaming, studio-quality voices |
| Orchestration | **Pipecat** | Real-time pipeline, VAD, metrics |
| Transport | **SmallWebRTCTransport** | ≈150 ms voice latency |
| Public URL | **Cloudflare Tunnel** | Free HTTPS exposure |

---

## One-command helper script

If you prefer, copy the contents below into `scripts/start_voice_chatbot.sh`, make it executable (`chmod +x ...`), and run it from the repo root.

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$ROOT_DIR"

source venv2/bin/activate

echo "[+] Starting Kokoro-FastAPI…"
docker compose -f Kokoro-FastAPI/docker/cpu/docker-compose.yml up -d

export OPENAI_API_BASE=http://localhost:8880/v1
export OPENAI_API_KEY=dummy

echo "[+] Launching Pipecat example…"
python examples/foundational/43-ollama-chatbot_rag.py --transport webrtc
```
