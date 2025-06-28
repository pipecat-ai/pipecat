# AI Voice Assistant – Technical Overview

_Last updated: 2025-06-20_

---

## 1. Goal
Build a **fully-local, privacy-first, multi-user voice assistant** that answers questions about Alex Covo and showcases his portfolio, while remaining free to run and simple to deploy on Apple-Silicon hardware.

## 2. Core Technologies
| Layer | Technology (default) | Notes |
|-------|----------------------|-------|
| Orchestration | **Pipecat** | Real-time voice pipeline, VAD, multi-session WebRTC |
| LLM | **Ollama** running `alexcovo-ai-rep-rag:1.1` fine-tune (fallback: `gemma3-4b:128k`) | fine-tuned on Alex-specific KB; `/no_think` enforced |
| RAG | **Stateless keyword-plus-heading scoring** (top-k = 3) | Built in `rag_utils.py`; zero chat history to curb hallucinations |
| STT | **Lightning-Whisper-MLX** `distil-medium.en` | Runs on Apple Neural Engine |
| TTS | **Kokoro-FastAPI** model `kokoro`, voice `shimmer` | 24 kHz WAV via `OpenAITTSService` |
| Front-end | **Next.js** app in `voice-chat-ui/` + built-in `/client/` | WebRTC/Mic, React hooks |
| Networking | **Cloudflare Tunnel** (`pipecat-tunnel`) | HTTPS → local port 27880 |

## 3. Runtime Ports
| Service | Port | Command |
|---------|------|---------|
| FastAPI + WebRTC | **27880** | `python src/pipecat/examples/run_multisession_ws_patch.py --stateless` |
| Kokoro-FastAPI | **8880** | `docker compose -f Kokoro-FastAPI/docker/cpu/docker-compose.yml up -d` |
| Ollama | **11434** | `ollama serve` |
| Cloudflare | 443 (public) → 27880 | `cloudflared tunnel run pipecat-tunnel` |

## 4. Important Files
| Path | Purpose |
|------|---------|
| `examples/foundational/ollama_chatbot_rag_multi.py` | Main multi-session entry; toggles `--stateless` flag |
| `examples/foundational/rag_utils.py` | Stateless RAG utilities (`retrieve_chunks`, `RAGProcessor`) |
| `src/pipecat/examples/run_multisession_ws_patch.py` | Generic launcher (same pipeline, used by CI) |
| `examples/foundational/ws_handler.py` | Broadcast captions to multiple WebSocket clients |
| `examples/foundational/voice-chat-ui/` | **Next.js** custom UI (React, Tailwind, hooks for WebRTC) |
| `examples/foundational/local-alex-covo-readme.md` | Personal ops cheat-sheet |
| `examples/foundational/43-ollama-chatbot_rag.md` | Full setup & deployment guide |
| `examples/foundational/.env` | Runtime config (`OLLAMA_MODEL`, `QWEN_THINK_MODE`, etc.) |

## 5. Current Progress
* ✅ Multi-user voice pipeline working (per-session isolation).
* ✅ Stateless RAG over local markdown docs (no context bloat).
* ✅ Lightning-Whisper-MLX integrated (STT latency ≈35 ms/chunk).
* ✅ Kokoro TTS (shimmer voice) streaming locally (~220 ms/utt).
* ✅ Cloudflare HTTPS tunnel online (`https://ai.alexcovo.com`).
* ⚠ Thinking artifacts occasionally leak; need robust strip before TTS.
* ⚠ RAG chunking is naïve; migrate to ANN vector search.

## 6. Issues / TODO
| Status | Task |
|--------|------|
| Done | Strip thinking content before TTS |
| High | Replace keyword RAG with FAISS/Qdrant vector search |
| Priority | Task |
|----------|------|
| High | Add `strip_thinking_content()` just before TTS in pipeline |
| High | Replace keyword RAG with FAISS/Qdrant ANN vector store |
| Med | Evaluate **mlx-audio** TTS (Kokoro-82M) to remove Docker overhead |
| Med | Structured logging + Prometheus exporter |
| Low | Gallery page in Next.js UI for artwork & audio captions |
| Low | Auth / rate limiting for public endpoint |

## 7. Nice-to-Have Features
* **Gallery tab** in `voice-chat-ui/` for image/audio showcase.
* **Conversation export** (markdown) per user.
* **Admin dashboard** (FastAPI + HTMX) for live session stats.
* **Emotion-based voice switching** (multiple Kokoro voices).
* **CI/CD** workflow to auto-build Kokoro image & push to GHCR.

## 8. Quick-Start (Single Node)
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
ollama serve &                       # pull gemma3n
docker compose -f Kokoro-FastAPI/docker/cpu/docker-compose.yml up -d
python examples/foundational/ollama_chatbot_rag_multi.py --transport webrtc
```
Open `http://localhost:27880/client/` or `http://localhost:27880/web/` (Next.js).

## 9. Handoff Checklist
1. Copy `.env.example` ⇒ `.env`; set `OLLAMA_MODEL`, `QWEN_THINK_MODE`, Cloudflare IDs.
2. Ensure Ollama & Kokoro running; verify ports (11434, 8880).
3. Launch FastAPI server; confirm captions WS OK.
4. Review TODO table; prioritise vector RAG + thinking strip.
5. Dive into `voice-chat-ui/` to continue UI/UX work (gallery, theme).

---
_Signed-off: Current maintainer_
