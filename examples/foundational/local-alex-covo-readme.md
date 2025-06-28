# Alex Covo – Local Ollama RAG Chatbot Multi-Session Setup (Default)

> **NOTE:** The default and recommended way to run your Pipecat voice assistant is now via the multi-session backend. All enhancements, bugfixes, and new features are in the multi-session files:
> - `src/pipecat/examples/run_multisession_ws_patch.py`
> - `examples/foundational/ollama_chatbot_rag_multi.py`
> - `examples/foundational/ws_handler.py`
>
> The old single-user scripts are legacy and not recommended for new work. Always use the multi-session backend.


A distilled, personal cheat-sheet for running my Pipecat + Ollama voice assistant and exposing it over HTTPS with ngrok.

---

## 1. Prerequisites (one-time)

| What | Command |
|------|---------|
| Python venv | `python -m venv venv && source venv/bin/activate` |
| Pipecat code (editable) | `pip install -e . && pip install -r requirements.txt` |
| Ollama server | `ollama serve`  (model `gemma3n` pulled) |
| Kokoro TTS | `docker run -p 8880:8080 ghcr.io/pipecat-ai/kokoro:latest` |
| Reserve ngrok sub-domain | In dashboard: `raven-feasible-gelding.ngrok-free.app` |

---

## 2. Start the chatbot

```bash
cd examples/foundational
source ../venv/bin/activate  # if not already
python 43-ollama-chatbot_rag.py \
       --host 0.0.0.0 --port 27880 --transport webrtc
```
Expect: `Uvicorn running on http://0.0.0.0:27880`.

---

## 3. Local testing

Open in Chrome:
* Built-in UI – http://localhost:27880/client/
* Custom UI – http://localhost:27880/web/

> Mic permissions only work on localhost or HTTPS.

---

## 4. Expose via ngrok

```bash
ngrok http --domain=raven-feasible-gelding.ngrok-free.app 27880
```
Browser URL → https://raven-feasible-gelding.ngrok-free.app/web/

Keep both terminals running.

---

## 5. Common fixes

* **Port in use** → `lsof -i :27880` then `kill <PID>` or change `--port` & ngrok command.
* **ERR_NGROK_8012** → Start Python server first or match port.
* **Mic blocked** → Use HTTPS ngrok URL, not LAN IP.

---

Happy hacking!

---

## Networking modes: Local, Cloud Tunnel, Public IP

| Mode | When to use | TURN / coturn | Cloudflared / Ngrok | Notes |
|------|-------------|---------------|--------------------|-------|
| **Local-only** | All devices on same Wi-Fi / LAN | _Disabled_ | _None_ | Fastest; skip coturn & tunnel. In browser, host ICE candidates work on the subnet. |
| **Cloud Tunnel (default)** | Any network, zero-config | Optional minimal coturn | ✅ Cloudflared (`pipecat-tunnel`) | Tunnel punches through NAT; external TURN rarely needed. |
| **Public IP (home lab)** | Static public IP with ports forwarded | Full coturn with `--external-ip` | _None_ | Requires router port-forward (3478 + relay range).|

### Switching strategies
1. **Local demo:** Comment out coturn & Cloudflared in `start_voice_chatbot.sh`.
2. **Travel / café:** Use the script as-is (Cloudflared on by default).
3. **Home server:** Replace the coturn `docker run` line with:
   ```bash
   docker run -d --name coturn \
     -p 3478:3478/udp -p 3478:3478/tcp \
     -p 49160-49200:49160-49200/udp \
     --restart=unless-stopped \
     instrumentisto/coturn:latest \
     --no-cli --log-file=stdout --lt-cred-mech \
     --user=alex:supersecret --realm=turn.local \
     --min-port=49160 --max-port=49200 \
     --external-ip <PUBLIC_IP>
   ```
   Adjust `<PUBLIC_IP>` or remove `--external-ip` when roaming.

