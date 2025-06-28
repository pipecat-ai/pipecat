
1. (venv2) (base) alexcovo@Athena pipecat % python src/pipecat/examples/run_multisession_ws_patch.py --host 0.0.0.0 --port 27880 --stateless

2. voice-chat-ui % npm run dev

3. pipecat % cloudflared tunnel run pipecat-tunnel



# Voice Chat UI (Pipecat Local WebRTC)

A minimal **Next.js + TypeScript** front-end that talks to a **FastAPI** back-end running the Pipecat example `43-ollama-chatbot_rag.py` via **SmallWebRTCTransport**.  
Everything runs **locally** – no paid cloud services or tunnelling required. 

> **NEW: 100% Private STUN/TURN with Docker coturn**
> 
> You can now run your own STUN/TURN server for WebRTC using Docker, with no dependency on Google or public infra. See below for instructions.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Front-end UI | Next.js 14 (React 18), Tailwind CSS |
| Voice / RTC | `@pipecat-ai/client-js` 0.4.1 & `@pipecat-ai/small-webrtc-transport` |
| Back-end API | FastAPI (Python 3.12) – example `43-ollama-chatbot_rag.py` |
| Signalling | HTTP POST `/api/offer` (SDP offer/answer) |
| Media | WebSocket `/ws` carrying RTP over WebRTC |

---

## Folder Structure

```
voice-chat-ui/
 ├─ components/VoiceAssistant.tsx   # React component with hooks
 ├─ pages/_app.tsx                  # RTVIClient initialisation
 ├─ .env.local                      # Front-end environment variables
 └─ README.md                       # ← you are here
```

Back-end changes live in `src/pipecat/examples/run.py` (the FastAPI wrapper).

---

## Full Local Startup Checklist (Copy-Paste Friendly)

This checklist lets you bring up the entire Pipecat voice chat stack from scratch, including Docker coturn, backend, frontend, Cloudflare tunnel, and ICE config. **Update your IP addresses if your LAN or public IP changes.**

### 1. Start coturn (STUN/TURN) with Docker
```bash
docker rm -f coturn 2>/dev/null # Stop/remove old container if needed
docker run -d --name coturn \
  -p 3478:3478/udp \
  -p 3478:3478/tcp \
  -p 49160-49200:49160-49200/udp \
  --restart=unless-stopped \
  instrumentisto/coturn:latest \
  --no-cli \
  --log-file=stdout \
  --lt-cred-mech \
  --user=alex:supersecret \
  --realm=turn.local \
  --min-port=49160 \
  --max-port=49200 \
  --external-ip YOUR_PUBLIC_IPV4
```
- Replace `YOUR_PUBLIC_IPV4` with your current public IPv4 (get with `curl -4 ifconfig.me` or https://whatismyipaddress.com/).
- Make sure your router forwards UDP 3478 and 49160-49200 to your Mac’s LAN IP.

### 2. Start the Pipecat Backend
```bash
# From repo root
python examples/foundational/43-ollama-chatbot_rag.py --transport webrtc
```
- By default, listens on `http://0.0.0.0:27880`.

### 3. Start the Next.js Frontend
```bash
cd examples/foundational/voice-chat-ui
npm install         # Only needed once
echo "NEXT_PUBLIC_PIPECAT_URL=http://localhost:27880" > .env.local
npm run dev         # Starts on http://localhost:3000
```

### 4. (Optional) Start Cloudflare Tunnel for Public Access
```bash
# Install cloudflared if needed
# brew install cloudflared
cloudflared tunnel login
cloudflared tunnel create pipecat-voice-chat
cloudflared tunnel route dns pipecat-voice-chat YOUR_DOMAIN
cloudflared tunnel run pipecat-voice-chat
```
- Update Cloudflare config to point to your frontend/backend ports as needed.

### 5. ICE Server Config (Frontend & Backend)
```js
iceServers: [
  { urls: 'stun:YOUR_PUBLIC_IPV4:3478' },
  {
    urls: ['turn:YOUR_PUBLIC_IPV4:3478?transport=udp'],
    username: 'alex',
    credential: 'supersecret',
  },
]
```
- Replace `YOUR_PUBLIC_IPV4` everywhere if your public IP changes.
- If your Mac’s LAN IP changes, update your router’s port forwarding rules accordingly.

---

## Running the Demo Locally

### (Recommended) Run your own STUN/TURN server with Docker

1. **Start coturn:**
   ```bash
   docker run -d --name coturn \
     -p 3478:3478/udp \
     -p 3478:3478/tcp \
     -p 49160-49200:49160-49200/udp \
     --restart=unless-stopped \
     instrumentisto/coturn:latest \
     --no-cli \
     --log-file=stdout \
     --lt-cred-mech \
     --user=alex:supersecret \
     --realm=turn.local \
     --min-port=49160 \
     --max-port=49200 \
     --external-ip YOUR_PUBLIC_IPV4
   ```
   Replace `YOUR_PUBLIC_IPV4` with your real public IPv4 address.

2. **Forward ports on your router:**
   - UDP: 3478 and 49160–49200 → your Mac’s LAN IP
   - TCP: 3478 (optional, for TURN-over-TCP)

3. **Update ICE config (frontend & backend):**
   ```js
   iceServers: [
     { urls: 'stun:YOUR_PUBLIC_IPV4:3478' },
     {
       urls: ['turn:YOUR_PUBLIC_IPV4:3478?transport=udp'],
       username: 'alex',
       credential: 'supersecret',
     },
   ]
   ```
   Replace `YOUR_PUBLIC_IPV4` everywhere.

---

### Quick-start sequence (three terminal tabs)

| Tab | Purpose | Command |
|-----|---------|---------|
| 1   | Backend (FastAPI + Pipecat) | `python examples/foundational/43-ollama-chatbot_rag.py --transport webrtc` |
| 2   | Front-end (Next.js dev server) | `cd examples/foundational/voice-chat-ui && npm run dev` |
| 3   | Browser (optional auto-open) | Open `http://localhost:3000` in Chrome/Edge/Firefox |

Wait for Tab 1 to print `Uvicorn running on http://0.0.0.0:27880` **before** pressing **Start** in the UI.

---

1. **Back-end** – start the Pipecat example with WebRTC transport:

   ```bash
   # From repo root
   python examples/foundational/43-ollama-chatbot_rag.py --transport webrtc
   # → listens on http://0.0.0.0:27880 by default
   ```

   The modified `/api/offer` handler now supports the initial *handshake* request
   (it returns `{}` when no `sdp` is present) and CORS is open to all origins for
   development.

2. **Front-end** – in another shell:

   ```bash
   cd examples/foundational/voice-chat-ui
   npm install      # first time only
   echo "NEXT_PUBLIC_PIPECAT_URL=http://localhost:27880" > .env.local
   npm run dev      # Next.js on http://localhost:3000
   ```

3. Open `http://localhost:3000` and click **Start**.

   * The browser sends a handshake `POST /api/offer` with `{ rtvi_client_version }`.
   * Back-end returns `{}` (200).  
   * The client then sends the real SDP offer.  
   * FastAPI responds with the SDP answer and WebRTC media flows via `/ws`.

---

## Important Implementation Details

### Front-end (`pages/_app.tsx` or ICE config)

```js
const ICE_SERVERS = [
  { urls: 'stun:YOUR_PUBLIC_IPV4:3478' },
  {
    urls: ['turn:YOUR_PUBLIC_IPV4:3478?transport=udp'],
    username: 'alex',
    credential: 'supersecret',
  },
];
```

*Use only your own coturn for both STUN and TURN. No Google or public infra required.*

*If `NEXT_PUBLIC_PIPECAT_URL` is undefined the code falls back to
`https://ai.alexcovo.com`, so **always** set `.env.local` during development.*

### Back-end (`run.py`)

```python
ice_servers = [
    IceServer(urls="stun:YOUR_PUBLIC_IPV4:3478"),
    IceServer(
        urls="turn:YOUR_PUBLIC_IPV4:3478?transport=udp",
        username="alex",
        credential="supersecret",
    ),
]
```

*Use only your own coturn for both STUN and TURN. No Google or public infra required.*

```py
@app.post('/api/offer')
async def offer(request: Request, background_tasks: BackgroundTasks):
    try:
        body = await request.json()
    except Exception:
        body = {}

    logger.debug('/api/offer payload received: %s', body)

    # Handshake request – no SDP yet
    if 'sdp' not in body:
        return {}

    # ... existing connection / renegotiation logic ...
```

*   Returns early for the first empty handshake.  
*   Continues with original logic once `sdp` appears.

### CORS

```py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # ok for local dev
    allow_methods=["*"],
    allow_headers=["*"],
)
```

The OPTIONS pre-flight therefore succeeds and `Access-Control-Allow-Origin`
headers are added **only** when the handler returns 2xx; if it crashes you will
see a misleading CORS error in the browser, so always check the back-end log.

---

## Troubleshooting Checklist

1. Front-end keeps hitting `https://ai.alexcovo.com` → `.env.local` missing.
2. 500 + CORS message → check back-end stack trace (probably `KeyError: 'sdp'`).
3. `ERR_NAME_NOT_RESOLVED` → bad DNS / tunnel URL – use localhost.
4. No audio – ensure browser granted microphone permission.

---

## Snapshot / Change Log

* **17 Jun 2025** – Added local handshake support in FastAPI `/api/offer` + docs.
* **17 Jun 2025** – Clean `VoiceAssistant.tsx`, removed custom visualiser, use
  built-in hooks.

---

Enjoy your fully-local Pipecat voice chat!
