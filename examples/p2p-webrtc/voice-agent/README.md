# Voice Agent

A Pipecat example demonstrating the simplest way to create a voice agent using `SmallWebRTCTransport`.

## 🚀 Quick Start

### 1️⃣ Start the Bot Server

#### 🔧 Set Up the Environment
1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   - Copy `env.example` to `.env`
   ```bash
   cp env.example .env
   ```
   - Add your API keys

#### ▶️ Run the Server
```bash
python server.py
```

### 2️⃣ Connect Using the Client App

Open your browser and visit:
```
http://localhost:7860
```

## 📌 Requirements

- Python **3.10+**
- Node.js **16+** (for JavaScript components)
- Google API Key
- Modern web browser with WebRTC support

---

## WebRTC ICE Servers Configuration

When implementing WebRTC in your project, **STUN** (Session Traversal Utilities for NAT) and **TURN** (Traversal Using Relays around NAT) 
servers are usually needed in cases where users are behind routers or firewalls.

In local networks (e.g., testing within the same home or office network), you usually don’t need to configure STUN or TURN servers. 
In such cases, WebRTC can often directly establish peer-to-peer connections without needing to traverse NAT or firewalls.

### What are STUN and TURN Servers?

- **STUN Server**: Helps clients discover their public IP address and port when they're behind a NAT (Network Address Translation) device (like a router). 
This allows WebRTC to attempt direct peer-to-peer communication by providing the public-facing IP and port.
  
- **TURN Server**: Used as a fallback when direct peer-to-peer communication isn't possible due to strict NATs or firewalls blocking connections. 
The TURN server relays media traffic between peers.

### Why are ICE Servers Important?

**ICE (Interactive Connectivity Establishment)** is a framework used by WebRTC to handle network traversal and NAT issues. 
The `iceServers` configuration provides a list of **STUN** and **TURN** servers that WebRTC uses to find the best way to connect two peers. 

### Example Configuration for ICE Servers

Here’s how you can configure a basic `iceServers` object in WebRTC for testing purposes, using Google's public STUN server:

```javascript
const config = {
  iceServers: [
    {
      urls: ["stun:stun.l.google.com:19302"], // Google's public STUN server
    }
  ],
};
```

> For testing purposes, you can either use public **STUN** servers (like Google's) or set up your own **TURN** server. 
If you're running your own TURN server, make sure to include your server URL, username, and credential in the configuration.

---

### 💡 Notes
- Ensure all dependencies are installed before running the server.
- Check the `.env` file for missing configurations.
- WebRTC requires a secure environment (HTTPS) for full functionality in production.

Happy coding! 🎉