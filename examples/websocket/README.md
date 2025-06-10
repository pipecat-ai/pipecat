# Voice Agent

A Pipecat example demonstrating the simplest way to create a voice agent using `WebsocketTransport`.

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
   - Choose what do you wish to use, 'fast_api' or 'websocket_server'

#### ▶️ Run the Server
```bash
python server/server.py
```

### 3️⃣ Connect Using a Custom Client App

For client-side setup, refer to the:
- [Typescript Guide](client/README.md).

## ⚠️ Important Note
Ensure the bot server is running before using any client implementations.

## 📌 Requirements

- Python **3.10+**
- Node.js **16+** (for JavaScript components)
- Google API Key

---

### 💡 Notes
- Ensure all dependencies are installed before running the server.
- Check the `.env` file for missing configurations.

Happy coding! 🎉