# SmallWebRTC and Daily

A Pipecat example demonstrating how to interoperate audio and video between `SmallWebRTCTransport` and `DailyTransport`.

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

###  1️⃣ Connect the first client using Daily Prebuilt

- Open your browser and navigate to the same URL that you configured inside your `.env` file:
  - `DAILY_SAMPLE_ROOM_URL`

### 2️⃣ Connect the second client using SmallWebRTC Prebuilt UI

- Open your browser and navigate to:
👉 http://localhost:7860
  - (Or use your custom port, if configured)

## ⚠️ Important Note
Ensure the bot server is running before using any client implementations.

## 📌 Requirements

- Python **3.10+**
- Node.js **16+** (for JavaScript components)
- Google API Key
- Modern web browser with WebRTC support

---

### 💡 Notes
- Ensure all dependencies are installed before running the server.
- Check the `.env` file for missing configurations.
- WebRTC requires a secure environment (HTTPS) for full functionality in production.

Happy coding! 🎉