# 🚀 Quick Start Guide

Get your Pipecat AI Backend running in 5 minutes!

## Prerequisites

✅ You have:
- AI rig with RTX 3090, 128GB RAM, i5-10400
- Python 3.10+ installed
- Internet connection

## Step-by-Step Setup

### 1. Install Ollama (if not already installed)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve &

# Pull the recommended model for low latency
ollama pull llama3.2
```

Wait for the model to download (this may take 5-10 minutes).

### 2. Setup Python Environment

```bash
# Navigate to pipecat directory
cd /home/user/pipecat

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate

# Install Pipecat framework
pip install -e .

# Install backend dependencies
cd backend
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit the .env file
nano .env  # or use your preferred editor
```

**Minimum required changes in .env:**

```env
# Get your Cartesia API key from: https://cartesia.ai
CARTESIA_API_KEY=your_actual_api_key_here

# Generate a secure secret key
SECRET_KEY=your_secure_random_key_here
```

To generate a secure secret key:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 4. Start the Server

```bash
# Easy way - use the startup script
./start.sh

# Or manually
python -m backend.main
```

### 5. Test It Works

Open your browser and visit:
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

You should see:
```json
{
  "status": "healthy",
  "active_pipelines": 0,
  "active_sessions": 0,
  "total_sessions": 0
}
```

## 🎉 You're Ready!

### Default Admin Login

- **Username**: `admin`
- **Password**: `admin123`

⚠️ **Change this in production!**

### Test Voice Conversation

1. Go to http://localhost:8000/docs
2. Find the `/api/voice/ws` endpoint
3. Click "Try it out"
4. Test with a WebSocket client

### Next Steps

1. **Get Cartesia API Key**: https://cartesia.ai (free tier available)
2. **Setup Twilio** (optional): https://console.twilio.com
3. **Build Frontend**: Connect your React app to the API
4. **Customize**: Edit voice configs and system prompts in `config.py`

## Common Issues

### "Ollama is not running"
```bash
# Start Ollama
ollama serve
```

### "Model not found"
```bash
# Pull the model
ollama pull llama3.2
```

### "Import errors"
```bash
# Make sure you installed Pipecat
cd /home/user/pipecat
pip install -e .
```

### "Cartesia API errors"
- Make sure you added your real API key to `.env`
- Get one from: https://cartesia.ai

## Performance Tips

For your hardware (RTX 3090, 128GB RAM):

- **llama3.2** - Fastest, ~100ms latency ⚡
- **llama3.1:8b** - Balanced, ~200ms latency ⚖️
- **llama3.1:70b** - Best quality, ~500ms latency 🎯

To use a different model:
```bash
# Pull the model
ollama pull llama3.1:8b

# Update .env
OLLAMA_MODEL=llama3.1:8b
```

## What's Next?

Check out the full [README.md](README.md) for:
- Complete API documentation
- User management
- Session tracking
- Twilio phone integration
- Production deployment tips
- Frontend integration examples

## Need Help?

- 📚 API Docs: http://localhost:8000/docs
- 📖 Full README: [README.md](README.md)
- 🌐 Pipecat Docs: https://docs.pipecat.ai
