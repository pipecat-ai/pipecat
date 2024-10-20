# podcastai

### Have a conversation about any article on the web

podcastai is a fast conversational AI built using [Daily](https://www.daily.co/) for real-time media transport and [Cartesia](https://cartesia.ai) for text-to-speech. Everything is orchestrated together (VAD -> STT -> LLM -> TTS) using [Pipecat](https://www.pipecat.ai/).

## Get started

```python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp env.example .env # and add your credentials

```

## Run the bot

```bash
python bot.py
```

While the app is running, go to the `https://<yourdomain>.daily.co/<room_url>` set in `DAILY_SAMPLE_ROOM_URL` and talk to studypal!
