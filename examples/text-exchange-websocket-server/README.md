# Websocket Server

This is an example that shows how to use `WebsocketServerTransport` to communicate with a web client.

It also shows how to create a chat interface that sends and receives both text and audio, which can be useful for debugging.

<img width="949" alt="Screenshot 2025-01-24 at 4 44 24 PM" src="https://github.com/user-attachments/assets/b87d68ba-2c30-4c5f-b07f-5cf5f44acf84" />


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

## Run the HTTP server

This will host the static web client:

```bash
python -m http.server
```

Then, visit `http://localhost:8000` in your browser to start a session.
