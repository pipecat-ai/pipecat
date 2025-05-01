# Daily Custom Tracks

This example shows how to send and receive Daily custom tracks. We will run a simple `daily-python` application to send an audio file with a custom track (named "pipecat") to a room. Then, the Pipecat bot will mirror that custom track into another custom track (named "pipecat-mirror") in the same room.

## Get started

```python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the bot

Start the bot by giving it a Daily room URL.

```bash
python bot.py -u ROOM_URL
```

The bot will wait for the first participant to join. Then, it will mirror a custom track named "pipecat" into a new custom track named "pipecat-mirror".

## Run the sender

Now, run the custom track sender. This is a simple `daily-python` application that opens and audio file and sends it as a custom track to the same Daily room.

```bash
python custom_track_sender.py -u ROOM_URL -i office-ambience-mono-16000.mp3
```

## Open client

Finally, open the client so you can hear both custom tracks.

```bash
open index.html
```

Once the client is opened, copy the URL of the Daily room and join it. You should be able to select which custom track you want to hear.
