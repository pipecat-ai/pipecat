# Daily Multi Translation

This example shows how to use Daily to stream multiple simultaneous translations using a single transport. Daily provides custom tracks and in this example we will simultaneously translate incoming audio in English to Spanish, French and German, each of them being sent to a custom track.

## Get started

```python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp env.example .env # and add your credentials

```

## Run the server

```bash
python server.py
```

Then, visit `http://localhost:7860/` in your browser. This will open a Daily Prebuilt room where you will speak in English (make sure you are not muted).

## Open client

Next, you need to open the client that will listen to the translations.

```bash
open index.html
```

Once the client is opened, copy the URL of the Daily room created above and join it. You should be able to select which translation you want to hear.

## Build and test the Docker image

```
docker build -t daily-multi-translation .
docker run --env-file .env -p 7860:7860 daily-multi-translation
```
