# Multi-Transport Chatbot for Pipecat and Pipecat Cloud

This project demonstrates a bot architecture that allows you to use different transports with the same bot, depending on how you run the botfile. This can be really useful for starting with one transport for early development and then transitioning to a different transport in production.

Here's how to use this bot with each of the supported transports.

## Step 1: Local development with SmallWebRTCTransport

To get started, let's run the bot with SmallWebRTCTransport, which makes a direct peer-to-peer WebRTC connection between your browser and the bot. 

```bash
# Start with the standard venv setup:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Rename the env example and add your keys:
mv example.env .env

# Now run the included webserver:
python server.py
```

Open a browser pointed at `http://localhost:7860` and click the **Connect** button to talk to the bot.

`server.py` helps set up the WebRTC connection, and then calls the `local_webrtc` function in bot.py with this line of code:

```python
        background_tasks.add_task(local_webrtc, pipecat_connection)
```

In `bot.py`, you can see that the `local_webrtc` function creates a `SmallWebRTCTransport` instance and passes it to the `main()` function.

## Step 2: Local development with Daily

After step 1, you can run the same bot using the Daily transport. Add a `DAILY_API_KEY` to your .env file. If you have a Daily account already, you can get your API key from  https://dashboard.daily.co/developers. If you have a Pipecat Cloud account, you have a Daily API key available at https://pipecat.daily.co/<your-org-slug>/settings/daily.

Run the bot using a different entrypoint:

```bash
LOCAL_RUN=1 python bot.py
```

This uses the `local_daily()` function in `bot.py`, which creates a `DailyTransport`.

### Step 3: Deploy to Pipecat Cloud

This repo already includes a Dockerfile you can use to build an image that works with Pipecat Cloud. You can do it in two steps:

```bash
./build.sh
pcc deploy

# Then start a session with your bot
pcc agent start multi-transport-chatbot --use-daily
```

This will give you a URL you can open in your browser to talk to the bot using Daily Prebuilt.

Behind the scenes, Pipecat Cloud loads your botfile and calls its `bot()` function. Since you used the `--use-daily` option, the `args` argument is a `DailySessionArguments` instance that includes the Daily room URL and token, so the bot uses a `DailyTransport`.

## Step 4: Use a Twilio phone number and websocket

Follow the [Pipecat Cloud Twilio docs](https://docs.pipecat.daily.co/pipecat-in-production/twilio-mediastreams) to configure a TwiML Bin that points one of your phone numbers to Pipecat Cloud. When you dial that number, Pipecat Cloud will start a session with your bot that includes a `WebsocketArguments` object, so the `bot()` function will start your bot with a `FastAPIWebsocketTransport`.
