# Server Example

This is an example server based on [Santa Cat](https://santacat.ai). You can run the server with this command:

```
flask --app daily-bot-manager.py --debug run
```

Once the server is started, you can load `http://127.0.0.1:5000/spin-up-kitty` in a browser, and the server will do the following:

- Create a new, randomly-named Daily room with `DAILY_API_KEY` from your .env file or environment
- Start the `10-wake-word.py` example and connect it to that room
- 301 redirect your browser to the room
