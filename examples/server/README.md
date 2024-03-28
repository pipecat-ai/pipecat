# Server Example

Use this server app to quickly host a bot on the web:

```
flask --app daily-bot-manager.py --debug run
```

It's currently configured to serve example apps defined in the APPS constant in the server file:

```
chatbot
patient-intake
storybot
translator
```

Once the server is started, you can create a bot instance by opening `http://127.0.0.1:5000/start/chatbot` in a browser, and the server will do the following:

- Create a new, randomly-named Daily room with `DAILY_API_KEY` from your .env file or environment
- Start an instance of `chatbot.py` and connect it to that room
- 301 redirect your browser to the room

### Options

The server supports several options, which can be set in the body of a POST request, or as params in the URL of a GET request.

- `room_url` (default: none): A room URL to join. If empty, the server will create a Daily room and return the URL in the response.
  room_properties (none): A JSON object (URL encoded if included as a GET parameter) for overriding default room creation properties, as described here: https://docs.daily.co/reference/rest-api/rooms/create-room This will be ignored if a room_url is provided.
- `token_properties` (none): A JSON object (URL encoded if included as a GET parameter) for overriding default token properties. By default, the server creates an owner token with an expiration time of one hour.
- `duration` (7200 seconds, or two hours): Use this property to set a time limit for the bot, as well as an expiration time for the room (if the server is creating one). This will not add an expiration time to an existing room. Expiration times in `token_properties` or `room_properties` will also take precedence over this value. You can set this property to `0` to disable timeouts, but this isn't recommended.
- `bot_args` (none): A string containing any additional command-line args to pass to the bot.
- `wait_for_bot` (true): Whether to wait for the bot to successfully join the room before returning a response from the server. If true, the server will start the bot script, then poll the room for up to 5 seconds to confirm the bot has joined the room. If it doesn't, the server will stop the bot and return a 500 response. If set to `false`, the server will start the bot, but immediately return a 200 response. This can be useful if the server is creating rooms for you, and you need the room URL to join the user to the room.
- `redirect` (true): Instead of returning a 200 for GET requests, the server will return a 301 redirect to the ROOM_URL. This is handy for testing by creating a bot with a GET request directly in the browser. POST requests will never return redirects. Set to `false` to get 200 responses with info in a JSON object even for GET requests.
