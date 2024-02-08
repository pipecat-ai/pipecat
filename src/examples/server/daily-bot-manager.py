import os
import requests
import subprocess
import time

from flask import Flask, jsonify, request, redirect
from flask_cors import CORS
from examples.server.auth import get_meeting_token

from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

print(f"I loaded an environment, and my FAL_KEY_ID is {os.getenv('FAL_KEY_ID')}")


def start_bot(bot_path, args=None):
    daily_api_key = os.getenv("DAILY_API_KEY")
    api_path = os.getenv("DAILY_API_PATH") or "https://api.daily.co/v1"

    timeout = int(os.getenv("DAILY_ROOM_TIMEOUT") or os.getenv("DAILY_BOT_MAX_DURATION") or 300)
    exp = time.time() + timeout
    res = requests.post(
        f"{api_path}/rooms",
        headers={"Authorization": f"Bearer {daily_api_key}"},
        json={
            "properties": {
                "exp": exp,
                "enable_chat": True,
                "enable_emoji_reactions": True,
                "eject_at_room_exp": True,
                "enable_prejoin_ui": False,
                "enable_recording": "cloud"
            }
        },
    )
    if res.status_code != 200:
        return (
            jsonify(
                {
                    "error": "Unable to create room",
                    "status_code": res.status_code,
                    "text": res.text,
                }
            ),
            500,
        )
    room_url = res.json()["url"]
    room_name = res.json()["name"]

    meeting_token = get_meeting_token(room_name, daily_api_key, exp)

    if args:
        extra_args = " ".join([f'-{x[0]} "{x[1]}"' for x in args])
    else:
        extra_args = ""

    proc = subprocess.Popen(
        [
            f"python {bot_path} -u {room_url} -t {meeting_token} -k {daily_api_key} {extra_args}"
        ],
        shell=True,
        bufsize=1,
    )

    # Don't return until the bot has joined the room, but wait for at most 2 seconds.
    attempts = 0
    while attempts < 20:
        time.sleep(0.1)
        attempts += 1
        res = requests.get(
            f"{api_path}/rooms/{room_name}/get-session-data",
            headers={"Authorization": f"Bearer {daily_api_key}"},
        )
        if res.status_code == 200:
            break
    print(f"Took {attempts} attempts to join room {room_name}")

    # Additional client config
    config = {}
    if os.getenv("CLIENT_VAD_TIMEOUT_SEC"):
        config['vad_timeout_sec'] = float(os.getenv("DAILY_CLIENT_VAD_TIMEOUT_SEC"))
    else:
        config['vad_timeout_sec'] = 1.5

    # return jsonify({"room_url": room_url, "token": meeting_token, "config": config}), 200
    return redirect(room_url, code=301)


@app.route("/spin-up-kitty", methods=["GET", "POST"])
def spin_up_kitty():
    return start_bot("./src/examples/foundational/10-wake-word.py")


@app.route("/healthz")
def health_check():
    return "ok", 200
