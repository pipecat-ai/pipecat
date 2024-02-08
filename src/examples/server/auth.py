import time
import urllib

from dotenv import load_dotenv
import requests
from flask import jsonify
import os

load_dotenv()


def get_meeting_token(room_name, daily_api_key, token_expiry):
    api_path = os.getenv('DAILY_API_PATH') or 'https://api.daily.co/v1'

    if not token_expiry:
        token_expiry = time.time() + 600
    res = requests.post(
        f'{api_path}/meeting-tokens',
        headers={
            'Authorization': f'Bearer {daily_api_key}'},
        json={
            'properties': {
                'room_name': room_name,
                'is_owner': True,
                'exp': token_expiry}})
    if res.status_code != 200:
        return jsonify({'error': 'Unable to create meeting token', 'detail': res.text}), 500
    meeting_token = res.json()['token']
    return meeting_token


def get_room_name(room_url):
    return urllib.parse.urlparse(room_url).path[1:]
