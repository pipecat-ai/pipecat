
import urllib.parse
import os
import time
import urllib
import requests

from dotenv import load_dotenv
load_dotenv()


daily_api_path = os.getenv("DAILY_API_URL") or "api.daily.co/v1"
daily_api_key = os.getenv("DAILY_API_KEY")


def create_room() -> tuple[str, str]:
    """
    Helper function to create a Daily room.
    # See: https://docs.daily.co/reference/rest-api/rooms

    Returns:
        tuple: A tuple containing the room URL and room name.

    Raises:
        Exception: If the request to create the room fails or if the response does not contain the room URL or room name.
    """
    room_props = {
        "exp": time.time() + 60 * 60,  # 1 hour
        "enable_chat": True,
        "enable_emoji_reactions": True,
        "eject_at_room_exp": True,
        "enable_prejoin_ui": False,  # Important for the bot to be able to join headlessly
    }
    res = requests.post(
        f"https://{daily_api_path}/rooms",
        headers={"Authorization": f"Bearer {daily_api_key}"},
        json={
            "properties": room_props
        },
    )
    if res.status_code != 200:
        raise Exception(f"Unable to create room: {res.text}")

    data = res.json()
    room_url: str = data.get("url")
    room_name: str = data.get("name")
    if room_url is None or room_name is None:
        raise Exception("Missing room URL or room name in response")

    return room_url, room_name


def get_name_from_url(room_url: str) -> str:
    """
    Extracts the name from a given room URL.

    Args:
        room_url (str): The URL of the room.

    Returns:
        str: The extracted name from the room URL.
    """
    return urllib.parse.urlparse(room_url).path[1:]


def get_token(room_url: str) -> str:
    """
    Retrieves a meeting token for the specified Daily room URL.
    # See: https://docs.daily.co/reference/rest-api/meeting-tokens

    Args:
        room_url (str): The URL of the Daily room.

    Returns:
        str: The meeting token.

    Raises:
        Exception: If no room URL is specified or if no Daily API key is specified.
        Exception: If there is an error creating the meeting token.
    """
    if not room_url:
        raise Exception(
            "No Daily room specified. You must specify a Daily room in order a token to be generated.")

    if not daily_api_key:
        raise Exception(
            "No Daily API key specified. set DAILY_API_KEY in your environment to specify a Daily API key, available from https://dashboard.daily.co/developers.")

    expiration: float = time.time() + 60 * 60
    room_name = get_name_from_url(room_url)

    res: requests.Response = requests.post(
        f"https://{daily_api_path}/meeting-tokens",
        headers={
            "Authorization": f"Bearer {daily_api_key}"},
        json={
            "properties": {
                "room_name": room_name,
                "is_owner": True,  # Owner tokens required for transcription
                "exp": expiration}},
    )

    if res.status_code != 200:
        raise Exception(
            f"Failed to create meeting token: {res.status_code} {res.text}")

    token: str = res.json()["token"]

    return token
