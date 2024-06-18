import os

import aiohttp
from utils.unit_conversion import convert_kelvin

async def fetch_weather_from_api(args):
    location = args.get("location")
    temp_format = args.get("format")

    api_url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"q={location}&appid={os.getenv('OPENWEATHERMAP_API_KEY')}"
    )

    async with aiohttp.ClientSession() as session:
        async with session.get(api_url) as resp:
            data = await resp.json()

    temperature_kelvin = data["main"]["temp"]

    temp = convert_kelvin(temp_format, temperature_kelvin)

    conditions = data["weather"][0]["description"]

    return {"conditions": conditions, "temperature": temp}
