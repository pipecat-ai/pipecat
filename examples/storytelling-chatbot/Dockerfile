FROM pipecatai/cloud-base:latest

COPY ./requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./src/bot.py bot.py
COPY ./src/processors.py processors.py
COPY ./src/prompts.py prompts.py
COPY ./src/assets assets
COPY ./src/utils utils