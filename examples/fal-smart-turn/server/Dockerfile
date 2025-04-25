FROM dailyco/pipecat-base:latest

COPY ./requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./assets assets
COPY ./bot.py bot.py
