FROM dailyco/pipecat-base:latest
RUN apt-get update && apt-get install ffmpeg -y

COPY ./pipecat pipecat

COPY ./requirements.txt requirements.txt

COPY ./assets assets

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./bot.py bot.py
