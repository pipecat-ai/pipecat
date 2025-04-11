FROM dailyco/pipecat-base:latest
RUN apt-get update && apt-get install ffmpeg -y

COPY ./requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./bot.py bot.py
