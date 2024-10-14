FROM python:3.10-bullseye

RUN mkdir /app

COPY *.py /app/
COPY requirements.txt /app/
COPY .env /app/

WORKDIR /app

RUN pip3 install -r requirements.txt

EXPOSE 7860

CMD ["python3", "bot.py"]
