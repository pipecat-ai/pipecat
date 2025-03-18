FROM python:3.10-bullseye

RUN mkdir /app
RUN mkdir /app/assets
RUN mkdir /app/utils
COPY requirements.txt /app/

WORKDIR /app
RUN pip3 install -r requirements.txt

COPY *.py /app/
COPY assets/* /app/assets/
COPY .env /app/.env

EXPOSE 7860

CMD ["python3", "server.py"]
