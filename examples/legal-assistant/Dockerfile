FROM python:3.10-bullseye

WORKDIR /app

COPY . /app/

RUN pip install -r requirements.txt

EXPOSE 7860

CMD ["python3", "server.py"]