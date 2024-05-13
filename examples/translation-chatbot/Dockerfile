FROM python:3.10-bullseye

RUN mkdir /app
RUN mkdir /app/utils
COPY *.py /app/
COPY requirements.txt /app/
copy utils/* /app/utils/

WORKDIR /app
RUN pip3 install -r requirements.txt

EXPOSE 7860

CMD ["python3", "server.py"]