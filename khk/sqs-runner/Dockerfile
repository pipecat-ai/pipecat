FROM python:3.10-bullseye

RUN apt-get update
RUN apt-get install -y python3-dev portaudio19-dev

RUN git clone https://github.com/pipecat-ai/pipecat.git
WORKDIR /pipecat
RUN git fetch
RUN git checkout khk/tuesday-sqs
RUN python3 -m venv venv
RUN . venv/bin/activate
RUN pip install -r dev-requirements.txt -r linux-py3.10-requirements.txt
RUN pip install boto3
RUN python -m build
RUN pip install --editable .

EXPOSE 7860

WORKDIR /pipecat/khk/sqs-runner
CMD ["python", "sqs-runner.py"]

