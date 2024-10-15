# setup
FROM python:3.11.5

WORKDIR /app
COPY requirements.txt /app
COPY *.py /app
COPY pyproject.toml /app

COPY src/ /app/src/
COPY examples/ /app/examples/

WORKDIR /app
RUN ls --recursive /app/
RUN pip3 install --upgrade -r requirements.txt
RUN python -m build .
RUN pip3 install .
RUN pip3 install gunicorn
# If running on Ubuntu, Azure TTS requires some extra config
# https://learn.microsoft.com/en-us/azure/ai-services/speech-service/quickstarts/setup-platform?pivots=programming-language-python&tabs=linux%2Cubuntu%2Cdotnetcli%2Cdotnet%2Cjre%2Cmaven%2Cnodejs%2Cmac%2Cpypi

RUN wget -O - https://www.openssl.org/source/openssl-1.1.1w.tar.gz | tar zxf -
WORKDIR openssl-1.1.1w
RUN ./config --prefix=/usr/local
RUN make -j $(nproc)
RUN make install_sw install_ssldirs
RUN ldconfig -v
ENV SSL_CERT_DIR=/etc/ssl/certs

#ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
RUN apt clean
RUN apt-get update
RUN apt-get -y install build-essential libssl-dev ca-certificates libasound2 wget

ENV PYTHONUNBUFFERED=1

WORKDIR /app

EXPOSE 8000
# run
CMD ["gunicorn", "--workers=2", "--log-level", "debug", "--chdir", "examples/server", "--capture-output", "daily-bot-manager:app", "--bind=0.0.0.0:8000"]
