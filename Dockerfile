# Base image with Python 3.11.5
FROM python:3.11.5

# Set working directory
WORKDIR /app
COPY requirements.txt /app
COPY *.py /app
COPY pyproject.toml /app

COPY src/ /app/src/

# Copy requirements file first (to use Docker's cache for dependency installation)
COPY requirements.txt /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libssl-dev \
    libasound2 \
    wget \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy remaining app files
COPY . /app

# OpenSSL Installation (for Azure TTS or any SSL-related requirements)
RUN wget -O - https://www.openssl.org/source/openssl-1.1.1w.tar.gz | tar zxf - && \
    cd openssl-1.1.1w && \
    ./config --prefix=/usr/local && \
    make -j $(nproc) && \
    make install_sw install_ssldirs && \
    ldconfig -v

ENV SSL_CERT_DIR=/etc/ssl/certs
ENV PYTHONUNBUFFERED=1

# Expose the FastAPI app port
EXPOSE 8000

# Command to run the FastAPI app using gunicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]