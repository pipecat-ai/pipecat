FROM python:3.11-bullseye

ARG DEBIAN_FRONTEND=noninteractive
ARG USE_PERSISTENT_DATA
ENV PYTHONUNBUFFERED=1
# Expose FastAPI port
ENV FAST_API_PORT=7860
EXPOSE 7860

# Install system dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    git \
    ffmpeg \
    google-perftools \
    ca-certificates curl gnupg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=$HOME/app \
    PYTHONUNBUFFERED=1

# Switch to the "user" user
USER user

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Install Python dependencies
COPY *.py .
COPY ./requirements.txt requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

# Start the FastAPI server
CMD python3 bot_runner.py --host "0.0.0.0" --port ${FAST_API_PORT}