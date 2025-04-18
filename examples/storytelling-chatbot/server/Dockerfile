FROM python:3.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ARG USE_PERSISTENT_DATA
ENV PYTHONUNBUFFERED=1
ENV NODE_MAJOR=20

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

# Install Node.js
RUN mkdir -p /etc/apt/keyrings 
RUN curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
RUN echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list > /dev/null
RUN apt-get update && apt-get install nodejs -y

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
COPY ./requirements.txt requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

# Copy everything else
COPY --chown=user ./server/ server/

# Copy client app and build
COPY --chown=user ./client/ client/
RUN cd client && npm install && npm run build

# Start the FastAPI server
CMD python3 server/bot_runner.py --port ${FAST_API_PORT}