# Use an official Python runtime as a parent image
FROM python:3.12-bullseye

# Set the working directory in the container (repo root inside the image)
WORKDIR /vonage-speech-to-speech

# Install ffmpeg for pydub at runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy the example's requirements file into the container (for layer caching)
COPY examples/vonage-speech-to-speech/requirements.txt ./requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire repo so local src/pipecat/* is available
COPY . .

# Install the local pipecat package (so imports like pipecat.serializers.vonage work)
RUN pip install --no-cache-dir -e .

# Expose the desired port (WebSocket server)
EXPOSE 8005

# Run the application from the example directory
WORKDIR /vonage-speech-to-speech/examples/vonage-speech-to-speech
CMD ["python", "server.py"]
