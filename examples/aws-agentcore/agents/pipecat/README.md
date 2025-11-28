# Need to update the Dockerfile after executing `agentcore configure`

Add this command to the Dockerfile:
- `RUN apt update && apt install -y libgl1 libglib2.0-0 && apt clean`