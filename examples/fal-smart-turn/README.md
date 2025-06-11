# Smart Turn Detection Demo

This demo showcases Pipecat's Smart Turn Detection feature - an advanced conversational turn detection system that uses machine learning to identify when a speaker has finished their turn in a conversation. Unlike basic Voice Activity Detection (VAD) which only detects speech vs. silence, Smart Turn detects natural conversational cues like intonation patterns, pacing, and linguistic signals.

This demo uses the [pipecat-ai/smart-turn](https://huggingface.co/pipecat-ai/smart-turn) model - an open-source, community-driven conversational turn detection model designed to provide more natural turn-taking in voice interactions. The model is being hosted on Fal's infrastructure for GPU acceleration, offering inference times between 50-60ms.

In the client UI, you can see the transcription messages along with the smart-turn model's prediction results in real-time.

## Try the demo

Try the hosted version of the demo here: https://pcc-smart-turn.vercel.app/.

## Run the demo locally

### Run the Server

1. Set up and activate your virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create your .env file and set your env vars:

   ```bash
   cp env.example .env
   ```

   Keys to provide:

   - GOOGLE_API_KEY
   - CARTESIA_API_KEY
   - DEEPGRAM_API_KEY
   - DAILY_API_KEY
   - FAL_SMART_TURN_API_KEY

4. Run the server:

   ```bash
   LOCAL_RUN=1 python server.py
   ```

### Run the client

1. Open a new terminal and navigate to the client directory:

   ```bash
   cd client
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Create your .env.local file:

   ```bash
   cp env.local.example .env.local
   ```

   > Note: No keys need to be modified. `NEXT_PUBLIC_API_BASE_URL` is already configured for local use.

4. Start the development server:

   ```bash
   npm run dev
   ```

5. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Deploy the app

### Deploy the server to Pipecat Cloud

1. Navigate to server

   ```bash
   cd server
   ```

2. You should already have a .env set up from running locally. If not, do that now.

3. Update your build and deploy scripts.

   - In build.sh, set `DOCKER_USERNAME` and `AGENT_NAME`.
   - In pcc-deploy.toml, set `image`, which specifies where your Docker image is stored.

4. Build your Docker image by running the build script:

   ```bash
   ./build.sh
   ```

   > Note: This builds, tags and pushes your docker image and assumes Docker Hub is the container registry.

5. Make sure you have the Pipecat Cloud CLI installed:

   ```bash
   pip install pipecatcloud
   ```

6. Login via the Pipecat Cloud CLI:

   ```bash
   pcc auth login
   ```

   > Note: If you don't have an account, sign up at https://pipecat.daily.co.

7. Add a secrets set:

   ```bash
   pcc secrets set pcc-smart-turn-secrets --file .env
   ```

8. Deploy your agent:

   ```bash
   pcc deploy
   ```

   > Note: This uses your pcc-deploy.toml settings. Modify as needed.

### Deploy the client to Vercel

This project uses TypeScript, React, and Next.js, making it a perfect fit for [Vercel](https://vercel.com/).

- In your client directory, install Vercel's CLI tool: `npm install -g vercel`
- Verify it's installed using `vercel --version`
- Log in your Vercel account using `vercel login`
- Deploy your client to Vercel using `vercel`

Follow the vercel prompts to deploy your project.

### Test your deployed app

Now with the client and server deployed, you can join the call using your Vercel URL.

See the debug information for the Smart Turn data. It prints a log line for each smart-turn inference:

```
Smart Turn: COMPLETE, Probability: 95.3%, Model inference: 65.23ms, Server processing: 82.09ms, End-to-end: 245.43ms
```
