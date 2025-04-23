# Smart Turn Demo

## Run the demo

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

3. Run the server (locally):

```bash
LOCAL=1 python server.py
```

### Run the client

1. Install dependencies:

   ```bash
   npm install
   ```

2. Created .env.local:

   ```bash
   cp env.example .env.local
   ```

3. Set up env vars as needed:

- Run locally:
  ```bash
  NEXT_PUBLIC_API_BASE_URL=http://localhost:7860
  ```
- Deployed:
  ```bash
  NEXT_PUBLIC_API_BASE_URL=/api
  PIPECAT_CLOUD_API_KEY=
  AGENT_NAME=
  ```

4. Start the development server:

```bash
npm run dev
```

5. Open [http://localhost:3000](http://localhost:3000) in your browser
