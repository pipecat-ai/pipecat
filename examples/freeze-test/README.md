# Freeze Test Client

The purpose of this example is to create an environment for testing the bot and try to create freezing conditions.

### Approach 1: Server-Side Testing with `SimulateFreezeInput`

- Utilize only the bot `freeze_test_bot.py` with the `SimulateFreezeInput` processor. This input continuously injects frames, simulating user speech interruptions at random intervals.
- This approach excludes the use of input transport and speech-to-text (STT) functionalities.

### Approach 2: Server-Side with TypeScript Client

- Combine server-side operations with a TypeScript client.
- The client initially records a segment of audio, e.g., 5–10 seconds long. It can be anything.
- After that, it replays this recorded audio to the server at random intervals, mimicking user input interruptions.
- This helps testing interruptions in the pipeline as if real users were interacting with the bot.

## Setup

Follow these steps to set up and run the Freeze Test Client:

1. **Run the Bot Server**  
   - Set up and activate your virtual environment:
       ```bash
       python3 -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate
       ```

   - Install dependencies:
      ```bash
      pip install -r requirements.txt
      ```

   - Create your `.env` file and set your env vars:
      ```bash
      cp env.example .env
      ```
   
   - Run the server:
      ```bash
      python freeze_test_bot.py
      ```

2. **Navigate to the Client Directory**
   ```bash
   cd client
   ```

3. **Install Dependencies**
   ```bash
   npm install
   ```

4. **Run the Client Application**
   ```bash
   npm run dev
   ```

5. **Access the Client in Your Browser**  
   Visit [http://localhost:5173](http://localhost:5173) to interact with the Freeze Test Client.
