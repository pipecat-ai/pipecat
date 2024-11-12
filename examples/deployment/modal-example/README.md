# Deploying Pipecat to Modal.com

Barebones deployment example for [modal.com](https://www.modal.com)

1. Install dependencies

```bash
python -m venv venv
source venv/bin/active # or OS equivalent
pip install -r requirements.txt
```

2. Setup .env

```bash
cp env.example .env
```

Alternatively, you can configure your Modal app to use [secrets](https://modal.com/docs/guide/secrets)

3. Test the app locally

```bash
modal serve app.py
```

4. Deploy to production

```bash
modal deploy app.py
```

## Configuration options

This app sets some sensible defaults for reducing cold starts, such as `minkeep_warm=1`, which will keep at least 1 warm instance ready for your bot function.

It has been configured to only allow a concurrency of 1 (`max_inputs=1`) as each user will require their own running function.