# dailyai SDK

This SDK can help you build applications that participate in WebRTC meetings and use various AI services to interact with other participants.

## Build/Install

_Note that you may need to set up a virtual environment before following the instructions below. For instance, you might need to run the following from the root of the repo:_

```
python3 -m venv env
source env/bin/activate
```

From the root of this repo, run the following:

```
pip install -r requirements.txt
python -m build
```

This builds the package. To use the package locally (eg to run sample files), run

```
pip install --editable .
```

If you want to use this package from another directory, you can run:

```
pip install path_to_this_repo
```

## Running the samples

Tou can run the simple sample like so:

```
python src/samples/theoretical-to-real/01-say-one-thing.py -u <url of your Daily meeting> -k <your Daily API Key>
```

Note that the sample uses Azure's TTS and LLM services. You'll need to set the following environment variables for the sample to work:

```
AZURE_SPEECH_SERVICE_KEY
AZURE_SPEECH_SERVICE_REGION
AZURE_CHATGPT_KEY
AZURE_CHATGPT_ENDPOINT
AZURE_CHATGPT_DEPLOYMENT_ID
```

If you have those environment variables stored in an .env file, you can quickly load them into your terminal's environment by running this:

```bash
export $(grep -v '^#' .env | xargs)
```
