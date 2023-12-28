# dailyai SDK

This SDK can help you build applications that participate in WebRTC meetings and use various AI services to interact with other participants.

## Build/Install

From the root of this repo, run the following:

```
pip install -r requirements.txt
python -m build
```

This builds the package. To use the package locally (eg to run sample files), run

```
pip install .
```

If you want to use this package from another directory, you can run:
```
pip install path_to_this_repo
```

## Running the samples

Tou can run the simple sample like so:

```
src/samples/simple-sample/simple-sample.py -u your_room_url -k your_daily_api_key
```

Note that the sample uses Azure's TTS and LLM services. You'll need to set the following environment variables for the sample to work:

```
AZURE_SPEECH_SERVICE_KEY
AZURE_SPEECH_SERVICE_REGION
AZURE_CHATGPT_KEY
AZURE_CHATGPT_ENDPOINT
AZURE_CHATGPT_DEPLOYMENT_ID
```
