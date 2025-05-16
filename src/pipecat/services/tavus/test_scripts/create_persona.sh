#!/bin/bash

API_URL="https://tavusapi.com/v2/personas"

curl --request POST \
  --url "$API_URL" \
  --header "Content-Type: application/json" \
  --header "x-api-key: $TAVUS_API_KEY" \
  --data '{
    "persona_name": "Pipecat",
    "system_prompt": "",
    "pipeline_mode": "echo",
    "context": "",
    "layers": {
      "transport":{
        "input_settings":{
            "microphone": {
                "isEnabled": true
            }
         },
         "output_settings":{
         },
        "room_settings": {
          "enable_chat": true,
          "enable_knocking": false,
          "enable_prejoin_ui": true,
          "enable_network_ui": true,
          "enable_screenshare": true,
          "eject_at_room_exp": false,
          "eject_after_elapsed": 0,
          "exp": 32503680000,
          "nbf": 0,
          "start_video_off": false,
          "start_audio_off": false,
          "enable_people_ui": true,
          "enable_noise_cancellation_ui": true,
          "enable_recording": "cloud",
          "recordings_bucket": null,
          "enable_multiparty_adaptive_simulcast": true,
          "enable_adaptive_simulcast": true,
          "enable_live_captions_ui": false,
          "enable_transcription": false
        },
        "transport_type": "daily"
      }
    }
  }'