#!/bin/bash

# DTMF frequency map (low, high)
declare -A DTMF=(
  [1]="697 1209"
  [2]="697 1336"
  [3]="697 1477"
  [4]="770 1209"
  [5]="770 1336"
  [6]="770 1477"
  [7]="852 1209"
  [8]="852 1336"
  [9]="852 1477"
  ["star"]="941 1209"
  [0]="941 1336"
  ["pound"]="941 1477"
)

# Tone duration (seconds) + gap after
DURATION=0.3
GAP=0.2
SAMPLERATE=8000

for key in "${!DTMF[@]}"; do
  freqs=(${DTMF[$key]})
  low=${freqs[0]}
  high=${freqs[1]}
  echo "Generating DTMF tone for $key ($low Hz + $high Hz)"
  ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i "sine=frequency=$low:duration=$DURATION:sample_rate=$SAMPLERATE" \
    -f lavfi -i "sine=frequency=$high:duration=$DURATION:sample_rate=$SAMPLERATE" \
    -f lavfi -i "anullsrc=r=$SAMPLERATE:cl=mono:d=$GAP" \
    -filter_complex "[0][1]amix=2[a];[a][2]concat=n=2:v=0:a=1[out]" \
    -map "[out]" -c:a pcm_s16le -ar $SAMPLERATE "dtmf-${key}.wav"
done
