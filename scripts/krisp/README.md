# Krisp Demo Tools

Demo tools for evaluating Krisp's Turn-Taking and Interrupt Prediction technologies.

## Setup

```bash
pip install soundfile numpy sounddevice pipecat-ai[krisp]
```

Create a `.env` file in this directory with model paths:

```
KRISP_VIVA_TURN_MODEL_PATH=C:\path\to\tt_model.kef
KRISP_VIVA_IP_MODEL_PATH=C:\path\to\ip_model.kef
KRISP_VIVA_FILTER_MODEL_PATH=C:\path\to\filter_model.kef
KRISP_VIVA_VAD_MODEL_PATH=C:\path\to\vad_model.kef
```

## Tools

### 1. Record Audio

```bash
python demo_audio_recorder.py                   # record to recording.wav
python demo_audio_recorder.py -o session.wav     # custom output name
python demo_audio_recorder.py -d 30              # stop after 30 seconds
python demo_audio_recorder.py --list-devices     # show input devices
```

### 2. Turn-Taking Demo

Compares turn analyzers (Krisp vs SmartTurn) on an audio file. Produces annotated WAVs with beeps at turn points, an HTML report with interactive timeline and playback, and a comparison table.

```bash
python demo_turn_taking.py input.wav
python demo_turn_taking.py input.wav --analyzer krisp --analyzer smart-turn-v3
python demo_turn_taking.py input.wav --vad krisp --viva-filter
```

Output goes to `./demo_output/` (override with `-o`).

### 3. Interrupt Prediction Demo

Demonstrates Krisp IP by comparing two modes: IP enabled (assistant stops on genuine interruption) vs IP disabled (assistant ignores interruptions).

```bash
python demo_interrupt_prediction.py input.wav
python demo_interrupt_prediction.py input.wav --bot-audio assistant.wav
python demo_interrupt_prediction.py input.wav --threshold 0.6
```

## Output

Each demo generates:
- **Annotated WAV files** -- listen to beeps/cuts at detection points
- **HTML report** -- open in browser for interactive timeline with audio playback
- **Terminal output** -- ASCII timeline and comparison tables
