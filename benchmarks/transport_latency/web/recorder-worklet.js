// Receive-side recorder for the browser measuring client.
//
// Accumulates the tapped bot-audio input into 20 ms chunks and stamps each
// chunk with the context-clock time at which its last sample was rendered —
// the same "arrival stamp of the chunk containing the onset" convention the
// Python client core uses. Chunks are batched to the main thread every ~0.5 s
// and on an explicit "flush" message.

const CHUNK_SAMPLES = 960; // 20 ms at 48 kHz
const BATCH_CHUNKS = 24;

class BenchRecorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Float32Array(CHUNK_SAMPLES);
    this._fill = 0;
    this._chunks = [];
    this._stamps = [];
    this.port.onmessage = (e) => {
      if (e.data === "flush") {
        this._post(true);
      }
    };
  }

  _post(final) {
    if (this._chunks.length || final) {
      const pcm = new Float32Array(this._chunks.length * CHUNK_SAMPLES);
      this._chunks.forEach((c, i) => pcm.set(c, i * CHUNK_SAMPLES));
      this.port.postMessage(
        { pcm, stamps: this._stamps.slice(), final: !!final },
        [pcm.buffer],
      );
      this._chunks = [];
      this._stamps = [];
    }
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch || ch.length === 0) return true;
    let i = 0;
    while (i < ch.length) {
      const n = Math.min(CHUNK_SAMPLES - this._fill, ch.length - i);
      this._buf.set(ch.subarray(i, i + n), this._fill);
      this._fill += n;
      i += n;
      if (this._fill === CHUNK_SAMPLES) {
        this._chunks.push(this._buf.slice(0));
        this._stamps.push((currentFrame + i) / sampleRate);
        this._fill = 0;
        if (this._chunks.length >= BATCH_CHUNKS) this._post(false);
      }
    }
    return true;
  }
}

registerProcessor("bench-recorder", BenchRecorder);
