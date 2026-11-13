export type VisualizerState = "connecting" | "silent" | "speaking" | "thinking";

/** One mel-spaced analyser band: its FFT bin range plus tilt boost. */
export interface VisualizerBand {
  startBin: number;
  endBin: number;
  /** dB-space boost countering speech's high-frequency rolloff. */
  tiltBoost: number;
}

/** Call dispose to release the AudioContext; smoothing here only reduces FFT jitter. */
export function createVisualizerAnalyser(track: MediaStreamTrack): {
  analyser: AnalyserNode;
  dispose: () => void;
} {
  const audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(new MediaStream([track]));
  const analyser = audioContext.createAnalyser();
  analyser.fftSize = 1024;
  analyser.smoothingTimeConstant = 0.6;
  source.connect(analyser);
  return { analyser, dispose: () => void audioContext.close() };
}

/** Mel bands cover 200 Hz–8 kHz. A per-octave boost compensates speech rolloff so low bands do not dominate. */
export function createVoiceBands(
  bandCount: number,
  sampleRate: number,
  frequencyBinCount: number,
): VisualizerBand[] {
  const nyquist = sampleRate / 2;
  const MIN_FREQ = 200;
  const MAX_FREQ = 8000;
  const toMel = (hz: number) => 2595 * Math.log10(1 + hz / 700);
  const toHz = (mel: number) => 700 * (Math.pow(10, mel / 2595) - 1);
  const melMin = toMel(MIN_FREQ);
  const melStep = (toMel(MAX_FREQ) - melMin) / bandCount;

  const TILT_DB_PER_OCTAVE = 4;
  const BYTES_PER_DB = 255 / 70;
  const refCenter = Math.sqrt(toHz(melMin) * toHz(melMin + melStep));

  return Array.from({ length: bandCount }, (_, i) => {
    const startFreq = toHz(melMin + i * melStep);
    const endFreq = toHz(melMin + (i + 1) * melStep);
    const startBin = Math.round(
      (startFreq / nyquist) * (frequencyBinCount - 1),
    );
    // Every band covers at least one bin, so high band counts can't
    // round a band into permanent silence.
    const endBin = Math.max(
      Math.round((endFreq / nyquist) * (frequencyBinCount - 1)),
      startBin + 1,
    );
    const center = Math.sqrt(startFreq * endFreq);
    const tiltBoost =
      TILT_DB_PER_OCTAVE * BYTES_PER_DB * Math.log2(center / refCenter);
    return { startBin, endBin, tiltBoost };
  });
}

/**
 * Averages a band's FFT bins and applies its tilt boost, returning 0–255.
 * The boost fades in with signal strength so the noise floor near
 * minDecibels isn't amplified into standing bars.
 */
export function readBand(spectrum: Uint8Array, band: VisualizerBand): number {
  let sum = 0;
  for (let j = band.startBin; j < band.endBin; j++) sum += spectrum[j] ?? 0;
  const value = sum / (band.endBin - band.startBin);
  return Math.min(255, value + band.tiltBoost * Math.min(1, value / 32));
}

/**
 * Resolves a visualizer color prop to a concrete canvas color:
 * "currentColor" resolves from the given element, "--css-var" names from
 * the document root (where the kit's theme tokens live).
 */
export function resolveVisualizerColor(
  color: string,
  element: Element | null,
): string {
  if (!color) return "black";
  if (color === "currentColor") {
    return element ? getComputedStyle(element).color || "black" : "black";
  }
  if (color.startsWith("--")) {
    return (
      getComputedStyle(document.documentElement)
        .getPropertyValue(color)
        .trim() || "black"
    );
  }
  return color;
}
