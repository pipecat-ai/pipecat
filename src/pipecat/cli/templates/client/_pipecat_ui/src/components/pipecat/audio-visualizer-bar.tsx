"use client";

import { usePipecatClientMediaTrack } from "@pipecat-ai/client-react";
import { memo, useEffect, useRef } from "react";

import {
  createVisualizerAnalyser,
  createVoiceBands,
  readBand,
  resolveVisualizerColor,
  type VisualizerState,
} from "@/lib/visualizer";

export type { VisualizerState };

type ParticipantType = Parameters<typeof usePipecatClientMediaTrack>[1];

export interface AudioVisualizerBarViewProps {
  /** Audio track to visualize. Renders resting dots when null. */
  track?: MediaStreamTrack | null;
  /** Canvas background. Any CSS color, or "transparent". */
  backgroundColor?: string;
  /** Bar color. Supports "currentColor" (default) and "--css-var" names. */
  barColor?: string;
  /** Number of frequency bands / bars. */
  barCount?: number;
  /** Bar thickness in px; also the resting dot diameter. */
  barWidth?: number;
  /** Gap between bars in px. */
  barGap?: number;
  /** Max bar height in px; the element sizes itself from this. */
  barMaxHeight?: number;
  /** Edge the bars grow from. */
  barOrigin?: "top" | "bottom" | "center";
  /** Bar end style; also shapes the resting dots. */
  barLineCap?: "round" | "square";
  /** Spectrum smoothing per frame (0–1, default 0.5); 1 disables smoothing. */
  barSpeed?: number;
  /** Disable the falling peak lines (on by default). */
  noPeaks?: boolean;
  /** Peak line color; same formats as barColor. */
  peakLineColor?: string;
  /** Peak fall speed. */
  peakLineSpeed?: number;
  /** Peak line thickness in px. */
  peakLineThickness?: number;
  /** Resting distance between a bar tip and its peak line, in px. */
  peakOffset?: number;
  /**
   * Seconds for a peak to fade out once it stops being pushed by its bar
   * (knocks restore full opacity). 0 keeps peaks visible forever.
   */
  peakFadeDuration?: number;
  /** Connecting takes precedence over thinking and ignores the audio track. */
  isConnecting?: boolean;
  /** Opacity-roll speed for the connecting override. */
  connectingSpeed?: number;
  /**
   * Override: the bot is working on a response — plays a traveling wave.
   * The track is ignored while set.
   */
  isThinking?: boolean;
  /** Wave speed multiplier for the thinking override. */
  thinkingSpeed?: number;
  /** Wave tightness; smaller values condense the wave. */
  thinkingWaveWidth?: number;
  /** Fraction of barMaxHeight the thinking wave rises to (0–1). */
  thinkingHeight?: number;
  /**
   * Opacity of the wave's lowest bars (0–1); crests render at 1 and the
   * rest fade toward this with the wave.
   */
  thinkingAlpha?: number;
  className?: string;
}

/** Props-driven voice spectrum; connecting and thinking override track-derived state. */
export const AudioVisualizerBarView = memo(function AudioVisualizerBarView({
  track = null,
  backgroundColor = "transparent",
  barColor = "currentColor",
  barCount = 5,
  barWidth = 30,
  barGap = 12,
  barMaxHeight = 120,
  barOrigin = "center",
  barLineCap = "round",
  barSpeed = 0.5,
  noPeaks = true,
  peakLineColor = "currentColor",
  peakLineSpeed = 0.2,
  peakLineThickness = 2,
  peakOffset = 0,
  peakFadeDuration = 0.8,
  isConnecting = false,
  connectingSpeed = 3,
  isThinking = false,
  thinkingSpeed = 5,
  thinkingWaveWidth = 1.5,
  thinkingHeight = 0.25,
  thinkingAlpha = 0.5,
  className,
}: AudioVisualizerBarViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const resolvedBarColorRef = useRef<string>("black");
  const resolvedPeakLineColorRef = useRef<string>("black");
  const analyserRef = useRef<AnalyserNode | null>(null);
  const frequencyDataRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
  // Displayed height/alpha per bar, persisted across renders so state
  // changes ease from wherever the bars currently are.
  const displayRef = useRef<{ h: number; a: number }[]>([]);

  // Resolve CSS-flavored colors to concrete values the canvas can use.
  // className is a dependency because it can change what currentColor
  // resolves to.
  useEffect(() => {
    resolvedBarColorRef.current = resolveVisualizerColor(
      barColor,
      canvasRef.current,
    );
    resolvedPeakLineColorRef.current = resolveVisualizerColor(
      peakLineColor,
      canvasRef.current,
    );
  }, [barColor, peakLineColor, className]);

  // Audio pipeline — only recreated when the track changes. The overrides
  // are synthetic, so they skip the pipeline entirely.
  useEffect(() => {
    if (!track || isConnecting || isThinking) {
      analyserRef.current = null;
      frequencyDataRef.current = null;
      return;
    }

    const { analyser, dispose } = createVisualizerAnalyser(track);
    analyserRef.current = analyser;
    frequencyDataRef.current = new Uint8Array(analyser.frequencyBinCount);

    return () => {
      analyserRef.current = null;
      frequencyDataRef.current = null;
      dispose();
    };
  }, [track, isConnecting, isThinking]);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const canvasWidth = barCount * barWidth + (barCount - 1) * barGap;
    // Peaks float beyond the bar tips; pad the canvas so they render
    // inside it. Bar math stays in content space via the translate.
    const peakClearance = noPeaks ? 0 : peakOffset + peakLineThickness;
    const contentHeight = barMaxHeight;
    const totalHeight = contentHeight + peakClearance * 2;
    const scaleFactor = 2;

    canvas.width = canvasWidth * scaleFactor;
    canvas.height = totalHeight * scaleFactor;
    canvas.style.width = `${canvasWidth}px`;
    canvas.style.height = `${totalHeight}px`;

    const canvasCtx = canvas.getContext("2d")!;
    canvasCtx.lineCap = barLineCap;
    canvasCtx.scale(scaleFactor, scaleFactor);
    canvasCtx.translate(0, peakClearance);

    // Mel-scale bands over the voice range (see visualizer.ts), plus the
    // peak line each bar may ride. Peak heights live in display space.
    const sampleRate = analyserRef.current?.context.sampleRate ?? 48000;
    const frequencyBinCount = analyserRef.current?.frequencyBinCount ?? 512;
    const bands = createVoiceBands(barCount, sampleRate, frequencyBinCount);
    const peaks = noPeaks ? null : bands.map(() => ({ height: 0, opacity: 0 }));

    // Displayed values persist across effect restarts (mode/track changes)
    // so transitions ease from the current picture; a barCount change
    // resets them.
    const display = displayRef.current;
    if (display.length !== barCount) {
      display.length = 0;
      for (let i = 0; i < barCount; i++) display.push({ h: 0, a: 1 });
    }

    const startX =
      (canvasWidth - (barCount * barWidth + (barCount - 1) * barGap)) / 2;
    // Line caps extend bars by half their width; doubles as dot radius.
    const capRadius = barWidth / 2;
    // Below this displayed height a bar renders as its resting dot.
    const DOT_THRESHOLD = 0.75;
    // Displayed values ease toward per-frame targets, so mode changes
    // blend (~120ms) instead of snapping. Live audio instead chases at
    // barSpeed — the configurable responsiveness of the speaking state.
    const EASE = 0.28;
    const chase = Math.min(1, Math.max(0.05, barSpeed));
    // Peak fall rate in px per frame, scaled so peakLineSpeed keeps the
    // same feel it had in the reference's 0–255 value space.
    const peakFall = peakLineSpeed * 5 * (barMaxHeight / 255);
    // peakFadeDuration is seconds-to-transparent at ~60fps; 0 disables
    // fading entirely.
    const peakFade = peakFadeDuration > 0 ? 1 / (peakFadeDuration * 60) : 0;

    /** Y of the resting dots for the current barOrigin. */
    function dotY(): number {
      switch (barOrigin) {
        case "top":
          return capRadius;
        case "bottom":
          return contentHeight - capRadius;
        case "center":
        default:
          return contentHeight / 2;
      }
    }

    /** Bar endpoints for a height, clamped to the content per barOrigin. */
    function barY(barHeight: number): { yTop: number; yBottom: number } {
      switch (barOrigin) {
        case "top": {
          const yTop = capRadius;
          const yBottom = Math.min(
            capRadius + barHeight,
            contentHeight - capRadius,
          );
          return { yTop, yBottom };
        }
        case "bottom": {
          const yBottom = contentHeight - capRadius;
          const yTop = Math.max(yBottom - barHeight, capRadius);
          return { yTop, yBottom };
        }
        case "center":
        default:
          return {
            yTop: Math.max(contentHeight / 2 - barHeight / 2, capRadius),
            yBottom: Math.min(
              contentHeight / 2 + barHeight / 2,
              contentHeight - capRadius,
            ),
          };
      }
    }

    /** The resting dot, shaped to match the bar line cap. */
    function drawDot(x: number, color: string) {
      const y = dotY();
      if (barLineCap === "square") {
        canvasCtx.fillStyle = color;
        canvasCtx.fillRect(x, y - capRadius, capRadius * 2, capRadius * 2);
        return;
      }
      canvasCtx.beginPath();
      canvasCtx.arc(x + capRadius, y, capRadius, 0, 2 * Math.PI);
      canvasCtx.fillStyle = color;
      canvasCtx.fill();
      canvasCtx.closePath();
    }

    let rafId: number;

    function drawSpectrum() {
      const analyser = analyserRef.current;
      const frequencyData = frequencyDataRef.current;
      const resolvedBarColor = resolvedBarColorRef.current;
      const time = performance.now() / 1000;

      canvasCtx.clearRect(0, -peakClearance, canvasWidth, totalHeight);
      canvasCtx.fillStyle = backgroundColor;
      canvasCtx.fillRect(0, -peakClearance, canvasWidth, totalHeight);

      let spectrum: Uint8Array<ArrayBuffer> | null = null;
      if (!isConnecting && !isThinking && analyser && frequencyData) {
        analyser.getByteFrequencyData(frequencyData);
        spectrum = frequencyData;
      }

      let settled = true;

      bands.forEach((band, i) => {
        let targetHeight = 0;
        let targetAlpha = 1;

        if (isConnecting) {
          // Roll opacity across the resting dots.
          targetAlpha =
            0.25 +
            0.75 *
              (0.5 +
                0.5 *
                  Math.sin(
                    time * connectingSpeed - i * ((Math.PI * 2) / barCount),
                  ));
        } else if (isThinking) {
          // Traveling wave, smoothstepped so crests swell and troughs
          // rest instead of tracking the sine linearly. Opacity follows
          // the wave: crests at 1, troughs at thinkingAlpha.
          const phaseOffset = (Math.PI * 2) / barCount / thinkingWaveWidth;
          const n = (Math.sin(time * thinkingSpeed - i * phaseOffset) + 1) / 2;
          const shaped = n * n * (3 - 2 * n);
          targetHeight = shaped * barMaxHeight * thinkingHeight;
          targetAlpha = thinkingAlpha + (1 - thinkingAlpha) * shaped;
        } else if (spectrum) {
          targetHeight = (readBand(spectrum, band) / 255) * barMaxHeight;
        }
        // Silent (no overrides, no analyser): targets stay 0 / 1.

        const d = display[i]!;
        d.h += (targetHeight - d.h) * (spectrum ? chase : EASE);
        d.a += (targetAlpha - d.a) * EASE;
        if (
          Math.abs(targetHeight - d.h) > 0.5 ||
          Math.abs(targetAlpha - d.a) > 0.02
        ) {
          settled = false;
        }

        const x = startX + i * (barWidth + barGap);

        canvasCtx.globalAlpha = d.a;
        if (d.h > DOT_THRESHOLD) {
          const { yTop, yBottom } = barY(d.h);
          canvasCtx.beginPath();
          canvasCtx.moveTo(x + capRadius, yTop);
          canvasCtx.lineTo(x + capRadius, yBottom);
          canvasCtx.lineWidth = barWidth;
          canvasCtx.strokeStyle = resolvedBarColor;
          canvasCtx.stroke();
        } else {
          drawDot(x, resolvedBarColor);
        }
        canvasCtx.globalAlpha = 1;

        const peak = peaks?.[i];
        if (spectrum && peak) {
          // The peak rides the drawn bar: knocked up instantly with it,
          // falling back linearly. Opacity fades whenever the peak isn't
          // being pushed — knocks restore it, so the line stays lit
          // through active audio and dissolves smoothly once it stops.
          if (d.h > peak.height) {
            peak.height = d.h;
            peak.opacity = 1;
          } else {
            peak.height = Math.max(peak.height - peakFall, d.h);
            peak.opacity = Math.max(0, peak.opacity - peakFade);
          }

          if (peak.opacity > 0) {
            // Visual tip of a bar peak.height tall — line caps extend it
            // by capRadius — plus the configured offset outward.
            const tip = barY(peak.height);
            const outward = capRadius + peakOffset;
            const peakY =
              barOrigin === "top" ? tip.yBottom + outward : tip.yTop - outward;

            const previousLineCap = canvasCtx.lineCap;
            const previousAlpha = canvasCtx.globalAlpha;
            canvasCtx.lineCap = "butt";
            canvasCtx.globalAlpha = peak.opacity;
            canvasCtx.beginPath();
            canvasCtx.moveTo(x, peakY);
            canvasCtx.lineTo(x + barWidth, peakY);
            canvasCtx.lineWidth = peakLineThickness;
            canvasCtx.strokeStyle = resolvedPeakLineColorRef.current;
            canvasCtx.stroke();
            canvasCtx.lineCap = previousLineCap;
            canvasCtx.globalAlpha = previousAlpha;
          }
        }
      });

      // Animated modes and unfinished transitions keep the loop alive; a
      // settled silent frame stops it (the effect restarts on changes).
      if (isConnecting || isThinking || spectrum || !settled) {
        rafId = requestAnimationFrame(drawSpectrum);
      }
    }

    rafId = requestAnimationFrame(drawSpectrum);

    return () => {
      cancelAnimationFrame(rafId);
    };
  }, [
    backgroundColor,
    // Colors are drawn from refs, but a settled silent frame stops the
    // loop — restart it when they change so it repaints. The resolution
    // effect above runs first (declaration order), so the refs are fresh.
    barColor,
    peakLineColor,
    barCount,
    barGap,
    barLineCap,
    barMaxHeight,
    barOrigin,
    barSpeed,
    barWidth,
    track,
    noPeaks,
    peakLineSpeed,
    peakLineThickness,
    peakOffset,
    peakFadeDuration,
    isConnecting,
    connectingSpeed,
    isThinking,
    thinkingSpeed,
    thinkingWaveWidth,
    thinkingHeight,
    thinkingAlpha,
  ]);

  return (
    <canvas
      ref={canvasRef}
      data-slot="audio-visualizer-bar"
      data-state={
        (isConnecting
          ? "connecting"
          : isThinking
            ? "thinking"
            : track
              ? "speaking"
              : "silent") satisfies VisualizerState
      }
      style={{ display: "block", width: "100%", height: "100%" }}
      className={className}
    />
  );
});

export interface AudioVisualizerBarProps extends Omit<
  AudioVisualizerBarViewProps,
  "track"
> {
  /** Which participant's audio to visualize. */
  participantType: ParticipantType;
}

/** Requires PipecatClientProvider; silence and speech follow the participant track. */
export function AudioVisualizerBar({
  participantType,
  ...props
}: AudioVisualizerBarProps) {
  const track = usePipecatClientMediaTrack("audio", participantType);
  return <AudioVisualizerBarView track={track} {...props} />;
}
