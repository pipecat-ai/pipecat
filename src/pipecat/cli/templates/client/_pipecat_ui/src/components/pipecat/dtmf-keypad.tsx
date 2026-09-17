"use client";

import { type DTMFButton } from "@pipecat-ai/client-js";
import {
  useDTMF,
  usePipecatClientTransportState,
} from "@pipecat-ai/client-react";
import { DeleteIcon, PhoneIcon } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

type ButtonProps = React.ComponentProps<typeof Button>;

/**
 * Ordered list of the twelve DTMF keys as they appear on a standard
 * telephone keypad, with the letters traditionally printed beneath each
 * digit. Rendered row by row into a 3-column grid.
 */
const DTMF_KEYS: { value: DTMFButton; sub?: string }[] = [
  { value: "1" },
  { value: "2", sub: "ABC" },
  { value: "3", sub: "DEF" },
  { value: "4", sub: "GHI" },
  { value: "5", sub: "JKL" },
  { value: "6", sub: "MNO" },
  { value: "7", sub: "PQRS" },
  { value: "8", sub: "TUV" },
  { value: "9", sub: "WXYZ" },
  { value: "*" },
  { value: "0", sub: "+" },
  { value: "#" },
];

const sanitizeDTMF = (value: string): string => value.replace(/[^0-9*#]/g, "");

/**
 * The (row, column) frequency pair defining each key's dual tone, per the
 * standard DTMF layout. Synthesized so no audio assets ship with the kit.
 */
const DTMF_FREQUENCIES: Record<DTMFButton, [number, number]> = {
  "1": [697, 1209],
  "2": [697, 1336],
  "3": [697, 1477],
  "4": [770, 1209],
  "5": [770, 1336],
  "6": [770, 1477],
  "7": [852, 1209],
  "8": [852, 1336],
  "9": [852, 1477],
  "*": [941, 1209],
  "0": [941, 1336],
  "#": [941, 1477],
};

const TONE_DURATION = 0.12;
/** Deliberately quiet: the tone plays over a live call. */
const DEFAULT_TONE_VOLUME = 0.15;
/** Attack/release ramp; without it the tone clicks. */
const TONE_RAMP = 0.01;

/**
 * Local press-feedback tone. Cosmetic sidetone only — the actual
 * signalling happens over the transport via sendTone.
 */
function useDTMFTone(enabled: boolean, volume: number) {
  const contextRef = useRef<AudioContext | null>(null);
  const peakGain = Math.min(Math.max(volume, 0), 1);

  useEffect(
    () => () => {
      void contextRef.current?.close();
      contextRef.current = null;
    },
    [],
  );

  return useCallback(
    (key: DTMFButton) => {
      if (
        !enabled ||
        peakGain <= 0 ||
        typeof window === "undefined" ||
        !window.AudioContext
      ) {
        return;
      }
      // Created lazily on first press so it's tied to a user gesture
      // (autoplay policy) and never constructed during SSR.
      const context = (contextRef.current ??= new window.AudioContext());
      if (context.state === "suspended") void context.resume();

      const now = context.currentTime;
      const gain = context.createGain();
      gain.connect(context.destination);
      gain.gain.setValueAtTime(0, now);
      gain.gain.linearRampToValueAtTime(peakGain, now + TONE_RAMP);
      gain.gain.setValueAtTime(peakGain, now + TONE_DURATION - TONE_RAMP);
      gain.gain.linearRampToValueAtTime(0, now + TONE_DURATION);

      const oscillators = DTMF_FREQUENCIES[key].map((frequency) => {
        const oscillator = context.createOscillator();
        oscillator.type = "sine";
        oscillator.frequency.setValueAtTime(frequency, now);
        oscillator.connect(gain);
        oscillator.start(now);
        oscillator.stop(now + TONE_DURATION);
        return oscillator;
      });

      const last = oscillators[oscillators.length - 1];
      if (last) {
        last.onended = () => {
          oscillators.forEach((oscillator) => oscillator.disconnect());
          gain.disconnect();
        };
      }
    },
    [enabled, peakGain],
  );
}

/**
 * - "buffered" (default): presses accumulate in an editable field and send
 *   as one sequence on submit — mistypes can be corrected before dialing.
 * - "immediate": each press sends right away — suits live IVR menus.
 */
export type DTMFKeypadMode = "immediate" | "buffered";

export interface DTMFKeypadViewProps {
  mode?: DTMFKeypadMode;
  variant?: ButtonProps["variant"];
  size?: ButtonProps["size"];
  disabled?: boolean;
  /** Hide the letters printed beneath each digit. */
  noSubLabels?: boolean;
  /** Disable the audible press-feedback tone. */
  noToneFeedback?: boolean;
  /** Peak tone volume, 0–1 (clamped). */
  toneVolume?: number;
  placeholder?: string;
  sendLabel?: string;
  className?: string;
  /** Called with each key as it is pressed (both modes). */
  onPress?: (button: DTMFButton) => void;
  /** Called with the full sequence on submit (buffered mode). */
  onSend?: (sequence: string) => void;
  /** Controlled buffer value (buffered mode). */
  value?: string;
  defaultValue?: string;
  onValueChange?: (value: string) => void;
}

/**
 * Telephone keypad, props-driven. Buffered mode adds an editable sequence
 * field, backspace, and a send button.
 */
export function DTMFKeypadView({
  mode = "buffered",
  variant = "secondary",
  size = "lg",
  disabled = false,
  noSubLabels = false,
  noToneFeedback = false,
  toneVolume = DEFAULT_TONE_VOLUME,
  placeholder = "Enter digits",
  sendLabel = "Send",
  className,
  onPress,
  onSend,
  value,
  defaultValue,
  onValueChange,
}: DTMFKeypadViewProps) {
  const buffered = mode === "buffered";

  const isControlled = value !== undefined;
  const [internalValue, setInternalValue] = useState(defaultValue ?? "");
  const buffer = isControlled ? (value ?? "") : internalValue;

  const setBuffer = useCallback(
    (next: string) => {
      if (!isControlled) setInternalValue(next);
      onValueChange?.(next);
    },
    [isControlled, onValueChange],
  );

  const playTone = useDTMFTone(!noToneFeedback, toneVolume);

  const handlePress = useCallback(
    (key: DTMFButton) => {
      playTone(key);
      onPress?.(key);
      if (buffered) setBuffer(buffer + key);
    },
    [buffered, buffer, onPress, setBuffer, playTone],
  );

  const handleSend = useCallback(() => {
    // Sanitize at send time too: controlled values bypass the input's
    // onChange sanitization.
    const sequence = sanitizeDTMF(buffer);
    if (!sequence) return;
    onSend?.(sequence);
    setBuffer("");
  }, [buffer, onSend, setBuffer]);

  const grid = (
    <div className="grid grid-cols-3 gap-2">
      {DTMF_KEYS.map(({ value: key, sub }) => (
        <Button
          key={key}
          type="button"
          variant={variant}
          size={size}
          aria-label={`DTMF ${key}`}
          disabled={disabled}
          onClick={() => handlePress(key)}
          className="flex h-auto flex-col items-center justify-center gap-0 py-1.5 tabular-nums"
        >
          <span className="text-base font-semibold">{key}</span>
          {!noSubLabels && (
            <span className="h-2.5 text-[0.6rem] leading-none tracking-widest opacity-60">
              {sub ?? ""}
            </span>
          )}
        </Button>
      ))}
    </div>
  );

  if (!buffered) {
    return (
      <div data-slot="dtmf-keypad" className={className}>
        {grid}
      </div>
    );
  }

  return (
    <div
      data-slot="dtmf-keypad"
      className={cn("flex flex-col gap-2", className)}
    >
      <div className="flex items-center gap-2">
        <Input
          value={buffer}
          onChange={(e) => setBuffer(sanitizeDTMF(e.target.value))}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              // Enter should dispatch the sequence, not submit a form.
              e.preventDefault();
              handleSend();
            }
          }}
          placeholder={placeholder}
          disabled={disabled}
          // The on-screen keypad is the intended input method on touch
          // devices; typing/paste on desktop still works.
          inputMode="none"
          aria-label="DTMF sequence"
          className="flex-1 tracking-widest tabular-nums"
        />
        <Button
          type="button"
          variant="secondary"
          size="icon"
          disabled={disabled || !buffer}
          aria-label="Delete last digit"
          onClick={() => setBuffer(buffer.slice(0, -1))}
        >
          <DeleteIcon />
        </Button>
      </div>
      {grid}
      <Button
        type="button"
        variant={variant}
        size={size}
        disabled={disabled || !buffer}
        onClick={handleSend}
        className="w-full"
      >
        <PhoneIcon />
        {sendLabel}
      </Button>
    </div>
  );
}

export interface DTMFKeypadProps extends Omit<DTMFKeypadViewProps, "onSend"> {
  /** Called after tone(s) are successfully sent. */
  onToneSent?: (sequence: string) => void;
  /**
   * Called if sending fails — sendTone throws when the transport isn't
   * ready or the bot doesn't support DTMF (RTVI < 2.0.0).
   */
  onError?: (error: unknown) => void;
}

/** Requires PipecatClientProvider. Buffered mode sends on submit; immediate mode sends each press. */
export function DTMFKeypad({
  mode = "buffered",
  disabled,
  onPress,
  onToneSent,
  onError,
  ...props
}: DTMFKeypadProps) {
  const { sendTone } = useDTMF();
  const transportState = usePipecatClientTransportState();
  const isConnected = transportState === "ready";

  const send = useCallback(
    (sequence: string) => {
      try {
        sendTone(sequence);
        onToneSent?.(sequence);
      } catch (error) {
        onError?.(error);
      }
    },
    [sendTone, onToneSent, onError],
  );

  return (
    <DTMFKeypadView
      mode={mode}
      disabled={disabled || !isConnected}
      onPress={(button) => {
        onPress?.(button);
        if (mode === "immediate") send(button);
      }}
      onSend={mode === "buffered" ? send : undefined}
      {...props}
    />
  );
}
