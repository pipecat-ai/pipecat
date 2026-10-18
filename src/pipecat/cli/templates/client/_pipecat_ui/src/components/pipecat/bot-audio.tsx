"use client";

import { RTVIEvent } from "@pipecat-ai/client-js";
import {
  usePipecatClientMediaTrack,
  usePipecatClientTransportState,
  useRTVIClientEvent,
} from "@pipecat-ai/client-react";
import {
  Volume1Icon,
  Volume2Icon,
  VolumeIcon,
  VolumeXIcon,
} from "lucide-react";
import { useCallback, useEffect, useRef } from "react";
import { create } from "zustand";

import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";

type ButtonProps = React.ComponentProps<typeof Button>;

// Nova's icon-* sizes render a square button; the trigger responds by
// hiding its label (the label still names the button for screen readers).
function isIconSize(size: ButtonProps["size"]): boolean {
  return typeof size === "string" && size.startsWith("icon");
}

function clampVolume(v: number): number {
  if (Number.isNaN(v)) return 0;
  return Math.min(1, Math.max(0, v));
}

interface BotAudioState {
  /** Current bot audio output volume, 0.0 – 1.0. */
  volume: number;
  /** Set the bot audio output volume. Values are clamped to [0, 1]. */
  setVolume: (volume: number) => void;
}

/**
 * Module-level store for bot audio output volume. Drives the <audio>
 * element rendered by BotAudioOutput; read/written by the volume controls,
 * so any number of control instances stay in sync without prop drilling.
 */
export const useBotAudio = create<BotAudioState>()((set) => ({
  volume: 1,
  setVolume: (volume) => set({ volume: clampVolume(volume) }),
}));

/** Mount once inside PipecatClientProvider, in place of PipecatClientAudio, to use shared volume controls. */
export function BotAudioOutput() {
  const audioRef = useRef<HTMLAudioElement>(null);
  const botAudioTrack = usePipecatClientMediaTrack("audio", "bot");
  const volume = useBotAudio((s) => s.volume);

  // Attach the bot's audio track, de-duping on track id so we don't tear
  // down an already-playing stream.
  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    if (!botAudioTrack) {
      el.srcObject = null;
      return;
    }
    const existing = el.srcObject as MediaStream | null;
    if (existing) {
      const oldTrack = existing.getAudioTracks()[0];
      if (oldTrack && oldTrack.id === botAudioTrack.id) return;
    }
    el.srcObject = new MediaStream([botAudioTrack]);
  }, [botAudioTrack]);

  useEffect(() => {
    const el = audioRef.current;
    return () => {
      if (el) el.srcObject = null;
    };
  }, []);

  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    el.volume = volume;
  }, [volume]);

  // setSinkId can reject (unsupported deviceId, permission denied) —
  // swallow and log so speaker-routing failures stay non-fatal.
  useRTVIClientEvent(
    RTVIEvent.SpeakerUpdated,
    useCallback((speaker: MediaDeviceInfo) => {
      const el = audioRef.current;
      if (!el || typeof el.setSinkId !== "function") return;
      el.setSinkId(speaker.deviceId).catch((err: unknown) => {
        console.warn("BotAudioOutput: setSinkId failed", err);
      });
    }, []),
  );

  return <audio ref={audioRef} autoPlay />;
}

type SliderProps = React.ComponentProps<typeof Slider>;

// onVolumeChange is also omitted from the passthrough: React's DOM typings
// declare it (the native volumechange media event), which would clash with
// our domain callback.
export interface BotVolumeSliderViewProps extends Omit<
  SliderProps,
  "value" | "defaultValue" | "onValueChange" | "onVolumeChange" | "min" | "max"
> {
  /** Current volume, 0.0 – 1.0. */
  volume?: number;
  /** Called with the new volume (0.0 – 1.0). */
  onVolumeChange?: (volume: number) => void;
  /** Hide the text label. */
  noLabel?: boolean;
  /** Hide the percent readout. */
  noPercent?: boolean;
  /** Hide the mute toggle button. */
  noMuteButton?: boolean;
  /** Label text; doubles as the slider's accessible name. */
  label?: string;
  /** Classes for the root element, not the slider itself. */
  className?: string;
}

/** Props-driven volume slider. Muting preserves the last audible volume for restoration. */
export function BotVolumeSliderView({
  volume = 1,
  onVolumeChange,
  noLabel = false,
  noPercent = false,
  noMuteButton = false,
  label = "Bot volume",
  orientation = "horizontal",
  step = 0.01,
  className,
  ...sliderProps
}: BotVolumeSliderViewProps) {
  const pct = Math.round(clampVolume(volume) * 100);
  const muted = volume <= 0;
  const vertical = orientation === "vertical";
  const accessibleLabel = label.trim() || "Bot volume";
  const MuteIcon = iconForVolume(volume);

  // Remember the last audible volume so the mute button can restore it.
  const lastVolume = useRef(volume > 0 ? volume : 1);
  useEffect(() => {
    if (volume > 0) lastVolume.current = volume;
  }, [volume]);

  return (
    <div
      data-slot="bot-volume-slider"
      className={cn(
        "flex flex-col gap-2",
        vertical && "items-center",
        className,
      )}
    >
      {!noLabel && (
        <span className="text-muted-foreground text-sm">{label}</span>
      )}
      {/* Same DOM both ways: column-reverse puts the percent on top and
          the mute button at the bottom of a vertical strip. */}
      <div
        className={cn(
          "flex items-center gap-2",
          vertical && "flex-col-reverse",
        )}
      >
        {!noMuteButton && (
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label={muted ? "Unmute bot" : "Mute bot"}
            aria-pressed={muted}
            onClick={() => onVolumeChange?.(muted ? lastVolume.current : 0)}
          >
            <MuteIcon />
          </Button>
        )}
        <Slider
          min={0}
          max={1}
          step={step}
          value={volume}
          onValueChange={(value) => {
            const next = Array.isArray(value) ? value[0] : value;
            if (typeof next === "number") onVolumeChange?.(clampVolume(next));
          }}
          orientation={orientation}
          aria-label={accessibleLabel}
          className={cn(!vertical && "flex-1")}
          {...sliderProps}
        />
        {!noPercent && (
          <span
            className={cn(
              "text-muted-foreground text-xs tabular-nums",
              !vertical && "w-8 text-right",
            )}
          >
            {pct}%
          </span>
        )}
      </div>
    </div>
  );
}

/**
 * Bot volume slider wired to the useBotAudio store. Requires
 * BotAudioOutput to be mounted for volume changes to be audible.
 */
export function BotVolumeSlider(
  props: Omit<BotVolumeSliderViewProps, "volume" | "onVolumeChange">,
) {
  const { volume, setVolume } = useBotAudio();
  return (
    <BotVolumeSliderView
      volume={volume}
      onVolumeChange={setVolume}
      {...props}
    />
  );
}

function iconForVolume(volume: number) {
  if (volume <= 0) return VolumeXIcon;
  if (volume < 0.34) return VolumeIcon;
  if (volume < 0.67) return Volume1Icon;
  return Volume2Icon;
}

export interface BotAudioControlViewProps {
  /** Current volume, 0.0 – 1.0. */
  volume?: number;
  /** Called with the new volume (0.0 – 1.0). */
  onVolumeChange?: (volume: number) => void;
  /** Disables the trigger button. */
  disabled?: boolean;
  /** Hide the volume icon. */
  noIcon?: boolean;
  /** Visible label next to the icon. */
  label?: string;
  /** Button variant for the trigger. */
  variant?: ButtonProps["variant"];
  /**
   * Button size for the trigger. Icon sizes ("icon", "icon-sm", …) render
   * it icon-only: the label is hidden but still read to screen readers.
   */
  size?: ButtonProps["size"];
  /** Popover alignment relative to the trigger. */
  align?: React.ComponentProps<typeof PopoverContent>["align"];
  /** Overrides merged onto the popover's volume slider (wins over defaults). */
  sliderProps?: Partial<BotVolumeSliderViewProps>;
  /** Classes for the trigger button. */
  className?: string;
  /** Extra content rendered inside the trigger, after the label. */
  children?: React.ReactNode;
}

/**
 * Bot volume control: a button that opens a popover with the mute toggle,
 * slider, and percent readout (no label). The trigger's icon tracks the
 * current volume level.
 */
export function BotAudioControlView({
  volume = 1,
  onVolumeChange,
  disabled = false,
  noIcon = false,
  label,
  variant = "outline",
  size = "default",
  align = "end",
  sliderProps,
  className,
  children,
}: BotAudioControlViewProps) {
  const Icon = iconForVolume(volume);
  const iconOnly = isIconSize(size);

  return (
    <Popover>
      <PopoverTrigger
        render={
          <Button
            data-slot="bot-audio-control"
            variant={variant}
            size={size}
            disabled={disabled}
            aria-label={label?.trim() || "Bot volume"}
            className={className}
          >
            {!noIcon && <Icon />}
            {!iconOnly && label}
            {children}
          </Button>
        }
      />
      <PopoverContent
        align={align}
        className={sliderProps?.orientation === "vertical" ? "w-auto" : "w-56"}
      >
        <BotVolumeSliderView
          noLabel
          volume={volume}
          onVolumeChange={onVolumeChange}
          {...sliderProps}
        />
      </PopoverContent>
    </Popover>
  );
}

/**
 * Bot volume control wired to the useBotAudio store and transport state.
 * Disabled while disconnected. Requires BotAudioOutput to be mounted.
 * Must be rendered inside a PipecatClientProvider.
 */
export function BotAudioControl(
  props: Omit<
    BotAudioControlViewProps,
    "volume" | "onVolumeChange" | "disabled"
  >,
) {
  const { volume, setVolume } = useBotAudio();
  const transportState = usePipecatClientTransportState();
  const disabled =
    transportState === "disconnected" || transportState === "initializing";

  return (
    <BotAudioControlView
      volume={volume}
      onVolumeChange={setVolume}
      disabled={disabled}
      {...props}
    />
  );
}
