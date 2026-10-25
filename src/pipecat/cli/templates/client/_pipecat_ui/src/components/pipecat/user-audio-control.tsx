"use client";

import type { DeviceErrorReason } from "@pipecat-ai/client-js";
import {
  type OptionalMediaDeviceInfo,
  PipecatClientMicToggle,
  useMediaState,
  usePipecatClient,
  usePipecatClientMediaTrack,
} from "@pipecat-ai/client-react";
import {
  ChevronDownIcon,
  Loader2Icon,
  MicIcon,
  MicOffIcon,
  Volume2Icon,
} from "lucide-react";
import { Fragment, useCallback, useEffect, useRef, useState } from "react";

import {
  formatDeviceLabel,
  usePipecatDevices,
} from "@/components/pipecat/device-select";
import {
  AudioVisualizerBarView,
  type AudioVisualizerBarViewProps,
} from "@/components/pipecat/audio-visualizer-bar";
import { Button } from "@/components/ui/button";
import { ButtonGroup } from "@/components/ui/button-group";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Kbd } from "@/components/ui/kbd";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";

type ButtonProps = React.ComponentProps<typeof Button>;

// Muted styling per variant, applied while muted (data-state also flips
// for consumer styling): solid variants fill with the inactive color,
// outline keeps its border + tint (nova's destructive-button idiom),
// secondary tints without a border, ghost only recolors content. Unknown
// (consumer-added) variants get the ghost treatment.
const INACTIVE_VARIANT_CLASSES: Record<string, string> = {
  default: "bg-inactive text-inactive-foreground hover:bg-inactive/80",
  outline:
    "border-inactive/30 bg-inactive/10 text-inactive hover:bg-inactive/20 hover:text-inactive dark:bg-inactive/20",
  secondary:
    "bg-inactive/10 text-inactive hover:bg-inactive/20 dark:bg-inactive/20",
  ghost: "text-inactive hover:text-inactive",
};

function inactiveClasses(variant: ButtonProps["variant"]): string {
  return (
    INACTIVE_VARIANT_CLASSES[variant ?? "default"] ??
    INACTIVE_VARIANT_CLASSES.ghost!
  );
}

// The visualizer draws in pixels: known Button sizes map to bar heights,
// anything else (including custom cva sizes) falls back to 18. Override
// per-instance via visualizerProps.barMaxHeight.
const VISUALIZER_HEIGHT: Partial<
  Record<NonNullable<ButtonProps["size"]>, number>
> = {
  xs: 12,
  sm: 14,
  lg: 22,
};

// Nova's icon-* sizes render a square button; the toggle responds by going
// icon-only (no label, hotkey hint, or visualizer — and no min-width).
function isIconSize(size: ButtonProps["size"]): boolean {
  return typeof size === "string" && size.startsWith("icon");
}

// Nova's icon sizes mirror the text sizes by suffix (sm ↔ icon-sm), so the
// picker trigger derives its size from the control's mechanically.
function iconSize(size: ButtonProps["size"]): ButtonProps["size"] {
  if (isIconSize(size)) return size;
  return !size || size === "default"
    ? "icon"
    : (`icon-${size}` as ButtonProps["size"]);
}

/** Human label for a KeyboardEvent.code ("KeyM" → "M", "Backquote" → "`"). */
function keyLabel(code: string): string {
  if (code.startsWith("Key")) return code.slice(3);
  if (code.startsWith("Digit")) return code.slice(5);
  return code === "Backquote" ? "`" : code;
}

/**
 * Renders the hotkey hint, replacing "[key]" with the key chip. Wrapped in
 * its own span so it's a single flex item inside the button — the button's
 * gap separates it from the label, same as the icon and visualizer.
 */
function renderKeyHint(template: string, code: string): React.ReactNode {
  return (
    <span data-slot="ptt-hint" className="inline-flex items-center gap-1">
      {template.split("[key]").map((part, i) => (
        <Fragment key={i}>
          {i > 0 && <Kbd>{keyLabel(code)}</Kbd>}
          {part}
        </Fragment>
      ))}
    </span>
  );
}

// Keys with native meaning on focused controls (activation, menu/list
// navigation). When the hotkey is one of these, push-to-talk yields to
// focused interactive elements; any other key can be claimed page-wide.
const ACTIVATION_KEYS = new Set([
  "Space",
  "Enter",
  "NumpadEnter",
  "ArrowUp",
  "ArrowDown",
  "ArrowLeft",
  "ArrowRight",
  "Home",
  "End",
  "PageUp",
  "PageDown",
  "Escape",
  "Tab",
]);

export interface UserAudioControlViewProps {
  /** Whether the microphone is currently enabled. */
  isMicEnabled?: boolean;
  /** Called when the user clicks the mute toggle. */
  onToggleMic?: () => void;
  /** Disables the control and shows a spinner (e.g. while devices initialize). */
  isLoading?: boolean;
  /** Content shown next to the spinner while loading. Spinner only by default. */
  loadingText?: React.ReactNode;
  /**
   * When set, the control renders disabled with this message instead of the
   * toggle (mic blocked, in use, not found, …). Takes precedence over the
   * interactive state; isLoading takes precedence over both.
   */
  unavailableText?: React.ReactNode;
  /** Microphones offered in the picker. */
  mics?: MediaDeviceInfo[];
  /** Currently selected microphone, matched by deviceId. */
  selectedMic?: OptionalMediaDeviceInfo;
  /** Called with the deviceId when the user picks a microphone. */
  onMicChange?: (deviceId: string) => void;
  /** Speakers offered in the picker. */
  speakers?: MediaDeviceInfo[];
  /** Currently selected speaker, matched by deviceId. */
  selectedSpeaker?: OptionalMediaDeviceInfo;
  /** Called with the deviceId when the user picks a speaker. */
  onSpeakerChange?: (deviceId: string) => void;
  /** Hide the device picker dropdown entirely. */
  noDevicePicker?: boolean;
  /** Hide the microphone section of the picker. */
  noMicrophones?: boolean;
  /** Hide the speaker section of the picker. */
  noSpeakers?: boolean;
  /** Label shown while the mic is live (toggle mode only). */
  activeText?: React.ReactNode;
  /** Label shown while the mic is muted (toggle mode only). */
  inactiveText?: React.ReactNode;
  /** Hide the mic icon. */
  noIcon?: boolean;
  /** Hide the audio visualizer. */
  noVisualizer?: boolean;
  /** Local audio track rendered by the visualizer while the mic is live. */
  audioTrack?: MediaStreamTrack | null;
  /** Overrides merged onto the embedded visualizer (wins over defaults). */
  visualizerProps?: Partial<AudioVisualizerBarViewProps>;
  /** Interaction mode: click to toggle, or hold to talk (default "toggle"). */
  mode?: "toggle" | "push-to-talk";
  /**
   * Called when the user flips the push-to-talk switch in the device
   * dropdown. The switch only renders when this is provided.
   */
  onModeChange?: (mode: "toggle" | "push-to-talk") => void;
  /** Hide the push-to-talk switch in the device dropdown. */
  noModeSwitch?: boolean;
  /**
   * Sets the mic to an absolute state. Push-to-talk prefers this over
   * onToggleMic (its fallback), which can race across quick presses. The
   * connected control wires it to client.enableMic().
   */
  onMicEnabledChange?: (enabled: boolean) => void;
  /** Global push-to-talk KeyboardEvent.code (default "Backquote"). Interactive elements retain their activation keys; null disables the hotkey. */
  pttKey?: string | null;
  /**
   * Text around the hotkey chip; "[key]" marks where the chip renders
   * (default "press [key] to talk"). "[key]" alone shows just the chip;
   * an empty string hides the hint.
   */
  pttKeyLabel?: string;
  /**
   * Quiet period in ms between releasing push-to-talk and re-muting.
   * Pressing again inside the window cancels the re-mute, so rapid taps
   * can't thrash the WebRTC track (default 200).
   */
  debounceMs?: number;
  /** Pulse the push-to-talk outline (default true); respects reduced motion. */
  pttActiveOutline?: boolean;
  /**
   * Called when push-to-talk engages the mic (e.g. to play an on-air
   * chime). Fires once per hold, not per press source.
   */
  onPttOn?: () => void;
  /**
   * Called when push-to-talk re-mutes after the debounced release — a
   * re-press inside the window keeps the hold alive without firing this.
   */
  onPttOff?: () => void;
  /** Button variant for the toggle and picker trigger. */
  variant?: ButtonProps["variant"];
  /**
   * Button size for the toggle; the visualizer and picker trigger scale
   * with it. Icon sizes ("icon", "icon-sm", …) render the toggle
   * icon-only: label, hotkey hint, and visualizer are hidden.
   */
  size?: ButtonProps["size"];
  /** Classes for the outer ButtonGroup. */
  className?: string;
  /** Extra content rendered inside the toggle, between label and visualizer. */
  children?: React.ReactNode;
}

/** Toggle or hold-to-talk microphone control. Loading takes precedence over unavailability. */
export function UserAudioControlView({
  isMicEnabled = false,
  onToggleMic,
  isLoading = false,
  loadingText,
  unavailableText,
  mics,
  selectedMic,
  onMicChange,
  speakers,
  selectedSpeaker,
  onSpeakerChange,
  noDevicePicker = false,
  noMicrophones = false,
  noSpeakers = false,
  activeText,
  inactiveText,
  noIcon = false,
  noVisualizer = false,
  audioTrack = null,
  visualizerProps,
  mode = "toggle",
  onModeChange,
  noModeSwitch = false,
  onMicEnabledChange,
  pttKey = "Backquote",
  pttKeyLabel = "press [key] to talk",
  debounceMs = 200,
  pttActiveOutline = true,
  onPttOn,
  onPttOff,
  variant = "outline",
  size = "default",
  className,
  children,
}: UserAudioControlViewProps) {
  const hasMics = !noMicrophones && (mics?.length ?? 0) > 0;
  const hasSpeakers = !noSpeakers && (speakers?.length ?? 0) > 0;
  const noDropdown = noDevicePicker || (noMicrophones && noSpeakers);
  const showModeSwitch = !noModeSwitch && !!onModeChange;

  const isPtt = mode === "push-to-talk";
  const pttActive = isPtt && !isLoading && !unavailableText;
  const iconOnly = isIconSize(size);

  // Push-to-talk shows the key hint instead of the toggle state labels;
  // icon sizes show neither.
  const stateText =
    isPtt || iconOnly ? undefined : isMicEnabled ? activeText : inactiveText;

  // Push-to-talk plumbing. The mic setter and chime callbacks read fresh
  // props through a ref so the hold handlers stay stable; holds are tracked
  // per source (global hotkey, pointer, focused-button keys) so overlapping
  // presses behave.
  const liveRef = useRef({
    isMicEnabled,
    onMicEnabledChange,
    onToggleMic,
    onPttOn,
    onPttOff,
  });
  liveRef.current = {
    isMicEnabled,
    onMicEnabledChange,
    onToggleMic,
    onPttOn,
    onPttOff,
  };
  const setMic = useCallback((enabled: boolean) => {
    const current = liveRef.current;
    if (enabled === current.isMicEnabled) return;
    if (current.onMicEnabledChange) current.onMicEnabledChange(enabled);
    else current.onToggleMic?.();
  }, []);

  const holds = useRef({ key: false, pointer: false, focus: false });
  const engaged = useRef(false);
  const releaseTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);

  const updateHold = useCallback(
    (source: "key" | "pointer" | "focus", held: boolean) => {
      const h = holds.current;
      if (h[source] === held) return;
      h[source] = held;
      if (releaseTimer.current) {
        clearTimeout(releaseTimer.current);
        releaseTimer.current = null;
      }
      if (h.key || h.pointer || h.focus) {
        setMic(true);
        // One engage edge per hold: re-presses inside the debounce window
        // and second press sources don't re-fire the chime.
        if (!engaged.current) {
          engaged.current = true;
          liveRef.current.onPttOn?.();
        }
      } else {
        // Debounced release: pressing again inside the window cancels the
        // re-mute instead of bouncing the WebRTC track.
        releaseTimer.current = setTimeout(() => {
          releaseTimer.current = null;
          const s = holds.current;
          if (s.key || s.pointer || s.focus) return;
          setMic(false);
          if (engaged.current) {
            engaged.current = false;
            liveRef.current.onPttOff?.();
          }
        }, debounceMs);
      }
    },
    [setMic, debounceMs],
  );

  const releaseAllHolds = useCallback(() => {
    updateHold("key", false);
    updateHold("pointer", false);
    updateHold("focus", false);
  }, [updateHold]);

  // Push-to-talk invariant: the mic is muted unless held. Entering the
  // mode — via the dropdown switch, a controlled mode change, mounting
  // with defaultMode, or devices becoming ready while in it — closes the
  // channel until the first press.
  useEffect(() => {
    if (!pttActive) return;
    const h = holds.current;
    if (!(h.key || h.pointer || h.focus)) setMic(false);
  }, [pttActive, setMic]);

  // Global hotkey. Text fields always win; keys with native control
  // semantics (Space, Enter, arrows) additionally yield to focused
  // interactive elements — the talk button handles its own held keys.
  // Window blur releases every hold so cmd-tabbing mid-press can't leave
  // the mic open.
  useEffect(() => {
    if (!pttActive) return;
    const editable = (t: EventTarget | null) => {
      const el = t as HTMLElement | null;
      return (
        !!el && !!el.closest?.("input, textarea, select, [contenteditable]")
      );
    };
    const handlesOwnKeys = (t: EventTarget | null) => {
      const el = t as HTMLElement | null;
      return (
        !!el &&
        !!el.closest?.(
          "button, a[href], " +
            '[role="menu"], [role="menuitem"], [role="menuitemcheckbox"], ' +
            '[role="listbox"], [role="option"], [role="combobox"], [role="dialog"]',
        )
      );
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (!pttKey || e.code !== pttKey) return;
      if (e.repeat || e.defaultPrevented || editable(e.target)) return;
      if (ACTIVATION_KEYS.has(pttKey) && handlesOwnKeys(e.target)) return;
      e.preventDefault();
      updateHold("key", true);
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (pttKey && e.code === pttKey) updateHold("key", false);
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", releaseAllHolds);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", releaseAllHolds);
      releaseAllHolds();
    };
  }, [pttActive, pttKey, updateHold, releaseAllHolds]);

  useEffect(
    () => () => {
      if (releaseTimer.current) clearTimeout(releaseTimer.current);
      if (engaged.current) {
        engaged.current = false;
        const current = liveRef.current;
        // Cancel even if the SDK has not reported the preceding enable yet.
        if (current.onMicEnabledChange) current.onMicEnabledChange(false);
        else if (current.isMicEnabled) current.onToggleMic?.();
        current.onPttOff?.();
      }
    },
    [],
  );

  if (isLoading || unavailableText) {
    return (
      <ButtonGroup data-slot="user-audio-control" className={className}>
        <Button
          variant={variant}
          size={size}
          disabled
          aria-busy={isLoading || undefined}
          data-state={isLoading ? "loading" : "unavailable"}
          aria-label={
            !iconOnly
              ? undefined
              : isLoading
                ? "Microphone loading"
                : typeof unavailableText === "string"
                  ? unavailableText
                  : "Microphone unavailable"
          }
          className={cn("grow", !iconOnly && "min-w-32")}
        >
          {isLoading ? (
            <>
              <Loader2Icon className="animate-spin" />
              {!iconOnly && loadingText}
            </>
          ) : (
            <>
              {!noIcon && <MicOffIcon />}
              {!iconOnly && unavailableText}
            </>
          )}
        </Button>
        {/* Keep the (disabled) picker trigger while loading so the split
            button doesn't collapse; it stays hidden for unavailable, which
            is a terminal state until permissions change. */}
        {isLoading && !noDropdown && (
          <Button
            variant={variant}
            size={iconSize(size)}
            disabled
            aria-label="Audio devices"
          >
            <ChevronDownIcon />
          </Button>
        )}
      </ButtonGroup>
    );
  }

  return (
    <ButtonGroup data-slot="user-audio-control" className={className}>
      <Button
        ref={toggleRef}
        variant={variant}
        size={size}
        data-state={isMicEnabled ? "active" : "inactive"}
        data-mode={mode}
        aria-pressed={isMicEnabled}
        aria-label={
          stateText
            ? undefined
            : isPtt
              ? "Hold to talk"
              : isMicEnabled
                ? "Mute microphone"
                : "Unmute microphone"
        }
        aria-keyshortcuts={isPtt && pttKey ? keyLabel(pttKey) : undefined}
        className={cn(
          // Fills a group given a width; content-sized groups are unaffected.
          "grow",
          !isMicEnabled && inactiveClasses(variant),
          !iconOnly && "min-w-32",
          isPtt && "touch-none",
          // outline-solid: the button base resets outline-none, which
          // outline-<width> alone would inherit. z-10 lifts the outline
          // above the adjacent picker trigger in the ButtonGroup.
          isPtt &&
            pttActiveOutline &&
            isMicEnabled &&
            "outline-active motion-safe:animate-ptt-pulse z-10 outline-3 outline-offset-2 outline-solid",
        )}
        {...(isPtt
          ? {
              onPointerDown: (e: React.PointerEvent<HTMLButtonElement>) => {
                if (e.button === 0) updateHold("pointer", true);
              },
              onPointerUp: () => updateHold("pointer", false),
              onPointerLeave: () => updateHold("pointer", false),
              onPointerCancel: () => updateHold("pointer", false),
              // Holding Space/Enter on the focused button talks too;
              // preventDefault suppresses the native click-toggle.
              onKeyDown: (e: React.KeyboardEvent<HTMLButtonElement>) => {
                if ((e.key === " " || e.key === "Enter") && !e.repeat) {
                  e.preventDefault();
                  updateHold("focus", true);
                }
              },
              onKeyUp: (e: React.KeyboardEvent<HTMLButtonElement>) => {
                if (e.key === " " || e.key === "Enter")
                  updateHold("focus", false);
              },
              // Tabbing away mid-hold would swallow the keyup.
              onBlur: () => updateHold("focus", false),
            }
          : { onClick: onToggleMic })}
      >
        {!noIcon && (isMicEnabled ? <MicIcon /> : <MicOffIcon />)}
        {stateText}
        {children}
        {isPtt &&
          !iconOnly &&
          pttKey &&
          pttKeyLabel &&
          renderKeyHint(pttKeyLabel, pttKey)}
        {!noVisualizer && !iconOnly && (
          <AudioVisualizerBarView
            track={isMicEnabled ? audioTrack : null}
            // On the solid inactive fill (default variant) the bars use the
            // foreground token — background-colored bars would disappear.
            barColor={
              isMicEnabled
                ? "--active-background"
                : variant === "default"
                  ? "--inactive-foreground"
                  : "--inactive-background"
            }
            barGap={2}
            barWidth={3}
            barMaxHeight={VISUALIZER_HEIGHT[size ?? "default"] ?? 18}
            barOrigin="center"
            className="w-auto"
            {...visualizerProps}
          />
        )}
      </Button>
      {!noDropdown && (
        <DropdownMenu>
          <DropdownMenuTrigger
            render={
              <Button
                variant={variant}
                size={iconSize(size)}
                aria-label="Audio devices"
              >
                <ChevronDownIcon />
              </Button>
            }
          />
          <DropdownMenuContent align="end" className="min-w-56">
            {/* Base UI requires GroupLabel to live inside a Group. */}
            {hasMics && (
              <DropdownMenuGroup>
                <DropdownMenuLabel className="text-muted-foreground flex items-center gap-1.5 text-xs">
                  <MicIcon className="size-3" />
                  Microphones
                </DropdownMenuLabel>
                {mics!.map((device, index) => (
                  <DropdownMenuCheckboxItem
                    key={`mic-${device.deviceId || index}`}
                    checked={selectedMic?.deviceId === device.deviceId}
                    onCheckedChange={(checked) => {
                      if (checked) onMicChange?.(device.deviceId);
                    }}
                  >
                    {formatDeviceLabel(device, "Microphone")}
                  </DropdownMenuCheckboxItem>
                ))}
              </DropdownMenuGroup>
            )}
            {hasMics && hasSpeakers && <DropdownMenuSeparator />}
            {hasSpeakers && (
              <DropdownMenuGroup>
                <DropdownMenuLabel className="text-muted-foreground flex items-center gap-1.5 text-xs">
                  <Volume2Icon className="size-3" />
                  Speakers
                </DropdownMenuLabel>
                {speakers!.map((device, index) => (
                  <DropdownMenuCheckboxItem
                    key={`speaker-${device.deviceId || index}`}
                    checked={selectedSpeaker?.deviceId === device.deviceId}
                    onCheckedChange={(checked) => {
                      if (checked) onSpeakerChange?.(device.deviceId);
                    }}
                  >
                    {formatDeviceLabel(device, "Speaker")}
                  </DropdownMenuCheckboxItem>
                ))}
              </DropdownMenuGroup>
            )}
            {!hasMics && !hasSpeakers && (
              <DropdownMenuGroup>
                <DropdownMenuLabel className="text-muted-foreground font-normal">
                  No devices found
                </DropdownMenuLabel>
              </DropdownMenuGroup>
            )}
            {showModeSwitch && (
              <>
                <DropdownMenuSeparator />
                <DropdownMenuGroup>
                  {/* Enabling push-to-talk closes the menu and moves focus
                      to the talk button — otherwise focus returns to the
                      dropdown trigger, where Space reopens the menu instead
                      of talking. Disabling keeps the menu open. */}
                  <DropdownMenuItem
                    closeOnClick={!isPtt}
                    onClick={() => {
                      const next = isPtt ? "toggle" : "push-to-talk";
                      onModeChange?.(next);
                      if (next === "push-to-talk") {
                        requestAnimationFrame(() => toggleRef.current?.focus());
                      }
                    }}
                  >
                    Push to talk
                    {/* Inert visual — the menu item handles the toggle. */}
                    <Switch
                      checked={isPtt}
                      tabIndex={-1}
                      className="pointer-events-none ml-auto"
                    />
                  </DropdownMenuItem>
                </DropdownMenuGroup>
              </>
            )}
          </DropdownMenuContent>
        </DropdownMenu>
      )}
    </ButtonGroup>
  );
}

/** Short, user-facing message for a mic device error. */
function micErrorText(reason: DeviceErrorReason): string {
  switch (reason) {
    case "blocked":
      return "Microphone blocked";
    case "already-in-use":
      return "Microphone in use";
    case "not-found":
      return "No microphone";
    case "not-supported":
      return "Audio not supported";
    case "unknown":
    default:
      return "Microphone unavailable";
  }
}

export interface UserAudioControlProps extends Omit<
  UserAudioControlViewProps,
  | "isMicEnabled"
  | "onToggleMic"
  | "onMicEnabledChange"
  | "mics"
  | "selectedMic"
  | "onMicChange"
  | "speakers"
  | "selectedSpeaker"
  | "onSpeakerChange"
  | "audioTrack"
  | "isLoading"
  | "unavailableText"
> {
  /** Replaces the automatically derived device-error message. */
  unavailableText?: React.ReactNode;
  /** Initial mode when `mode` is left uncontrolled (default "toggle"). */
  defaultMode?: "toggle" | "push-to-talk";
}

/** Requires PipecatClientProvider. Manages device state, selection, and toggle or push-to-talk interaction. */
export function UserAudioControl({
  unavailableText,
  mode: controlledMode,
  defaultMode = "toggle",
  onModeChange,
  ...props
}: UserAudioControlProps) {
  const client = usePipecatClient();
  const {
    devices: mics,
    selectedDevice: selectedMic,
    updateDevice: onMicChange,
  } = usePipecatDevices("audioinput");
  const {
    devices: speakers,
    selectedDevice: selectedSpeaker,
    updateDevice: onSpeakerChange,
  } = usePipecatDevices("audiooutput");
  const audioTrack = usePipecatClientMediaTrack("audio", "local");

  const [uncontrolledMode, setUncontrolledMode] = useState(defaultMode);
  const mode = controlledMode ?? uncontrolledMode;
  const handleModeChange = useCallback(
    (next: "toggle" | "push-to-talk") => {
      setUncontrolledMode(next);
      onModeChange?.(next);
    },
    [onModeChange],
  );

  // "uninitialized" is deliberately not treated as loading: it also covers
  // post-init setups where the transport didn't acquire the mic. In those
  // cases nothing is in flight, so the control just renders muted.
  const { mic } = useMediaState();
  const isLoading = mic.state === "initializing";
  const derivedUnavailable =
    mic.state === "error" ? micErrorText(mic.reason) : undefined;

  return (
    <PipecatClientMicToggle>
      {({ isMicEnabled, onClick }) => (
        <UserAudioControlView
          isMicEnabled={isMicEnabled}
          onToggleMic={onClick}
          onMicEnabledChange={(enabled) => client?.enableMic(enabled)}
          mode={mode}
          onModeChange={handleModeChange}
          isLoading={isLoading}
          unavailableText={unavailableText ?? derivedUnavailable}
          mics={mics}
          selectedMic={selectedMic}
          onMicChange={onMicChange}
          speakers={speakers}
          selectedSpeaker={selectedSpeaker}
          onSpeakerChange={onSpeakerChange}
          audioTrack={audioTrack}
          {...props}
        />
      )}
    </PipecatClientMicToggle>
  );
}
