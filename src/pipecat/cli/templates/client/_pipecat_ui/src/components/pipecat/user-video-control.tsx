"use client";

import type { DeviceErrorReason } from "@pipecat-ai/client-js";
import {
  type OptionalMediaDeviceInfo,
  PipecatClientCamToggle,
  PipecatClientVideo,
  useMediaState,
} from "@pipecat-ai/client-react";
import {
  ChevronDownIcon,
  Loader2Icon,
  VideoIcon,
  VideoOffIcon,
} from "lucide-react";

import {
  DeviceDropdownContent,
  DeviceDropdownTrigger,
  DeviceDropdownView,
  usePipecatDevices,
} from "@/components/pipecat/device-select";
import { Button } from "@/components/ui/button";
import { ButtonGroup } from "@/components/ui/button-group";
import { cn } from "@/lib/utils";

type ButtonProps = React.ComponentProps<typeof Button>;

// Camera on keeps the plain button surface; camera off styles per variant
// while off (data-state also flips for consumer styling), matching the
// user audio control: solid variants fill with the inactive color, outline
// keeps its border + tint (nova's destructive-button idiom), secondary
// tints without a border, ghost only recolors content. Unknown
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

// Nova's icon-* sizes render a square button; the toggle responds by going
// icon-only (no label, no min-width).
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

export interface UserVideoControlViewProps {
  /** Whether the camera is currently enabled. */
  isCamEnabled?: boolean;
  /** Called when the user clicks the camera toggle. */
  onToggleCam?: () => void;
  /** Disables the control and shows a spinner (e.g. while devices initialize). */
  isLoading?: boolean;
  /** Content shown next to the spinner while loading. Spinner only by default. */
  loadingText?: React.ReactNode;
  /**
   * When set, the control renders disabled with this message instead of the
   * toggle (camera blocked, in use, not found, …). Takes precedence over the
   * interactive state; isLoading takes precedence over both.
   */
  unavailableText?: React.ReactNode;
  /** Cameras offered in the picker. */
  cams?: MediaDeviceInfo[];
  /** Currently selected camera, matched by deviceId. */
  selectedCam?: OptionalMediaDeviceInfo;
  /** Called with the deviceId when the user picks a camera. */
  onCamChange?: (deviceId: string) => void;
  /** Hide the camera picker dropdown. */
  noDevicePicker?: boolean;
  /** Render only the toggle button, without the video preview tile. */
  noVideo?: boolean;
  /** Video element rendered in the tile while the camera is on. */
  video?: React.ReactNode;
  /** Label shown while the camera is on. */
  activeText?: React.ReactNode;
  /** Label shown while the camera is off. */
  inactiveText?: React.ReactNode;
  /** Hide the camera icon. */
  noIcon?: boolean;
  /** Button variant for the toggle and picker trigger. */
  variant?: ButtonProps["variant"];
  /**
   * Button size for the toggle; the picker trigger scales with it. Icon
   * sizes ("icon", "icon-sm", …) render the toggle icon-only: the label
   * is hidden.
   */
  size?: ButtonProps["size"];
  /** Classes for the tile, or for the ButtonGroup when noVideo is set. */
  className?: string;
  /** Extra content rendered inside the toggle, after the label. */
  children?: React.ReactNode;
}

/** Camera toggle and device picker with an optional preview. Loading takes precedence over unavailability. */
export function UserVideoControlView({
  isCamEnabled = false,
  onToggleCam,
  isLoading = false,
  loadingText,
  unavailableText,
  cams,
  selectedCam,
  onCamChange,
  noDevicePicker = false,
  noVideo = false,
  video,
  activeText,
  inactiveText,
  noIcon = false,
  variant = "outline",
  size = "default",
  className,
  children,
}: UserVideoControlViewProps) {
  const iconOnly = isIconSize(size);
  const stateText = iconOnly
    ? undefined
    : isCamEnabled
      ? activeText
      : inactiveText;

  let toggle: React.ReactNode;
  if (isLoading || unavailableText) {
    toggle = (
      <ButtonGroup
        data-slot="user-video-control"
        className={noVideo ? className : undefined}
      >
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
                ? "Camera loading"
                : typeof unavailableText === "string"
                  ? unavailableText
                  : "Camera unavailable"
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
              {!noIcon && <VideoOffIcon />}
              {!iconOnly && unavailableText}
            </>
          )}
        </Button>
        {/* Keep the (disabled) picker trigger while loading so the split
            button doesn't collapse; it stays hidden for unavailable, which
            is a terminal state until permissions change. */}
        {isLoading && !noDevicePicker && (
          <Button
            variant={variant}
            size={iconSize(size)}
            disabled
            aria-label="Camera devices"
          >
            <ChevronDownIcon />
          </Button>
        )}
      </ButtonGroup>
    );
  } else {
    toggle = (
      <ButtonGroup
        data-slot="user-video-control"
        className={noVideo ? className : undefined}
      >
        <Button
          variant={variant}
          size={size}
          data-state={isCamEnabled ? "active" : "inactive"}
          aria-pressed={isCamEnabled}
          aria-label={
            stateText
              ? undefined
              : isCamEnabled
                ? "Turn off camera"
                : "Turn on camera"
          }
          onClick={onToggleCam}
          className={cn(
            // Fills a group given a width; content-sized groups are unaffected.
            "grow",
            !isCamEnabled && inactiveClasses(variant),
            !iconOnly && "min-w-32",
          )}
        >
          {!noIcon && (isCamEnabled ? <VideoIcon /> : <VideoOffIcon />)}
          {stateText}
          {children}
        </Button>
        {!noDevicePicker && (
          <DeviceDropdownView
            kind="videoinput"
            devices={cams}
            selectedDevice={selectedCam}
            onDeviceChange={onCamChange}
          >
            <DeviceDropdownTrigger
              render={
                <Button
                  variant={variant}
                  size={iconSize(size)}
                  aria-label="Camera devices"
                >
                  <ChevronDownIcon />
                </Button>
              }
            />
            <DeviceDropdownContent />
          </DeviceDropdownView>
        )}
      </ButtonGroup>
    );
  }

  if (noVideo) return toggle;

  return (
    <div
      data-slot="user-video-tile"
      className={cn(
        "bg-muted relative aspect-video overflow-hidden rounded-xl",
        className,
      )}
    >
      <div className={cn("absolute inset-0", !isCamEnabled && "hidden")}>
        {video}
      </div>
      <div className="absolute bottom-2 left-2">{toggle}</div>
    </div>
  );
}

/** Short, user-facing message for a camera device error. */
function camErrorText(reason: DeviceErrorReason): string {
  switch (reason) {
    case "blocked":
      return "Camera blocked";
    case "already-in-use":
      return "Camera in use";
    case "not-found":
      return "No camera";
    case "not-supported":
      return "Video not supported";
    case "unknown":
    default:
      return "Camera unavailable";
  }
}

export interface UserVideoControlProps extends Omit<
  UserVideoControlViewProps,
  | "isCamEnabled"
  | "onToggleCam"
  | "cams"
  | "selectedCam"
  | "onCamChange"
  | "video"
  | "isLoading"
  | "unavailableText"
> {
  /** Replaces the automatically derived device-error message. */
  unavailableText?: React.ReactNode;
  /** Props for the underlying PipecatClientVideo element. */
  videoProps?: Partial<React.ComponentProps<typeof PipecatClientVideo>>;
}

/** Requires PipecatClientProvider; manages camera state, devices, and preview. */
export function UserVideoControl({
  unavailableText,
  videoProps,
  ...props
}: UserVideoControlProps) {
  const {
    devices: cams,
    selectedDevice: selectedCam,
    updateDevice: onCamChange,
  } = usePipecatDevices("videoinput");

  // "uninitialized" is deliberately not treated as loading: it also covers
  // post-init setups where the transport didn't acquire the camera (e.g.
  // enableCam false). In those cases nothing is in flight, so the control
  // just renders off.
  const { cam } = useMediaState();
  const isLoading = cam.state === "initializing";
  const derivedUnavailable =
    cam.state === "error" ? camErrorText(cam.reason) : undefined;

  return (
    <PipecatClientCamToggle>
      {({ isCamEnabled, onClick }) => (
        <UserVideoControlView
          isCamEnabled={isCamEnabled}
          onToggleCam={onClick}
          isLoading={isLoading}
          unavailableText={unavailableText ?? derivedUnavailable}
          cams={cams}
          selectedCam={selectedCam}
          onCamChange={onCamChange}
          video={
            <PipecatClientVideo
              participant="local"
              className="h-full w-full object-cover"
              {...videoProps}
            />
          }
          {...props}
        />
      )}
    </PipecatClientCamToggle>
  );
}
