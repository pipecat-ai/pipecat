"use client";

import {
  PipecatClientScreenShareToggle,
  PipecatClientVideo,
  usePipecatClientTransportState,
} from "@pipecat-ai/client-react";
import { Loader2Icon, MonitorIcon, MonitorOffIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type ButtonProps = React.ComponentProps<typeof Button>;

// Inverse of the mic/camera controls: not-sharing is the neutral resting
// state (plain surface), while sharing turns the button into a red
// stop-sharing affordance, applied while sharing (data-state also flips
// for consumer styling). Styled per variant like the other controls:
// solid variants fill with the inactive color, outline keeps its border +
// tint (nova's destructive-button idiom), secondary tints without a
// border, ghost only recolors content. Unknown (consumer-added) variants
// get the ghost treatment.
const ACTIVE_VARIANT_CLASSES: Record<string, string> = {
  default: "bg-inactive text-inactive-foreground hover:bg-inactive/80",
  outline:
    "border-inactive/30 bg-inactive/10 text-inactive hover:bg-inactive/20 hover:text-inactive dark:bg-inactive/20",
  secondary:
    "bg-inactive/10 text-inactive hover:bg-inactive/20 dark:bg-inactive/20",
  ghost: "text-inactive hover:text-inactive",
};

function activeClasses(variant: ButtonProps["variant"]): string {
  return (
    ACTIVE_VARIANT_CLASSES[variant ?? "default"] ??
    ACTIVE_VARIANT_CLASSES.ghost!
  );
}

// Nova's icon-* sizes render a square button; the toggle responds by going
// icon-only (no label, no min-width).
function isIconSize(size: ButtonProps["size"]): boolean {
  return typeof size === "string" && size.startsWith("icon");
}

export interface UserScreenControlViewProps {
  /** Whether screen sharing is currently active. */
  isScreenEnabled?: boolean;
  /** Called when the user clicks the toggle. */
  onToggleScreen?: () => void;
  /** Disables the control (e.g. while disconnected). */
  disabled?: boolean;
  /** Disables the control and shows a spinner. */
  isLoading?: boolean;
  /** Content shown next to the spinner while loading. Spinner only by default. */
  loadingText?: React.ReactNode;
  /**
   * When set, the control renders disabled with this message instead of the
   * toggle (sharing blocked, not supported, …). Takes precedence over the
   * interactive state; isLoading takes precedence over both.
   */
  unavailableText?: React.ReactNode;
  /** Never render the preview tile, even while sharing. */
  noPreview?: boolean;
  /** Video element rendered in the tile while sharing. */
  video?: React.ReactNode;
  /** Label shown while sharing. */
  activeText?: React.ReactNode;
  /** Label shown while not sharing. */
  inactiveText?: React.ReactNode;
  /** Hide the monitor icon. */
  noIcon?: boolean;
  /** Button variant for the toggle. */
  variant?: ButtonProps["variant"];
  /**
   * Button size for the toggle. Icon sizes ("icon", "icon-sm", …) render
   * it icon-only: the label is hidden.
   */
  size?: ButtonProps["size"];
  /** Classes for the tile while sharing, otherwise for the button. */
  className?: string;
  /** Extra content rendered inside the toggle, after the label. */
  children?: React.ReactNode;
}

/** Screen-share toggle with a preview while sharing. Loading takes precedence over unavailability. */
export function UserScreenControlView({
  isScreenEnabled = false,
  onToggleScreen,
  disabled = false,
  isLoading = false,
  loadingText,
  unavailableText,
  noPreview = false,
  video,
  activeText,
  inactiveText,
  noIcon = false,
  variant = "outline",
  size = "default",
  className,
  children,
}: UserScreenControlViewProps) {
  const iconOnly = isIconSize(size);
  const stateText = iconOnly
    ? undefined
    : isScreenEnabled
      ? activeText
      : inactiveText;
  const showTile =
    !noPreview && isScreenEnabled && !isLoading && !unavailableText;

  let toggle: React.ReactNode;
  if (isLoading || unavailableText) {
    toggle = (
      <Button
        data-slot="user-screen-control"
        variant={variant}
        size={size}
        disabled
        aria-busy={isLoading || undefined}
        data-state={isLoading ? "loading" : "unavailable"}
        aria-label={
          !iconOnly
            ? undefined
            : isLoading
              ? "Screen share loading"
              : typeof unavailableText === "string"
                ? unavailableText
                : "Screen share unavailable"
        }
        className={cn(!iconOnly && "min-w-32", className)}
      >
        {isLoading ? (
          <>
            <Loader2Icon className="animate-spin" />
            {!iconOnly && loadingText}
          </>
        ) : (
          <>
            {!noIcon && <MonitorOffIcon />}
            {!iconOnly && unavailableText}
          </>
        )}
      </Button>
    );
  } else {
    toggle = (
      <Button
        data-slot="user-screen-control"
        variant={variant}
        size={size}
        data-state={isScreenEnabled ? "active" : "inactive"}
        aria-pressed={isScreenEnabled}
        aria-label={
          stateText
            ? undefined
            : isScreenEnabled
              ? "Stop screen sharing"
              : "Start screen sharing"
        }
        disabled={disabled}
        onClick={onToggleScreen}
        className={cn(
          isScreenEnabled && activeClasses(variant),
          !iconOnly && "min-w-32",
          !showTile && className,
        )}
      >
        {!noIcon && (isScreenEnabled ? <MonitorIcon /> : <MonitorOffIcon />)}
        {stateText}
        {children}
      </Button>
    );
  }

  if (!showTile) return toggle;

  return (
    <div
      data-slot="user-screen-tile"
      className={cn(
        "bg-muted relative aspect-video overflow-hidden rounded-xl",
        className,
      )}
    >
      <div className="absolute inset-0">{video}</div>
      <div className="absolute bottom-2 left-2">{toggle}</div>
    </div>
  );
}

const CONNECTED_STATES = ["connected", "ready"];

export interface UserScreenControlProps extends Omit<
  UserScreenControlViewProps,
  "isScreenEnabled" | "onToggleScreen" | "video" | "disabled"
> {
  /** Props for the underlying PipecatClientVideo element. */
  videoProps?: Partial<React.ComponentProps<typeof PipecatClientVideo>>;
}

/** Requires PipecatClientProvider; sharing is disabled until the transport is connected. */
export function UserScreenControl({
  videoProps,
  ...props
}: UserScreenControlProps) {
  const transportState = usePipecatClientTransportState();
  const isConnected = CONNECTED_STATES.includes(transportState);

  return (
    <PipecatClientScreenShareToggle>
      {({ isScreenShareEnabled, onClick }) => (
        <UserScreenControlView
          isScreenEnabled={isScreenShareEnabled}
          onToggleScreen={onClick}
          disabled={!isConnected}
          video={
            <PipecatClientVideo
              participant="local"
              trackType="screenVideo"
              className="h-full w-full object-cover"
              {...videoProps}
            />
          }
          {...props}
        />
      )}
    </PipecatClientScreenShareToggle>
  );
}
