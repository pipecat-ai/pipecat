"use client";

import type { TransportState } from "@pipecat-ai/client-js";
import {
  usePipecatClient,
  usePipecatClientTransportState,
} from "@pipecat-ai/client-react";
import {
  Loader2Icon,
  PhoneIcon,
  PhoneOffIcon,
  RefreshCwIcon,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type ButtonProps = React.ComponentProps<typeof Button>;

/**
 * Appearance and behavior of the button in a single transport state.
 * Everything is optional — an unset field falls back to the matching
 * top-level prop, then to the built-in default for that state.
 */
export interface ConnectButtonStateProps {
  /** Label for this state. Pass null to render no label (add an aria-label). */
  children?: React.ReactNode;
  /** Leading icon. Transitional states default to a spinner; null removes it. */
  icon?: React.ReactNode;
  /**
   * Button variant for this state. Setting a variant — here or at the top
   * level — replaces the default token styling for the state.
   */
  variant?: ButtonProps["variant"];
  /** Extra classes for this state, merged after the top-level className. */
  className?: string;
  /** Whether the button is disabled. Transitional states default to true. */
  disabled?: boolean;
  /** Replaces the default connect/disconnect action for this state. */
  onClick?: React.MouseEventHandler<HTMLButtonElement>;
}

/**
 * Sparse per-state overrides, keyed by Pipecat's TransportState. Specify
 * only the states you want to change; everything else keeps its default.
 */
export type ConnectButtonStateMap = Partial<
  Record<TransportState, ConnectButtonStateProps>
>;

const SPINNER = <Loader2Icon className="animate-spin" />;

// Nova's icon-* sizes render a square button; the view responds by going
// icon-only (no label, no min-width).
function isIconSize(size: ButtonProps["size"]): boolean {
  return typeof size === "string" && size.startsWith("icon");
}

// Default glyphs for icon-only rendering. Text sizes stay label-first
// (spinner only on transitional states), so these apply only when an
// icon-* size is used and neither the state nor the consumer set an icon.
const DEFAULT_STATE_ICONS: Record<TransportState, React.ReactNode> = {
  disconnected: <PhoneIcon />,
  initializing: SPINNER,
  initialized: <PhoneIcon />,
  authenticating: SPINNER,
  authenticated: SPINNER,
  connecting: SPINNER,
  connected: <PhoneOffIcon />,
  ready: <PhoneOffIcon />,
  disconnecting: SPINNER,
  error: <RefreshCwIcon />,
};

// Idle states: solid active-token fill — the kit's "go" affordance.
const CONNECT_CLASSES =
  "bg-active text-active-foreground hover:bg-active/85 focus-visible:ring-active/40 focus-visible:border-active";
// Connected states: tinted inactive-token surface — the kit's "stop"
// affordance (nova's destructive-button idiom, as on the screen control).
const DISCONNECT_CLASSES =
  "border-inactive/30 bg-inactive/10 text-inactive hover:bg-inactive/20 hover:text-inactive dark:bg-inactive/20";

const DEFAULT_STATE_PROPS: Record<TransportState, ConnectButtonStateProps> = {
  disconnected: { children: "Connect", className: CONNECT_CLASSES },
  initializing: {
    children: "Initializing…",
    icon: SPINNER,
    variant: "secondary",
    disabled: true,
  },
  initialized: { children: "Connect", className: CONNECT_CLASSES },
  authenticating: {
    children: "Connecting…",
    icon: SPINNER,
    variant: "secondary",
    disabled: true,
  },
  authenticated: {
    children: "Connecting…",
    icon: SPINNER,
    variant: "secondary",
    disabled: true,
  },
  connecting: {
    children: "Connecting…",
    icon: SPINNER,
    variant: "secondary",
    disabled: true,
  },
  connected: {
    children: "Disconnect",
    variant: "outline",
    className: DISCONNECT_CLASSES,
  },
  ready: {
    children: "Disconnect",
    variant: "outline",
    className: DISCONNECT_CLASSES,
  },
  disconnecting: {
    children: "Disconnecting…",
    icon: SPINNER,
    variant: "secondary",
    disabled: true,
  },
  error: { children: "Retry", variant: "destructive" },
};

/** States where the transport is between stable endpoints. */
const TRANSITIONAL_STATES: TransportState[] = [
  "initializing",
  "authenticating",
  "authenticated",
  "connecting",
  "disconnecting",
];

/** States where clicking tears the session down rather than starting one. */
const DISCONNECT_STATES: TransportState[] = [
  "authenticating",
  "authenticated",
  "connecting",
  "connected",
  "ready",
];

export interface ConnectButtonViewProps extends Omit<
  ButtonProps,
  "children" | "onClick"
> {
  /** Transport state driving the button. */
  transportState?: TransportState;
  /** Called when clicked in a state that starts a session (disconnected, initialized, error). */
  onConnect?: () => void;
  /** Called when clicked in a state that ends one (connecting through ready). */
  onDisconnect?: () => void;
  /** Called on every click, before the state action. */
  onClick?: React.MouseEventHandler<HTMLButtonElement>;
  /** Label used for every state, unless a state overrides it. */
  children?: React.ReactNode;
  /** Sparse per-state overrides for label, icon, style, and action. */
  stateProps?: ConnectButtonStateMap;
}

/** State overrides take precedence over component props, then defaults. Icon-only buttons use the label as their accessible name. */
export function ConnectButtonView({
  transportState = "disconnected",
  onConnect,
  onDisconnect,
  onClick,
  children,
  stateProps,
  variant,
  size,
  className,
  disabled,
  ...props
}: ConnectButtonViewProps) {
  const defaults = DEFAULT_STATE_PROPS[transportState];
  const overrides = stateProps?.[transportState] ?? {};
  const iconOnly = isIconSize(size);

  // null is meaningful for content (renders nothing); undefined falls back.
  const resolvedLabel =
    overrides.children !== undefined
      ? overrides.children
      : children !== undefined
        ? children
        : defaults.children;
  const label = iconOnly ? null : resolvedLabel;
  const icon =
    overrides.icon !== undefined
      ? overrides.icon
      : iconOnly
        ? (defaults.icon ?? DEFAULT_STATE_ICONS[transportState])
        : defaults.icon;
  // An explicit variant opts out of the default token styling, which is
  // designed around the state's default variant.
  const stateClassName =
    (overrides.variant ?? variant) ? undefined : defaults.className;

  const handleClick: React.MouseEventHandler<HTMLButtonElement> = (event) => {
    onClick?.(event);
    if (overrides.onClick) {
      overrides.onClick(event);
    } else if (DISCONNECT_STATES.includes(transportState)) {
      onDisconnect?.();
    } else {
      onConnect?.();
    }
  };

  return (
    <Button
      data-slot="connect-button"
      data-state={transportState}
      variant={overrides.variant ?? variant ?? defaults.variant}
      size={size}
      disabled={disabled || (overrides.disabled ?? defaults.disabled)}
      aria-busy={TRANSITIONAL_STATES.includes(transportState) || undefined}
      aria-label={
        iconOnly && typeof resolvedLabel === "string"
          ? resolvedLabel
          : undefined
      }
      onClick={handleClick}
      className={cn(
        !iconOnly && "min-w-32",
        stateClassName,
        className,
        overrides.className,
      )}
      {...props}
    >
      {icon}
      {label}
    </Button>
  );
}

export type ConnectButtonProps = Omit<ConnectButtonViewProps, "transportState">;

/** Requires PipecatClientProvider. Pass onConnect when your app owns bot startup. */
export function ConnectButton({
  onConnect,
  onDisconnect,
  ...props
}: ConnectButtonProps) {
  const client = usePipecatClient();
  const transportState = usePipecatClientTransportState();

  const handleConnect = () => {
    if (onConnect) {
      onConnect();
      return;
    }
    try {
      // Connection failures surface through the transport's "error" state.
      client?.connect().catch(() => {});
    } catch {
      // connect() can also throw synchronously (e.g. missing connection
      // params); the client reports it, nothing more to do here.
    }
  };

  const handleDisconnect = () => {
    if (onDisconnect) {
      onDisconnect();
      return;
    }
    client?.disconnect().catch(() => {});
  };

  return (
    <ConnectButtonView
      transportState={transportState}
      onConnect={handleConnect}
      onDisconnect={handleDisconnect}
      {...props}
    />
  );
}
