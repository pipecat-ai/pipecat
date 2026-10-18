"use client";

import type {
  APIRequest,
  Participant,
  PipecatClient,
  PipecatClientOptions,
  TransportConnectionParams,
  TransportState,
} from "@pipecat-ai/client-js";
import { RTVIEvent } from "@pipecat-ai/client-js";
import {
  PipecatClientProvider,
  useConversationContext,
  useRTVIClientEvent,
} from "@pipecat-ai/client-react";
import {
  BotIcon,
  ChevronsLeftRightEllipsisIcon,
  InfoIcon,
  MessagesSquareIcon,
  PanelLeftCloseIcon,
  PanelRightCloseIcon,
  XIcon,
} from "lucide-react";
import * as React from "react";
import {
  useDefaultLayout,
  usePanelRef,
  type LayoutStorage,
} from "react-resizable-panels";

import { BotAudioOutput } from "@/components/pipecat/bot-audio";
import { ConnectButton } from "@/components/pipecat/connect-button";
import { ConsoleBotAudioPanel } from "@/components/pipecat/console/bot-audio-panel";
import { ConsoleBotVideoPanel } from "@/components/pipecat/console/bot-video-panel";
import { SmallWebRTCCodecSetter } from "@/components/pipecat/console/codec-setter";
import { ConsoleConversationPanel } from "@/components/pipecat/console/conversation-panel";
import { ConsoleEventsPanel } from "@/components/pipecat/console/events-panel";
import { ConsoleInfoPanel } from "@/components/pipecat/console/info-panel";
import { ConsoleKeypadToggle } from "@/components/pipecat/console/keypad-toggle";
import { PipecatLogo } from "@/components/pipecat/console/logo";
import { useMinWidth } from "@/components/pipecat/console/panel";
import type { ConversationProps } from "@/components/pipecat/conversation";
import type { TextRenderMode } from "@/components/pipecat/conversation-message";
import type { DTMFKeypadMode } from "@/components/pipecat/dtmf-keypad";
import {
  Alert,
  AlertAction,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { Spinner } from "@/components/ui/spinner";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  usePipecatApp,
  type UsePipecatAppReturn,
} from "@/hooks/use-pipecat-app";
import { usePipecatEventStream } from "@/hooks/use-pipecat-event-stream";
import { usePipecatMetricValue } from "@/hooks/use-pipecat-metrics";
import type {
  TransportFactory,
  TransportOptions,
  TransportType,
} from "@/lib/transports";
import { cn } from "@/lib/utils";

// Connection-URL helpers: resolve the URL the connect button will hit, for
// its hover tooltip.

function extractUrlFromOptions(options: unknown): string | undefined {
  if (typeof options !== "object" || options === null) return undefined;
  const record = options as Record<string, unknown>;
  const requestParams = record.webrtcRequestParams;
  if (typeof requestParams === "object" && requestParams !== null) {
    const endpoint = (requestParams as Record<string, unknown>).endpoint;
    if (typeof endpoint === "string") return endpoint;
  }
  if (typeof record.webrtcUrl === "string") return record.webrtcUrl;
  if (typeof record.connectionUrl === "string") return record.connectionUrl;
  return undefined;
}

function resolveUrl(url: string): string {
  if (typeof window === "undefined") return url;
  try {
    return new URL(url, window.location.origin).toString();
  } catch {
    return url;
  }
}

function getConnectionUrl(
  startBotParams?: APIRequest,
  connectParams?: unknown,
  transportOptions?: unknown,
): string | undefined {
  const candidate =
    (typeof startBotParams?.endpoint === "string" && startBotParams.endpoint) ||
    extractUrlFromOptions(connectParams) ||
    extractUrlFromOptions(transportOptions);
  return candidate ? resolveUrl(candidate) : undefined;
}

// The metrics tab and the mobile events tab unmount while inactive. These keep
// their shared stores collecting, so opening either shows the whole session.
function MetricsCollector() {
  usePipecatMetricValue("ttfb");
  return null;
}

function EventStreamCollector() {
  usePipecatEventStream();
  return null;
}

/** Inert storage handed to useDefaultLayout when persistence is off. */
const MEMORY_STORAGE: LayoutStorage = {
  getItem: () => null,
  setItem: () => {},
} as LayoutStorage;

export interface ConsoleProps {
  // -- Bootstrap (forwarded to usePipecatApp) --------------------------------
  /** Transport backing the client (default "smallwebrtc"). Match it to transportFactory; codec and ICE handling depend on it. */
  transportType?: TransportType;
  /** Creates your installed transport. Read once; remount to change. Without it, the loader registered for transportType is used. */
  transportFactory?: TransportFactory;
  /** Constructor options for the selected transport. */
  transportOptions?: TransportOptions;
  /** Overrides merged into the PipecatClient constructor. */
  clientOptions?: Partial<PipecatClientOptions>;
  /** Connection params, or an APIRequest (with endpoint) for start-and-connect. */
  connectParams?: TransportConnectionParams | APIRequest;
  /** When set, connecting starts a bot via startBot() first. */
  startBotParams?: APIRequest;
  /** Transforms the startBot response before connecting. */
  startBotResponseTransformer?: (
    response: TransportConnectionParams,
  ) => TransportConnectionParams | Promise<TransportConnectionParams>;
  /** Connect automatically once the client is ready. Default false. */
  connectOnMount?: boolean;
  /** Initialize devices as soon as the client exists. Default true. */
  initDevicesOnMount?: boolean;
  /** Fired once per created client, before device init or connect. */
  onClient?: (client: PipecatClient) => void;

  // -- Header ---------------------------------------------------------------
  /** Header title (default "Pipecat Console"); hidden on narrow viewports. */
  titleText?: string;
  /** Replaces the Pipecat logo. */
  logo?: React.ReactNode;
  /** Hides the logo entirely. Default false. */
  noLogo?: boolean;
  /**
   * Extra header content rendered before the connect button — e.g. your
   * app's theme toggle. Rendered only when provided.
   */
  headerSlot?: React.ReactNode;

  // -- Region toggles (all default false unless noted) -----------------------
  noUserAudio?: boolean;
  noUserVideo?: boolean;
  noScreenControl?: boolean;
  /** Hides the DTMF keypad button. */
  noDTMF?: boolean;
  /** Hides the text input under the transcript. */
  noTextInput?: boolean;
  /** Disables bot audio playback entirely (also hides its volume control). */
  noAudioOutput?: boolean;
  /** Hides the bot audio pane. */
  noBotAudio?: boolean;
  /** Hides the volume control in the bot audio pane header. */
  noBotAudioControls?: boolean;
  /** Hides the bot video pane. Default false. */
  noBotVideo?: boolean;
  noConversation?: boolean;
  noMetrics?: boolean;
  /** Hides the events strip. */
  noEvents?: boolean;
  noStatusInfo?: boolean;
  noSessionInfo?: boolean;

  // -- Behavior --------------------------------------------------------------
  /** How the DTMF keypad dispatches presses (default "buffered"). */
  keypadMode?: DTMFKeypadMode;
  /** Seeds the transcript render mode (default "karaoke"). */
  textRenderMode?: TextRenderMode;
  /** Hides the render-mode select. Default false. */
  noTextRenderModeSwitch?: boolean;
  /** Overrides merged onto the Conversation (labels, renderers, …). */
  conversationProps?: Partial<ConversationProps>;
  /** Preferred audio codec (smallwebrtc only). */
  audioCodec?: string;
  /** Preferred video codec (smallwebrtc only). */
  videoCodec?: string;
  /** Start with the info pane collapsed. Default false. */
  collapseInfoPanel?: boolean;
  /** Start with the bot media pane collapsed. Default false. */
  collapseMediaPanel?: boolean;
  /** Start with the events strip collapsed. Default false. */
  collapseEventsPanel?: boolean;
  /**
   * Persist the resizable layout under this key (localStorage). Off when
   * unset.
   */
  layoutPersistenceKey?: string;

  // -- Callbacks -------------------------------------------------------------
  /** Fired on every transport state change. */
  onConnectionStateChange?: (state: TransportState) => void;
  /** Fired for RTVI server messages. */
  onServerMessage?: (data: unknown) => void;
  /** Hands you the transcript's injectMessage function once available. */
  onInjectMessage?: (
    inject: ReturnType<typeof useConversationContext>["injectMessage"],
  ) => void;

  className?: string;
}

/**
 * Full-page Pipecat debugging console: header with connect flow, resizable
 * bot-media / conversation+metrics / info panes over an events strip, and a
 * single-tree mobile layout using bottom tabs.
 *
 * The console builds its own client via usePipecatApp and renders its own
 * PipecatClientProvider — do not nest it inside another provider. Supply
 * the transport with `transportFactory` or `registerTransport`; load
 * failures surface in the error banner.
 */
export function Console(props: ConsoleProps) {
  const {
    transportType = "smallwebrtc",
    transportFactory,
    transportOptions,
    clientOptions,
    connectParams,
    startBotParams,
    startBotResponseTransformer,
    connectOnMount = false,
    initDevicesOnMount = true,
    onClient,
    className,
  } = props;

  const app = usePipecatApp({
    transportType,
    transportFactory,
    transportOptions,
    clientOptions,
    connectParams,
    startBotParams,
    startBotResponseTransformer,
    connectOnMount,
    initDevicesOnMount,
    onClient,
  });

  if (!app.client) {
    return (
      <div
        data-slot="console"
        data-state={app.error ? "error" : "loading"}
        className={cn(
          "bg-background flex h-full min-h-64 items-center justify-center p-4",
          className,
        )}
      >
        {app.error ? (
          <Alert variant="destructive" className="max-w-lg">
            <AlertTitle>Failed to start the console</AlertTitle>
            <AlertDescription>{app.error}</AlertDescription>
          </Alert>
        ) : (
          <Spinner className="size-6" />
        )}
      </div>
    );
  }

  return (
    <PipecatClientProvider client={app.client}>
      <TooltipProvider>
        <ConsoleShell {...props} app={app} />
      </TooltipProvider>
      {!props.noAudioOutput && <BotAudioOutput />}
      {!props.noMetrics && <MetricsCollector />}
      {!props.noEvents && <EventStreamCollector />}
      {transportType === "smallwebrtc" && (
        <SmallWebRTCCodecSetter
          audioCodec={props.audioCodec}
          videoCodec={props.videoCodec}
        />
      )}
    </PipecatClientProvider>
  );
}

function ConsoleShell({
  app,
  transportOptions,
  connectParams,
  startBotParams,
  titleText = "Pipecat Console",
  logo,
  noLogo = false,
  headerSlot,
  noUserAudio = false,
  noUserVideo = false,
  noScreenControl = false,
  noDTMF = false,
  noTextInput = false,
  noAudioOutput = false,
  noBotAudio = false,
  noBotAudioControls = false,
  noBotVideo = false,
  noConversation = false,
  noMetrics = false,
  noEvents = false,
  noStatusInfo = false,
  noSessionInfo = false,
  keypadMode = "buffered",
  textRenderMode,
  noTextRenderModeSwitch = false,
  conversationProps,
  collapseInfoPanel = false,
  collapseMediaPanel = false,
  collapseEventsPanel = false,
  layoutPersistenceKey,
  onConnectionStateChange,
  onServerMessage,
  onInjectMessage,
  className,
}: ConsoleProps & { app: UsePipecatAppReturn }) {
  // Three resizable columns need about 1024px; narrower screens get tabs.
  const isDesktop = useMinWidth(1024);

  // -- Session facts collected from RTVI events ------------------------------
  const [participantId, setParticipantId] = React.useState<string>();
  const [sessionId, setSessionId] = React.useState<string>();

  useRTVIClientEvent(
    RTVIEvent.ParticipantConnected,
    React.useCallback((participant: Participant) => {
      if (participant.local) setParticipantId(participant.id);
    }, []),
  );
  // Belt and braces for transports that never fire ParticipantConnected.
  useRTVIClientEvent(
    RTVIEvent.TrackStarted,
    React.useCallback((_track: MediaStreamTrack, participant?: Participant) => {
      if (participant?.local && participant.id) {
        setParticipantId(participant.id);
      }
    }, []),
  );
  useRTVIClientEvent(
    RTVIEvent.BotStarted,
    React.useCallback((response: unknown) => {
      const id = (response as { sessionId?: string } | null)?.sessionId;
      if (typeof id === "string") setSessionId(id);
    }, []),
  );

  const onServerMessageRef = React.useRef(onServerMessage);
  onServerMessageRef.current = onServerMessage;
  useRTVIClientEvent(
    RTVIEvent.ServerMessage,
    React.useCallback((data: unknown) => {
      onServerMessageRef.current?.(data);
    }, []),
  );

  const onConnectionStateChangeRef = React.useRef(onConnectionStateChange);
  onConnectionStateChangeRef.current = onConnectionStateChange;
  useRTVIClientEvent(
    RTVIEvent.TransportStateChanged,
    React.useCallback((state: TransportState) => {
      onConnectionStateChangeRef.current?.(state);
    }, []),
  );

  const { injectMessage } = useConversationContext();
  const onInjectMessageRef = React.useRef(onInjectMessage);
  onInjectMessageRef.current = onInjectMessage;
  React.useEffect(() => {
    onInjectMessageRef.current?.(injectMessage);
  }, [injectMessage]);

  // -- Region math -----------------------------------------------------------
  const noBotArea = noBotAudio && noBotVideo;
  const noConversationPanel = noConversation && noMetrics;
  const noDevices = noUserAudio && noUserVideo && noScreenControl;
  const noInfoPanel = noStatusInfo && noDevices && noSessionInfo;

  // react-resizable-panels v4 reads bare numbers as pixels, so every size
  // below is a percentage string. Conversation takes whatever the side panes
  // leave so the horizontal defaults always sum to 100.
  const mediaDefaultSize = noBotArea ? 0 : collapseMediaPanel ? 8 : 26;
  const infoDefaultSize = noInfoPanel ? 0 : collapseInfoPanel ? 4 : 27;
  const conversationDefaultSize = 100 - mediaDefaultSize - infoDefaultSize;

  // -- Resizable layout state ------------------------------------------------
  const persist = Boolean(layoutPersistenceKey);
  const storage = persist ? undefined : MEMORY_STORAGE;
  const verticalLayout = useDefaultLayout({
    id: `${layoutPersistenceKey ?? "pipecat-console"}:v`,
    storage,
  });
  const horizontalLayout = useDefaultLayout({
    id: `${layoutPersistenceKey ?? "pipecat-console"}:h`,
    storage,
  });

  const mediaPanelRef = usePanelRef();
  const infoPanelRef = usePanelRef();
  const eventsPanelRef = usePanelRef();
  const [isMediaCollapsed, setIsMediaCollapsed] =
    React.useState(collapseMediaPanel);
  const [isInfoCollapsed, setIsInfoCollapsed] =
    React.useState(collapseInfoPanel);
  const [isEventsCollapsed, setIsEventsCollapsed] =
    React.useState(collapseEventsPanel);

  const connectionUrl = getConnectionUrl(
    startBotParams,
    connectParams,
    transportOptions,
  );

  // -- Shared panel elements (instantiated once per render, one branch) ------
  const mediaStack = (collapsed: boolean) =>
    noBotArea ? null : (
      <div className="flex h-full min-h-0 flex-col gap-2">
        {!noBotAudio && (
          <ConsoleBotAudioPanel
            collapsed={collapsed}
            noControls={noBotAudioControls || noAudioOutput}
            className={noBotVideo ? undefined : "flex-1"}
          />
        )}
        {!noBotVideo && (
          <ConsoleBotVideoPanel
            collapsed={collapsed}
            className={noBotAudio ? undefined : "flex-1"}
          />
        )}
      </div>
    );

  const conversationPanel = noConversationPanel ? null : (
    <ConsoleConversationPanel
      noConversation={noConversation}
      noMetrics={noMetrics}
      noTextInput={noTextInput}
      noTextRenderModeSwitch={noTextRenderModeSwitch}
      textRenderMode={textRenderMode}
      conversationProps={conversationProps}
      className="h-full"
    />
  );

  const infoPanel = (collapsed: boolean) =>
    noInfoPanel ? null : (
      <ConsoleInfoPanel
        noStatusInfo={noStatusInfo}
        noUserAudio={noUserAudio}
        noUserVideo={noUserVideo}
        noScreenControl={noScreenControl}
        noSessionInfo={noSessionInfo}
        sessionId={sessionId}
        participantId={participantId}
        collapsed={collapsed}
        className="h-full"
      />
    );

  const eventsPanel = (collapsed: boolean) =>
    noEvents ? null : (
      <ConsoleEventsPanel collapsed={collapsed} className="h-full" />
    );

  const toggleInfoPanel = () => {
    const handle = infoPanelRef.current;
    if (!handle) return;
    if (handle.isCollapsed()) handle.expand();
    else handle.collapse();
  };

  const connectButton = (
    <ConnectButton
      size="sm"
      onConnect={app.connect}
      onDisconnect={app.disconnect}
    />
  );

  return (
    <div
      data-slot="console"
      data-state="ready"
      className={cn(
        "bg-background text-foreground flex h-full min-h-0 flex-col",
        className,
      )}
    >
      <header
        data-slot="console-header"
        className="flex items-center gap-2 border-b px-3 py-2"
      >
        <div className="flex shrink-0 items-center">
          {noLogo ? (
            <span className="h-6" />
          ) : (
            (logo ?? <PipecatLogo height={20} />)
          )}
        </div>
        <strong className="hidden min-w-0 truncate text-sm md:block">
          {titleText}
        </strong>
        <div className="ml-auto flex min-w-0 items-center justify-end gap-1">
          {/* Slot content may shrink; the console's own controls never do. */}
          {headerSlot && (
            <div className="flex min-w-0 items-center gap-1">{headerSlot}</div>
          )}
          {!noDTMF && <ConsoleKeypadToggle mode={keypadMode} />}
          {!noInfoPanel && (
            <Button
              variant="ghost"
              size="icon-sm"
              className="hidden lg:inline-flex"
              aria-label={
                isInfoCollapsed ? "Expand info panel" : "Collapse info panel"
              }
              onClick={toggleInfoPanel}
            >
              {isInfoCollapsed ? (
                <PanelLeftCloseIcon />
              ) : (
                <PanelRightCloseIcon />
              )}
            </Button>
          )}
          {connectionUrl ? (
            <Tooltip>
              <TooltipTrigger render={<span className="inline-flex" />}>
                {connectButton}
              </TooltipTrigger>
              <TooltipContent side="bottom" align="end">
                {connectionUrl}
              </TooltipContent>
            </Tooltip>
          ) : (
            connectButton
          )}
        </div>
      </header>

      {app.error && (
        <Alert
          variant="destructive"
          className="animate-in fade-in m-2 duration-300"
        >
          <AlertTitle>Session error</AlertTitle>
          <AlertDescription>{app.error}</AlertDescription>
          <AlertAction>
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label="Dismiss error"
              onClick={app.clearError}
            >
              <XIcon />
            </Button>
          </AlertAction>
        </Alert>
      )}

      <div className="min-h-0 flex-1 p-2">
        {isDesktop ? (
          <ResizablePanelGroup
            orientation="vertical"
            defaultLayout={verticalLayout.defaultLayout}
            onLayoutChanged={verticalLayout.onLayoutChanged}
            className="gap-2"
          >
            <ResizablePanel id="main" defaultSize="70%" minSize="50%">
              <ResizablePanelGroup
                orientation="horizontal"
                defaultLayout={horizontalLayout.defaultLayout}
                onLayoutChanged={horizontalLayout.onLayoutChanged}
                className="gap-2"
              >
                {!noBotArea && (
                  <ResizablePanel
                    id="media"
                    defaultSize={`${mediaDefaultSize}%`}
                    minSize="10%"
                    maxSize="30%"
                    collapsible
                    collapsedSize="8%"
                    panelRef={mediaPanelRef}
                    onResize={() =>
                      setIsMediaCollapsed(
                        mediaPanelRef.current?.isCollapsed() ?? false,
                      )
                    }
                  >
                    {/* p-px keeps each card's ring inside the panel's scroll box, which clips it. */}
                    <div className="h-full p-px">
                      {mediaStack(isMediaCollapsed)}
                    </div>
                  </ResizablePanel>
                )}
                {!noBotArea && (!noConversationPanel || !noInfoPanel) && (
                  <ResizableHandle withHandle />
                )}
                {!noConversationPanel && (
                  <ResizablePanel
                    id="conversation"
                    defaultSize={`${conversationDefaultSize}%`}
                    minSize="30%"
                  >
                    <div className="h-full p-px">{conversationPanel}</div>
                  </ResizablePanel>
                )}
                {!noConversationPanel && !noInfoPanel && (
                  <ResizableHandle withHandle />
                )}
                {!noInfoPanel && (
                  <ResizablePanel
                    id="info"
                    defaultSize={`${infoDefaultSize}%`}
                    minSize="15%"
                    collapsible
                    collapsedSize="4%"
                    panelRef={infoPanelRef}
                    onResize={() =>
                      setIsInfoCollapsed(
                        infoPanelRef.current?.isCollapsed() ?? false,
                      )
                    }
                  >
                    <div className="h-full p-px">
                      {infoPanel(isInfoCollapsed)}
                    </div>
                  </ResizablePanel>
                )}
              </ResizablePanelGroup>
            </ResizablePanel>
            {!noEvents && <ResizableHandle withHandle />}
            {!noEvents && (
              <ResizablePanel
                id="events"
                defaultSize={collapseEventsPanel ? "6%" : "30%"}
                minSize="7%"
                collapsible
                collapsedSize="6%"
                panelRef={eventsPanelRef}
                onResize={() =>
                  setIsEventsCollapsed(
                    eventsPanelRef.current?.isCollapsed() ?? false,
                  )
                }
              >
                <div className="h-full p-px">
                  {eventsPanel(isEventsCollapsed)}
                </div>
              </ResizablePanel>
            )}
          </ResizablePanelGroup>
        ) : (
          <Tabs
            defaultValue={
              !noConversationPanel
                ? "conversation"
                : !noBotArea
                  ? "bot"
                  : !noInfoPanel
                    ? "info"
                    : "events"
            }
            className="flex h-full min-h-0 flex-col"
          >
            {!noBotArea && (
              <TabsContent value="bot" className="min-h-0 flex-1">
                {mediaStack(false)}
              </TabsContent>
            )}
            {!noConversationPanel && (
              <TabsContent
                value="conversation"
                keepMounted
                className="min-h-0 flex-1 data-[hidden]:hidden"
              >
                {conversationPanel}
              </TabsContent>
            )}
            {!noInfoPanel && (
              <TabsContent value="info" className="min-h-0 flex-1">
                {infoPanel(false)}
              </TabsContent>
            )}
            {!noEvents && (
              <TabsContent value="events" className="min-h-0 flex-1">
                {eventsPanel(false)}
              </TabsContent>
            )}
            <TabsList className="mt-2 w-full">
              {!noConversationPanel && (
                <TabsTrigger value="conversation" aria-label="Conversation">
                  <MessagesSquareIcon />
                </TabsTrigger>
              )}
              {!noBotArea && (
                <TabsTrigger value="bot" aria-label="Bot media">
                  <BotIcon />
                </TabsTrigger>
              )}
              {!noInfoPanel && (
                <TabsTrigger value="info" aria-label="Info">
                  <InfoIcon />
                </TabsTrigger>
              )}
              {!noEvents && (
                <TabsTrigger value="events" aria-label="Events">
                  <ChevronsLeftRightEllipsisIcon />
                </TabsTrigger>
              )}
            </TabsList>
          </Tabs>
        )}
      </div>
    </div>
  );
}
