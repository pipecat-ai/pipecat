"use client";

import { usePipecatClientCamControl } from "@pipecat-ai/client-react";
import { ChevronsLeftRightEllipsisIcon, InfoIcon, MicIcon } from "lucide-react";
import * as React from "react";

import { ClientStatus } from "@/components/pipecat/client-status";
import {
  ConsolePanel,
  ConsolePanelContent,
  ConsolePanelTitle,
} from "@/components/pipecat/console/panel";
import { SessionInfo } from "@/components/pipecat/session-info";
import { UserAudioControl } from "@/components/pipecat/user-audio-control";
import { UserScreenControl } from "@/components/pipecat/user-screen-control";
import { UserVideoControl } from "@/components/pipecat/user-video-control";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

export interface ConsoleInfoPanelProps {
  /** Hides the connection status section. Default false. */
  noStatusInfo?: boolean;
  /** Hides the microphone control. Default false. */
  noUserAudio?: boolean;
  /** Hides the camera control. Default false. */
  noUserVideo?: boolean;
  /** Hides the screen-share control. Default false. */
  noScreenControl?: boolean;
  /** Hides the session info section. Default false. */
  noSessionInfo?: boolean;
  /** Session id collected by the console shell (BotStarted). */
  sessionId?: string;
  /** Local participant id collected by the console shell. */
  participantId?: string;
  /** Icon-strip rendering for a collapsed pane; sections open in popovers. */
  collapsed?: boolean;
  className?: string;
}

function CollapsedSection({
  label,
  icon,
  children,
}: {
  label: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <Popover>
      <PopoverTrigger
        render={
          <Button variant="ghost" size="icon-sm" aria-label={label}>
            {icon}
          </Button>
        }
      />
      <PopoverContent side="left" align="start" className="w-80 max-w-[90vw]">
        {children}
      </PopoverContent>
    </Popover>
  );
}

/**
 * Console info pane: connection status, local device controls, and session
 * details. Renders null when every section is disabled; the collapsed
 * variant is a vertical icon strip whose sections open in popovers. Must be
 * rendered inside a PipecatClientProvider (with a TooltipProvider ancestor
 * for the session copy buttons).
 */
export function ConsoleInfoPanel({
  noStatusInfo = false,
  noUserAudio = false,
  noUserVideo = false,
  noScreenControl = false,
  noSessionInfo = false,
  sessionId,
  participantId,
  collapsed = false,
  className,
}: ConsoleInfoPanelProps) {
  // Most agents never use the camera, so its preview tile only takes space
  // once the camera is on.
  const { isCamEnabled } = usePipecatClientCamControl();
  const noDevices = noUserAudio && noUserVideo && noScreenControl;
  if (noStatusInfo && noDevices && noSessionInfo) return null;

  const devices = (
    <div className="flex flex-col gap-2">
      {!noUserAudio && <UserAudioControl className="w-full" />}
      {!noUserVideo && (
        <UserVideoControl className="w-full" noVideo={!isCamEnabled} />
      )}
      {!noScreenControl && <UserScreenControl className="w-full" />}
    </div>
  );

  if (collapsed) {
    return (
      <div
        data-slot="console-info-panel"
        data-state="collapsed"
        className={className}
      >
        <div className="flex h-full flex-col items-center gap-1 py-2">
          {!noStatusInfo && (
            <CollapsedSection
              label="Connection status"
              icon={<ChevronsLeftRightEllipsisIcon />}
            >
              <ClientStatus />
            </CollapsedSection>
          )}
          {!noDevices && (
            <CollapsedSection label="Devices" icon={<MicIcon />}>
              {devices}
            </CollapsedSection>
          )}
          {!noSessionInfo && (
            <CollapsedSection label="Session info" icon={<InfoIcon />}>
              <SessionInfo
                sessionId={sessionId}
                participantId={participantId}
              />
            </CollapsedSection>
          )}
        </div>
      </div>
    );
  }

  return (
    <ConsolePanel
      className={className}
      data-slot="console-info-panel"
      data-state="expanded"
    >
      <ConsolePanelContent className="flex flex-col gap-(--card-spacing)">
        {!noStatusInfo && (
          <section className="flex flex-col gap-2">
            <ConsolePanelTitle>Status</ConsolePanelTitle>
            <ClientStatus />
          </section>
        )}
        {!noDevices && (
          <section className="flex flex-col gap-2">
            <ConsolePanelTitle>Devices</ConsolePanelTitle>
            {devices}
          </section>
        )}
        {!noSessionInfo && (
          <section className="flex flex-col gap-2">
            <ConsolePanelTitle>Session</ConsolePanelTitle>
            <SessionInfo sessionId={sessionId} participantId={participantId} />
          </section>
        )}
      </ConsolePanelContent>
    </ConsolePanel>
  );
}
