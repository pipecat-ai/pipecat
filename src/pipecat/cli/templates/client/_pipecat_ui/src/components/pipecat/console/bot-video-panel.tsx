"use client";

import {
  PipecatClientVideo,
  usePipecatClientMediaTrack,
} from "@pipecat-ai/client-react";
import { VideoOffIcon } from "lucide-react";
import {
  ConsolePanel,
  ConsolePanelContent,
  ConsolePanelHeader,
  ConsolePanelTitle,
} from "@/components/pipecat/console/panel";

export interface ConsoleBotVideoPanelProps {
  /** Compact rendering for a collapsed pane: no header. */
  collapsed?: boolean;
  className?: string;
}

/**
 * Console pane rendering the bot's video track, with a quiet placeholder
 * until one exists. Must be rendered inside a PipecatClientProvider.
 */
export function ConsoleBotVideoPanel({
  collapsed = false,
  className,
}: ConsoleBotVideoPanelProps) {
  const track = usePipecatClientMediaTrack("video", "bot");

  return (
    <ConsolePanel
      className={className}
      data-slot="console-bot-video-panel"
      data-state={collapsed ? "collapsed" : "expanded"}
    >
      {!collapsed && (
        <ConsolePanelHeader>
          <ConsolePanelTitle>Bot video</ConsolePanelTitle>
        </ConsolePanelHeader>
      )}
      <ConsolePanelContent className="flex items-center justify-center overflow-hidden">
        {track ? (
          <PipecatClientVideo
            participant="bot"
            fit="contain"
            className="h-full w-full rounded-md"
          />
        ) : (
          <div className="text-muted-foreground flex aspect-video w-full flex-col items-center justify-center gap-1 text-xs">
            <VideoOffIcon className="size-4" />
            {!collapsed && <span>No video</span>}
          </div>
        )}
      </ConsolePanelContent>
    </ConsolePanel>
  );
}
