"use client";

import { usePipecatClientMediaTrack } from "@pipecat-ai/client-react";
import { MicOffIcon } from "lucide-react";
import { AudioVisualizerBar } from "@/components/pipecat/audio-visualizer-bar";
import { BotAudioControl } from "@/components/pipecat/bot-audio";
import {
  ConsolePanel,
  ConsolePanelActions,
  ConsolePanelContent,
  ConsolePanelHeader,
  ConsolePanelTitle,
} from "@/components/pipecat/console/panel";
import { cn } from "@/lib/utils";

export interface ConsoleBotAudioPanelProps {
  /** Compact rendering for a collapsed pane: no header, tighter bars. */
  collapsed?: boolean;
  /** Hides the volume control in the header. Default false. */
  noControls?: boolean;
  className?: string;
}

/**
 * Console pane visualizing the bot's audio track, with a volume control in
 * the header. Shows a quiet placeholder until a bot audio track exists.
 * Must be rendered inside a PipecatClientProvider.
 */
export function ConsoleBotAudioPanel({
  collapsed = false,
  noControls = false,
  className,
}: ConsoleBotAudioPanelProps) {
  const track = usePipecatClientMediaTrack("audio", "bot");

  return (
    <ConsolePanel
      className={className}
      data-slot="console-bot-audio-panel"
      data-state={collapsed ? "collapsed" : "expanded"}
    >
      {!collapsed && (
        <ConsolePanelHeader>
          <ConsolePanelTitle>Bot audio</ConsolePanelTitle>
          {!noControls && (
            <ConsolePanelActions>
              <BotAudioControl size="icon-sm" variant="ghost" />
            </ConsolePanelActions>
          )}
        </ConsolePanelHeader>
      )}
      <ConsolePanelContent className="text-agent-foreground flex items-center justify-center overflow-hidden">
        {track ? (
          <AudioVisualizerBar
            participantType="bot"
            barCount={collapsed ? 3 : 8}
            barWidth={collapsed ? 4 : 8}
            barGap={collapsed ? 3 : 6}
            barMaxHeight={collapsed ? 32 : 96}
            className={cn("w-auto", collapsed && "opacity-80")}
          />
        ) : (
          <div className="text-muted-foreground flex flex-col items-center gap-1 text-xs">
            <MicOffIcon className="size-4" />
            {!collapsed && <span>No audio</span>}
          </div>
        )}
      </ConsolePanelContent>
    </ConsolePanel>
  );
}
