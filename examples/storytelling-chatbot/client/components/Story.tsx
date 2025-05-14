import React, { useState, useEffect } from "react";
import {
  useDaily,
  useParticipantIds,
  useAppMessage,
  DailyAudio,
} from "@daily-co/daily-react";
import { IconLogout, IconLoader2 } from "@tabler/icons-react";

import VideoTile from "@/components/VideoTile";
import { Button } from "@/components/ui/button";
import UserInputIndicator from "@/components/UserInputIndicator";
import WaveText from "@/components/WaveText";

interface StoryProps {
  handleLeave: () => void;
}

const Story: React.FC<StoryProps> = ({ handleLeave }) => {
  const daily = useDaily();
  const participantIds = useParticipantIds({ filter: "remote" });
  const [storyState, setStoryState] = useState<"user" | "assistant">(
    "assistant"
  );

  useAppMessage({
    onAppMessage: (e) => {
      if (!daily || !e.data?.cue) return;

      // Determine the UI state from the cue sent by the bot
      if (e.data?.cue === "user_turn") {
        // Delay enabling local mic input to avoid feedback from LLM
        setTimeout(() => daily.setLocalAudio(true), 500);
        setStoryState("user");
      } else {
        // Uncomment the next line to mute the mic while the 
        // assistant it talking. Leave it commented to allow for interruptions
        // daily.setLocalAudio(false);
        setStoryState("assistant");
      }
    },
  });

  return (
    <div className="w-full flex flex-col flex-1 self-stretch">
      {/* Absolute elements */}
      <div className="absolute top-20 w-full text-center z-50">
        <WaveText active={storyState === "user"} />
      </div>
      <header className="flex absolute top-0 w-full z-50 p-6 justify-end">
        <Button variant="secondary" onClick={() => handleLeave()}>
          <IconLogout size={21} className="mr-2" />
          Exit
        </Button>
      </header>
      <div className="absolute inset-0 bg-gray-800 bg-opacity-90 z-10 fade-in"></div>

      {/* Static elements */}
      <div className="relative z-20 flex-1 flex items-center justify-center">
        {participantIds.length >= 1 ? (
          <VideoTile
            sessionId={participantIds[0]}
            inactive={false}
          />
        ) : (
          <span className="p-3 rounded-full bg-gray-900/60 animate-pulse">
            <IconLoader2
              size={42}
              stroke={2}
              className="animate-spin text-white z-20 self-center"
            />
          </span>
        )}
        <DailyAudio />
      </div>
      <UserInputIndicator active={true} />
    </div>
  );
};

export default Story;
