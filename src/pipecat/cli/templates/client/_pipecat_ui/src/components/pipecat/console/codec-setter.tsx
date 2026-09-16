"use client";

import { usePipecatClient } from "@pipecat-ai/client-react";
import { useEffect } from "react";

export interface SmallWebRTCCodecSetterProps {
  /** Preferred audio codec, e.g. "opus". Default "default" (no override). */
  audioCodec?: string;
  /** Preferred video codec, e.g. "VP8". Default "default" (no override). */
  videoCodec?: string;
}

/**
 * Headless helper that applies audio/video codec preferences to a
 * SmallWebRTC transport. Renders nothing; transports without codec setters
 * are left untouched, so it works whether the transport came from a factory
 * or a registered loader. Must be rendered inside a PipecatClientProvider.
 */
export function SmallWebRTCCodecSetter({
  audioCodec = "default",
  videoCodec = "default",
}: SmallWebRTCCodecSetterProps) {
  const client = usePipecatClient();

  useEffect(() => {
    // Structural check: the transport package's types can't be named here
    // (transports are optional installs).
    const transport = client?.transport as
      | {
          setAudioCodec?: (codec: string) => void;
          setVideoCodec?: (codec: string) => void;
        }
      | undefined;
    if (typeof transport?.setAudioCodec === "function") {
      transport.setAudioCodec(audioCodec);
    }
    if (typeof transport?.setVideoCodec === "function") {
      transport.setVideoCodec(videoCodec);
    }
  }, [audioCodec, client, videoCodec]);

  return null;
}
