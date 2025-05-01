"use client";

import { RTVIClient } from "@pipecat-ai/client-js";
import { DailyTransport } from "@pipecat-ai/daily-transport";
import { RTVIClientProvider } from "@pipecat-ai/client-react";
import { PropsWithChildren, useEffect, useState, useRef } from "react";
import { useConfigurationSettings } from "@/contexts/Configuration";

// Get the API base URL from environment variables
// Default to "/api" if not specified
// "/api" is the default for Next.js API routes and used
// for the Pipecat Cloud deployed agent
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "/api";

console.log("Using API base URL:", API_BASE_URL);

export function RTVIProvider({ children }: PropsWithChildren) {
  const [client, setClient] = useState<RTVIClient | null>(null);
  const config = useConfigurationSettings();
  const clientCreated = useRef(false);

  useEffect(() => {
    // Only create the client once
    if (clientCreated.current) return;

    const transport = new DailyTransport();

    const rtviClient = new RTVIClient({
      transport,
      params: {
        baseUrl: API_BASE_URL,
        endpoints: {
          connect: "/connect",
        },
        requestData: {
          personality: config.personality,
        },
      },
      enableMic: true,
      enableCam: false,
    });

    setClient(rtviClient);
    clientCreated.current = true;

    // Cleanup when component unmounts
    return () => {
      if (rtviClient) {
        rtviClient.disconnect().catch((err) => {
          console.error("Error disconnecting client:", err);
        });
      }
      clientCreated.current = false;
    };
  }, []);

  // Update the connectParams when config changes
  useEffect(() => {
    if (!client) return;

    // Update the connect params without recreating the client
    client.params.requestData = {
      personality: config.personality,
    };
  }, [client, config.personality]);

  if (!client) {
    return null;
  }

  return <RTVIClientProvider client={client}>{children}</RTVIClientProvider>;
}
