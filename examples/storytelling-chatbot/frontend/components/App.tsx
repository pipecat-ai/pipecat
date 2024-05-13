"use client";

import React, { useState } from "react";

import { useDaily } from "@daily-co/daily-react";
import Setup from "./Setup";
import Story from "./Story";

type State =
  | "idle"
  | "connecting"
  | "connected"
  | "started"
  | "finished"
  | "error";

export default function Call() {
  const daily = useDaily();

  const [state, setState] = useState<State>("idle");
  const [room, setRoom] = useState<string | null>(null);

  async function start() {
    setState("connecting");

    if (!daily) return;

    // Create a new room for the story session
    try {
      const response = await fetch("/create", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          room_url: process.env.NEXT_PUBLIC_ROOM_URL || null,
        }),
      });

      const { room_url, token } = await response.json();

      // Keep a reference to the room url for later
      setRoom(room_url);

      // Join the WebRTC session
      await daily.join({
        url: room_url,
        token,
        videoSource: false,
        startAudioOff: true,
      });

      setState("connected");

      // Disable local audio, the bot will say hello first
      daily.setLocalAudio(false);

      // Start the bot
      const resp = await fetch("/start", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          room_url,
        }),
      });

      setState("started");
    } catch (error) {
      setState("error");
      leave();
    }
  }

  async function leave() {
    await daily?.leave();
    setState("finished");
  }

  if (state === "error") {
    return <div>An Error occured</div>;
  }

  if (state === "started") {
    return <Story handleLeave={() => leave()} />;
  }

  return <Setup handleStart={() => start()} />;
}
