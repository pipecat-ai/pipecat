import { useState } from "react";
import { useDaily } from "@daily-co/daily-react";

import { Alert } from "./components/alert";
import { Button } from "./components/button";
import { ArrowRight, Loader2 } from "lucide-react";
import { DeviceSelect } from "./components/DeviceSelect";
import Session from "./components/Session";

type State =
  | "idle"
  | "configuring"
  | "requesting_agent"
  | "connecting"
  | "connected"
  | "started"
  | "finished"
  | "error";

export default function App() {
  // Use Daily as our agent transport
  const daily = useDaily();

  const [state, setState] = useState<State>("idle");
  const [error, setError] = useState<string | null>(null);
  //const [room, setRoom] = useState<string | null>(null);

  async function start() {
    if (!daily) return;

    setState("requesting_agent");

    const serverUrl =
      import.meta.env.VITE_SERVER_URL || import.meta.env.BASE_URL;

    // Request a bot to join your session
    let data;

    try {
      const res = await fetch(`${serverUrl}/start_bot`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          room_url: import.meta.env.VITE_TRANSPORT_ROOM_URL || null,
        }),
      });

      data = await res.json();

      if (!res.ok) {
        setError(data.detail);
        setState("error");
      }
    } catch (e) {
      setError(
        `Unable to connect to the server at '${serverUrl}' - is it running?`
      );
      setState("error");
      return;
    }

    setState("connecting");

    await daily.join({
      url: data.room_url,
      token: data.user_token,
      videoSource: false,
      startAudioOff: true,
    });

    setState("connected");
  }

  /*async function leave() {
    await daily?.leave();
    setState("finished");
  }*/

  if (state === "error") {
    return (
      <Alert intent="danger" title="An error occurred">
        {error}
      </Alert>
    );
  }

  if (state === "connected") {
    return <Session />;
  }

  const status_text = {
    configuring: "Start",
    requesting_agent: "Requesting agent...",
    connecting: "Connecting to agent...",
  };

  if (state !== "idle") {
    return (
      <div className="card card-appear">
        <div className="card-inner">
          <h1 className="card-header">Configure your devices</h1>
          <p className="card-text">
            Please configure your microphone and speakers below
          </p>
          <DeviceSelect />
          <Button
            key="start"
            onClick={() => start()}
            disabled={state !== "configuring"}
          >
            {state !== "configuring" && <Loader2 className="animate-spin" />}
            {status_text[state as keyof typeof status_text]}
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="card card-appear">
      <div className="card-inner">
        <h1 className="card-header">Pipecat Simple Chatbot</h1>
        <p className="card-text">
          Please ensure you microphone and speakers are connected and ready to
          go
        </p>
        {import.meta.env.DEV && !import.meta.env.VITE_SERVER_URL && (
          <div>
            Warning: you have not set a server URL for local development. Please
            set <code>VITE_SERVER_URL</code> in{" "}
            <code>.env.development.local</code>
          </div>
        )}
        <Button key="next" onClick={() => setState("configuring")}>
          Next <ArrowRight />
        </Button>
      </div>
    </div>
  );
}
