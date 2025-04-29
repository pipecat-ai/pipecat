import {
  RTVIClientAudio,
  RTVIClientVideo,
  useRTVIClient,
  useRTVIClientTransportState,
} from "@pipecat-ai/client-react";
import { RTVIProvider } from "./providers/RTVIProvider";
import { ConnectButton } from "./components/ConnectButton";
import { StatusDisplay } from "./components/StatusDisplay";
import { DebugDisplay } from "./components/DebugDisplay";
import "./App.css";

function BotVideo() {
  const transportState = useRTVIClientTransportState();
  const isConnected = transportState !== "disconnected";

  return (
    <div className="bot-container">
      <div className="video-container">
        {isConnected && <RTVIClientVideo participant="bot" fit="cover" />}
      </div>
    </div>
  );
}

function AppContent() {
  const client = useRTVIClient();
  return (
    <div className="app">
      <div className="status-bar">
        <StatusDisplay />
        <ConnectButton />
        <div
          className="controls"
          onClick={async () => {
            if (!client) {
              console.error("RTVI client is not initialized");
              return;
            }
            client.action({
              service: "tts",
              action: "say",
              arguments: [
                { name: "text", value: "Hello, world!" },
                { name: "interrupt", value: false },
              ],
            });
          }}
        >
          <button className="connect-btn">Say something</button>
        </div>
      </div>

      <div className="main-content">
        <BotVideo />
      </div>

      <DebugDisplay />
      <RTVIClientAudio />
    </div>
  );
}

function App() {
  return (
    <RTVIProvider>
      <AppContent />
    </RTVIProvider>
  );
}

export default App;
