import {
  RTVIClientAudio,
  useRTVIClient,
  useRTVIClientTransportState,
} from '@pipecat-ai/client-react';
import { RTVIProvider } from './providers/RTVIProvider';
import { DebugDisplay } from './components/DebugDisplay';
import { Waveform } from './components/Waveform';
import StaticMemoryPanel from './components/StaticMemoryPanel';
import './App.css';
import Navbar from './components/Navbar';
import { ScrollArea } from './components/ui/scroll-area';
import { InteractiveHoverButton } from "@/components/magicui/interactive-hover-button";
import { useState, useEffect } from 'react';
import { CaretRight, Memory } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import { Button } from "./components/ui/button";

function AppContent() {
  const client = useRTVIClient();
  const transportState = useRTVIClientTransportState();
  const isConnected = ['connected', 'ready'].includes(transportState);
  const isConnecting = ['connecting', 'disconnecting'].includes(transportState);
  const [speakingState, setSpeakingState] = useState<'user' | 'assistant' | 'system' | 'idle'>('idle');
  const [leftPanelCollapsed, setLeftPanelCollapsed] = useState(true);
  const [rightPanelCollapsed, setRightPanelCollapsed] = useState(false);
  const [isFirstConnectionConnected, setIsFirstConnectionConnected] = useState(false);

  const handleReset = () => {
    setIsFirstConnectionConnected(false);
  }

  // Simulated memory counts - replace with actual counts from your components
  const leftMemoryCount = 3;
  const rightMemoryCount = 3;

  const handleConnect = async () => {
    if (!client) {
      console.error('RTVI client is not initialized');
      return;
    }

    try {
      if (isConnected) {
        await client.disconnect();
      } else {
        await client.connect();
      }
      setIsFirstConnectionConnected(true);
    } catch (error) {
      console.error('Connection error:', error);
    }
  };

  // Listen for message events from DebugDisplay
  useEffect(() => {
    const handleMessage = (event: CustomEvent<{ type: 'user' | 'assistant' | 'system' }>) => {
      setSpeakingState(event.detail.type);
      // Reset to idle after animation
      setTimeout(() => setSpeakingState('idle'), 2000);
    };

    window.addEventListener('newMessage', handleMessage as EventListener);
    return () => window.removeEventListener('newMessage', handleMessage as EventListener);
  }, []);

  return (
    <div className="app">
      <div className="mb-24">
        <Navbar onReset={handleReset} />
      </div>

      {!isConnected && !isFirstConnectionConnected ? (
        <div className="h-[calc(100vh-12rem)] flex items-center justify-center">
          <InteractiveHoverButton
            onClick={handleConnect}
            disabled={isConnecting}
            className="px-8 py-4 text-xl animate-pulse-shadow"
          >
            {isConnecting ? 'Connecting...' : 'Connect'}
          </InteractiveHoverButton>
        </div>
      ) : (
        <div className="flex px-4 relative">
          {/* Main Content */}
          <div className={cn(
            "transition-all duration-300 ease-in-out flex-1 relative",
            leftPanelCollapsed && rightPanelCollapsed ? "mx-4" : "",
            !leftPanelCollapsed && !rightPanelCollapsed ? "mx-4" : "",
            !leftPanelCollapsed && rightPanelCollapsed ? "ml-4 mr-4" : "",
            leftPanelCollapsed && !rightPanelCollapsed ? "ml-4 mr-4" : ""
          )}>
            <ScrollArea className="w-full h-[calc(100vh-10rem)] rounded-xl bg-zinc-50 dark:bg-neutral-950">
              <DebugDisplay onNewMessage={(type) => setSpeakingState(type)} />
            </ScrollArea>
            
            <div className="absolute bottom-6 left-1/2 -translate-x-1/2">
              <div className="p-4 rounded-full shadow-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 transition-all duration-300 hover:shadow-xl">
                <Waveform speakingState={speakingState} />
              </div>
            </div>
          </div>

          {/* Right Panel */}
          <div className={cn(
            "transition-all duration-300 ease-in-out group",
            rightPanelCollapsed ? "w-12" : "w-[25rem]"
          )}>
            {/* Collapsed State with Memory Icons */}
            <div className={cn(
              "absolute top-0 right-0 h-full w-12 flex flex-col items-center pt-4 gap-2",
              rightPanelCollapsed ? "opacity-100" : "opacity-0 pointer-events-none"
            )}>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setRightPanelCollapsed(false)}
                className="h-10 w-10 rounded-xl bg-violet-100 dark:bg-violet-900/30 hover:bg-violet-200 dark:hover:bg-violet-900/50"
              >
                <CaretRight
                  className="h-5 w-5 text-violet-600 dark:text-violet-400 rotate-180"
                />
              </Button>
              <div className="flex flex-col items-center gap-1 mt-2">
                <Memory className="h-5 w-5 text-violet-500" weight="duotone" />
                <span className="text-xs font-medium text-violet-600 dark:text-violet-400">{rightMemoryCount}</span>
              </div>
            </div>

            {/* Expanded State */}
            <div className={cn(
              "transition-all duration-300 w-[25rem]",
              rightPanelCollapsed ? "opacity-0 pointer-events-none" : "opacity-100"
            )}>
              <div className="relative">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setRightPanelCollapsed(true)}
                  className="absolute left-2 top-2 z-10 h-10 w-10 rounded-xl bg-violet-100 dark:bg-violet-900/30 hover:bg-violet-200 dark:hover:bg-violet-900/50"
                >
                  <CaretRight
                    className="h-5 w-5 text-violet-600 dark:text-violet-400"
                  />
                </Button>
                <ScrollArea>
                  <StaticMemoryPanel />
                </ScrollArea>
              </div>
            </div>
          </div>
        </div>
      )}
      
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
