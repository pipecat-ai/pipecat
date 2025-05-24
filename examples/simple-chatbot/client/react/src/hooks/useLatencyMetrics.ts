import { useState, useCallback, useRef, useEffect } from 'react';

interface LatencyEvent {
  timestamp: number;
  event: 'user_started_speaking' | 'user_stopped_speaking' | 'bot_started_speaking' | 'bot_stopped_speaking';
  sessionId?: string; // Make optional initially
}

interface LatencyMetric {
  type: 'response_latency' | 'interruption_latency' | 'user_latency';
  value: number;
  timestamp: string;
  sessionId?: string; // Make optional initially
  details: any;
}

interface ComputedMetrics {
  response_latency: {
    count: number;
    total: number;
    average: number;
    last: number;
  };
  interruption_latency: {
    count: number;
    total: number;
    average: number;
    last: number;
  };
  user_latency: {
    count: number;
    total: number;
    average: number;
    last: number;
  };
}

export const useLatencyMetrics = (sessionId: string, onClearComparison?: () => void) => {
  const [latencyMetrics, setLatencyMetrics] = useState<LatencyMetric[]>([]);
  const [computedMetrics, setComputedMetrics] = useState<ComputedMetrics>({
    response_latency: { count: 0, total: 0, average: 0, last: 0 },
    interruption_latency: { count: 0, total: 0, average: 0, last: 0 },
    user_latency: { count: 0, total: 0, average: 0, last: 0 }
  });
  
  const eventsRef = useRef<LatencyEvent[]>([]);
  const sessionIdRef = useRef<string>(sessionId);
  const metricsSentRef = useRef<boolean>(false);
  const botSpeakingRef = useRef(false);
  const isActiveSessionRef = useRef<boolean>(false);
  
  // Calculate computed metrics whenever latencyMetrics changes
  useEffect(() => {
    const responseMetrics = latencyMetrics.filter(m => m.type === 'response_latency');
    const interruptionMetrics = latencyMetrics.filter(m => m.type === 'interruption_latency');
    const userMetrics = latencyMetrics.filter(m => m.type === 'user_latency');
    
    const responseTotal = responseMetrics.reduce((sum, m) => sum + m.value, 0);
    const interruptionTotal = interruptionMetrics.reduce((sum, m) => sum + m.value, 0);
    const userTotal = userMetrics.reduce((sum, m) => sum + m.value, 0);
    
    setComputedMetrics({
      response_latency: {
        count: responseMetrics.length,
        total: responseTotal,
        average: responseMetrics.length > 0 ? Math.round(responseTotal / responseMetrics.length) : 0,
        last: responseMetrics.length > 0 ? responseMetrics[responseMetrics.length - 1].value : 0
      },
      interruption_latency: {
        count: interruptionMetrics.length,
        total: interruptionTotal,
        average: interruptionMetrics.length > 0 ? Math.round(interruptionTotal / interruptionMetrics.length) : 0,
        last: interruptionMetrics.length > 0 ? interruptionMetrics[interruptionMetrics.length - 1].value : 0
      },
      user_latency: {
        count: userMetrics.length,
        total: userTotal,
        average: userMetrics.length > 0 ? Math.round(userTotal / userMetrics.length) : 0,
        last: userMetrics.length > 0 ? userMetrics[userMetrics.length - 1].value : 0
      }
    });
  }, [latencyMetrics]);
  
  // Handle session ID changes - only assign session ID to existing metrics
  useEffect(() => {
    const newSessionId = sessionId;
    const previousSessionId = sessionIdRef.current;
    
    sessionIdRef.current = newSessionId;
    
    // Update session ID on existing metrics if we have metrics but no session ID on them
    if (newSessionId && !previousSessionId && latencyMetrics.length > 0) {
      const updatedMetrics = latencyMetrics.map(metric => ({
        ...metric,
        sessionId: newSessionId
      }));
      
      setLatencyMetrics(updatedMetrics);
      
      eventsRef.current = eventsRef.current.map(event => ({
        ...event,
        sessionId: newSessionId
      }));
    }
  }, [sessionId, latencyMetrics]);
  
  // Function to clear metrics when starting a new session
  const startNewSession = useCallback(() => {
    setLatencyMetrics([]);
    eventsRef.current = [];
    metricsSentRef.current = false;
    isActiveSessionRef.current = true;
    
    // Clear comparison data in ClientMetricsDisplay
    if (onClearComparison) {
      onClearComparison();
    }
  }, [onClearComparison]);
  
  // Function to mark session as ended (preserve metrics)
  const endSession = useCallback(() => {
    isActiveSessionRef.current = false;
  }, []);

  // Define sendMetricsDirectly first
  const sendMetricsDirectly = useCallback(async (metricsToSend: LatencyMetric[], sessionIdToUse: string) => {
    if (metricsToSend.length === 0 || !sessionIdToUse) {
      return;
    }
    
    const payload = {
      session_id: sessionIdToUse,
      metrics: metricsToSend
    };
    
    try {
      const response = await fetch(`http://localhost:7860/api/sessions/${sessionIdToUse}/latency-metrics`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      });
      
      await response.text();
    } catch (error) {
      // Error handling silently fails
    }
  }, []);
  
  // Function to send metrics on disconnect
  const sendMetricsOnDisconnect = useCallback(async () => {
    const currentSessionId = sessionIdRef.current;
    const currentMetrics = latencyMetrics;
    
    if (metricsSentRef.current) {
      return;
    }
    
    if (currentMetrics.length > 0 && currentSessionId) {
      await sendMetricsDirectly(currentMetrics, currentSessionId);
      metricsSentRef.current = true;
    }
  }, [latencyMetrics, sendMetricsDirectly]);

  const logEvent = useCallback((event: LatencyEvent['event']) => {
    const currentSessionId = sessionIdRef.current;
    const timestamp = Date.now();
    
    const eventObj: LatencyEvent = {
      timestamp,
      event,
      sessionId: currentSessionId || undefined
    };
    
    eventsRef.current.push(eventObj);
    
    if (event === 'bot_started_speaking') {
      botSpeakingRef.current = true;
    } else if (event === 'bot_stopped_speaking') {
      botSpeakingRef.current = false;
    }
    
    calculateLatencyMetrics(eventObj);
  }, []);

  const calculateLatencyMetrics = useCallback((newEvent: LatencyEvent) => {
    const events = eventsRef.current;
    const currentSessionId = sessionIdRef.current;
    
    if (newEvent.event === 'bot_started_speaking') {
      // Check if this is a response (bot_started_speaking right after user_stopped_speaking)
      const lastEvent = events[events.length - 2]; // Get previous event (current event is already pushed)
      
      if (lastEvent && lastEvent.event === 'user_stopped_speaking') {
        // This is a response latency
        const responseLatency = newEvent.timestamp - lastEvent.timestamp;
        const metric: LatencyMetric = {
          type: 'response_latency',
          value: responseLatency,
          timestamp: new Date().toISOString(),
          sessionId: currentSessionId || undefined,
          details: {
            userStopTime: lastEvent.timestamp,
            botStartTime: newEvent.timestamp
          }
        };
        
        setLatencyMetrics(prev => [...prev, metric]);
      }
    }
    
    if (newEvent.event === 'user_started_speaking') {
      // Check if this is user latency (user_started_speaking right after bot_stopped_speaking)
      const lastEvent = events[events.length - 2]; // Get previous event (current event is already pushed)
      
      if (lastEvent && lastEvent.event === 'bot_stopped_speaking') {
        // This is a user latency
        const userLatency = newEvent.timestamp - lastEvent.timestamp;
        const metric: LatencyMetric = {
          type: 'user_latency',
          value: userLatency,
          timestamp: new Date().toISOString(),
          sessionId: currentSessionId || undefined,
          details: {
            botStopTime: lastEvent.timestamp,
            userStartTime: newEvent.timestamp
          }
        };
        
        setLatencyMetrics(prev => [...prev, metric]);
      }
    }
    
    if (newEvent.event === 'bot_stopped_speaking') {
      // Check if this is ending an interruption
      // We need to find the most recent bot_started_speaking and check if there was a user_started_speaking after it
      
      // Find the most recent bot_started_speaking before this bot_stopped_speaking
      let mostRecentBotStart: LatencyEvent | null = null;
      for (let i = events.length - 2; i >= 0; i--) { // Start from second-to-last (current event is last)
        if (events[i].event === 'bot_started_speaking') {
          mostRecentBotStart = events[i];
          break;
        }
      }
      
      if (mostRecentBotStart) {
        // Look for user_started_speaking between the most recent bot_started_speaking and now
        let interruptingUserStart: LatencyEvent | null = null;
        
        for (let i = events.length - 2; i >= 0; i--) {
          const event = events[i];
          
          // Stop searching when we reach the bot_started_speaking
          if (event.timestamp <= mostRecentBotStart.timestamp) {
            break;
          }
          
          // Found user_started_speaking after bot_started_speaking
          if (event.event === 'user_started_speaking') {
            interruptingUserStart = event;
            break; // Take the most recent user_started_speaking
          }
        }
        
        // If we found an interrupting user_started_speaking, calculate interrupt latency
        if (interruptingUserStart) {
          const interruptLatency = newEvent.timestamp - interruptingUserStart.timestamp;
          const metric: LatencyMetric = {
            type: 'interruption_latency',
            value: interruptLatency,
            timestamp: new Date().toISOString(),
            sessionId: currentSessionId || undefined,
            details: {
              userInterruptTime: interruptingUserStart.timestamp,
              botStoppedTime: newEvent.timestamp,
              botStartTime: mostRecentBotStart.timestamp
            }
          };
          
          setLatencyMetrics(prev => [...prev, metric]);
        }
      }
    }
  }, []);
  
  return {
    logEvent,
    latencyMetrics,
    computedMetrics,
    sendMetricsOnDisconnect,
    startNewSession,
    endSession,
    clearMetrics: () => {
      setLatencyMetrics([]);
      setComputedMetrics({
        response_latency: { count: 0, total: 0, average: 0, last: 0 },
        interruption_latency: { count: 0, total: 0, average: 0, last: 0 },
        user_latency: { count: 0, total: 0, average: 0, last: 0 }
      });
      eventsRef.current = [];
      metricsSentRef.current = false;
      isActiveSessionRef.current = false;
      
      // Clear comparison data in ClientMetricsDisplay
      if (onClearComparison) {
        onClearComparison();
      }
    },
    isSessionActive: isActiveSessionRef.current
  };
};