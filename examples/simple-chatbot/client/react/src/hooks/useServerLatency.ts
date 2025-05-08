// src/hooks/useServerLatency.ts
import { useState, useEffect } from 'react'
import {
  useRTVIClientEvent,
  useRTVIClient,
} from '@pipecat-ai/client-react'
import { RTVIEvent, ServerMessageData } from '@pipecat-ai/client-js'

export interface ServerLatency {
  turnId:    number
  latencyMs: number
}

export function useServerLatency() {
  const [latencies, setLatencies] = useState<ServerLatency[]>([])

  // Listen to the raw serverMessage event
  useRTVIClientEvent(
    RTVIEvent.ServerMessage,
    (msg: ServerMessageData) => {
      // msg.type is the frame name you pushed, e.g. "LatencyFrame"
    //   if (msg.type !== 'LatencyFrame') return

    //   const { turn_id: turnId, latency_ms: latencyMs } = msg.data as {
    //     turn_id: number
    //     latency_ms: number
    //   }

    //   setLatencies((prev) => [...prev, { turnId, latencyMs }])
        console.log('ServerMessage', msg)
    }
  )

  return latencies
}
