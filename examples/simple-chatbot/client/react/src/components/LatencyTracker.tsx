import { useRef } from 'react'
import { useRTVIClientEvent, } from '@pipecat-ai/client-react'
import { RTVIEvent } from '@pipecat-ai/client-js'

export type Interval = {
  start: number  // absolute ms timestamp when the user finished speaking
  end:   number  // absolute ms timestamp when the bot began speaking
}

export interface LatencyTrackerProps {
  /** Called once per turn with the latency interval */
  onLatency: (interval: Interval) => void
}

export function LatencyTracker({ onLatency }: LatencyTrackerProps) {
  // hold the timestamp when the user stopped speaking
  const userEndRef = useRef<number|null>(null)

  // 1) capture when the user stops speaking
  useRTVIClientEvent(RTVIEvent.UserStoppedSpeaking, () => {
    userEndRef.current = performance.now()
  })

  // 2) listen for our server message carrying latency_ms
  useRTVIClientEvent(
    RTVIEvent.ServerMessage,
    (data: any) => {
      // guard: do we have a numeric latency_ms?
      if (data && typeof data.latency_ms === 'number') {
        const start = userEndRef.current
        if (start !== null) {
          const end = start + data.latency_ms
          onLatency({ start, end })
          // clear for next turn
          userEndRef.current = null
        }
      }
    }
  )

  return null
}
