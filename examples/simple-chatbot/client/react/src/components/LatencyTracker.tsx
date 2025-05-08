// src/components/LatencyTracker.tsx
import { useRef } from 'react'
import { useRTVIClientEvent } from '@pipecat-ai/client-react'
import { RTVIEvent } from '@pipecat-ai/client-js'

// make sure you export this so App.tsx can import it
export type Interval = {
  start: number  // relative ms from recording start when user stopped speaking
  end:   number  // relative ms from recording start when bot started speaking
}

export interface LatencyTrackerProps {
  /** called once per turn with the new [start,end] interval */
  onLatency: (latency: Interval) => void
}

export function LatencyTracker({ onLatency }: LatencyTrackerProps) {
  // stamp the time when recording begins
  const recordingStartRef = useRef<number>(performance.now())
  // hold the last user-end relative timestamp
  const userEndRef = useRef<number|null>(null)

  // 1) when the user stops speaking, capture relative ms
  useRTVIClientEvent(RTVIEvent.UserStoppedSpeaking, () => {
    userEndRef.current = performance.now() - recordingStartRef.current
  })

  // 2) when the bot starts speaking, emit the interval
  useRTVIClientEvent(RTVIEvent.BotStartedSpeaking, () => {
    const ue = userEndRef.current
    if (ue != null) {
      const bs = performance.now() - recordingStartRef.current
      onLatency({ start: ue, end: bs })
      userEndRef.current = null
    }
  })

  // this component doesn't render anything
  return null
}
