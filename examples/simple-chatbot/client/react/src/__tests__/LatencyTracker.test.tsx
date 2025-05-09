// src/__tests__/LatencyTracker.test.tsx

import React from 'react'
import { render } from '@testing-library/react'
import { LatencyTracker, Interval } from '../components/LatencyTracker'
import { RTVIEvent } from '@pipecat-ai/client-js'

// We’ll capture the two handlers so tests can fire them
const handlers: Partial<Record<RTVIEvent, (data?: any) => void>> = {}

// Mock out the hook to register our handlers
jest.mock('@pipecat-ai/client-react', () => ({
  useRTVIClientEvent: jest.fn(
    (eventType: RTVIEvent, callback: (data: any) => void) => {
      handlers[eventType] = callback
    }
  ),
}))

// Fix performance.now to a constant
beforeAll(() => {
  jest.spyOn(performance, 'now').mockReturnValue(1000)
})

beforeEach(() => {
  // Clear handler registrations & mocks
  jest.clearAllMocks()
  for (const k of Object.keys(handlers)) {
    delete handlers[k as unknown as RTVIEvent]
  }
})

describe('LatencyTracker', () => {
  it('smoke: mounts and registers two handlers', () => {
    const onLatency = jest.fn()
    render(<LatencyTracker onLatency={onLatency} />)

    // Should register one handler for each event
    expect(handlers[RTVIEvent.UserStoppedSpeaking]).toBeDefined()
    expect(handlers[RTVIEvent.ServerMessage]).toBeDefined()
  })

  it('does NOT call onLatency if ServerMessage arrives first', () => {
    const onLatency = jest.fn()
    render(<LatencyTracker onLatency={onLatency} />)

    // Fire server message with valid latency before user stopped
    handlers[RTVIEvent.ServerMessage]!({ latency_ms: 200 })

    expect(onLatency).not.toHaveBeenCalled()
  })

  it('calls onLatency(start, end) on valid cycle', () => {
    const onLatency = jest.fn<void, [Interval]>()
    render(<LatencyTracker onLatency={onLatency} />)

    // 1) user stops speaking
    handlers[RTVIEvent.UserStoppedSpeaking]!()

    // 2) server sends latency
    handlers[RTVIEvent.ServerMessage]!({ latency_ms: 300 })

    // Expect start = performance.now() = 1000
    //        end   = start + 300 = 1300
    expect(onLatency).toHaveBeenCalledTimes(1)
    expect(onLatency).toHaveBeenCalledWith({ start: 1000, end: 1300 })
  })

  it('does NOT call onLatency twice for the same cycle', () => {
    const onLatency = jest.fn<void, [Interval]>()
    render(<LatencyTracker onLatency={onLatency} />)

    // Complete one cycle
    handlers[RTVIEvent.UserStoppedSpeaking]!()
    handlers[RTVIEvent.ServerMessage]!({ latency_ms: 150 })
    expect(onLatency).toHaveBeenCalledTimes(1)

    // Fire server message again — should be ignored
    handlers[RTVIEvent.ServerMessage]!({ latency_ms: 150 })
    expect(onLatency).toHaveBeenCalledTimes(1)
  })

  it('guards against non‐numeric or missing latency_ms', () => {
    const onLatency = jest.fn<void, [Interval]>()
    render(<LatencyTracker onLatency={onLatency} />)

    handlers[RTVIEvent.UserStoppedSpeaking]!()

    // Missing field
    handlers[RTVIEvent.ServerMessage]!({})
    // Non‐numeric
    handlers[RTVIEvent.ServerMessage]!({ latency_ms: 'foo' })
    // Negative or zero is still numeric, but you can add a test if you want to guard that too

    expect(onLatency).not.toHaveBeenCalled()
  })
})
