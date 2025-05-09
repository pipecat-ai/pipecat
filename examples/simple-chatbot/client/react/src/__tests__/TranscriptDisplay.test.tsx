// src/__tests__/TranscriptDisplay.test.tsx
/// <reference types="@testing-library/jest-dom" />

import React from 'react'
import { render, screen, act, cleanup } from '@testing-library/react'
import { TranscriptDisplay } from '../components/TranscriptDisplay'
import { RTVIEvent } from '@pipecat-ai/client-js'
jest.mock('../components/TranscriptDisplay.css', () => ({}))

// -------------- Mocks & Handlers --------------

let transportState: string = 'disconnected'
const handlers: Partial<Record<RTVIEvent, (data?: any) => void>> = {}

jest.mock('@pipecat-ai/client-react', () => ({
  useRTVIClientTransportState: () => transportState,
  useRTVIClientEvent: (eventType: RTVIEvent, callback: (data: any) => void) => {
    handlers[eventType] = callback
  },
}))

afterEach(() => {
  cleanup()
  jest.clearAllMocks()
  transportState = 'disconnected'
  for (const k of Object.keys(handlers)) {
    delete handlers[k as RTVIEvent]
  }
})

// -------------- Tests --------------

describe('TranscriptDisplay', () => {
  it('smoke: mounts and registers handlers', () => {
    render(<TranscriptDisplay />)
    expect(handlers[RTVIEvent.UserTranscript]).toBeDefined()
    expect(handlers[RTVIEvent.BotTranscript]).toBeDefined()
  })

  it('clears log when transport becomes connected', () => {
    const { container, rerender } = render(<TranscriptDisplay />)
    // Pre-populate the log container
    const logDiv = container.querySelector('.transcript-log')!
    logDiv.innerHTML = '<div>dummy</div>'

    // Simulate transport → connected
    act(() => {
      transportState = 'connected'
      rerender(<TranscriptDisplay />)
    })

    expect(logDiv.innerHTML).toBe('')
  })

  it('logs only final user transcripts and ignores non-final', () => {
    render(<TranscriptDisplay />)
    const logDiv = document.querySelector('.transcript-log')!

    // Non-final should be ignored
    act(() => {
      handlers[RTVIEvent.UserTranscript]!({ final: false, text: 'skip me' })
    })
    expect(logDiv.childNodes.length).toBe(0)

    // Final should be logged
    act(() => {
      handlers[RTVIEvent.UserTranscript]!({ final: true, text: 'hello' })
    })
    expect(screen.getByText('User: hello')).toBeInTheDocument()

    // Duplicate final should be ignored
    act(() => {
      handlers[RTVIEvent.UserTranscript]!({ final: true, text: 'hello' })
    })
    // still only one entry
    expect(logDiv.childNodes.length).toBe(1)
  })

  it('logs bot transcripts as they arrive', () => {
    render(<TranscriptDisplay />)
    // No initial entries
    expect(document.querySelectorAll('.transcript-log > div').length).toBe(0)

    act(() => {
      handlers[RTVIEvent.BotTranscript]!({ text: 'world' })
    })
    expect(screen.getByText('Bot: world')).toBeInTheDocument()
  })

  it('applies correct color styling for user and bot messages', () => {
    render(<TranscriptDisplay />)

    act(() => {
      handlers[RTVIEvent.UserTranscript]!({ final: true, text: 'u' })
      handlers[RTVIEvent.BotTranscript]!({ text: 'b' })
    })

    const userEntry = screen.getByText('User: u')
    const botEntry  = screen.getByText('Bot: b')

    expect(userEntry).toHaveStyle('color: #2196F3')
    expect(botEntry).toHaveStyle('color: #4CAF50')
  })
})
