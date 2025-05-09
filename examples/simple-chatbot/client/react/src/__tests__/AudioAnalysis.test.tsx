// src/__tests__/AudioAnalysis.test.tsx
/// <reference types="@testing-library/jest-dom" />

/**
 * Comprehensive tests for AudioAnalysis Component
 */

// Stub wavesurfer.js before imports
jest.mock('wavesurfer.js', () => ({
    __esModule: true,
    default: {
      create: jest.fn(() => {
        const events: Record<string, Function[]> = {}
        return {
          load: jest.fn(),
          on: jest.fn((evt: string, cb: Function) => {
            events[evt] = events[evt] || []
            events[evt].push(cb)
          }),
          destroy: jest.fn(),
          playPause: jest.fn(function () {
            events['play']?.forEach(cb => cb())
          }),
          seekTo: jest.fn(),
          getDuration: jest.fn(() => 5),
          _events: events,
        }
      }),
    },
  }))
  
  import React from 'react'
  import WaveSurfer from 'wavesurfer.js'
  import {
    render,
    screen,
    fireEvent,
    cleanup,
    act,
  } from '@testing-library/react'
  import { AudioAnalysis } from '../components/AudioAnalysis'
  
  // Clean up and reset mocks between tests
  afterEach(() => {
    cleanup()
    jest.clearAllMocks()
  })
  
  describe('AudioAnalysis Component', () => {
    it('Smoke: mounts and calls WaveSurfer.create', () => {
      render(
        <AudioAnalysis playbackUrl="url" startTime={0} latencies={[]} />
      )
      expect(WaveSurfer.create).toHaveBeenCalledTimes(1)
    })

  
    it('"ready" event: hides overlay, shows controls and waveform, sets duration', () => {
      render(
        <AudioAnalysis playbackUrl="url" startTime={0} latencies={[]} />
      )
      const instance = (WaveSurfer.create as jest.Mock).mock.results[0].value
      // simulate ready event inside act
      act(() => {
        instance._events['ready'][0]()
      })
  
      expect(screen.queryByText('Generating waveform...')).toBeNull()
      const downloadBtn = screen.getByRole('button', { name: /download/i })
      const playBtn = screen.getByRole('button', { name: /play/i })
      expect(downloadBtn).not.toBeDisabled()
      expect(playBtn).not.toBeDisabled()
      const waveform = document.querySelector('.waveform-container')
      expect(waveform).toHaveStyle('visibility: visible')
    })
  
    it('Play/Pause lifecycle: toggles label on play and pause', () => {
      render(
        <AudioAnalysis playbackUrl="url" startTime={0} latencies={[]} />
      )
      const instance = (WaveSurfer.create as jest.Mock).mock.results[0].value
      act(() => {
        instance._events['ready'][0]()
      })
  
      const playBtn = screen.getByRole('button', { name: /play/i })
      act(() => {
        fireEvent.click(playBtn)
      })
      // playPause stub emits play event
      expect(screen.getByRole('button', { name: /pause/i })).toBeInTheDocument()
  
      act(() => {
        instance._events['pause'][0]()
      })
      expect(screen.getByRole('button', { name: /play/i })).toBeInTheDocument()
    })
  
    it('"finish" event resets playback: seekTo(0) and shows Play', () => {
      render(
        <AudioAnalysis playbackUrl="url" startTime={0} latencies={[]} />
      )
      const instance = (WaveSurfer.create as jest.Mock).mock.results[0].value
      act(() => {
        instance._events['ready'][0]()
      })
  
      // start playback
      const playBtn = screen.getByRole('button', { name: /play/i })
      act(() => {
        fireEvent.click(playBtn)
      })
      // simulate finish inside act
      act(() => {
        instance._events['finish'][0]()
      })
  
      expect(instance.seekTo).toHaveBeenCalledWith(0)
      expect(screen.getByRole('button', { name: /play/i })).toBeInTheDocument()
    })
  
    it('Cleanup on unmount: destroy called', () => {
      const { unmount } = render(
        <AudioAnalysis playbackUrl="url" startTime={0} latencies={[]} />
      )
      const instance = (WaveSurfer.create as jest.Mock).mock.results[0].value
      unmount()
      expect(instance.destroy).toHaveBeenCalled()
    })
  
    it('Prop-change handling: recreates WaveSurfer', () => {
      const { rerender } = render(
        <AudioAnalysis playbackUrl="url1" startTime={0} latencies={[]} />
      )
      rerender(
        <AudioAnalysis playbackUrl="url2" startTime={0} latencies={[]} />
      )
      expect(WaveSurfer.create).toHaveBeenCalledTimes(2)
      const oldInstance = (WaveSurfer.create as jest.Mock).mock.results[0].value
      expect(oldInstance.destroy).toHaveBeenCalled()
    })
  
    it('Marker overlay: none before ready or duration=0', () => {
      const { container } = render(
        <AudioAnalysis playbackUrl="url" startTime={0} latencies={[{start:1000,end:1100}]} />
      )
      expect(container.querySelectorAll('div[style*="pointer-events: auto"]').length).toBe(0)
    })
  })
  