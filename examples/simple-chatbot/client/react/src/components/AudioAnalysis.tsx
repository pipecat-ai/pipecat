// src/components/AudioAnalysis.tsx
import React, { useEffect, useRef, useState} from 'react'
import WaveSurfer from 'wavesurfer.js'
import { Interval } from './LatencyTracker'

/**
 * Props for the AudioAnalysis component
 */
type AudioAnalysisProps = {
  playbackUrl: string           // URL of the audio file to play back
  startTime:   number           // absolute timestamp (ms) when the recording began
  latencies:   Interval[]       // array of latency intervals between user end and bot start
  waveColor?:     string        // optional color for the waveform
  progressColor?: string        // optional color for the played portion of the waveform
  height?:        number        // optional height of the waveform container
}

/**
 * AudioAnalysis renders a waveform of the audio, playback controls,
 * and tappable latency markers showing intervals annotated in time.
 */
export const AudioAnalysis: React.FC<AudioAnalysisProps> = ({
  playbackUrl,
  startTime,
  latencies,
  waveColor = '#ddd',
  progressColor = '#4F46E5',
  height = 80,
}) => {
    // Ref to the DOM element that will contain the waveform
    const containerRef = useRef<HTMLDivElement>(null)
    // Ref to the WaveSurfer instance for controlling playback
    const waveSurferRef = useRef<WaveSurfer|null>(null)

    // Loading state while waveform is being generated
    const [loading, setLoading] = useState(true)
    // Playback state: playing vs paused
    const [isPlaying, setIsPlaying] = useState(false)
    // Duration of the audio in seconds, set when WaveSurfer is ready
    const [duration, setDuration] = useState<number>(0)
    // Index of the currently active latency marker (for tooltip), or null
    const [activeMarker, setActiveMarker] = useState<number | null>(null)

    /**
     * Initialize WaveSurfer when component mounts or when playbackUrl/config changes.
     * Sets up event handlers and cleans up on unmount.
     */
    useEffect(() => {
        if (!containerRef.current) return

        // Create a new WaveSurfer instance with provided colors and height
        const ws = WaveSurfer.create({
          container: containerRef.current,
          waveColor,
          progressColor,
          cursorColor: '#333',     // caret color when hovering waveform
          height,
        })
        waveSurferRef.current = ws

        // Load the audio from the URL
        ws.load(playbackUrl)

        // When waveform is generated and audio metadata loaded
        ws.on('ready', () => {
          setLoading(false)            // hide loading overlay
          setDuration(ws.getDuration()) // store total duration
        })

        // Update playback state on play/pause/finish
        ws.on('play', () => setIsPlaying(true))
        ws.on('pause', () => setIsPlaying(false))
        ws.on('finish', () => {
          setIsPlaying(false)
          ws.seekTo(0)                // rewind to start after finishing
        })

        // Cleanup WaveSurfer instance on unmount or props change
        return () => {
          ws.destroy()
          waveSurferRef.current = null
        }
    }, [playbackUrl, waveColor, progressColor, height])

  const togglePlayback = () => {
    const ws = waveSurferRef.current
    if (!ws) return
    ws.playPause()
  }

  return (
    <div style={{ position: 'relative', width: '100%' }}>
      {loading && (
        <div
          style={{
            position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            backgroundColor: 'rgba(255,255,255,0.8)', zIndex: 10,
            fontSize: '1.2rem', fontWeight: 'bold',
          }}
        >Generating waveform...</div>
      )}
      <a href={playbackUrl} download="call-recording.webm">
              <button type="button" style={{
          marginBottom: 8, marginRight:6, padding: '8px 16px', borderRadius: 4,
          border: '1px solid #ccc', backgroundColor: '#fff',
          cursor: loading ? 'not-allowed' : 'pointer',
        }}>Download</button>
      </a>
      {/* Play/Pause Button */}
      <button
        onClick={togglePlayback}
        disabled={loading}
        style={{
          marginBottom: 8, padding: '8px 16px', borderRadius: 4,
          border: '1px solid #ccc', backgroundColor: '#fff',
          cursor: loading ? 'not-allowed' : 'pointer',
        }}
      >{isPlaying ? 'Pause' : 'Play'}</button>

      {/* Waveform Container */}
      <div
        ref={containerRef}
        className="waveform-container"
        style={{ width: '100%', height, visibility: loading ? 'hidden' : 'visible' }}
      />

      {/* Overlay latency markers as tappable dots */}
      {!loading && duration > 0 && (
        <div
          style={{ position: 'absolute', top: 0, left: 0, width: '100%', height, pointerEvents: 'none' }}
        >
          {latencies.map(({ start, end }, idx) => {
            const relStartSec = (start - startTime) / 1000
            const leftPercent = (relStartSec / duration) * 100
            const latencyMs = end - start
            const isActive = activeMarker === idx

            return (
              <React.Fragment key={idx}>
                <div
                  onClick={() => setActiveMarker(isActive ? null : idx)}
                  onTouchStart={() => setActiveMarker(isActive ? null : idx)}
                  style={{
                    position: 'absolute', left: `${leftPercent}%`, top: '102%', zIndex: 20,
                    transform: 'translate(-50%, -50%)', width: 10, height: 10,
                    borderRadius: '50%', backgroundColor: 'rgba(220,38,38, 1)',
                    pointerEvents: 'auto',
                    cursor: 'pointer',
                    touchAction: 'manipulation',
                  }}
                />

                {isActive && (
                  <div
                    style={{
                      position: 'absolute', left: `${leftPercent}%`, top: '150%', zIndex: 30,
                      transform: 'translate(-50%,-100%)',
                      padding: '4px 8px', backgroundColor: '#fff',
                      border: '1px solid #ccc', borderRadius: 4,
                      whiteSpace: 'nowrap', pointerEvents: 'none',
                      fontSize: '0.875rem',
                    }}
                  >
                    {`${latencyMs.toFixed(1)} ms`}
                  </div>
                )}
              </React.Fragment>
            )
          })}
        </div>
      )}
    </div>
  )
}
