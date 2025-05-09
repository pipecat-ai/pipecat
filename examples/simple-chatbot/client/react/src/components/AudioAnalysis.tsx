// src/components/AudioAnalysis.tsx
import React, { useEffect, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import { Interval } from './LatencyTracker'

type AudioAnalysisProps = {
  playbackUrl: string
  startTime:   number       // absolute ms timestamp when recording began
  latencies:   Interval[]   // latency intervals between user end and bot start
  waveColor?:     string
  progressColor?: string
  height?:        number
}

export const AudioAnalysis: React.FC<AudioAnalysisProps> = ({
  playbackUrl,
  startTime,
  latencies,
  waveColor = '#ddd',
  progressColor = '#4F46E5',
  height = 80,
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const waveSurferRef = useRef<WaveSurfer|null>(null)
  const [loading, setLoading] = useState(true)
  const [isPlaying, setIsPlaying] = useState(false)
  const [duration, setDuration] = useState<number>(0)
  const [activeMarker, setActiveMarker] = useState<number | null>(null)

  useEffect(() => {
    if (!containerRef.current) return

    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor,
      progressColor,
      cursorColor: '#333',
      height,
    })
    waveSurferRef.current = ws

    ws.load(playbackUrl)

    ws.on('ready', () => {
      setLoading(false)
      setDuration(ws.getDuration())
    })

    ws.on('play', () => setIsPlaying(true))
    ws.on('pause', () => setIsPlaying(false))
    ws.on('finish', () => {
      setIsPlaying(false)
      ws.seekTo(0)
    })

    return () => {
      ws.destroy()
      waveSurferRef.current = null
    }
  }, [playbackUrl, waveColor, height])

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
