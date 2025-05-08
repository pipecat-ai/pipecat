// src/components/AudioAnalysis.tsx
import React, { useEffect, useRef } from 'react'
import WaveSurfer from 'wavesurfer.js'
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.esm.js'
import { Interval } from './LatencyTracker'

type AudioAnalysisProps = {
  playbackUrl: string
  startTime:   number    // absolute ms timestamp when the recording began
  latencies:   Interval[] 
  waveColor?:      string
  progressColor?:  string
  height?:         number
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

  useEffect(() => {
    if (!containerRef.current) return

    const regions = RegionsPlugin.create()
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor,
      progressColor,
      cursorColor: '#333',
      height,
      plugins: [ regions ],
    })
    waveSurferRef.current = ws
    ws.load(playbackUrl)

    ws.on('ready', () => {
      latencies.forEach(({ start, end }, idx) => {
        // compute relative seconds from absolute ms
        const relStart = (8000 - startTime) / 1000
        const relEnd   = (16000   - startTime) / 1000

        regions.addRegion({
          id:    `latency-${idx}`,
          start: relStart,
          end:   relEnd,
          content: `Latency: ${(end - start).toFixed(0)} ms`,
          color: 'rgba(220,38,38,0.3)',  // semi-transparent red
          drag:   false,
          resize: false,
        })
      })
    })

    return () => {
      ws.destroy()
      waveSurferRef.current = null
    }
  }, [playbackUrl, startTime, latencies, waveColor, progressColor, height])
  console.log("Latencies: ", latencies)
  return (
    <div>
      <div
        ref={containerRef}
        className="waveform-container"
        style={{ width: '100%', marginBottom: 16 }}
      />
      <ul style={{ listStyle: 'none', padding: 0, fontSize: 14 }}>
        {latencies.map(({ start, end }, i) => (
          <li key={i}>
            <strong>Turn {i + 1}:</strong>{' '}
            {((start - startTime).toFixed(0))} ms → {(end - startTime).toFixed(0)} ms (
            <em>{(end - start).toFixed(0)} ms</em>)
          </li>
        ))}
      </ul>
    </div>
  )
}
