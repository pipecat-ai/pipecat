import { render, act, waitFor, cleanup } from '@testing-library/react'
import { AudioRecorder } from '../components/AudioRecorder'

// --- Driveable hook mocks ---
let transportState: 'ready' | 'disconnected' | null = null
let localTrack: MediaStreamTrack | null = null
let botTrack: MediaStreamTrack | null = null

jest.mock('@pipecat-ai/client-react', () => ({
  useRTVIClientTransportState: () => transportState,
  useRTVIClientMediaTrack: (_kind: string, who: 'local' | 'bot') =>
    who === 'local' ? localTrack : botTrack,
}))

// --- Fix performance.now() ---
beforeAll(() => {
  jest.spyOn(performance, 'now').mockReturnValue(123456)
})

// --- Teardown ---
afterEach(() => {
  cleanup()
  jest.clearAllMocks()
  transportState = null
  localTrack = botTrack = null
})

// --- Grab our mocked constructor ---
const RecorderMock = (globalThis.MediaRecorder as unknown) as jest.Mock

describe('AudioRecorder (incremental)', () => {
  it('Constructs a MediaRecorder once on mount', async () => {
    // Arrange: make tracks available and transport immediately 'ready'
    localTrack = {} as any
    botTrack   = {} as any
    transportState = 'ready'

    // Act
    render(<AudioRecorder onStopRecording={jest.fn()} />)

    // Assert constructor was called
    await waitFor(() => {
      expect(RecorderMock).toHaveBeenCalledTimes(1)
    })
  })

  it('Instance has start() & stop() functions', async () => {
    // Arrange
    localTrack = {} as any
    botTrack   = {} as any
    transportState = 'ready'
    render(<AudioRecorder onStopRecording={jest.fn()} />)

    // Wait for constructor
    await waitFor(() => {
      expect(RecorderMock).toHaveBeenCalled()
    })

    // Grab the actual instance returned by our mock
    const instance = RecorderMock.mock.results[0].value
    expect(typeof instance.start).toBe('function')
    expect(typeof instance.stop).toBe('function')
  })

  it('Invokes onStopRecording(url, startTime) when recorder stops', async () => {
    // Arrange
    localTrack = {} as any
    botTrack   = {} as any
    transportState = 'ready'
    const onStop = jest.fn()
    render(<AudioRecorder onStopRecording={onStop} />)

    // Wait for and grab the instance
    await waitFor(() => expect(RecorderMock).toHaveBeenCalled())
    const instance = RecorderMock.mock.results[0].value

    // Simulate one chunk + stop
    const fakeBlob = new Blob(['audio'], { type: 'audio/webm' })
    act(() => {
      instance.ondataavailable({ data: fakeBlob })
      instance.onstop()
    })

    // onStopRecording should have been called with our URL + perf timestamp
    expect(onStop).toHaveBeenCalledWith('blob://test', 123456)
  })

  it('Cleans up by revoking the blob URL on unmount', async () => {
    // Arrange + start+stop cycle
    localTrack = {} as any
    botTrack   = {} as any
    transportState = 'ready'
    const onStop = jest.fn()
    const { unmount } = render(<AudioRecorder onStopRecording={onStop} />)

    await waitFor(() => expect(RecorderMock).toHaveBeenCalled())
    const instance = RecorderMock.mock.results[0].value
    act(() => {
      instance.ondataavailable({ data: new Blob() })
      instance.onstop()
    })

    // Confirm we created one URL
    expect(URL.createObjectURL).toHaveBeenCalledTimes(1)

    // Unmount → our cleanup effect should revoke it
    unmount()
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob://test')
  })

  it('starts the MediaRecorder when transport transitions to ready', async () => {
    const onStop = jest.fn()
  
    // 1. Arrange: tracks exist, but initial transport is "disconnected"
    localTrack = {} as any
    botTrack   = {} as any
    transportState = 'disconnected'
  
    // Mount the component
    const { rerender } = render(<AudioRecorder onStopRecording={onStop} />)
  
    // Grab the created instance
    await waitFor(() => expect(RecorderMock).toHaveBeenCalledTimes(1))
    const instance = RecorderMock.mock.results[0].value
  
    // Sanity: it should start out inactive
    expect(instance.state).toBe('inactive')
  
    // 2. Act: switch transport to 'ready' and rerender
    act(() => {
      transportState = 'ready'
      rerender(<AudioRecorder onStopRecording={onStop} />)
    })
  
    // 3. Assert: now the instance.state should be "recording"
    expect(instance.state).toBe('recording')
  })
  
  
})
