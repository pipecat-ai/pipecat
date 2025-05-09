// jest.setup.ts
import '@testing-library/jest-dom'
import { jest } from '@jest/globals'

// ——— Mock performance.now() first, so tests are deterministic ———
jest
  .spyOn(performance, 'now')
  .mockReturnValue(123456)

// ——— Mock MediaRecorder with real start/stop behavior ———
const MockRecorder = jest.fn().mockImplementation(function (stream, options) {
  // initialize instance state
  ;(this as any).state = 'inactive'
  ;(this as any).ondataavailable = null
  ;(this as any).onstop = null

  // start() flips to "recording"
  this.start = jest.fn().mockImplementation(function () {
    ;(this as any).state = 'recording'
  })

  // stop() flips back to "inactive"
  this.stop = jest.fn().mockImplementation(function () {
    ;(this as any).state = 'inactive'
    // simulate a data event so onstop handler can fire if needed
    this.ondataavailable?.({ data: new Blob() })
    this.onstop?.()
  })

  return this
})
Object.defineProperty(globalThis, 'MediaRecorder', {
  writable: true,
  configurable: true,
  value: MockRecorder,
})

// ——— Mock AudioContext & its nodes ———
class MockAudioContext {
  createMediaStreamDestination = jest.fn(() => ({ stream: {} as MediaStream }))
  createMediaStreamSource = jest.fn(() => ({ connect: jest.fn() }))
  close = jest.fn()
}
Object.defineProperty(globalThis, 'AudioContext', {
  writable: true,
  configurable: true,
  value: MockAudioContext,
})

// ——— Mock MediaStream so `new MediaStream([track])` works ———
class MockMediaStream {
  tracks: any[]
  constructor(tracks: any[]) {
    this.tracks = tracks
  }
}
Object.defineProperty(globalThis, 'MediaStream', {
  writable: true,
  configurable: true,
  value: MockMediaStream,
})

// ——— Mock URL helpers ———
Object.defineProperty(globalThis.URL, 'createObjectURL', {
  writable: true,
  configurable: true,
  value: jest.fn(() => 'blob://test'),
})
Object.defineProperty(globalThis.URL, 'revokeObjectURL', {
  writable: true,
  configurable: true,
  value: jest.fn(),
})

// ——— (Optional) Stub requestAnimationFrame to silence warnings ———
Object.defineProperty(globalThis, 'requestAnimationFrame', {
  writable: true,
  configurable: true,
  value: (cb: FrameRequestCallback) => setTimeout(cb, 0),
})
