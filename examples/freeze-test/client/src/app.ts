/**
 * Copyright (c) 2024–2025, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

/**
 * RTVI Client Implementation
 *
 * This client connects to an RTVI-compatible bot server using WebSocket.
 *
 * Requirements:
 * - A running RTVI bot server (defaults to http://localhost:7860)
 */

import {
  RTVIClient,
  RTVIClientOptions,
  RTVIEvent,
} from '@pipecat-ai/client-js';
import {
  ProtobufFrameSerializer,
  WebSocketTransport,
} from '@pipecat-ai/websocket-transport';

class RecordingSerializer extends ProtobufFrameSerializer {
  private lastTimestamp: number | null = null;
  private recordingAudioToSend: boolean = false;
  private _recordedAudio: { data: ArrayBuffer; delay: number }[] = [];

  public startRecording() {
    this.recordingAudioToSend = true;
    this._recordedAudio = [];
    this.lastTimestamp = null;
  }

  public stopRecording() {
    this.recordingAudioToSend = false;
  }

  // @ts-ignore
  serializeAudio(
    data: ArrayBuffer,
    sampleRate: number,
    numChannels: number
  ): Uint8Array | null {
    if (this.recordingAudioToSend) {
      const now = Date.now();
      // Compute delay since last packet
      const delay = this.lastTimestamp ? now - this.lastTimestamp : 0;
      this.lastTimestamp = now;
      // Save audio chunk and delay
      this._recordedAudio.push({ data, delay });
      return null;
    } else {
      return super.serializeAudio(data, sampleRate, numChannels);
    }
  }

  public get recordedAudio() {
    return this._recordedAudio;
  }
}

class WebsocketClientApp {
  private ENABLE_RECORDING_MODE = false;
  private RECORDING_TIME_MS = 10000;

  private rtviClient: RTVIClient | null = null;
  private connectBtn: HTMLButtonElement | null = null;
  private disconnectBtn: HTMLButtonElement | null = null;
  private statusSpan: HTMLElement | null = null;
  private debugLog: HTMLElement | null = null;
  private botAudio: HTMLAudioElement;

  private declare websocketTransport: WebSocketTransport;
  private sendRecordedAudio: boolean = false;
  private declare recordingSerializer: RecordingSerializer;

  private playBtn: HTMLButtonElement | null = null;
  private stopBtn: HTMLButtonElement | null = null;

  constructor() {
    this.botAudio = document.createElement('audio');
    this.botAudio.autoplay = true;
    //this.botAudio.playsInline = true;
    document.body.appendChild(this.botAudio);

    this.setupDOMElements();
    this.setupEventListeners();
  }

  /**
   * Set up references to DOM elements and create necessary media elements
   */
  private setupDOMElements(): void {
    this.connectBtn = document.getElementById(
      'connect-btn'
    ) as HTMLButtonElement;
    this.disconnectBtn = document.getElementById(
      'disconnect-btn'
    ) as HTMLButtonElement;
    this.statusSpan = document.getElementById('connection-status');
    this.debugLog = document.getElementById('debug-log');
    this.playBtn = document.getElementById('play-btn') as HTMLButtonElement;
    this.stopBtn = document.getElementById('stop-btn') as HTMLButtonElement;
  }

  /**
   * Set up event listeners for connect/disconnect buttons
   */
  private setupEventListeners(): void {
    this.connectBtn?.addEventListener('click', () => this.connect());
    this.disconnectBtn?.addEventListener('click', () => this.disconnect());
    this.playBtn?.addEventListener('click', () =>
      this.startSendingRecordedAudio()
    );
    this.stopBtn?.addEventListener('click', () =>
      this.stopSendingRecordedAudio()
    );
  }

  /**
   * Add a timestamped message to the debug log
   */
  private log(message: string): void {
    if (!this.debugLog) return;
    const entry = document.createElement('div');
    entry.textContent = `${new Date().toISOString()} - ${message}`;
    if (message.startsWith('User: ')) {
      entry.style.color = '#2196F3';
    } else if (message.startsWith('Bot: ')) {
      entry.style.color = '#4CAF50';
    }
    this.debugLog.appendChild(entry);
    this.debugLog.scrollTop = this.debugLog.scrollHeight;
    console.log(message);
  }

  /**
   * Update the connection status display
   */
  private updateStatus(status: string): void {
    if (this.statusSpan) {
      this.statusSpan.textContent = status;
    }
    this.log(`Status: ${status}`);
  }

  /**
   * Check for available media tracks and set them up if present
   * This is called when the bot is ready or when the transport state changes to ready
   */
  setupMediaTracks() {
    if (!this.rtviClient) return;
    const tracks = this.rtviClient.tracks();
    if (tracks.bot?.audio) {
      this.setupAudioTrack(tracks.bot.audio);
    }
  }

  /**
   * Set up listeners for track events (start/stop)
   * This handles new tracks being added during the session
   */
  setupTrackListeners() {
    if (!this.rtviClient) return;

    // Listen for new tracks starting
    this.rtviClient.on(RTVIEvent.TrackStarted, (track, participant) => {
      // Only handle non-local (bot) tracks
      if (!participant?.local && track.kind === 'audio') {
        this.setupAudioTrack(track);
      }
    });

    // Listen for tracks stopping
    this.rtviClient.on(RTVIEvent.TrackStopped, (track, participant) => {
      this.log(
        `Track stopped: ${track.kind} from ${participant?.name || 'unknown'}`
      );
    });
  }

  /**
   * Set up an audio track for playback
   * Handles both initial setup and track updates
   */
  private setupAudioTrack(track: MediaStreamTrack): void {
    this.log('Setting up audio track');
    if (
      this.botAudio.srcObject &&
      'getAudioTracks' in this.botAudio.srcObject
    ) {
      const oldTrack = this.botAudio.srcObject.getAudioTracks()[0];
      if (oldTrack?.id === track.id) return;
    }
    this.botAudio.srcObject = new MediaStream([track]);
  }

  /**
   * Initialize and connect to the bot
   * This sets up the RTVI client, initializes devices, and establishes the connection
   */
  public async connect(): Promise<void> {
    try {
      const startTime = Date.now();

      this.recordingSerializer = new RecordingSerializer();
      const ws_opts = {
        serializer: this.ENABLE_RECORDING_MODE
          ? this.recordingSerializer
          : new ProtobufFrameSerializer(),
        recorderSampleRate: 8000,
        playerSampleRate: 8000,
      };

      const RTVIConfig: RTVIClientOptions = {
        transport: new WebSocketTransport(ws_opts),
        enableMic: true,
        enableCam: false,
        callbacks: {
          onConnected: () => {
            this.updateStatus('Connected');
            if (this.connectBtn) this.connectBtn.disabled = true;
            if (this.disconnectBtn) this.disconnectBtn.disabled = false;
          },
          onDisconnected: () => {
            this.updateStatus('Disconnected');
            if (this.connectBtn) this.connectBtn.disabled = false;
            if (this.disconnectBtn) this.disconnectBtn.disabled = true;
            this.log('Client disconnected');
          },
          onBotReady: (data) => {
            this.log(`Bot ready: ${JSON.stringify(data)}`);
            this.setupMediaTracks();
          },
          onUserTranscript: (data) => {
            if (data.final) {
              this.log(`User: ${data.text}`);
            }
          },
          onBotTranscript: (data) => this.log(`Bot: ${data.text}`),
          onMessageError: (error) => console.error('Message error:', error),
          onError: (error) => console.error('Error:', error),
        },
      };
      this.rtviClient = new RTVIClient(RTVIConfig);
      this.websocketTransport = this.rtviClient.transport;
      this.setupTrackListeners();

      this.log('Initializing devices...');
      await this.rtviClient.initDevices();

      this.log('Connecting to bot...');
      await this.rtviClient.connect({
        endpoint: 'http://localhost:7860/connect',
      });

      const timeTaken = Date.now() - startTime;
      this.log(`Connection complete, timeTaken: ${timeTaken}`);

      if (this.ENABLE_RECORDING_MODE) {
        this.log(
          `Starting to recording the next ${
            this.RECORDING_TIME_MS / 1000
          }s of audio`
        );
        this.recordingSerializer.startRecording();
        await this.sleep(this.RECORDING_TIME_MS);
        this.recordingSerializer.stopRecording();
        this.log('Recording stopped');
        this.rtviClient.enableMic(false);
        this.startSendingRecordedAudio();
      }
    } catch (error) {
      this.log(`Error connecting: ${(error as Error).message}`);
      this.updateStatus('Error');
      // Clean up if there's an error
      if (this.rtviClient) {
        try {
          await this.rtviClient.disconnect();
        } catch (disconnectError) {
          this.log(`Error during disconnect: ${disconnectError}`);
        }
      }
    }
  }

  /**
   * Disconnect from the bot and clean up media resources
   */
  public async disconnect(): Promise<void> {
    if (this.rtviClient) {
      try {
        this.stopSendingRecordedAudio();
        await this.rtviClient.disconnect();
        this.rtviClient = null;
        if (
          this.botAudio.srcObject &&
          'getAudioTracks' in this.botAudio.srcObject
        ) {
          this.botAudio.srcObject
            .getAudioTracks()
            .forEach((track) => track.stop());
          this.botAudio.srcObject = null;
        }
      } catch (error) {
        this.log(`Error disconnecting: ${(error as Error).message}`);
      }
    }
  }

  private startSendingRecordedAudio() {
    this.sendRecordedAudio = true;
    if (this.playBtn) this.playBtn.disabled = true;
    if (this.stopBtn) this.stopBtn.disabled = false;
    void this.replayAudio();
  }

  private stopSendingRecordedAudio() {
    if (this.stopBtn) this.stopBtn.disabled = true;
    if (this.playBtn) this.playBtn.disabled = false;
    this.sendRecordedAudio = false;
  }

  private async replayAudio() {
    if (this.sendRecordedAudio) {
      this.log('Sending recorded audio');
      for (const chunk of this.recordingSerializer.recordedAudio) {
        await this.sleep(chunk.delay);
        this.websocketTransport.handleUserAudioStream(chunk.data);
      }
      const randomDelay = 1000 + Math.random() * (10000 - 500);
      await this.sleep(randomDelay);

      void this.replayAudio();
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

declare global {
  interface Window {
    WebsocketClientApp: typeof WebsocketClientApp;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.WebsocketClientApp = WebsocketClientApp;
  new WebsocketClientApp();
});
