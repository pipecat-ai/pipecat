/**
 * Copyright (c) 2024–2025, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

/**
 * RTVI Client Implementation
 *
 * This client connects to an RTVI-compatible bot server using WebRTC (via Daily).
 * It handles audio/video streaming and manages the connection lifecycle.
 *
 * Requirements:
 * - A running RTVI bot server (defaults to http://localhost:7860)
 * - The server must implement the /connect endpoint that returns Daily.co room credentials
 * - Browser with WebRTC support
 */

import {
  Participant,
  RTVIClient,
  RTVIClientOptions,
  RTVIEvent,
} from '@pipecat-ai/client-js';
import { DailyTransport } from '@pipecat-ai/daily-transport';
import SoundUtils from './util/soundUtils';
import { InstantVoiceHelper } from './util/instantVoiceHelper';

/**
 * InstantVoiceClient handles the connection and media management for a real-time
 * voice and video interaction with an AI bot.
 */
class InstantVoiceClient {
  private declare rtviClient: RTVIClient;
  private connectBtn: HTMLButtonElement | null = null;
  private disconnectBtn: HTMLButtonElement | null = null;
  private statusSpan: HTMLElement | null = null;
  private bufferingAudioSpan: HTMLElement | null = null;
  private debugLog: HTMLElement | null = null;
  private botAudio: HTMLAudioElement;
  private declare startTime: number;

  constructor() {
    this.botAudio = document.createElement('audio');
    this.botAudio.autoplay = true;
    document.body.appendChild(this.botAudio);
    this.setupDOMElements();
    this.setupEventListeners();
    this.initializeRTVIClient();
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
    this.bufferingAudioSpan = document.getElementById('buffering-status');
    this.debugLog = document.getElementById('debug-log');
  }

  /**
   * Set up event listeners for connect/disconnect buttons
   */
  private setupEventListeners(): void {
    this.connectBtn?.addEventListener('click', () => this.connect());
    this.disconnectBtn?.addEventListener('click', () => this.disconnect());
  }

  private initializeRTVIClient(): void {
    const RTVIConfig: RTVIClientOptions = {
      transport: new DailyTransport({
        bufferLocalAudioUntilBotReady: true,
      }),
      params: {
        // The baseURL and endpoint of your bot server that the client will connect to
        baseUrl: 'http://localhost:7860',
        endpoints: { connect: '/connect' },
      },
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
          this.updateBufferingStatus('No');
          if (this.connectBtn) this.connectBtn.disabled = false;
          if (this.disconnectBtn) this.disconnectBtn.disabled = true;
          this.log('Client disconnected');
        },
        onBotConnected: (participant: Participant) => {
          this.log(`onBotConnected, timeTaken: ${Date.now() - this.startTime}`);
        },
        onBotReady: (data) => {
          this.log(`onBotReady, timeTaken: ${Date.now() - this.startTime}`);
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
    this.rtviClient.registerHelper(
      'transport',
      new InstantVoiceHelper({
        callbacks: {
          onAudioBufferingStarted: () => {
            SoundUtils.beep();
            this.updateBufferingStatus('Yes');
            this.log(
              `onMicCaptureStarted, timeTaken: ${Date.now() - this.startTime}`
            );
          },
          onAudioBufferingStopped: () => {
            this.updateBufferingStatus('No');
            this.log(
              `onMicCaptureStopped, timeTaken: ${Date.now() - this.startTime}`
            );
          },
        },
      })
    );
    this.setupTrackListeners();
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
   * Update the connection status display
   */
  private updateBufferingStatus(status: string): void {
    if (this.bufferingAudioSpan) {
      this.bufferingAudioSpan.textContent = status;
    }
    this.log(`BufferingStatus: ${status}`);
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
      this.startTime = Date.now();
      this.log('Connecting to bot...');
      await this.rtviClient.connect();
    } catch (error) {
      this.log(`Error connecting: ${(error as Error).message}`);
      this.updateStatus('Error');
      this.updateBufferingStatus('No');

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
    try {
      await this.rtviClient.disconnect();
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

declare global {
  interface Window {
    InstantVoiceClient: typeof InstantVoiceClient;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.InstantVoiceClient = InstantVoiceClient;
  new InstantVoiceClient();
});
