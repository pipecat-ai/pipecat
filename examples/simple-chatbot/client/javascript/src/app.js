/**
 * Copyright (c) 2024–2025, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

/**
 * Pipecat Client Implementation
 *
 * This client connects to an RTVI-compatible bot server using WebRTC (via Daily).
 * It handles audio/video streaming and manages the connection lifecycle.
 *
 * Requirements:
 * - A running RTVI bot server (defaults to http://localhost:7860)
 * - The server must implement the /connect endpoint that returns Daily.co room credentials
 * - Browser with WebRTC support
 */

import { PipecatClient, RTVIEvent } from '@pipecat-ai/client-js';
import { DailyTransport } from '@pipecat-ai/daily-transport';

/**
 * ChatbotClient handles the connection and media management for a real-time
 * voice and video interaction with an AI bot.
 */
class ChatbotClient {
  constructor() {
    // Initialize client state
    this.pcClient = null;
    this.setupDOMElements();
    this.initializeClientAndTransport();
  }

  /**
   * Set up references to DOM elements and create necessary media elements
   */
  setupDOMElements() {
    // Get references to UI control elements
    this.connectBtn = document.getElementById('connect-btn');
    this.disconnectBtn = document.getElementById('disconnect-btn');
    this.statusSpan = document.getElementById('connection-status');
    this.debugLog = document.getElementById('debug-log');
    this.botVideoContainer = document.getElementById('bot-video-container');
    this.deviceSelector = document.getElementById('device-selector');
    this.micToggleBtn = document.getElementById('mic-toggle-btn');

    // Create an audio element for bot's voice output
    this.botAudio = document.createElement('audio');
    this.botAudio.autoplay = true;
    this.botAudio.playsInline = true;
    document.body.appendChild(this.botAudio);
  }

  /**
   * Set up event listeners for connect/disconnect buttons
   */
  setupEventListeners() {
    this.connectBtn.addEventListener('click', () => {
      console.log('click');
      this.connect();
    });
    this.disconnectBtn.addEventListener('click', () => this.disconnect());

    // Populate device selector
    this.pcClient.getAllMics().then((mics) => {
      console.log('Available mics:', mics);
      mics.forEach((device) => {
        const option = document.createElement('option');
        option.value = device.deviceId;
        option.textContent = device.label || `Microphone ${device.deviceId}`;
        this.deviceSelector.appendChild(option);
      });
    });
    this.deviceSelector.addEventListener('change', (event) => {
      const selectedDeviceId = event.target.value;
      console.log('Selected device ID:', selectedDeviceId);
      this.pcClient.updateMic(selectedDeviceId);
    });

    // Handle mic mute/unmute toggle
    const micToggleBtn = document.getElementById('mic-toggle-btn');

    micToggleBtn.addEventListener('click', async () => {
      if (this.pcClient.state === 'disconnected') {
        await this.pcClient.initDevices();
      } else {
        this.pcClient.enableMic(!this.pcClient.isMicEnabled);
      }
    });
  }

  updateMicToggleButton(micEnabled) {
    console.log('Mic enabled:', micEnabled, this.pcClient?.isMicEnabled);
    this.micToggleBtn.textContent = micEnabled ? 'Mute Mic' : 'Unmute Mic';
  }
  /**
   * Set up the Pipecat client and Daily transport
   */
  async initializeClientAndTransport() {
    console.log('Initializing Pipecat client and transport...');
    // Initialize the Pipecat client with a DailyTransport and our configuration
    this.pcClient = new PipecatClient({
      transport: new DailyTransport(),
      enableMic: true,
      enableCam: false,
      callbacks: {
        // Handle connection state changes
        onConnected: () => {
          this.updateStatus('Connected');
          this.connectBtn.disabled = true;
          this.disconnectBtn.disabled = false;
          this.log('Client connected');
        },
        onDisconnected: () => {
          this.updateStatus('Disconnected');
          this.connectBtn.disabled = false;
          this.disconnectBtn.disabled = true;
          this.log('Client disconnected');
          this.updateMicToggleButton(false);
        },
        // Handle transport state changes
        onTransportStateChanged: (state) => {
          this.updateStatus(`Transport: ${state}`);
          this.log(`Transport state changed: ${state}`);
          if (state === 'ready') {
            this.setupMediaTracks();
          }
        },
        // Handle bot connection events
        onBotConnected: (participant) => {
          this.log(`Bot connected: ${JSON.stringify(participant)}`);
        },
        onBotDisconnected: (participant) => {
          this.log(`Bot disconnected: ${JSON.stringify(participant)}`);
        },
        onBotReady: (data) => {
          this.log(`Bot ready: ${JSON.stringify(data)}`);
          this.setupMediaTracks();
        },
        // Transcript events
        onUserTranscript: (data) => {
          // Only log final transcripts
          if (data.final) {
            this.log(`User: ${data.text}`);
          }
        },
        onBotTranscript: (data) => {
          this.log(`Bot: ${data.text}`);
        },
        // Error handling
        onMessageError: (error) => {
          console.log('Message error:', error);
        },
        onMicUpdated: (data) => {
          console.log('Mic updated:', data);
          this.deviceSelector.value = data.deviceId;
        },
        onError: (error) => {
          console.log('Error:', JSON.stringify(error));
        },
      },
    });
    window.client = this; // Expose client globally for debugging

    // Set up listeners for media track events
    this.setupTrackListeners();
    this.setupEventListeners();
  }

  /**
   * Add a timestamped message to the debug log
   */
  log(message) {
    const entry = document.createElement('div');
    entry.textContent = `${new Date().toISOString()} - ${message}`;

    // Add styling based on message type
    if (message.startsWith('User: ')) {
      entry.style.color = '#2196F3'; // blue for user
    } else if (message.startsWith('Bot: ')) {
      entry.style.color = '#4CAF50'; // green for bot
    }

    this.debugLog.appendChild(entry);
    this.debugLog.scrollTop = this.debugLog.scrollHeight;
    console.log(message);
  }

  /**
   * Update the connection status display
   */
  updateStatus(status) {
    this.statusSpan.textContent = status;
    this.log(`Status: ${status}`);
  }

  /**
   * Check for available media tracks and set them up if present
   * This is called when the bot is ready or when the transport state changes to ready
   */
  setupMediaTracks() {
    if (!this.pcClient) return;

    // Get current tracks from the client
    const tracks = this.pcClient.tracks();

    // Set up any available bot tracks
    if (tracks.bot?.audio) {
      this.setupAudioTrack(tracks.bot.audio);
    }
    if (tracks.bot?.video) {
      this.setupVideoTrack(tracks.bot.video);
    }
  }

  /**
   * Set up listeners for track events (start/stop)
   * This handles new tracks being added during the session
   */
  setupTrackListeners() {
    if (!this.pcClient) return;

    // Listen for new tracks starting
    this.pcClient.on(RTVIEvent.TrackStarted, (track, participant) => {
      if (!participant?.local) {
        if (track.kind === 'audio') {
          this.setupAudioTrack(track);
        } else if (track.kind === 'video') {
          this.setupVideoTrack(track);
        }
      } else if (track.kind === 'audio') {
        console.log(`Local audio track started: `, this.pcClient.tracks());
        // If local audio track starts, update mic
        this.updateMicToggleButton(true);
      }
    });

    // Listen for tracks stopping
    this.pcClient.on(RTVIEvent.TrackStopped, (track, participant) => {
      this.log(
        `Track stopped event: ${track.kind} from ${
          participant ? (participant.local ? 'local' : 'bot') : 'unknown'
        }`
      );
      if (participant?.local && track.kind === 'audio') {
        // If local audio track stops, update mic toggle button
        this.updateMicToggleButton(false);
      }
    });
  }

  /**
   * Set up an audio track for playback
   * Handles both initial setup and track updates
   */
  setupAudioTrack(track) {
    this.log('Setting up audio track');
    // Check if we're already playing this track
    if (this.botAudio.srcObject) {
      const oldTrack = this.botAudio.srcObject.getAudioTracks()[0];
      if (oldTrack?.id === track.id) return;
    }
    // Create a new MediaStream with the track and set it as the audio source
    this.botAudio.srcObject = new MediaStream([track]);
  }

  /**
   * Set up a video track for display
   * Handles both initial setup and track updates
   */
  setupVideoTrack(track) {
    this.log('Setting up video track');
    const videoEl = document.createElement('video');
    videoEl.autoplay = true;
    videoEl.playsInline = true;
    videoEl.muted = true;
    videoEl.style.width = '100%';
    videoEl.style.height = '100%';
    videoEl.style.objectFit = 'cover';

    // Check if we're already displaying this track
    if (this.botVideoContainer.querySelector('video')?.srcObject) {
      const oldTrack = this.botVideoContainer
        .querySelector('video')
        .srcObject.getVideoTracks()[0];
      if (oldTrack?.id === track.id) return;
    }

    // Create a new MediaStream with the track and set it as the video source
    videoEl.srcObject = new MediaStream([track]);
    this.botVideoContainer.innerHTML = '';
    this.botVideoContainer.appendChild(videoEl);
  }

  /**
   * Initialize and connect to the bot
   * This sets up the Pipecat client, initializes devices, and establishes the connection
   */
  async connect() {
    try {
      // Connect to the bot
      this.log('Connecting to bot...');
      await this.pcClient.connect({
        endpoint: 'http://localhost:7860/connect',
        timeout: 25000,
      });

      this.log('Connection complete');
    } catch (error) {
      // Handle any errors during connection
      this.log(`Error connecting: ${error.message}`);
      this.log(`Error stack: ${error.stack}`);
      this.updateStatus('Error');

      // Clean up if there's an error
      if (this.pcClient) {
        try {
          await this.pcClient.disconnect();
        } catch (disconnectError) {
          this.log(`Error during disconnect: ${disconnectError.message}`);
        }
      }
    }
  }

  /**
   * Disconnect from the bot and clean up media resources
   */
  async disconnect() {
    if (this.pcClient) {
      try {
        // Disconnect the Pipecat client
        await this.pcClient.disconnect();

        // Clean up audio
        if (this.botAudio.srcObject) {
          this.botAudio.srcObject.getTracks().forEach((track) => track.stop());
          this.botAudio.srcObject = null;
        }

        // Clean up video
        if (this.botVideoContainer.querySelector('video')?.srcObject) {
          const video = this.botVideoContainer.querySelector('video');
          video.srcObject.getTracks().forEach((track) => track.stop());
          video.srcObject = null;
        }
        this.botVideoContainer.innerHTML = '';
      } catch (error) {
        this.log(`Error disconnecting: ${error.message}`);
      }
    }
  }
}

// Initialize the client when the page loads
window.addEventListener('DOMContentLoaded', () => {
  new ChatbotClient();
});
