/**
 * Copyright (c) 2024–2025, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

/**
 * RTVI Client Implementation
 *
 * This client connects to an RTVI-compatible bot server using WebRTC (via Daily).
 * It handles text and manages the connection lifecycle.
 *
 * Requirements:
 * - A running RTVI bot server (defaults to http://localhost:7860)
 * - The server must implement the /connect endpoint that returns Daily.co room credentials
 */

import { RTVIClient, RTVIEvent, RTVIMessage } from '@pipecat-ai/client-js';
import { DailyTransport } from '@pipecat-ai/daily-transport';

/**
 * ChatbotClient handles the connection and media management for a real-time
 * voice and video interaction with an AI bot.
 */
class ChatbotClient {
  constructor() {
    // Initialize client state
    this.rtviClient = null;
    this.setupDOMElements();
    this.setupEventListeners();
    this.initializeClientAndTransport();
  }

  /**
   * Set up references to DOM elements and create necessary media elements
   */
  setupDOMElements() {
    // Get references to UI control elements
    this.sendBtn = document.getElementById('send-btn');
    this.connectBtn = document.getElementById('connect-btn');
    this.disconnectBtn = document.getElementById('disconnect-btn');
    this.statusSpan = document.getElementById('connection-status');
    this.debugLog = document.getElementById('debug-log');
    this.msgInput = document.getElementById('message-input');
  }

  /**
   * Set up event listeners for connect/disconnect buttons
   */
  setupEventListeners() {
    this.connectBtn.addEventListener('click', () => this.connect());
    this.disconnectBtn.addEventListener('click', () => this.disconnect());
    this.sendBtn.addEventListener('click', () => this.handleTextSubmit());
    const _this = this;
    this.msgInput.addEventListener('keydown', function(event) {
      if (event.key === 'Enter') {
        _this.handleTextSubmit();
      }
    });
  }

  /**
   * Set up the RTVI client and Daily transport
   */
  initializeClientAndTransport() {
    // Initialize the RTVI client with a DailyTransport and our configuration
    this.rtviClient = new RTVIClient({
      transport: new DailyTransport(),
      params: {
        // The baseURL and endpoint of your bot server that the client will connect to
        baseUrl: 'http://localhost:7860',
        endpoints: {
          connect: '/connect',
        },
      },
      callbacks: {
        // Handle connection state changes
        onConnected: () => {
          this.updateStatus('Connected');
        },
        onDisconnected: () => {
          this.updateStatus('Disconnected');
          this.connectBtn.disabled = false;
          this.disconnectBtn.disabled = true;
          this.msgInput.disabled = true;
          this.sendBtn.disabled = true;
        },
        // Handle transport state changes
        onTransportStateChanged: (state) => {
          this.updateStatus(`Transport: ${state}`);
          this.log(`Transport state changed: ${state}`);
          if (state === 'ready') {
            this.connectBtn.disabled = true;
            this.disconnectBtn.disabled = false;
            this.msgInput.disabled = false;
            this.sendBtn.disabled = false;
          }
        },
        // Transcript events
        onBotTranscript: (data) => {
          this.log(`Bot: ${data.text}`);
        },
        // Error handling
        onMessageError: (error) => {
          console.log('Message error:', error);
        },
        onError: (error) => {
          console.log('Error:', JSON.stringify(error));
        },
        // Text only chatbot
        onServerMessage: (data) => {
          console.log('server message:', JSON.stringify(data));
        },        
      },
    });
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

  async handleTextSubmit() {
    const text = this.msgInput.value;
    const message = text.trim();

    try {
      const d = await this.rtviClient?.action({
        service: "llm",
        action: "append_to_messages",
        arguments: [
          {
            name: "messages",
            value: [
              {
                role: "user",
                content: message,
              },
            ],
          },
        ],
      });
      this.log(`User: ${message}`);
    } catch (e) {
      if (e instanceof RTVIMessage) {
        console.error(e.data);
      } else {
        console.error(e);
      }      
    } finally {
      this.msgInput.value = ""
    }    
  };

  /**
   * Initialize and connect to the bot
   * This sets up the RTVI client, initializes devices, and establishes the connection
   */
  async connect() {
    try {
      // Connect to the bot
      this.log('Connecting to bot...');
      await this.rtviClient.connect();

      this.log('Connection complete');
    } catch (error) {
      // Handle any errors during connection
      this.log(`Error connecting: ${error.message}`);
      this.log(`Error stack: ${error.stack}`);
      this.updateStatus('Error');

      // Clean up if there's an error
      if (this.rtviClient) {
        try {
          await this.rtviClient.disconnect();
        } catch (disconnectError) {
          this.log(`Error during disconnect: ${disconnectError.message}`);
        }
      }
    }
  }

  /**
   * Disconnect from the bot
   */
  async disconnect() {
    if (this.rtviClient) {
      try {
        // Disconnect the RTVI client
        await this.rtviClient.disconnect();

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
