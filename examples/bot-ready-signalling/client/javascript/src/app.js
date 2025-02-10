/**
 * Copyright (c) 2024–2025, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

import Daily from "@daily-co/daily-js";

/**
 * ChatbotClient handles the connection and media management for a real-time
 * voice interaction with an AI bot.
 */
class ChatbotClient {
  constructor() {
    // Initialize client state
    this.dailyCallObject = null;
    this.setupDOMElements();
    this.setupEventListeners();
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
    this.connectBtn.addEventListener('click', () => this.connect());
    this.disconnectBtn.addEventListener('click', () => this.disconnect());
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

  handleEventToConsole (evt) {
    this.log(`Received event: ${evt.action}`);
  };

  /**
   * Set up listeners for track events (start/stop)
   * This handles new tracks being added during the session
   */
  setupTrackListeners() {
    if (!this.dailyCallObject) return;

    this.dailyCallObject.on("joined-meeting", () => {
      this.updateStatus('Connected');
      this.connectBtn.disabled = true;
      this.disconnectBtn.disabled = false;
      this.log('Client connected');
    });
    this.dailyCallObject.on("track-started", (evt) => {
      if (evt.track.kind === "audio" && evt.participant.local === false) {
        this.log("Audio track started.")
        this.setupAudioTrack(evt.track);
      }
    });
    this.dailyCallObject.on("track-stopped", this.handleEventToConsole.bind(this));
    this.dailyCallObject.on("participant-joined", this.handleEventToConsole.bind(this));
    this.dailyCallObject.on("participant-updated", this.handleEventToConsole.bind(this));
    this.dailyCallObject.on("participant-left", () => {
      // When the bot leaves, we are also disconnecting from the call
      this.disconnect()
    });
    this.dailyCallObject.on("left-meeting", () => {
      this.updateStatus('Disconnected');
      this.connectBtn.disabled = false;
      this.disconnectBtn.disabled = true;
      this.log('Client disconnected');
    });
    this.dailyCallObject.on("error", this.handleEventToConsole.bind(this));
  }

  /**
   * Set up an audio track for playback
   * Handles both initial setup and track updates
   */
  setupAudioTrack(track) {
    this.log(`Setting up audio track, track state: ${track.readyState}, muted: ${track.muted}`);

    // Check if we're already playing this track
    if (this.botAudio.srcObject) {
      const oldTrack = this.botAudio.srcObject.getAudioTracks()[0];
      if (oldTrack?.id === track.id) return;
    }
    // Create a new MediaStream with the track and set it as the audio source
    this.botAudio.srcObject = new MediaStream([track]);
    this.botAudio.onplaying = async (event) => {
      this.log("onplaying")
      this.log("Will send the audio message to play the audio at the next tick")
      this.dailyCallObject.sendAppMessage("playable")
    }
  }

  async fetchRoomInfo() {
    let connectUrl = '/connect'
    let res = await fetch(connectUrl, {
      method: "POST",
      mode: "cors",
      headers: new Headers({
        "Content-Type": "application/json"
      }),
    })
    if (res.ok) {
      return res.json();
    }
  }

  /**
   * Initialize and connect to the bot
   * This sets up the RTVI client, initializes devices, and establishes the connection
   */
  async connect() {
    try {
      // Initialize the client
      this.dailyCallObject = Daily.createCallObject({
        subscribeToTracksAutomatically: true,
      });

      // Set up listeners for media track events
      this.setupTrackListeners();

      this.log('Creating the bot...');
      let roomInfo = await this.fetchRoomInfo()

      // Connect to the bot
      this.log('Connecting to bot...');
      // Only for making debugger easier
      window.callObject = this.dailyCallObject;
      await this.dailyCallObject.join({
        url: roomInfo.room_url,
      });

      this.log('Connection complete');
    } catch (error) {
      // Handle any errors during connection
      this.log(`Error connecting: ${error.message}`);
      this.log(`Error stack: ${error.stack}`);
      this.updateStatus('Error');

      // Clean up if there's an error
      if (this.dailyCallObject) {
        try {
          await this.dailyCallObject.leave();
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
    if (this.dailyCallObject) {
      try {
        // Disconnect the RTVI client
        await this.dailyCallObject.leave();
        await this.dailyCallObject.destroy();
        this.dailyCallObject = null;

        // Clean up audio
        if (this.botAudio.srcObject) {
          this.botAudio.srcObject.getTracks().forEach((track) => track.stop());
          this.botAudio.srcObject = null;
        }
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
