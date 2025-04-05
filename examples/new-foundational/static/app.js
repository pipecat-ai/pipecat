/**
 * Copyright (c) 2024–2025, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

class WebRTCApp {
  constructor() {
    this.setupDOMElements();
    this.setupDOMEventListeners();
    this.connected = false;
    this.videoTrack = null;
    this.audioTrack = null;
    this.videoContainer = document.getElementById('bot-video-container');

    // Set initial mute states - camera starts muted/off
    this.micMuted = false;
    this.cameraMuted = true;

    // Start with both hidden - neither video nor visualizer
    this.videoContainer.classList.remove('video-hidden');
    this.videoContainer.classList.remove('video-visible');

    void this.populateDevices();
  }

  setupDOMElements() {
    this.connectionBtn = document.getElementById('connection-btn');
    this.micToggleBtn = document.getElementById('mic-toggle');
    this.cameraToggleBtn = document.getElementById('camera-toggle');
    this.micChevronBtn = document.getElementById('mic-chevron');
    this.cameraChevronBtn = document.getElementById('camera-chevron');
    this.micPopover = document.getElementById('mic-popover');
    this.cameraPopover = document.getElementById('camera-popover');

    this.audioInput = document.getElementById('audio-input');
    this.videoInput = document.getElementById('video-input');

    this.currentAudioDevice = document.getElementById('current-audio-device');
    this.currentVideoDevice = document.getElementById('current-video-device');

    this.videoElement = document.getElementById('bot-video');
    this.audioElement = document.getElementById('bot-audio');

    this.selfViewContainer = document.getElementById('self-view-container');
    this.selfViewVideo = document.getElementById('self-view');

    this.debugLog = document.getElementById('debug-log');
  }

  setupDOMEventListeners() {
    // Connection button handler
    this.connectionBtn.addEventListener('click', () => {
      const currentState = this.connectionBtn.getAttribute('data-state');
      if (currentState === 'disconnected') {
        this.start();
      } else {
        this.stop();
      }
    });

    // Microphone toggle button
    this.micToggleBtn.addEventListener('click', () => {
      this.toggleMicrophone();
    });

    // Camera toggle button
    this.cameraToggleBtn.addEventListener('click', () => {
      this.toggleCamera();
    });

    // Mic chevron click
    this.micChevronBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.togglePopover(this.micPopover, this.micChevronBtn);

      // Hide the other popover if it's open
      if (this.cameraPopover.classList.contains('show')) {
        this.togglePopover(this.cameraPopover, this.cameraChevronBtn);
      }
    });

    // Camera chevron click
    this.cameraChevronBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.togglePopover(this.cameraPopover, this.cameraChevronBtn);

      // Hide the other popover if it's open
      if (this.micPopover.classList.contains('show')) {
        this.togglePopover(this.micPopover, this.micChevronBtn);
      }
    });

    // Click outside to close popovers
    document.addEventListener('click', (e) => {
      if (!e.target.closest('.control-wrapper')) {
        if (this.micPopover.classList.contains('show')) {
          this.togglePopover(this.micPopover, this.micChevronBtn);
        }
        if (this.cameraPopover.classList.contains('show')) {
          this.togglePopover(this.cameraPopover, this.cameraChevronBtn);
        }
      }
    });

    // Device selection changes
    this.audioInput.addEventListener('change', () => {
      this.log('Audio input changed');
      this.updateCurrentDeviceDisplay();
      if (this.connected) {
        this.stop();
        setTimeout(() => this.start(), 500);
      }
    });

    this.videoInput.addEventListener('change', () => {
      this.log('Video input changed');
      this.updateCurrentDeviceDisplay();
      if (this.connected) {
        this.stop();
        setTimeout(() => this.start(), 500);
      }
    });
  }

  togglePopover(popover, chevronBtn) {
    popover.classList.toggle('show');
    chevronBtn.classList.toggle('active');
  }

  updateCurrentDeviceDisplay() {
    // Update the displayed device names
    if (this.audioInput.selectedIndex > 0) {
      this.currentAudioDevice.textContent =
        this.audioInput.options[this.audioInput.selectedIndex].text;
    } else {
      this.currentAudioDevice.textContent = 'Default device';
    }

    if (this.videoInput.selectedIndex > 0) {
      this.currentVideoDevice.textContent =
        this.videoInput.options[this.videoInput.selectedIndex].text;
    } else {
      this.currentVideoDevice.textContent = 'Default device';
    }
  }

  updateSelfViewVisibility() {
    // Show self view when:
    // 1. Connected
    // 2. Camera is not muted
    if (this.connected && !this.cameraMuted) {
      this.selfViewContainer.classList.add('active');
    } else {
      this.selfViewContainer.classList.remove('active');
    }
  }

  toggleMicrophone() {
    if (!this.connected) {
      this.log('Cannot toggle microphone when not connected', 'error');
      return;
    }

    // Get all audio tracks from the local stream
    if (this.stream) {
      const audioTracks = this.stream.getAudioTracks();
      if (audioTracks.length > 0) {
        const track = audioTracks[0];
        this.micMuted = !this.micMuted;
        track.enabled = !this.micMuted;

        // Update UI
        if (this.micMuted) {
          this.micToggleBtn.setAttribute('data-state', 'muted');
          this.micToggleBtn.title = 'Unmute microphone';
          this.log('Microphone muted');
        } else {
          this.micToggleBtn.setAttribute('data-state', 'unmuted');
          this.micToggleBtn.title = 'Mute microphone';
          this.log('Microphone unmuted');
        }
      } else {
        this.log('No audio track available', 'error');
      }
    }
  }

  toggleCamera() {
    if (!this.connected) {
      this.log('Cannot toggle camera when not connected', 'error');
      return;
    }

    // Get all video tracks from the local stream
    if (this.stream) {
      const videoTracks = this.stream.getVideoTracks();
      if (videoTracks.length > 0) {
        const track = videoTracks[0];
        this.cameraMuted = !this.cameraMuted;
        track.enabled = !this.cameraMuted;

        // Update UI
        if (this.cameraMuted) {
          this.cameraToggleBtn.setAttribute('data-state', 'muted');
          this.cameraToggleBtn.title = 'Turn on camera';
          this.log('Camera turned off');
        } else {
          this.cameraToggleBtn.setAttribute('data-state', 'unmuted');
          this.cameraToggleBtn.title = 'Turn off camera';
          this.log('Camera turned on');
        }

        // Update self view visibility
        this.updateSelfViewVisibility();
      } else {
        this.log('No video track available', 'error');
      }
    }
  }

  log(message, type = 'normal') {
    if (!this.debugLog) return;

    const now = new Date();
    const timeString = now.toISOString().replace('T', ' ').substring(0, 19);

    const entry = document.createElement('div');
    entry.textContent = `${timeString} - ${message}`;

    // Apply styling based on message type
    if (type === 'status' || message.includes('Status:')) {
      entry.classList.add('status-message');
    } else if (message.includes('User transcript:')) {
      entry.classList.add('user-message');
    } else if (message.includes('Bot transcript:')) {
      entry.classList.add('bot-message');
    } else if (type === 'error') {
      entry.classList.add('error-message');
    }

    this.debugLog.appendChild(entry);
    this.debugLog.scrollTop = this.debugLog.scrollHeight;
  }

  clearAllLogs() {
    if (this.debugLog) this.debugLog.innerText = '';
    this.log('Log cleared', 'status');
  }

  updateStatus(status) {
    if (status === 'Connected') {
      this.connectionBtn.textContent = 'Disconnect';
      this.connectionBtn.setAttribute('data-state', 'connected');
      this.connected = true;

      // Enable media control buttons
      this.micToggleBtn.disabled = false;
      this.cameraToggleBtn.disabled = false;
      this.micChevronBtn.disabled = false;
      this.cameraChevronBtn.disabled = false;

      // Set initial UI state for media controls based on mute states
      this.micToggleBtn.setAttribute(
        'data-state',
        this.micMuted ? 'muted' : 'unmuted'
      );
      this.cameraToggleBtn.setAttribute(
        'data-state',
        this.cameraMuted ? 'muted' : 'unmuted'
      );

      // If we have a video track, check its state
      if (this.videoTrack) {
        this.updateVideoVisibility(this.videoTrack, !this.videoTrack.muted);
      } else {
        // If no video track yet, assume we want visualizer
        this.videoContainer.classList.remove('video-visible');
        this.videoContainer.classList.add('video-hidden');
      }

      // Update self view visibility
      this.updateSelfViewVisibility();
    } else {
      this.connectionBtn.textContent = 'Connect';
      this.connectionBtn.setAttribute('data-state', 'disconnected');
      this.connected = false;

      // Disable and reset media control buttons
      this.micToggleBtn.disabled = true;
      this.cameraToggleBtn.disabled = true;
      this.micChevronBtn.disabled = true;
      this.cameraChevronBtn.disabled = true;

      // Close any open popovers
      if (this.micPopover.classList.contains('show')) {
        this.togglePopover(this.micPopover, this.micChevronBtn);
      }
      if (this.cameraPopover.classList.contains('show')) {
        this.togglePopover(this.cameraPopover, this.cameraChevronBtn);
      }

      // Reset UI state when disconnected - hide both
      this.videoContainer.classList.remove('video-visible');
      this.videoContainer.classList.remove('video-hidden');

      // Hide self view
      this.selfViewContainer.classList.remove('active');
    }

    this.log(`Status: ${status}`, 'status');
  }

  updateVideoVisibility(track, enabled) {
    this.log(`Video track ${enabled ? 'enabled' : 'disabled'}`);

    // Only update visibility if we're connected
    if (this.connected) {
      if (enabled) {
        // Show video, hide visualizer
        this.videoContainer.classList.remove('video-hidden');
        this.videoContainer.classList.add('video-visible');
      } else {
        // Hide video, show visualizer
        this.videoContainer.classList.remove('video-visible');
        this.videoContainer.classList.add('video-hidden');
      }
    } else {
      // If not connected, hide both
      this.videoContainer.classList.remove('video-hidden');
      this.videoContainer.classList.remove('video-visible');
    }
  }

  async populateDevices() {
    try {
      // Request permissions first to get labeled devices
      await navigator.mediaDevices
        .getUserMedia({ audio: true, video: true })
        .catch((e) => console.log('Permission partially denied: ', e));

      const allDevices = await navigator.mediaDevices.enumerateDevices();

      const audioDevices = allDevices.filter(
        (device) => device.kind === 'audioinput'
      );
      this.populateSelect(this.audioInput, audioDevices);

      const videoDevices = allDevices.filter(
        (device) => device.kind === 'videoinput'
      );
      this.populateSelect(this.videoInput, videoDevices);

      // Select the first devices as default
      this.selectDefaultDevices(audioDevices, videoDevices);
      this.updateCurrentDeviceDisplay();
    } catch (e) {
      this.log(`Error getting devices: ${e.message}`, 'error');
      console.error(e);
    }
  }

  populateSelect(select, devices) {
    // Clear existing options
    select.innerHTML = '';

    // Add default option
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.text = 'Default device';
    select.appendChild(defaultOption);

    // Add devices
    devices.forEach((device, index) => {
      const option = document.createElement('option');
      option.value = device.deviceId;
      option.text = device.label || `Device #${index + 1}`;
      select.appendChild(option);
    });
  }

  selectDefaultDevices(audioDevices, videoDevices) {
    // Select first audio device if available
    if (audioDevices.length > 0 && this.audioInput.options.length > 1) {
      this.audioInput.selectedIndex = 1; // Select first actual device (after "Default device")
      this.log(
        `Default audio device selected: ${this.audioInput.options[1].text}`
      );
    }

    // Select first video device if available
    if (videoDevices.length > 0 && this.videoInput.options.length > 1) {
      this.videoInput.selectedIndex = 1; // Select first actual device (after "Default device")
      this.log(
        `Default video device selected: ${this.videoInput.options[1].text}`
      );
    }
  }

  async start() {
    this.clearAllLogs();
    this.updateStatus('Connecting');

    try {
      // Get media based on selected devices (use default if none selected)
      const constraints = {
        audio:
          this.audioInput.selectedIndex > 0
            ? { deviceId: { exact: this.audioInput.value } }
            : true,
        video:
          this.videoInput.selectedIndex > 0
            ? { deviceId: { exact: this.videoInput.value } }
            : true,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);

      // Immediately mute camera if camera is set to be muted
      if (this.cameraMuted) {
        const videoTracks = stream.getVideoTracks();
        if (videoTracks.length > 0) {
          videoTracks[0].enabled = false;
          this.log('Camera initially muted');
        }
      }

      // Immediately mute microphone if mic is set to be muted
      if (this.micMuted) {
        const audioTracks = stream.getAudioTracks();
        if (audioTracks.length > 0) {
          audioTracks[0].enabled = false;
          this.log('Microphone initially muted');
        }
      }

      // Set up self view video
      this.selfViewVideo.srcObject = stream;

      // Update self view visibility - should be hidden since camera is muted
      this.updateSelfViewVisibility();

      const pc = new RTCPeerConnection();

      // Set up track handling
      pc.ontrack = (e) => {
        this.log(`Track started: ${e.track.kind}`);

        if (e.track.kind === 'audio') {
          this.audioTrack = e.track;

          const audioStream = new MediaStream([e.track]);
          this.audioElement.srcObject = audioStream;
          this.audioElement.play().catch((e) => {
            this.log(`Error playing audio: ${e.message}`, 'error');
          });

          // Connect the voice visualizer to the audio track
          if (window.voiceVisualizer) {
            window.voiceVisualizer.connectToAudioTrack(e.track);
          }
        } else if (e.track.kind === 'video') {
          this.videoTrack = e.track;

          const videoStream = new MediaStream([e.track]);
          this.videoElement.srcObject = videoStream;

          // Initial check if video is enabled
          this.updateVideoVisibility(e.track, !e.track.muted);

          // Listen for mute/unmute events
          e.track.onmute = () => this.updateVideoVisibility(e.track, false);
          e.track.onunmute = () => this.updateVideoVisibility(e.track, true);
        }
      };

      // Add tracks from our stream to the connection
      stream.getTracks().forEach((track) => {
        this.log(`Adding local track: ${track.kind}`);
        pc.addTransceiver(track, { direction: 'sendrecv' });
      });

      // Connection state change handler
      pc.onconnectionstatechange = () => {
        this.log(`Connection state: ${pc.connectionState}`);
        if (pc.connectionState === 'connected') {
          this.updateStatus('Connected');
        } else if (
          pc.connectionState === 'disconnected' ||
          pc.connectionState === 'failed' ||
          pc.connectionState === 'closed'
        ) {
          this.updateStatus('Disconnected');
          // If not explicitly stopped by user, try to stop/clean up
          if (this.connected) {
            this.stop();
          }
        }
      };

      // Create and set local description
      await pc.setLocalDescription(await pc.createOffer());

      // Exchange SDP with server
      const response = await fetch('/api/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type,
        }),
      });

      // Process server response
      const answer = await response.json();
      if (answer.error) {
        throw new Error(answer.error);
      }

      await pc.setRemoteDescription(answer);
      this.pc = pc;
      this.stream = stream;

      // Set up data channel for ping-pong
      this.keepAliveInterval = setInterval(() => {
        if (pc.connectionState === 'connected') {
          const dc = pc.createDataChannel('ping');
          dc.onopen = () => {
            dc.send('ping-' + Date.now());
            setTimeout(() => dc.close(), 1000);
          };
        }
      }, 15000);

      // User transcript handler
      pc.addEventListener('datachannel', (event) => {
        const dc = event.channel;
        dc.onmessage = (msg) => {
          try {
            const data = JSON.parse(msg.data);
            if (data.type === 'transcript') {
              if (data.role === 'user') {
                this.log(`User transcript: ${data.text}`);
              } else if (data.role === 'assistant') {
                this.log(`Bot transcript: ${data.text}`);
              }
            }
          } catch (e) {
            // Ignore non-JSON messages
          }
        };
      });
    } catch (e) {
      this.log(`Error connecting: ${e.message}`, 'error');
      console.error(e);
      this.updateStatus('Disconnected');
    }
  }

  stop() {
    // Stop all media tracks
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }

    // Reset track references
    this.videoTrack = null;
    this.audioTrack = null;

    // Clear self view
    this.selfViewVideo.srcObject = null;
    this.selfViewContainer.classList.remove('active');

    // Close the peer connection
    if (this.pc) {
      this.pc.close();
      this.pc = null;
    }

    // Clear intervals
    if (this.keepAliveInterval) {
      clearInterval(this.keepAliveInterval);
    }

    // Don't reset mute states - maintain them between connections
    // this.micMuted = false;
    // this.cameraMuted = true;

    // Reset UI
    this.updateStatus('Disconnected');
    this.videoElement.srcObject = null;
    this.audioElement.srcObject = null;

    // Reset UI state - hide both video and visualizer
    this.videoContainer.classList.remove('video-visible');
    this.videoContainer.classList.remove('video-hidden');

    // Disconnect visualizer
    if (window.voiceVisualizer) {
      window.voiceVisualizer.disconnectAudio();
    }
  }

  initializeMediaControls() {
    // Initially disable media controls until connected
    this.micToggleBtn.disabled = true;
    this.cameraToggleBtn.disabled = true;
    this.micChevronBtn.disabled = true;
    this.cameraChevronBtn.disabled = true;

    // Set initial UI states based on mute states
    this.micToggleBtn.setAttribute(
      'data-state',
      this.micMuted ? 'muted' : 'unmuted'
    );
    this.cameraToggleBtn.setAttribute(
      'data-state',
      this.cameraMuted ? 'muted' : 'unmuted'
    );

    this.micToggleBtn.title = this.micMuted
      ? 'Unmute microphone'
      : 'Mute microphone';
    this.cameraToggleBtn.title = this.cameraMuted
      ? 'Turn on camera'
      : 'Turn off camera';
  }
}

// Create the WebRTCConnection instance on page load
document.addEventListener('DOMContentLoaded', () => {
  window.webRTCApp = new WebRTCApp();
  window.webRTCApp.initializeMediaControls();

  // Cleanup when leaving the page
  window.addEventListener('beforeunload', () => {
    window.webRTCApp.stop();
  });
});
