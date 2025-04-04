class WebRTCApp {
  constructor() {
    this.setupDOMElements();
    this.setupDOMEventListeners();
    this.connected = false;
    this.videoTrack = null;
    this.audioTrack = null;
    this.videoContainer = document.getElementById('bot-video-container');

    // Start with both hidden - neither video nor visualizer
    this.videoContainer.classList.remove('video-hidden');
    this.videoContainer.classList.remove('video-visible');

    void this.populateDevices();
  }

  setupDOMElements() {
    this.connectionBtn = document.getElementById('connection-btn');

    this.audioInput = document.getElementById('audio-input');
    this.videoInput = document.getElementById('video-input');

    this.videoElement = document.getElementById('bot-video');
    this.audioElement = document.getElementById('bot-audio');

    this.debugLog = document.getElementById('debug-log');
  }

  setupDOMEventListeners() {
    // Single button handler that checks the current state
    this.connectionBtn.addEventListener('click', () => {
      const currentState = this.connectionBtn.getAttribute('data-state');
      if (currentState === 'disconnected') {
        this.start();
      } else {
        this.stop();
      }
    });

    // Update when device selections change
    this.audioInput.addEventListener('change', () => {
      this.log('Audio input changed');
    });

    this.videoInput.addEventListener('change', () => {
      this.log('Video input changed');
    });
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

      // If we have a video track, check its state
      if (this.videoTrack) {
        this.updateVideoVisibility(this.videoTrack, !this.videoTrack.muted);
      } else {
        // If no video track yet, assume we want visualizer
        this.videoContainer.classList.remove('video-visible');
        this.videoContainer.classList.add('video-hidden');
      }
    } else {
      this.connectionBtn.textContent = 'Connect';
      this.connectionBtn.setAttribute('data-state', 'disconnected');
      this.connected = false;

      // Reset UI state when disconnected - hide both
      this.videoContainer.classList.remove('video-visible');
      this.videoContainer.classList.remove('video-hidden');
    }

    this.log(`Status: ${status}`, 'status');
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

    // Close the peer connection
    if (this.pc) {
      this.pc.close();
      this.pc = null;
    }

    // Clear intervals
    if (this.keepAliveInterval) {
      clearInterval(this.keepAliveInterval);
    }

    // Reset UI
    this.updateStatus('Disconnected');
    this.videoElement.srcObject = null;
    this.audioElement.srcObject = null;

    // Reset UI - hide both video and visualizer
    this.videoContainer.classList.remove('video-visible');
    this.videoContainer.classList.remove('video-hidden');

    // Disconnect visualizer
    if (window.voiceVisualizer) {
      window.voiceVisualizer.disconnectAudio();
    }
  }
}

// Create the WebRTCConnection instance on page load
document.addEventListener('DOMContentLoaded', () => {
  window.webRTCApp = new WebRTCApp();

  // Cleanup when leaving the page
  window.addEventListener('beforeunload', () => {
    window.webRTCApp.stop();
  });
});
