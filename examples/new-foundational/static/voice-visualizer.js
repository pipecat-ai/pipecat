/**
 * Copyright (c) 2024–2025, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

class VoiceVisualizer {
  constructor(options = {}) {
    this.options = {
      backgroundColor: options.backgroundColor || 'transparent',
      barColor: options.barColor || 'rgba(255, 255, 255, 0.8)',
      barWidth: options.barWidth || 30,
      barGap: options.barGap || 12,
      barMaxHeight: options.barMaxHeight || 120,
      container: options.container || 'voice-visualizer-container',
    };

    this.canvas = null;
    this.canvasCtx = null;
    this.audioContext = null;
    this.analyser = null;
    this.source = null;
    this.isActive = false;
    this.track = null;
    this.animationId = null;

    this.bands = [
      { startFreq: 85, endFreq: 255, smoothValue: 0 },
      { startFreq: 255, endFreq: 500, smoothValue: 0 },
      { startFreq: 500, endFreq: 2000, smoothValue: 0 },
      { startFreq: 2000, endFreq: 4000, smoothValue: 0 },
      { startFreq: 4000, endFreq: 8000, smoothValue: 0 },
    ];

    this.init();
  }

  init() {
    const container = document.getElementById(this.options.container);
    if (!container) {
      console.error('Visualizer container not found');
      return;
    }

    // Create canvas element
    this.canvas = document.createElement('canvas');
    this.canvas.id = 'voice-visualizer';
    container.appendChild(this.canvas);

    // Set up canvas
    this.setupCanvas();

    // Add resize handler
    window.addEventListener('resize', this.handleResize.bind(this));
  }

  setupCanvas() {
    if (!this.canvas) return;

    const { barWidth, barGap, barMaxHeight } = this.options;
    const canvasWidth = 5 * barWidth + 4 * barGap;
    const canvasHeight = barMaxHeight;
    const scaleFactor = 2;

    this.canvas.width = canvasWidth * scaleFactor;
    this.canvas.height = canvasHeight * scaleFactor;
    this.canvas.style.width = `${canvasWidth}px`;
    this.canvas.style.height = `${canvasHeight}px`;

    this.canvasCtx = this.canvas.getContext('2d');
    if (this.canvasCtx) {
      this.canvasCtx.lineCap = 'round';
      this.canvasCtx.scale(scaleFactor, scaleFactor);
    }

    // Draw initial inactive state
    this.drawInactiveCircles();
  }

  handleResize() {
    this.setupCanvas();
    // Only draw circles if we're active and connected
    if (this.isActive && window.webRTCApp && window.webRTCApp.connected) {
      this.drawInactiveCircles();
    }
  }

  connectToAudioTrack(track) {
    if (!track) {
      console.log('No audio track provided');
      this.disconnectAudio();
      return;
    }

    this.track = track;

    // Clean up existing audio context if any
    this.disconnectAudio();

    try {
      this.audioContext = new (window.AudioContext ||
        window.webkitAudioContext)();
      const stream = new MediaStream([track]);
      this.source = this.audioContext.createMediaStreamSource(stream);
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 1024;
      this.source.connect(this.analyser);

      this.isActive = true;
      this.startVisualization();

      // Log to debug panel
      if (window.webRTCApp) {
        window.webRTCApp.log('Voice visualizer connected to audio track');
      }
    } catch (error) {
      console.error('Error connecting to audio track:', error);
    }
  }

  disconnectAudio() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }

    if (this.source) {
      this.source.disconnect();
      this.source = null;
    }

    if (this.audioContext) {
      if (this.audioContext.state !== 'closed') {
        this.audioContext.close();
      }
      this.audioContext = null;
    }

    this.analyser = null;
    this.isActive = false;

    // Draw inactive state
    this.drawInactiveCircles();
  }

  getFrequencyBinIndex(frequency) {
    if (!this.audioContext || !this.analyser) return 0;
    const nyquist = this.audioContext.sampleRate / 2;
    return Math.round(
      (frequency / nyquist) * (this.analyser.frequencyBinCount - 1)
    );
  }

  startVisualization() {
    if (!this.canvasCtx || !this.analyser) return;

    const frequencyData = new Uint8Array(this.analyser.frequencyBinCount);
    const scaleFactor = 2;
    const { barWidth, barGap, barMaxHeight, backgroundColor, barColor } =
      this.options;

    const drawSpectrum = () => {
      if (!this.analyser) return;

      this.analyser.getByteFrequencyData(frequencyData);
      this.canvasCtx.clearRect(
        0,
        0,
        this.canvas.width / scaleFactor,
        this.canvas.height / scaleFactor
      );
      this.canvasCtx.fillStyle = backgroundColor;
      this.canvasCtx.fillRect(
        0,
        0,
        this.canvas.width / scaleFactor,
        this.canvas.height / scaleFactor
      );

      let isActive = false;

      const totalBarsWidth =
        this.bands.length * barWidth + (this.bands.length - 1) * barGap;
      const startX = (this.canvas.width / scaleFactor - totalBarsWidth) / 2;

      const adjustedCircleRadius = barWidth / 2;

      this.bands.forEach((band, i) => {
        const startIndex = this.getFrequencyBinIndex(band.startFreq);
        const endIndex = this.getFrequencyBinIndex(band.endFreq);
        const bandData = frequencyData.slice(startIndex, endIndex);
        const bandValue =
          bandData.reduce((acc, val) => acc + val, 0) / bandData.length;

        const smoothingFactor = 0.2;

        if (bandValue < 1) {
          band.smoothValue = Math.max(
            band.smoothValue - smoothingFactor * 5,
            0
          );
        } else {
          band.smoothValue =
            band.smoothValue + (bandValue - band.smoothValue) * smoothingFactor;
          isActive = true;
        }

        const x = startX + i * (barWidth + barGap);
        const barHeight = Math.min(
          (band.smoothValue / 255) * barMaxHeight,
          barMaxHeight
        );

        const yTop = Math.max(
          this.canvas.height / scaleFactor / 2 - barHeight / 2,
          adjustedCircleRadius
        );
        const yBottom = Math.min(
          this.canvas.height / scaleFactor / 2 + barHeight / 2,
          this.canvas.height / scaleFactor - adjustedCircleRadius
        );

        if (band.smoothValue > 0) {
          this.canvasCtx.beginPath();
          this.canvasCtx.moveTo(x + barWidth / 2, yTop);
          this.canvasCtx.lineTo(x + barWidth / 2, yBottom);
          this.canvasCtx.lineWidth = barWidth;
          this.canvasCtx.strokeStyle = barColor;
          this.canvasCtx.stroke();
        } else {
          this.canvasCtx.beginPath();
          this.canvasCtx.arc(
            x + barWidth / 2,
            this.canvas.height / scaleFactor / 2,
            adjustedCircleRadius,
            0,
            2 * Math.PI
          );
          this.canvasCtx.fillStyle = barColor;
          this.canvasCtx.fill();
          this.canvasCtx.closePath();
        }
      });

      if (!isActive) {
        this.drawInactiveCircles();
      }

      this.animationId = requestAnimationFrame(drawSpectrum);
    };

    this.animationId = requestAnimationFrame(drawSpectrum);
  }

  drawInactiveCircles() {
    if (!this.canvasCtx) return;

    const scaleFactor = 2;
    const { barWidth, barGap, barColor } = this.options;
    const circleRadius = barWidth / 2;

    this.canvasCtx.clearRect(
      0,
      0,
      this.canvas.width / scaleFactor,
      this.canvas.height / scaleFactor
    );
    this.canvasCtx.fillStyle = this.options.backgroundColor;
    this.canvasCtx.fillRect(
      0,
      0,
      this.canvas.width / scaleFactor,
      this.canvas.height / scaleFactor
    );

    const totalBarsWidth =
      this.bands.length * barWidth + (this.bands.length - 1) * barGap;
    const startX = (this.canvas.width / scaleFactor - totalBarsWidth) / 2;
    const y = this.canvas.height / scaleFactor / 2;

    this.bands.forEach((_, i) => {
      const x = startX + i * (barWidth + barGap);

      this.canvasCtx.beginPath();
      this.canvasCtx.arc(x + barWidth / 2, y, circleRadius, 0, 2 * Math.PI);
      this.canvasCtx.fillStyle = barColor;
      this.canvasCtx.fill();
      this.canvasCtx.closePath();
    });
  }
}

// Initialize the visualizer when the page loads
document.addEventListener('DOMContentLoaded', () => {
  // Create the visualizer with white bars on transparent background
  window.voiceVisualizer = new VoiceVisualizer({
    backgroundColor: 'transparent',
    barColor: 'rgba(255, 255, 255, 0.8)',
    barWidth: 30,
    barGap: 12,
    barMaxHeight: 120,
  });
});
