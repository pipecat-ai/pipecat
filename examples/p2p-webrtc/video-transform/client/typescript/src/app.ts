import { SmallWebRTCTransport } from '@pipecat-ai/small-webrtc-transport';
import {
  BotLLMTextData,
  Participant,
  PipecatClient,
  PipecatClientOptions,
  TranscriptData,
  TransportState,
} from '@pipecat-ai/client-js';

class WebRTCApp {
  private declare connectBtn: HTMLButtonElement;
  private declare disconnectBtn: HTMLButtonElement;
  private declare muteBtn: HTMLButtonElement;

  private declare audioInput: HTMLSelectElement;
  private declare videoInput: HTMLSelectElement;
  private declare audioCodec: HTMLSelectElement;
  private declare videoCodec: HTMLSelectElement;

  private declare videoElement: HTMLVideoElement;
  private declare audioElement: HTMLAudioElement;

  private debugLog: HTMLElement | null = null;
  private statusSpan: HTMLElement | null = null;

  private declare smallWebRTCTransport: SmallWebRTCTransport;
  private declare pcClient: PipecatClient;

  constructor() {
    this.setupDOMElements();
    this.setupDOMEventListeners();
    this.initializePipecatClient();
    void this.populateDevices();
  }

  private initializePipecatClient(): void {
    const opts: PipecatClientOptions = {
      transport: new SmallWebRTCTransport({ connectionUrl: '/api/offer' }),
      enableMic: true,
      enableCam: true,
      callbacks: {
        onTransportStateChanged: (state: TransportState) => {
          this.log(`Transport state: ${state}`);
        },
        onConnected: () => {
          this.onConnectedHandler();
        },
        onBotReady: () => {
          this.log('Bot is ready.');
        },
        onDisconnected: () => {
          this.onDisconnectedHandler();
        },
        onUserStartedSpeaking: () => {
          this.log('User started speaking.');
        },
        onUserStoppedSpeaking: () => {
          this.log('User stopped speaking.');
        },
        onBotStartedSpeaking: () => {
          this.log('Bot started speaking.');
        },
        onBotStoppedSpeaking: () => {
          this.log('Bot stopped speaking.');
        },
        onUserTranscript: (transcript: TranscriptData) => {
          if (transcript.final) {
            this.log(`User transcript: ${transcript.text}`);
          }
        },
        onBotTranscript: (data: BotLLMTextData) => {
          this.log(`Bot transcript: ${data.text}`);
        },
        onTrackStarted: (
          track: MediaStreamTrack,
          participant?: Participant
        ) => {
          if (participant?.local) {
            return;
          }
          this.onBotTrackStarted(track);
        },
        onServerMessage: (msg: unknown) => {
          this.log(`Server message: ${msg}`);
        },
      },
    };
    this.pcClient = new PipecatClient(opts);
    this.smallWebRTCTransport = this.pcClient.transport as SmallWebRTCTransport;
  }

  private setupDOMElements(): void {
    this.connectBtn = document.getElementById(
      'connect-btn'
    ) as HTMLButtonElement;
    this.disconnectBtn = document.getElementById(
      'disconnect-btn'
    ) as HTMLButtonElement;
    this.muteBtn = document.getElementById('mute-btn') as HTMLButtonElement;

    this.audioInput = document.getElementById(
      'audio-input'
    ) as HTMLSelectElement;
    this.videoInput = document.getElementById(
      'video-input'
    ) as HTMLSelectElement;
    this.audioCodec = document.getElementById(
      'audio-codec'
    ) as HTMLSelectElement;
    this.videoCodec = document.getElementById(
      'video-codec'
    ) as HTMLSelectElement;

    this.videoElement = document.getElementById(
      'bot-video'
    ) as HTMLVideoElement;
    this.audioElement = document.getElementById(
      'bot-audio'
    ) as HTMLAudioElement;

    this.debugLog = document.getElementById('debug-log');
    this.statusSpan = document.getElementById('connection-status');
  }

  private setupDOMEventListeners(): void {
    this.connectBtn.addEventListener('click', () => this.start());
    this.disconnectBtn.addEventListener('click', () => this.stop());
    this.audioInput.addEventListener('change', (e) => {
      // @ts-ignore
      let audioDevice = e.target?.value;
      this.pcClient.updateMic(audioDevice);
    });
    this.videoInput.addEventListener('change', (e) => {
      // @ts-ignore
      let videoDevice = e.target?.value;
      this.pcClient.updateCam(videoDevice);
    });
    this.muteBtn.addEventListener('click', () => {
      let isCamEnabled = this.pcClient.isCamEnabled;
      this.pcClient.enableCam(!isCamEnabled);
      this.muteBtn.textContent = isCamEnabled ? '📵' : '📷';
    });
  }

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
  }

  private clearAllLogs() {
    this.debugLog!.innerText = '';
  }

  private updateStatus(status: string): void {
    if (this.statusSpan) {
      this.statusSpan.textContent = status;
    }
    this.log(`Status: ${status}`);
  }

  private onConnectedHandler() {
    this.updateStatus('Connected');
    if (this.connectBtn) this.connectBtn.disabled = true;
    if (this.disconnectBtn) this.disconnectBtn.disabled = false;
  }

  private onDisconnectedHandler() {
    this.updateStatus('Disconnected');
    if (this.connectBtn) this.connectBtn.disabled = false;
    if (this.disconnectBtn) this.disconnectBtn.disabled = true;
  }

  private onBotTrackStarted(track: MediaStreamTrack) {
    if (track.kind === 'video') {
      this.videoElement.srcObject = new MediaStream([track]);
    } else {
      this.audioElement.srcObject = new MediaStream([track]);
    }
  }

  private async populateDevices(): Promise<void> {
    const populateSelect = (
      select: HTMLSelectElement,
      devices: MediaDeviceInfo[]
    ): void => {
      let counter = 1;
      devices.forEach((device) => {
        const option = document.createElement('option');
        option.value = device.deviceId;
        option.text = device.label || 'Device #' + counter;
        select.appendChild(option);
        counter += 1;
      });
    };

    try {
      const audioDevices = await this.pcClient.getAllMics();
      populateSelect(this.audioInput, audioDevices);
      const videoDevices = await this.pcClient.getAllCams();
      populateSelect(this.videoInput, videoDevices);
    } catch (e) {
      alert(e);
    }
  }

  private async start(): Promise<void> {
    this.clearAllLogs();

    this.connectBtn.disabled = true;
    this.updateStatus('Connecting');

    this.smallWebRTCTransport.setAudioCodec(this.audioCodec.value);
    this.smallWebRTCTransport.setVideoCodec(this.videoCodec.value);
    try {
      await this.pcClient.connect();
    } catch (e) {
      console.log(`Failed to connect ${e}`);
      this.stop();
    }
  }

  private stop(): void {
    void this.pcClient.disconnect();
  }
}

// Create the WebRTCConnection instance
const webRTCConnection = new WebRTCApp();
