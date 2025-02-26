class WebRTCConnection {

    private declare dataChannelLog: HTMLElement;
    private declare iceConnectionLog: HTMLElement;
    private declare iceGatheringLog: HTMLElement;
    private declare signalingLog: HTMLElement;

    private declare startButton: HTMLButtonElement;
    private declare stopButton: HTMLButtonElement;

    private declare audioInput: HTMLSelectElement;
    private declare videoInput: HTMLSelectElement;
    private declare audioCodec: HTMLSelectElement;
    private declare videoCodec: HTMLSelectElement;
    private declare videoResolution: HTMLSelectElement;

    private declare video: HTMLVideoElement;
    private declare audio: HTMLAudioElement;
    private declare media: HTMLElement;

    private pc: RTCPeerConnection | null = null;
    private dc: RTCDataChannel | null = null;
    private dcInterval: number | null = null;

    constructor() {
        this.setupDOMElements();
        this.setupDOMEventListeners();
        void this.enumerateInputDevices();
    }

    private setupDOMElements(): void {
        this.dataChannelLog = document.getElementById('data-channel')!;
        this.iceConnectionLog = document.getElementById('ice-connection-state')!;
        this.iceGatheringLog = document.getElementById('ice-gathering-state')!;
        this.signalingLog = document.getElementById('signaling-state')!;

        this.startButton = document.getElementById('start') as HTMLButtonElement;
        this.stopButton = document.getElementById('stop') as HTMLButtonElement;

        this.audioInput = document.getElementById('audio-input') as HTMLSelectElement;
        this.videoInput = document.getElementById('video-input') as HTMLSelectElement;
        this.audioCodec = document.getElementById('audio-codec') as HTMLSelectElement;
        this.videoCodec = document.getElementById('video-codec') as HTMLSelectElement;
        this.videoResolution = document.getElementById('video-resolution') as HTMLSelectElement;

        this.video = document.getElementById('video') as HTMLVideoElement;
        this.audio = document.getElementById('audio') as HTMLAudioElement;
        this.media = document.getElementById('media') as HTMLElement;
    }

    private setupDOMEventListeners(): void {
        this.startButton.addEventListener("click", () => this.start());
        this.stopButton.addEventListener("click", () => this.stop());
    }

    private createPeerConnection(): RTCPeerConnection {
        const config: RTCConfiguration = {
            // sdpSemantics: 'unified-plan'
            iceServers: [{ urls: ['stun:stun.l.google.com:19302'] }]
        };

        this.pc = new RTCPeerConnection(config);

        this.pc.addEventListener('icegatheringstatechange', () => {
            this.iceGatheringLog.textContent += ' -> ' + this.pc!.iceGatheringState;
        });
        this.iceGatheringLog.textContent = this.pc!.iceGatheringState;

        this.pc.addEventListener('iceconnectionstatechange', () => {
            this.iceConnectionLog.textContent += ' -> ' + this.pc!.iceConnectionState;
        });
        this.iceConnectionLog.textContent = this.pc!.iceConnectionState;

        this.pc.addEventListener('signalingstatechange', () => {
            this.signalingLog.textContent += ' -> ' + this.pc!.signalingState;
        });
        this.signalingLog.textContent = this.pc!.signalingState;

        this.pc.addEventListener('track', (evt: RTCTrackEvent) => {
            if (evt.track.kind === 'video') {
                this.video.srcObject = evt.streams[0];
            } else {
                this.audio.srcObject = evt.streams[0];
            }
        });

        return this.pc;
    }

    private async enumerateInputDevices(): Promise<void> {
        const populateSelect = (select: HTMLSelectElement, devices: MediaDeviceInfo[]): void => {
            let counter = 1;
            devices.forEach((device) => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.text = device.label || ('Device #' + counter);
                select.appendChild(option);
                counter += 1;
            });
        };

        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            populateSelect(this.audioInput, devices.filter((device) => device.kind === 'audioinput'));
            populateSelect(this.videoInput, devices.filter((device) => device.kind === 'videoinput'));
        } catch (e) {
            alert(e);
        }
    }

    private async negotiate(): Promise<void> {
        if (!this.pc) {
            return Promise.reject('Peer connection is not initialized');
        }

        try {
            // Create offer
            const offer = await this.pc.createOffer();
            await this.pc.setLocalDescription(offer);

            // Wait for ICE gathering to complete
            await new Promise<void>((resolve) => {
                if (this.pc!.iceGatheringState === 'complete') {
                    resolve();
                } else {
                    const checkState = () => {
                        if (this.pc!.iceGatheringState === 'complete') {
                            this.pc!.removeEventListener('icegatheringstatechange', checkState);
                            resolve();
                        }
                    };
                    this.pc!.addEventListener('icegatheringstatechange', checkState);
                }
            });

            let offerSdp = this.pc!.localDescription!;
            let codec: string;

            // Filter audio codec
            codec = this.audioCodec.value;
            if (codec !== 'default') {
                // @ts-ignore
                offerSdp.sdp = this.sdpFilterCodec('audio', codec, offerSdp.sdp);
            }

            // Filter video codec
            codec = this.videoCodec.value;
            if (codec !== 'default') {
                // @ts-ignore
                offerSdp.sdp = this.sdpFilterCodec('video', codec, offerSdp.sdp);
            }

            // Send offer to server
            const response = await fetch('/api/offer', {
                body: JSON.stringify({
                    sdp: offerSdp.sdp,
                    type: offerSdp.type,
                    video_transform: "none", //Look the possible options at the server example
                }),
                headers: {
                    'Content-Type': 'application/json',
                },
                method: 'POST',
            });

            const answer: RTCSessionDescriptionInit = await response.json();
            await this.pc!.setRemoteDescription(answer);
        } catch (e) {
            alert(e);
        }
    }


    private async start(): Promise<void> {
        this.startButton.style.display = 'none';

        this.pc = this.createPeerConnection();
        this.dc = this.createDataChannel('chat', { ordered: true });

        const constraints = this.createMediaConstraints();

        if (constraints.audio || constraints.video) {
            if (constraints.video) this.media.style.display = 'block';

            try {
                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                stream.getTracks().forEach(track => this.pc!.addTrack(track, stream));
                await this.negotiate();
            } catch (err) {
                alert(`Could not acquire media: ${err}`);
            }
        } else {
            await this.negotiate();
        }

        this.stopButton.style.display = 'inline-block';
    }

    private createDataChannel(label: string, options: RTCDataChannelInit): RTCDataChannel {
        const dc = this.pc!.createDataChannel(label, options);
        let timeStart: number | null = null;

        const getCurrentTimestamp = (): number => {
            if (timeStart === null) {
                timeStart = Date.now();
                return 0;
            }
            return Date.now() - timeStart;
        };

        dc.addEventListener('close', () => {
            if (this.dcInterval) clearInterval(this.dcInterval);
            this.dataChannelLog.textContent += '- close\n';
        });

        dc.addEventListener('open', () => {
            this.dataChannelLog.textContent += '- open\n';
            // @ts-ignore
            this.dcInterval = setInterval(() => {
                const message = `ping ${getCurrentTimestamp()}`;
                this.dataChannelLog.textContent += `> ${message}\n`;
                dc.send(message);
            }, 1000);
        });

        dc.addEventListener('message', (evt: MessageEvent) => {
            this.dataChannelLog.textContent += `< ${evt.data}\n`;
            if (evt.data.startsWith('pong')) {
                const elapsedMs = getCurrentTimestamp() - parseInt(evt.data.substring(5), 10);
                this.dataChannelLog.textContent += ` RTT ${elapsedMs} ms\n`;
            }
        });

        return dc;
    }

    private createMediaConstraints(): MediaStreamConstraints {
        const constraints: MediaStreamConstraints = { audio: false, video: false };

        const audioConstraints: MediaTrackConstraints = {};
        const audioDevice = this.audioInput.value;
        if (audioDevice) audioConstraints.deviceId = { exact: audioDevice };

        constraints.audio = Object.keys(audioConstraints).length ? audioConstraints : true;

        const videoConstraints: MediaTrackConstraints = {};
        const videoDevice = this.videoInput.value;
        if (videoDevice) videoConstraints.deviceId = { exact: videoDevice };

        const resolution = this.videoResolution.value;
        if (resolution) {
            const [width, height] = resolution.split('x').map(Number);
            videoConstraints.width = width;
            videoConstraints.height = height;
        }

        constraints.video = Object.keys(videoConstraints).length ? videoConstraints : true;

        return constraints;
    }

    private stop(): void {
        this.stopButton.style.display = 'none';

        if (this.dc) {
            this.dc.close();
        }

        this.pc!.getTransceivers().forEach((transceiver) => {
            if (transceiver.stop) {
                transceiver.stop();
            }
        });

        this.pc!.getSenders().forEach((sender) => {
            sender.track?.stop();
        });

        setTimeout(() => {
            this.pc!.close();
        }, 500);
    }

    private sdpFilterCodec(kind: string, codec: string, realSdp: string): string {
        const allowed: number[] = [];
        const rtxRegex = new RegExp('a=fmtp:(\\d+) apt=(\\d+)\\r$');
        const codecRegex = new RegExp('a=rtpmap:([0-9]+) ' + this.escapeRegExp(codec));
        const videoRegex = new RegExp('(m=' + kind + ' .*?)( ([0-9]+))*\\s*$');

        const lines = realSdp.split('\n');

        let isKind = false;
        for (let i = 0; i < lines.length; i++) {
            if (lines[i].startsWith('m=' + kind + ' ')) {
                isKind = true;
            } else if (lines[i].startsWith('m=')) {
                isKind = false;
            }

            if (isKind) {
                const match = lines[i].match(codecRegex);
                if (match) {
                    allowed.push(parseInt(match[1]));
                }

                const matchRtx = lines[i].match(rtxRegex);
                if (matchRtx && allowed.includes(parseInt(matchRtx[2]))) {
                    allowed.push(parseInt(matchRtx[1]));
                }
            }
        }

        const skipRegex = 'a=(fmtp|rtcp-fb|rtpmap):([0-9]+)';
        let sdp = '';

        isKind = false;
        for (let i = 0; i < lines.length; i++) {
            if (lines[i].startsWith('m=' + kind + ' ')) {
                isKind = true;
            } else if (lines[i].startsWith('m=')) {
                isKind = false;
            }

            if (isKind) {
                const skipMatch = lines[i].match(skipRegex);
                if (skipMatch && !allowed.includes(parseInt(skipMatch[2]))) {
                    continue;
                } else if (lines[i].match(videoRegex)) {
                    sdp += lines[i].replace(videoRegex, '$1 ' + allowed.join(' ')) + '\n';
                } else {
                    sdp += lines[i] + '\n';
                }
            } else {
                sdp += lines[i] + '\n';
            }
        }

        return sdp;
    }

    private escapeRegExp(string: string): string {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
}

// Create the WebRTCConnection instance
const webRTCConnection = new WebRTCConnection();
