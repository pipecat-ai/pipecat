class WebRTCConnection {

    private declare connectBtn: HTMLButtonElement;
    private declare disconnectBtn: HTMLButtonElement;

    private declare audioInput: HTMLSelectElement;
    private declare videoInput: HTMLSelectElement;
    private declare audioCodec: HTMLSelectElement;
    private declare videoCodec: HTMLSelectElement;

    private declare videoElement: HTMLVideoElement;
    private declare audioElement: HTMLAudioElement;

    private pc: RTCPeerConnection | null = null;
    private dc: RTCDataChannel | null = null;

    private debugLog: HTMLElement | null = null;
    private statusSpan: HTMLElement | null = null;

    constructor() {
        this.setupDOMElements();
        this.setupDOMEventListeners();
        void this.enumerateInputDevices();
    }

    private setupDOMElements(): void {
        this.connectBtn = document.getElementById('connect-btn') as HTMLButtonElement;
        this.disconnectBtn = document.getElementById('disconnect-btn') as HTMLButtonElement;

        this.audioInput = document.getElementById('audio-input') as HTMLSelectElement;
        this.videoInput = document.getElementById('video-input') as HTMLSelectElement;
        this.audioCodec = document.getElementById('audio-codec') as HTMLSelectElement;
        this.videoCodec = document.getElementById('video-codec') as HTMLSelectElement;

        this.videoElement = document.getElementById('bot-video') as HTMLVideoElement;
        this.audioElement = document.getElementById('bot-audio') as HTMLAudioElement;

        this.debugLog = document.getElementById('debug-log');
        this.statusSpan = document.getElementById('connection-status');
    }

    private setupDOMEventListeners(): void {
        this.connectBtn.addEventListener("click", () => this.start());
        this.disconnectBtn.addEventListener("click", () => this.stop());
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
        console.log(message);
    }

    private clearAllLogs() {
        this.debugLog!.innerText = ''
    }

    private updateStatus(status: string): void {
        if (this.statusSpan) {
            this.statusSpan.textContent = status;
        }
        this.log(`Status: ${status}`);
   }

    private createPeerConnection(): RTCPeerConnection {
        const config: RTCConfiguration = {
            // sdpSemantics: 'unified-plan'
            iceServers: [{ urls: ['stun:stun.l.google.com:19302'] }]
        };

        let pc = new RTCPeerConnection(config);

        pc.addEventListener('icegatheringstatechange', () => {
            this.log(`iceGatheringState: ${this.pc!.iceGatheringState}`)
        });
        this.log(`iceGatheringState: ${pc.iceGatheringState}`)

        pc.addEventListener('iceconnectionstatechange', () => {
            let connectionState = this.pc!.iceConnectionState
            this.log(`iceConnectionState: ${connectionState}`)
        });
        this.log(`iceConnectionState: ${pc.iceConnectionState}`)

        pc.addEventListener('signalingstatechange', () => {
            this.log(`signalingState: ${this.pc!.signalingState}`)
        });
        this.log(`signalingState: ${pc.signalingState}`)

        pc.addEventListener('track', (evt: RTCTrackEvent) => {
            this.log(`Received new track ${evt.track.kind}`)
            if (evt.track.kind === 'video') {
                this.videoElement.srcObject = evt.streams[0];
            } else {
                this.audioElement.srcObject = evt.streams[0];
            }
        });

        pc.onconnectionstatechange = () => {
            let connectionState = this.pc?.connectionState
            this.log(`connectionState: ${connectionState}`)
            if (connectionState == 'connected') {
                this.onConnectedHandler()
            } else if (connectionState == 'disconnected') {
                this.onDisconnectedHandler()
            }
        }

        return pc;
    }

    private onConnectedHandler() {
        this.updateStatus('Connected');
        if (this.connectBtn) this.connectBtn.disabled = true;
        if (this.disconnectBtn) this.disconnectBtn.disabled = false;

        //Since we are using transceivers, the event with a new track is not triggered
        this.log("onConnectedHandler, trying to setup the tracks")
        this.audioElement.srcObject = new MediaStream([this.getAudioTransceiver().receiver.track]);
        this.videoElement.srcObject = new MediaStream([this.getVideoTransceiver().receiver.track]);
    }

    private onDisconnectedHandler() {
        this.updateStatus('Disconnected');
        if (this.connectBtn) this.connectBtn.disabled = false;
        if (this.disconnectBtn) this.disconnectBtn.disabled = true;
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
                }),
                headers: {
                    'Content-Type': 'application/json',
                },
                method: 'POST',
            });

            const answer: RTCSessionDescriptionInit = await response.json();
            // @ts-ignore
            this.log(`Received answer for peer connection id ${answer.pc_id}`)
            await this.pc!.setRemoteDescription(answer);
        } catch (e) {
            alert(e);
        }
    }

    private addInitialTransceivers() {
        // Transceivers always appear in creation-order for both peers
        // For now we are only considering that we are going to have 02 transceivers,
        // one for audio and one for video
        this.pc!.addTransceiver('audio', { direction: 'sendrecv' });
        this.pc!.addTransceiver('video', { direction: 'sendrecv' });
    }

    private getAudioTransceiver() {
        // Transceivers always appear in creation-order for both peers
        // Look at addInitialTransceivers
        return this.pc!.getTransceivers()[0];
    }

    private getVideoTransceiver() {
        // Transceivers always appear in creation-order for both peers
        // Look at addInitialTransceivers
        return this.pc!.getTransceivers()[1];
    }

    // TODO: we need to implement to renegotiate to support add and remove transceivers on the flight

    private async start(): Promise<void> {
        this.clearAllLogs()

        this.pc = this.createPeerConnection();
        this.addInitialTransceivers();
        this.dc = this.createDataChannel('chat', { ordered: true });

        const constraints = this.createMediaConstraints();

        if (constraints.audio || constraints.video) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia(constraints);

                let audioTrack = stream.getAudioTracks()[0]
                await this.getAudioTransceiver().sender.replaceTrack(audioTrack);

                let videoTrack = stream.getVideoTracks()[0]
                await this.getVideoTransceiver().sender.replaceTrack(videoTrack);
            } catch (err) {
                alert(`Could not acquire media: ${err}`);
            }
        }
        await this.negotiate();
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
            this.log("datachannel closed")
        });

        dc.addEventListener('open', () => {
            this.log("datachannel opened")
            // Sending message that the client is ready
            dc.send(JSON.stringify({type:'client-ready'}))
        });

        dc.addEventListener('message', (evt: MessageEvent) => {
            let message = evt.data
            this.log(message)
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

        constraints.video = Object.keys(videoConstraints).length ? videoConstraints : true;

        return constraints;
    }

    private stop(): void {
        if (!this.pc) {
            this.log("Peer connection is already closed or null.");
            return;
        }

        if (this.dc) {
            this.dc.close();
        }

        this.pc.getTransceivers().forEach((transceiver) => {
            if (transceiver.stop) {
                transceiver.stop();
            }
        });

        this.pc.getSenders().forEach((sender) => {
            sender.track?.stop();
        });

        this.log(`Before closing: 
            connectionState=${this.pc.connectionState}, 
            iceConnectionState=${this.pc.iceConnectionState}, 
            signalingState=${this.pc.signalingState}`
        );

        this.pc.close();

        this.log(`After closing: 
            connectionState=${this.pc.connectionState}, 
            iceConnectionState=${this.pc.iceConnectionState}, 
            signalingState=${this.pc.signalingState}`
        );

        // TODO: For some reason after we close the peer connection, it is not trigerring the listeners
        // need to investigate why. For now, forcing it here.
        this.onDisconnectedHandler()
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
