// TODO: we should refactor everything that is inside this class,
// and create a Pipecat client transport here
// https://github.com/pipecat-ai/pipecat-client-web-transports

const SIGNALLING_TYPE = "signalling";

enum SignallingMessage {
    RENEGOTIATE = "renegotiate",
}

// Interface for the structure of the signalling message
interface SignallingMessageObject {
    type: string;
    message: SignallingMessage;
}

export type SmallWebRTCTransportCallbacks = {
    onLog(message: string): void;
    onTrackStarted(track: MediaStreamTrack): void;
    onConnected(): void;
    onDisconnected(): void;
}

export class SmallWebRTCTransport {

    private _callbacks: SmallWebRTCTransportCallbacks;

    private pc: RTCPeerConnection | null = null;
    private dc: RTCDataChannel | null = null;
    private audioCodec: string | null = null;
    private videoCodec: string | null = null;
    private pc_id: string | null = null;

    constructor(callbacks: SmallWebRTCTransportCallbacks) {
        this._callbacks = callbacks
    }

    private log(message: string): void {
        console.log(message);
        this._callbacks.onLog(message)
    }

    private createPeerConnection(): RTCPeerConnection {
        const config: RTCConfiguration = {
            //iceServers: [{ urls: ['stun:stun.l.google.com:19302'] }]
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
            this._callbacks.onTrackStarted(evt.track)
        });

        pc.onconnectionstatechange = () => {
            let connectionState = this.pc?.connectionState
            this.log(`connectionState: ${connectionState}`)
            if (connectionState == 'connected') {
                this._callbacks.onConnected()
            } else if (connectionState == 'disconnected') {
                this.handleDisconnected()
            }
        }

        return pc;
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
            if (this.audioCodec && this.audioCodec !== 'default') {
                // @ts-ignore
                offerSdp.sdp = this.sdpFilterCodec('audio', this.audioCodec, offerSdp.sdp);
            }

            // Filter video codec
            if (this.videoCodec && this.videoCodec !== 'default') {
                // @ts-ignore
                offerSdp.sdp = this.sdpFilterCodec('video', this.videoCodec, offerSdp.sdp);
            }

            this.log(`Will create offer for peerId: ${this.pc_id}`)

            // Send offer to server
            const response = await fetch('/api/offer', {
                body: JSON.stringify({
                    sdp: offerSdp.sdp,
                    type: offerSdp.type,
                    pc_id: this.pc_id
                }),
                headers: {
                    'Content-Type': 'application/json',
                },
                method: 'POST',
            });

            const answer: RTCSessionDescriptionInit = await response.json();
            // @ts-ignore
            this.pc_id = answer.pc_id
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

    private handleDisconnected() {
        this.pc_id = null
        this._callbacks.onDisconnected()
    }

    async start(audioDevice: string | undefined, audioCodec: string, videoCodec: string, videoDevice: string | undefined): Promise<void> {
        this.pc = this.createPeerConnection();
        this.addInitialTransceivers();
        this.dc = this.createDataChannel('chat', { ordered: true });
        await this.addUserMedias(audioDevice, videoDevice);
        this.audioCodec = audioCodec
        this.videoCodec = videoCodec
        await this.negotiate();
    }

    private async addUserMedias(audioDevice: string|undefined, videoDevice:string|undefined): Promise<void> {
        this.log("Will send the audio and video");
        const constraints = this.createMediaConstraints(audioDevice, videoDevice);

        if (constraints.audio || constraints.video) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia(constraints);

                let videoTrack = stream.getVideoTracks()[0]
                await this.getVideoTransceiver().sender.replaceTrack(videoTrack);

                let audioTrack = stream.getAudioTracks()[0]
                await this.getAudioTransceiver().sender.replaceTrack(audioTrack);
            } catch (err) {
                alert(`Could not acquire media: ${err}`);
            }
        }
    }

    // Method to handle a general message (this can be expanded for other types of messages)
    handleMessage(message: string): void {
        try {
            this.log(message)

            const messageObj = JSON.parse(message); // Type is `any` initially

            // Check if it's a signalling message
            if (messageObj.type === SIGNALLING_TYPE) {
                void this.handleSignallingMessage(messageObj as SignallingMessageObject); // Delegate to handleSignallingMessage
            } else {
                // implement to handle the other messages in the future
            }
        } catch (error) {
            console.error("Failed to parse JSON message:", error);
        }
    }

    // Method to handle signalling messages specifically
    async handleSignallingMessage(messageObj: SignallingMessageObject): Promise<void> {
        // Cast the object to the correct type after verification
        const signallingMessage = messageObj as SignallingMessageObject;

        // Handle different signalling message types
        switch (signallingMessage.message) {
            case SignallingMessage.RENEGOTIATE:
                await this.negotiate()
                break;

            default:
                console.warn("Unknown signalling message:", signallingMessage.message);
        }
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
            // Sending message that the client is ready, just for testing
            dc.send(JSON.stringify({id: 'clientReady', label: 'rtvi-ai', type:'client-ready'}))
        });

        dc.addEventListener('message', (evt: MessageEvent) => {
            let message = evt.data
            this.handleMessage(message)
        });

        return dc;
    }

    private createMediaConstraints(audioDevice: string|undefined, videoDevice:string|undefined): MediaStreamConstraints {
        const constraints: MediaStreamConstraints = { audio: false, video: false };

        const audioConstraints: MediaTrackConstraints = {};
        if (audioDevice) audioConstraints.deviceId = { exact: audioDevice };

        constraints.audio = Object.keys(audioConstraints).length ? audioConstraints : true;

        const videoConstraints: MediaTrackConstraints = {};
        if (videoDevice) videoConstraints.deviceId = { exact: videoDevice };

        constraints.video = Object.keys(videoConstraints).length ? videoConstraints : true;

        return constraints;
    }

    stop(): void {
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

        this.pc.close();

        // For some reason after we close the peer connection, it is not triggering the listeners
        this.handleDisconnected()
    }

    private async getAllDevices() {
        return await navigator.mediaDevices.enumerateDevices();
    }

    async getAllCams() {
        const devices = await this.getAllDevices();
        return devices.filter((d) => d.kind === "videoinput");
    }

    async getAllMics() {
        const devices = await this.getAllDevices();
        return devices.filter((d) => d.kind === "audioinput");
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