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

    private reconnectionAttempts = 0;
    private maxReconnectionAttempts = 3;
    private isReconnecting = false;
    private keepAliveInterval: number | null = null;
    private audioDevice: string | undefined;
    private videoDevice: string | undefined;

    constructor(callbacks: SmallWebRTCTransportCallbacks) {
        this._callbacks = callbacks
        // for testing reconnections
        // @ts-ignore
        window.attemptReconnection = this.attemptReconnection.bind(this)
    }

    private log(message: string): void {
        console.log(message);
        this._callbacks.onLog(message)
    }

    private createPeerConnection(): RTCPeerConnection {
        const config: RTCConfiguration = {
            iceServers: [{ urls: ['stun:stun.l.google.com:19302'] }]
        };

        let pc = new RTCPeerConnection(config);

        pc.addEventListener('icegatheringstatechange', () => {
            this.log(`iceGatheringState: ${this.pc!.iceGatheringState}`)
        });
        this.log(`iceGatheringState: ${pc.iceGatheringState}`)

        pc.addEventListener("iceconnectionstatechange", () => this.handleICEConnectionStateChange());

        pc.addEventListener("connectionstatechange", () => this.handleConnectionStateChange());

        this.log(`iceConnectionState: ${pc.iceConnectionState}`)

        pc.addEventListener('signalingstatechange', () => {
            this.log(`signalingState: ${this.pc!.signalingState}`)
            if (this.pc!.signalingState == 'stable') {
                this.handleReconnectionCompleted()
            }
        });
        this.log(`signalingState: ${pc.signalingState}`)

        pc.addEventListener('track', (evt: RTCTrackEvent) => {
            this.log(`Received new track ${evt.track.kind}`)
            this._callbacks.onTrackStarted(evt.track)
        });

        return pc;
    }

    private handleICEConnectionStateChange(): void {
        if (!this.pc) return;
        this.log(`ICE Connection State: ${this.pc.iceConnectionState}`);

        if (this.pc.iceConnectionState === "failed") {
            this.log("ICE connection failed, attempting restart.");
            void this.attemptReconnection(true);
        } else if (this.pc.iceConnectionState === "disconnected") {
            // Waiting before trying to reconnect to see if it handles it automatically
            setTimeout(() => {
                if (this.pc?.iceConnectionState === "disconnected") {
                    this.log("Still disconnected, attempting reconnection.");
                    void this.attemptReconnection(true);
                }
            }, 5000);
        }
    }

    private handleReconnectionCompleted() {
        this.reconnectionAttempts = 0;
        this.isReconnecting = false;
    }

    private handleConnectionStateChange(): void {
        if (!this.pc) return;
        this.log(`Connection State: ${this.pc.connectionState}`);

        if (this.pc.connectionState === "connected") {
            this.handleReconnectionCompleted()
            this._callbacks.onConnected();
        } else if (this.pc.connectionState === "failed") {
            void this.attemptReconnection(true);
        }
    }

    private async attemptReconnection(recreatePeerConnection: boolean = false): Promise<void> {
        if (this.isReconnecting) {
            this.log("Reconnection already in progress, skipping.");
            return;
        }
        if (this.reconnectionAttempts >= this.maxReconnectionAttempts) {
            this.log("Max reconnection attempts reached. Stopping transport.");
            this.stop();
            return;
        }
        this.isReconnecting = true;
        this.reconnectionAttempts++;
        this.log(`Reconnection attempt ${this.reconnectionAttempts}...`);
        // aiortc it is not working fine when just trying to restart the ice
        // so in this case we are creating a new peer connection on both sides
        if (recreatePeerConnection) {
            const oldPC = this.pc
            await this.startNewPeerConnection(recreatePeerConnection)
            if (oldPC) {
                this.log("closing old peer connection")
                this.closePeerConnection(oldPC)
            }
        } else {
            await this.negotiate();
        }
    }

    private async negotiate(recreatePeerConnection: boolean = false): Promise<void> {
        if (!this.pc) {
            return Promise.reject('Peer connection is not initialized');
        }

        try {
            // Create offer
            const offer = await this.pc.createOffer();
            await this.pc.setLocalDescription(offer);

            // Wait for ICE gathering to complete
            /*await new Promise<void>((resolve) => {
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
            });*/

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
                    pc_id: this.pc_id,
                    restart_pc: recreatePeerConnection
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
            this.log(`Remote candidate supports trickle ice: ${this.pc.canTrickleIceCandidates}`)
        } catch (e) {
            this.log(`Reconnection attempt ${this.reconnectionAttempts} failed: ${e}`);
            this.isReconnecting = false
            setTimeout(() => this.attemptReconnection(true), 2000);
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

    async start(audioDevice: string | undefined, audioCodec: string, videoCodec: string, videoDevice: string | undefined): Promise<void> {
        this.audioDevice = audioDevice
        this.videoDevice = videoDevice
        this.audioCodec = audioCodec
        this.videoCodec = videoCodec
        await this.startNewPeerConnection()
    }

    private async startNewPeerConnection(recreatePeerConnection: boolean = false) {
        this.pc = this.createPeerConnection();
        this.addInitialTransceivers();
        this.dc = this.createDataChannel('chat', { ordered: true });
        await this.addUserMedias();
        await this.negotiate(recreatePeerConnection);
    }

    private async addUserMedias(): Promise<void> {
        this.log("Will send the audio and video");
        const constraints = this.createMediaConstraints();

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
                void this.attemptReconnection(false)
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
            if (this.keepAliveInterval) {
                clearInterval(this.keepAliveInterval)
                this.keepAliveInterval = null
            }
        });

        dc.addEventListener('open', () => {
            this.log("datachannel opened")
            // Sending message that the client is ready, just for testing
            dc.send(JSON.stringify({id: 'clientReady', label: 'rtvi-ai', type:'client-ready'}))
            // @ts-ignore
            this.keepAliveInterval = setInterval(() => {
                const message = 'ping: ' + new Date().getTime()
                dc.send(message);
            }, 1000);

        });

        dc.addEventListener('message', (evt: MessageEvent) => {
            let message = evt.data
            this.handleMessage(message)
        });

        return dc;
    }

    private createMediaConstraints(): MediaStreamConstraints {
        const constraints: MediaStreamConstraints = { audio: false, video: false };

        const audioConstraints: MediaTrackConstraints = {};
        if (this.audioDevice) audioConstraints.deviceId = { exact: this.audioDevice };

        constraints.audio = Object.keys(audioConstraints).length ? audioConstraints : true;

        const videoConstraints: MediaTrackConstraints = {};
        if (this.videoDevice) videoConstraints.deviceId = { exact: this.videoDevice };

        constraints.video = Object.keys(videoConstraints).length ? videoConstraints : true;

        return constraints;
    }

    private closePeerConnection(pc:RTCPeerConnection) {
        pc.getTransceivers().forEach((transceiver) => {
            if (transceiver.stop) {
                transceiver.stop();
            }
        });

        pc.getSenders().forEach((sender) => {
            sender.track?.stop();
        });

        pc.close();
    }

    stop(): void {
        if (!this.pc) {
            this.log("Peer connection is already closed or null.");
            return;
        }

        if (this.dc) {
            this.dc.close();
        }

        this.closePeerConnection(this.pc)

        // For some reason after we close the peer connection, it is not triggering the listeners
        this.pc_id = null
        this.reconnectionAttempts = 0
        this.isReconnecting = false;
        this._callbacks.onDisconnected()
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