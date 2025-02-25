class WebRTCConnection {
    private dataChannelLog: HTMLElement;
    private iceConnectionLog: HTMLElement;
    private iceGatheringLog: HTMLElement;
    private signalingLog: HTMLElement;

    private pc: RTCPeerConnection | null = null;
    private dc: RTCDataChannel | null = null;
    private dcInterval: number | null = null;

    private startButton: HTMLButtonElement;
    private stopButton: HTMLButtonElement;

    constructor() {
        this.dataChannelLog = document.getElementById('data-channel')!;
        this.iceConnectionLog = document.getElementById('ice-connection-state')!;
        this.iceGatheringLog = document.getElementById('ice-gathering-state')!;
        this.signalingLog = document.getElementById('signaling-state')!;

        this.startButton = <HTMLButtonElement>document.getElementById('start')!;
        this.startButton.addEventListener("click", () => {
            this.start()
        })

        this.stopButton = <HTMLButtonElement>document.getElementById('stop')!;
        this.stopButton.addEventListener("click", () => {
            this.stop()
        })

        this.enumerateInputDevices();
    }

    private createPeerConnection(): RTCPeerConnection {
        const config: RTCConfiguration = {
            //sdpSemantics: 'unified-plan'
        };

        let stunElement:HTMLInputElement = <HTMLInputElement> document.getElementById('use-stun')
        if (stunElement && stunElement.checked) {
            config.iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }];
        }

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
                // @ts-ignore
                document.getElementById('video')!.srcObject = evt.streams[0];
            } else {
                // @ts-ignore
                document.getElementById('audio')!.srcObject = evt.streams[0];
            }
        });

        return this.pc;
    }

    private enumerateInputDevices(): void {
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

        navigator.mediaDevices.enumerateDevices().then((devices) => {
            populateSelect(
                document.getElementById('audio-input') as HTMLSelectElement,
                devices.filter((device) => device.kind === 'audioinput')
            );
            populateSelect(
                document.getElementById('video-input') as HTMLSelectElement,
                devices.filter((device) => device.kind === 'videoinput')
            );
        }).catch((e) => {
            alert(e);
        });
    }

    private negotiate(): Promise<void> {
        if (!this.pc) return Promise.reject('Peer connection is not initialized');

        return this.pc.createOffer().then((offer) => {
            return this.pc!.setLocalDescription(offer);
        }).then(() => {
            return new Promise<void>((resolve) => {
                if (this.pc!.iceGatheringState === 'complete') {
                    resolve();
                } else {
                    const checkState = () => {
                        if (this.pc!.iceGatheringState === 'complete') {
                            this.pc!.removeEventListener('icegatheringstatechange', checkState);
                            resolve();
                        }
                    }
                    this.pc!.addEventListener('icegatheringstatechange', checkState);
                }
            });
        }).then(() => {
            let offer = this.pc!.localDescription!;
            let codec: string;

            codec = (document.getElementById('audio-codec') as HTMLSelectElement).value;
            if (codec !== 'default') {
                // @ts-ignore
                offer.sdp = this.sdpFilterCodec('audio', codec, offer.sdp);
            }

            codec = (document.getElementById('video-codec') as HTMLSelectElement).value;
            if (codec !== 'default') {
                // @ts-ignore
                offer.sdp = this.sdpFilterCodec('video', codec, offer.sdp);
            }

            (document.getElementById('offer-sdp') as HTMLElement).textContent = offer.sdp;

            return fetch('/api/offer', {
                body: JSON.stringify({
                    sdp: offer.sdp,
                    type: offer.type,
                    video_transform: (document.getElementById('video-transform') as HTMLInputElement).value
                }),
                headers: {
                    'Content-Type': 'application/json'
                },
                method: 'POST'
            });
        }).then((response) => {
            return response.json();
        }).then((answer: RTCSessionDescriptionInit) => {
            (document.getElementById('answer-sdp') as HTMLElement).textContent = answer.sdp || '';
            return this.pc!.setRemoteDescription(answer);
        }).catch((e) => {
            alert(e);
        });
    }

    private start(): void {
        (document.getElementById('start') as HTMLElement).style.display = 'none';

        this.pc = this.createPeerConnection();

        let time_start: number | null = null;

        const current_stamp = (): number => {
            if (time_start === null) {
                time_start = new Date().getTime();
                return 0;
            } else {
                return new Date().getTime() - time_start;
            }
        };

        if ((document.getElementById('use-datachannel') as HTMLInputElement).checked) {
            const parameters = JSON.parse((document.getElementById('datachannel-parameters') as HTMLTextAreaElement).value);

            this.dc = this.pc.createDataChannel('chat', parameters);
            this.dc.addEventListener('close', () => {
                if (this.dcInterval) clearInterval(this.dcInterval);
                this.dataChannelLog.textContent += '- close\n';
            });
            this.dc.addEventListener('open', () => {
                this.dataChannelLog.textContent += '- open\n';
                // @ts-ignore
                this.dcInterval = setInterval(() => {
                    const message = 'ping ' + current_stamp();
                    this.dataChannelLog.textContent += '> ' + message + '\n';
                    this.dc!.send(message);
                }, 1000);
            });
            this.dc.addEventListener('message', (evt: MessageEvent) => {
                this.dataChannelLog.textContent += '< ' + evt.data + '\n';

                if (evt.data.substring(0, 4) === 'pong') {
                    const elapsed_ms = current_stamp() - parseInt(evt.data.substring(5), 10);
                    this.dataChannelLog.textContent += ' RTT ' + elapsed_ms + ' ms\n';
                }
            });
        }

        const constraints: MediaStreamConstraints = {
            audio: false,
            video: false
        };

        if ((document.getElementById('use-audio') as HTMLInputElement).checked) {
            const audioConstraints: MediaTrackConstraints = {};

            const device = (document.getElementById('audio-input') as HTMLSelectElement).value;
            if (device) {
                audioConstraints.deviceId = { exact: device };
            }

            constraints.audio = Object.keys(audioConstraints).length ? audioConstraints : true;
        }

        if ((document.getElementById('use-video') as HTMLInputElement).checked) {
            const videoConstraints: MediaTrackConstraints = {};

            const device = (document.getElementById('video-input') as HTMLSelectElement).value;
            if (device) {
                videoConstraints.deviceId = { exact: device };
            }

            const resolution = (document.getElementById('video-resolution') as HTMLSelectElement).value;
            if (resolution) {
                const dimensions = resolution.split('x');
                videoConstraints.width = parseInt(dimensions[0], 10);
                videoConstraints.height = parseInt(dimensions[1], 10);
            }

            constraints.video = Object.keys(videoConstraints).length ? videoConstraints : true;
        }

        if (constraints.audio || constraints.video) {
            if (constraints.video) {
                (document.getElementById('media') as HTMLElement).style.display = 'block';
            }
            navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
                stream.getTracks().forEach((track) => {
                    this.pc!.addTrack(track, stream);
                });
                return this.negotiate();
            }, (err) => {
                alert('Could not acquire media: ' + err);
            });
        } else {
            this.negotiate();
        }

        (document.getElementById('stop') as HTMLElement).style.display = 'inline-block';
    }

    private stop(): void {
        (document.getElementById('stop') as HTMLElement).style.display = 'none';

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
