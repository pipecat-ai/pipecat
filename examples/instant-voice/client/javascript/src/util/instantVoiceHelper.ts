import {RTVIClientHelper, RTVIClientHelperOptions, RTVIMessage} from "@pipecat-ai/client-js";
import {DailyRTVIMessageType} from '@pipecat-ai/daily-transport';

export type InstantVoiceHelperCallbacks = Partial<{
    onAudioBufferingStarted: () => void;
    onAudioBufferingStopped: () => void;
}>;

// --- Interface and class
export interface InstantVoiceHelperOptions extends RTVIClientHelperOptions {
    callbacks?: InstantVoiceHelperCallbacks;
}
export class InstantVoiceHelper extends RTVIClientHelper {

    protected declare _options: InstantVoiceHelperOptions;

    constructor(options: InstantVoiceHelperOptions) {
        super(options);
    }

    handleMessage(rtviMessage: RTVIMessage): void {
        switch (rtviMessage.type) {
            case DailyRTVIMessageType.AUDIO_BUFFERING_STARTED:
                if (this._options.callbacks?.onAudioBufferingStarted) {
                    this._options.callbacks?.onAudioBufferingStarted()
                }
                break;
            case DailyRTVIMessageType.AUDIO_BUFFERING_STOPPED:
                if (this._options.callbacks?.onAudioBufferingStopped) {
                    this._options.callbacks?.onAudioBufferingStopped()
                }
                break;
        }
    }

    getMessageTypes(): string[] {
        return [DailyRTVIMessageType.AUDIO_BUFFERING_STARTED, DailyRTVIMessageType.AUDIO_BUFFERING_STOPPED];
    }
}
