import SwiftUI

import PipecatClientIOSSmallWebrtc
import PipecatClientIOS

class CallContainerModel: ObservableObject {
    
    @Published var voiceClientStatus: String = TransportState.disconnected.description
    @Published var isInCall: Bool = false
    @Published var isBotReady: Bool = false
    
    @Published var isMicEnabled: Bool = false
    @Published var isCamEnabled: Bool = false
    @Published var localCamId: MediaTrackId? = nil
    @Published var botCamId: MediaTrackId? = nil
    
    @Published var toastMessage: String? = nil
    @Published var showToast: Bool = false
    
    @Published var messages: [LiveMessage] = []
    @Published var liveBotMessage: LiveMessage?
    @Published var liveUserMessage: LiveMessage?
    
    var rtviClientIOS: RTVIClient?
    
    @Published var selectedMic: MediaDeviceId? = nil {
        didSet {
            guard let selectedMic else { return } // don't store nil
            var settings = SettingsManager.getSettings()
            settings.selectedMic = selectedMic.id
            SettingsManager.updateSettings(settings: settings)
        }
    }
    @Published var availableMics: [MediaDeviceInfo] = []
    
    init() {
        // Changing the log level
        PipecatClientIOS.setLogLevel(.warn)
        PipecatClientIOSSmallWebrtc.setLogLevel(.info)
    }
    
    @MainActor
    func connect(backendURL: String) {
        self.resetLiveMessages()
        
        let baseUrl = backendURL.trimmingCharacters(in: .whitespacesAndNewlines)
        if(baseUrl.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty){
            self.showError(message: "Need to fill the backendURL")
            return
        }
        
        let currentSettings = SettingsManager.getSettings()
        let rtviClientOptions = RTVIClientOptions.init(
            enableMic: currentSettings.enableMic,
            enableCam: currentSettings.enableCam,
            params: RTVIClientParams(
                config: [
                    .init(
                        service: SmallWebRTCTransport.SERVICE_NAME,
                        options: [
                            .init(name: "server_url", value: .string(baseUrl))
                        ]
                    )
                ]
            )
        )
        self.rtviClientIOS = RTVIClient.init(
            transport: SmallWebRTCTransport.init(options: rtviClientOptions),
            options: rtviClientOptions
        )
        self.rtviClientIOS?.delegate = self
        
        // Registering the llm helper, we will need this to handle the function calling
        let llmHelper = try? self.rtviClientIOS?.registerHelper(service: "llm", helper: LLMHelper.self)
        llmHelper?.delegate = self
        
        self.rtviClientIOS?.start() { result in
            switch result {
            case .failure(let error):
                self.showError(message: error.localizedDescription)
                self.rtviClientIOS = nil
            case .success():
                // Apply initial mic preference
                if let selectedMic = SettingsManager.getSettings().selectedMic {
                    self.selectMic(MediaDeviceId(id: selectedMic))
                }
                // Populate available devices list
                self.availableMics = self.rtviClientIOS?.getAllMics() ?? []
            }
        }
        self.saveCredentials(backendURL: backendURL)
    }
    
    @MainActor
    func disconnect() {
        self.rtviClientIOS?.disconnect(completion: nil)
        self.rtviClientIOS?.release()
        self.rtviClientIOS = nil
    }
    
    func showError(message: String) {
        self.toastMessage = message
        self.showToast = true
        // Hide the toast after 5 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
            self.showToast = false
            self.toastMessage = nil
        }
    }
    
    @MainActor
    func toggleMicInput() {
        self.rtviClientIOS?.enableMic(enable: !self.isMicEnabled) { result in
            switch result {
            case .success():
                self.isMicEnabled = self.rtviClientIOS?.isMicEnabled ?? false
            case .failure(let error):
                self.showError(message: error.localizedDescription)
            }
        }
    }
    
    @MainActor
    func toggleCamInput() {
        self.rtviClientIOS?.enableCam(enable: !self.isCamEnabled) { result in
            switch result {
            case .success():
                self.isCamEnabled = self.rtviClientIOS?.isCamEnabled ?? false
            case .failure(let error):
                self.showError(message: error.localizedDescription)
            }
        }
    }
    
    func saveCredentials(backendURL: String) {
        var currentSettings = SettingsManager.getSettings()
        currentSettings.backendURL = backendURL
        // Saving the settings
        SettingsManager.updateSettings(settings: currentSettings)
    }
    
    @MainActor
    func selectMic(_ mic: MediaDeviceId) {
        self.selectedMic = mic
        self.rtviClientIOS?.updateMic(micId: mic, completion: nil)
    }
    
    private func createLiveMessage(content:String = "", type:MessageType) {
        // Creating a new one
        DispatchQueue.main.async {
            let liveMessage = LiveMessage(content: content, type: type, updatedAt: Date())
            self.messages.append(liveMessage)
            if type == .bot {
                self.liveBotMessage = liveMessage
            } else if type == .user {
                self.liveUserMessage = liveMessage
            }
        }
    }
    
    private func appendTextToLiveMessage(fromBot: Bool, content:String) {
        DispatchQueue.main.async {
            // Updating the last message with the new content
            if fromBot {
                self.liveBotMessage?.content += content
            } else {
                self.liveUserMessage?.content += content
            }
        }
    }
    
    private func resetLiveMessages() {
        DispatchQueue.main.async {
            self.messages = []
        }
    }
}

extension CallContainerModel:RTVIClientDelegate, LLMHelperDelegate {
    
    private func handleEvent(eventName: String, eventValue: Any? = nil) {
        if let value = eventValue {
            print("Pipecat Demo, received event: \(eventName), value:\(value)")
        } else {
            print("Pipecat Demo, received event: \(eventName)")
        }
    }
    
    func onTransportStateChanged(state: TransportState) {
        Task { @MainActor in
            self.handleEvent(eventName: "onTransportStateChanged", eventValue: state)
            self.voiceClientStatus = state.description
            self.isInCall = ( state == .connecting || state == .connected || state == .ready || state == .authenticating )
            self.createLiveMessage(content: state.description, type: .system)
        }
    }
    
    func onBotReady(botReadyData: BotReadyData) {
        Task { @MainActor in
            self.handleEvent(eventName: "onBotReady")
            self.isBotReady = true
        }
    }
    
    func onConnected() {
        Task { @MainActor in
            self.handleEvent(eventName: "onConnected")
            self.isMicEnabled = self.rtviClientIOS?.isMicEnabled ?? false
            self.isCamEnabled = self.rtviClientIOS?.isCamEnabled ?? false
        }
    }
    
    func onDisconnected() {
        Task { @MainActor in
            self.handleEvent(eventName: "onDisconnected")
            self.isBotReady = false
        }
    }
    
    func onError(message: String) {
        Task { @MainActor in
            self.handleEvent(eventName: "onError", eventValue: message)
            self.showError(message: message)
        }
    }
    
    func onAvailableMicsUpdated(mics: [MediaDeviceInfo]) {
        Task { @MainActor in
            self.availableMics = mics
        }
    }
    
    func onMicUpdated(mic: MediaDeviceInfo?) {
        Task { @MainActor in
            self.selectedMic = mic?.id
        }
    }
    
    func onBotTranscript(data: String) {
        self.handleEvent(eventName: "onBotTranscript", eventValue: data)
    }
    
    func onTracksUpdated(tracks: Tracks) {
        self.handleEvent(eventName: "onTracksUpdated", eventValue: tracks)
        Task { @MainActor in
            self.localCamId = tracks.local.video
            self.botCamId = tracks.bot?.video ?? nil
        }
    }
    
    func onUserStartedSpeaking() {
        self.createLiveMessage(content: "User started speaking", type: .system)
        self.handleEvent(eventName: "onUserStartedSpeaking")
        self.createLiveMessage(type: .user)
    }
    
    func onUserStoppedSpeaking() {
        self.createLiveMessage(content: "User stopped speaking", type: .system)
        self.handleEvent(eventName: "onUserStoppedSpeaking")
    }
    
    func onBotStartedSpeaking() {
        self.createLiveMessage(content: "Bot started speaking", type: .system)
        self.handleEvent(eventName: "onBotStartedSpeaking")
        self.createLiveMessage(type: .bot)
    }
    
    func onBotStoppedSpeaking() {
        self.createLiveMessage(content: "Bot stopped speaking", type: .system)
        self.handleEvent(eventName: "onBotStoppedSpeaking")
    }
    
    func onUserTranscript(data: Transcript) {
        if data.final ?? false {
            self.handleEvent(eventName: "onUserTranscript", eventValue: data.text)
            self.appendTextToLiveMessage(fromBot: false, content: data.text)
        }
    }
    
    func onBotTTSText(data: BotTTSText) {
        self.appendTextToLiveMessage(fromBot: true, content: data.text)
    }
    
}
