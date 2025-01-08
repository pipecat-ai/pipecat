import SwiftUI

import RTVIClientIOSDaily
import RTVIClientIOS

class CallContainerModel: ObservableObject {
    
    @Published var voiceClientStatus: String = TransportState.disconnected.description
    @Published var isInCall: Bool = false
    @Published var isBotReady: Bool = false
    @Published var timerCount = 0
    
    @Published var isMicEnabled: Bool = false
    
    @Published var toastMessage: String? = nil
    @Published var showToast: Bool = false
    
    @Published
    var remoteAudioLevel: Float = 0
    @Published
    var localAudioLevel: Float = 0
    
    private var meetingTimer: Timer?
    
    var rtviClientIOS: RTVIClient?
    
    init() {
        // Changing the log level
        RTVIClientIOS.setLogLevel(.warn)
    }
    
    @MainActor
    func connect(backendURL: String) {
        let baseUrl = backendURL.trimmingCharacters(in: .whitespacesAndNewlines)
        if(baseUrl.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty){
            self.showError(message: "Need to fill the backendURL. For more info visit: https://bots.daily.co")
            return
        }
        
        let currentSettings = SettingsManager.getSettings()
        let rtviClientOptions = RTVIClientOptions.init(
            enableMic: currentSettings.enableMic,
            enableCam: false,
            params: RTVIClientParams(
                baseUrl: baseUrl,
                endpoints: RTVIURLEndpoints(connect: "/connect")
            )
        )
        self.rtviClientIOS = RTVIClient.init(
            transport: DailyTransport.init(options: rtviClientOptions),
            options: rtviClientOptions
        )
        self.rtviClientIOS?.delegate = self
        self.rtviClientIOS?.start() { result in
            if case .failure(let error) = result {
                self.showError(message: error.localizedDescription)
                self.rtviClientIOS = nil
            }
        }
        // Selecting the mic based on the preferences
        if let selectedMic = currentSettings.selectedMic {
            self.rtviClientIOS?.updateMic(micId: MediaDeviceId(id:selectedMic), completion: nil)
        }
        self.saveCredentials(backendURL: baseUrl)
    }
    
    @MainActor
    func disconnect() {
        self.rtviClientIOS?.disconnect(completion: nil)
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
    
    private func startTimer(withExpirationTime expirationTime: Int) {
        let currentTime = Int(Date().timeIntervalSince1970)
        self.timerCount = expirationTime - currentTime
        self.meetingTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { timer in
            DispatchQueue.main.async {
                self.timerCount -= 1
            }
        }
    }
    
    private func stopTimer() {
        self.meetingTimer?.invalidate()
        self.meetingTimer = nil
        self.timerCount = 0
    }
    
    func saveCredentials(backendURL: String) {
        var currentSettings = SettingsManager.getSettings()
        currentSettings.backendURL = backendURL
        // Saving the settings
        SettingsManager.updateSettings(settings: currentSettings)
    }
    
}

extension CallContainerModel:RTVIClientDelegate, LLMHelperDelegate {
    
    private func handleEvent(eventName: String, eventValue: Any? = nil) {
        if let value = eventValue {
            print("RTVI Demo, received event:\(eventName), value:\(value)")
        } else {
            print("RTVI Demo, received event: \(eventName)")
        }
    }
    
    func onTransportStateChanged(state: TransportState) {
        self.handleEvent(eventName: "onTransportStateChanged", eventValue: state)
        self.voiceClientStatus = state.description
        self.isInCall = ( state == .connecting || state == .connected || state == .ready || state == .authenticating )
    }
    
    @MainActor
    func onBotReady(botReadyData: BotReadyData) {
        self.handleEvent(eventName: "onBotReady.")
        self.isBotReady = true
        if let expirationTime = self.rtviClientIOS?.expiry() {
            self.startTimer(withExpirationTime: expirationTime)
        }
    }
    
    @MainActor
    func onConnected() {
        self.isMicEnabled = self.rtviClientIOS?.isMicEnabled ?? false
    }
    
    func onDisconnected() {
        self.stopTimer()
        self.isBotReady = false
    }
    
    func onRemoteAudioLevel(level: Float, participant: Participant) {
        self.remoteAudioLevel = level
    }
    
    func onUserAudioLevel(level: Float) {
        self.localAudioLevel = level
    }
    
    func onUserTranscript(data: Transcript) {
        if (data.final ?? false) {
            self.handleEvent(eventName: "onUserTranscript", eventValue: data.text)
        }
    }
    
    func onBotTranscript(data: String) {
        self.handleEvent(eventName: "onBotTranscript", eventValue: data)
    }
    
    func onError(message: String) {
        self.handleEvent(eventName: "onError", eventValue: message)
        self.showError(message: message)
    }
    
    func onTracksUpdated(tracks: Tracks) {
        self.handleEvent(eventName: "onTracksUpdated", eventValue: tracks)
    }
    
}
