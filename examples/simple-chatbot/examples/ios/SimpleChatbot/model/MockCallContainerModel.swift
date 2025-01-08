import SwiftUI
import RTVIClientIOS

class MockCallContainerModel: CallContainerModel {

    override init() {
    }

    override func connect(backendURL: String) {
        print("connect")
    }

    override func disconnect() {
        print("disconnect")
    }

    override func showError(message: String) {
        self.toastMessage = message
        self.showToast = true
        // Hide the toast after 5 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
            self.showToast = false
            self.toastMessage = nil
        }
    }

    func startAudioLevelSimulation() {
        // Simulate audio level changes
        Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            let newLevel = Float.random(in: 0...1)
            self.remoteAudioLevel = newLevel
            self.localAudioLevel = newLevel
        }
    }
}
