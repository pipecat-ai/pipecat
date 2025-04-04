import SwiftUI
import PipecatClientIOS

class MockCallContainerModel: CallContainerModel {
    
    override init() {
        super.init()
        let liveMessageFromSystem = LiveMessage(
            content: "System message",
            type: .system,
            updatedAt: Date()
        )
        let liveMessageFromUser = LiveMessage(
            content: "Message from User",
            type: .user,
            updatedAt: Date()
        )
        let liveMessageFromBot = LiveMessage(
            content: "Message from bot",
            type: .bot,
            updatedAt: Date()
        )
        self.messages = [ liveMessageFromSystem, liveMessageFromUser, liveMessageFromBot ]
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
}
