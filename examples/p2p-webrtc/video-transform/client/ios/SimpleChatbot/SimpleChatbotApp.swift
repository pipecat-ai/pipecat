import SwiftUI

@main
struct SimpleChatbotApp: App {

    @StateObject var callContainerModel = CallContainerModel()

    var body: some Scene {
        WindowGroup {
            if (!callContainerModel.isInCall) {
                PreJoinView().environmentObject(callContainerModel)
            } else {
                MeetingView().environmentObject(callContainerModel)
            }
        }
    }

}
