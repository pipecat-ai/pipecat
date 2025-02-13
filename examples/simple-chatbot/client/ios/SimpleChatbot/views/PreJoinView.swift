import SwiftUI

struct PreJoinView: View {
    
    @State var backendURL: String

    @EnvironmentObject private var model: CallContainerModel
    
    init() {
        let currentSettings = SettingsManager.getSettings()
        self.backendURL = currentSettings.backendURL
    }

    var body: some View {
        VStack(spacing: 20) {
            Image("pipecat")
                .resizable()
                .frame(width: 80, height: 80)
            Text("Pipecat Client iOS.")
                .font(.headline)
            TextField("Server URL", text: $backendURL)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .frame(maxWidth: .infinity)
                .padding([.bottom, .horizontal])
            Button("Connect") {
                self.model.connect(backendURL: self.backendURL)
            }
            .padding()
            .background(Color.black)
            .foregroundColor(.white)
            .cornerRadius(8)
        }
        .padding()
        .frame(maxHeight: .infinity)
        .background(Color.backgroundApp)
        .toast(message: model.toastMessage, isShowing: model.showToast)
    }
}

#Preview {
    PreJoinView().environmentObject(MockCallContainerModel() as CallContainerModel)
}
