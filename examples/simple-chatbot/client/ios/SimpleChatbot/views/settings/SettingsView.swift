import SwiftUI
import PipecatClientIOS

struct SettingsView: View {
    
    @EnvironmentObject private var model: CallContainerModel
    
    @Binding var showingSettings: Bool
    
    @State private var isMicEnabled: Bool = true
    @State private var backendURL: String = ""
    
    var body: some View {
        NavigationView {
            Form {
                Section {
                    List(model.availableMics, id: \.self.id.id) { mic in
                        Button(action: {
                            model.selectMic(mic.id)
                        }) {
                            HStack {
                                Text(mic.name)
                                Spacer()
                                if mic.id == model.selectedMic {
                                    Image(systemName: "checkmark")
                                }
                            }
                        }
                    }
                } header: {
                    VStack(alignment: .leading) {
                        Text("Audio Settings")
                        Text("(No selection = system default)")
                    }
                }
                Section(header: Text("Start options")) {
                    Toggle("Enable Microphone", isOn: $isMicEnabled)
                }
                Section(header: Text("Server")) {
                    TextField("Backend URL", text: $backendURL)
                        .keyboardType(.URL)
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") {
                        self.saveSettings()
                        self.showingSettings = false
                    }
                }
            }
            .onAppear {
                self.loadSettings()
            }
        }
    }
    
    private func saveSettings() {
        let newSettings = SettingsPreference(
            selectedMic: model.selectedMic?.id,
            enableMic: isMicEnabled,
            backendURL: backendURL
        )
        SettingsManager.updateSettings(settings: newSettings)
    }
    
    private func loadSettings() {
        let savedSettings = SettingsManager.getSettings()
        self.isMicEnabled = savedSettings.enableMic
        self.backendURL = savedSettings.backendURL
    }
}

#Preview {
    let mockModel = MockCallContainerModel()
    let result = SettingsView(showingSettings: .constant(true)).environmentObject(mockModel as CallContainerModel)
    mockModel.startAudioLevelSimulation()
    return result
}
