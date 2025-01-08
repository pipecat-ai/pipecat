import SwiftUI

struct MeetingView: View {
    
    @State private var showingSettings = false
    @EnvironmentObject private var model: CallContainerModel
    
    var body: some View {
        VStack {
            // Header Toolbar
            HStack {
                Image("dailyBot")
                    .resizable()
                    .frame(width: 48, height: 48)
                Spacer()
                HStack {
                    Image(systemName: "stopwatch")
                        .resizable()
                        .frame(width: 24, height: 24)
                    Text(timerString(from: self.model.timerCount))
                        .font(.headline)
                }.padding()
                    .background(Color.timer)
                    .cornerRadius(12)
            }
            .padding()
            
            // Main Panel
            VStack {
                VStack {
                    WaveformView(audioLevel: model.remoteAudioLevel, isBotReady: model.isBotReady, voiceClientStatus: model.voiceClientStatus)
                }
                .frame(maxHeight: .infinity)
                VStack {
                    HStack {
                        MicrophoneView(audioLevel: model.localAudioLevel, isMuted: !self.model.isMicEnabled)
                            .frame(width: 160, height: 160)
                            .onTapGesture {
                                self.model.toggleMicInput()
                            }
                    }
                }
                .frame(height: 120)
            }
            .frame(maxHeight: .infinity)
            .padding()
            
            // Bottom Panel
            VStack {
                HStack {
                    Button(action: {
                        self.showingSettings = true
                    }) {
                        HStack {
                            Image(systemName: "gearshape")
                                .resizable()
                                .frame(width: 24, height: 24)
                            Text("Settings")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .sheet(isPresented: $showingSettings) {
                            SettingsView(showingSettings: $showingSettings).environmentObject(self.model)
                        }
                    }
                    .border(Color.buttonsBorder, width: 1)
                    .cornerRadius(12)
                }
                .foregroundColor(.black)
                .padding([.top, .horizontal])
                
                Button(action: {
                    self.model.disconnect()
                }) {
                    HStack {
                        Image(systemName: "rectangle.portrait.and.arrow.right")
                            .resizable()
                            .frame(width: 24, height: 24)
                        Text("End")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                }
                .foregroundColor(.white)
                .background(Color.black)
                .cornerRadius(12)
                .padding([.bottom, .horizontal])
            }
        }
        .background(Color.backgroundApp)
        .toast(message: model.toastMessage, isShowing: model.showToast)
    }
    
    func timerString(from count: Int) -> String {
        let hours = count / 3600
        let minutes = (count % 3600) / 60
        let seconds = count % 60
        return String(format: "%02d:%02d:%02d", hours, minutes, seconds)
    }
}

#Preview {
    let mockModel = MockCallContainerModel()
    let result = MeetingView().environmentObject(mockModel as CallContainerModel)
    mockModel.startAudioLevelSimulation()
    return result
}
