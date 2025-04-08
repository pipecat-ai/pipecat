import SwiftUI
import PipecatClientIOSSmallWebrtc

struct MeetingView: View {
    
    @State private var showingSettings = false
    @EnvironmentObject private var model: CallContainerModel
    
    var body: some View {
        VStack {
            ZStack {
                SmallWebRTCVideoViewSwiftUI(videoTrack: self.model.botCamId, videoScaleMode: .fill)
                    .edgesIgnoringSafeArea(.all)
                
                VStack {
                    ChatView()
                        .frame(maxHeight: .infinity)
                    
                    HStack {
                        MicrophoneView(audioLevel: 0, isMuted: !self.model.isMicEnabled)
                            .frame(width: 100, height: 100)
                            .onTapGesture {
                                self.model.toggleMicInput()
                            }
                        CameraButtonView(trackId: self.model.localCamId, isMuted: !self.model.isCamEnabled)
                            .frame(width: 120, height: 120)
                            .onTapGesture {
                                self.model.toggleCamInput()
                            }
                    }
                    .padding()
                }
            }
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
            .foregroundColor(.black)
            .background(Color.white)
            .border(Color.buttonsBorder, width: 1)
            .cornerRadius(12)
            .padding([.horizontal])
            
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
        .background(Color.backgroundApp)
        .toast(message: model.toastMessage, isShowing: model.showToast)
    }
}

#Preview {
    let mockModel = MockCallContainerModel()
    let result = MeetingView().environmentObject(mockModel as CallContainerModel)
    return result
}
