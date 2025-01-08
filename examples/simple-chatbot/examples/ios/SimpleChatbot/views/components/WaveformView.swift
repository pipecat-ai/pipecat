import SwiftUI

struct WaveformView: View {
    
    var audioLevel: Float
    var isBotReady: Bool
    var voiceClientStatus: String
    
    @State
    private var audioLevels: [Float] = Array(repeating: 0, count: 5)
    private let dotCount = 5
    
    var body: some View {
        GeometryReader { geometry in
            VStack {
                Spacer()
                HStack {
                    Spacer()
                    ZStack {
                        // Outer gray border
                        Circle()
                            .stroke(Color.gray, lineWidth: 1)
                            .frame(width: geometry.size.width * 0.9, height: geometry.size.width * 0.9)
                        
                        // Gray middle
                        Circle()
                            .fill(isBotReady ? Color.backgroundCircle : Color.backgroundCircleNotConnected)
                            .frame(width: geometry.size.width * 0.82, height: geometry.size.width * 0.82)
                        
                        if isBotReady {
                            if audioLevel > 0 {
                                // Waveform bars inside the circle
                                HStack(spacing: 10) {
                                    ForEach(0..<dotCount, id: \.self) { index in
                                        Rectangle()
                                            .fill(Color.white)
                                            .frame(height: CGFloat(audioLevels[index]) * (geometry.size.height))
                                            .cornerRadius(12)
                                            .animation(.easeInOut(duration: 0.2), value: audioLevels[index])
                                    }
                                    .frame(maxWidth: .infinity)
                                    .padding(.horizontal, 5)
                                }
                                .frame(width: geometry.size.width * 0.5, height: geometry.size.width * 0.5)
                                .mask(Circle().frame(width: geometry.size.width * 0.82, height: geometry.size.width * 0.82))
                            } else {
                                // Dots inside the circle
                                HStack(spacing: 10) {
                                    ForEach(0..<dotCount, id: \.self) { _ in
                                        Circle()
                                            .fill(Color.white)
                                            .frame(maxWidth: .infinity)
                                    }
                                    .frame(maxWidth: .infinity)
                                }
                                .frame(width: geometry.size.width * 0.5, height: geometry.size.height * 0.5)
                            }
                        } else {
                            // Gray circle with loading icon when not connected
                            VStack {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                    .scaleEffect(2) // Adjust size of the loading spinner
                                    .padding()
                                Text(voiceClientStatus)
                                    .foregroundColor(.white)
                                    .font(.headline)
                            }
                        }
                    }
                    Spacer()
                }
                Spacer()
            }
        }
        .onChange(of: audioLevel) { oldLevel, newLevel in
            // The audio level that we receive from the bot is usually too low
            // so just increasing it so we can see a better graph but
            // making sure that it is not higher than the maximum 1
            var audioLevel = audioLevel + 0.4
            if(audioLevel > 1) {
                audioLevel = 1
            }
            // Update the array and shift values
            audioLevels.removeFirst()
            audioLevels.append(newLevel)
        }
    }
}

#Preview {
    WaveformView(audioLevel: 0, isBotReady: false, voiceClientStatus: "idle")
}
