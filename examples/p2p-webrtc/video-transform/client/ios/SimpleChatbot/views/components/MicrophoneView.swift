import SwiftUI

struct MicrophoneView: View {
    var audioLevel: Float // Current audio level
    var isMuted: Bool // Muted state

    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let circleSize = width * 0.9
            let innerCircleSize = width * 0.82
            let audioCircleSize = CGFloat(audioLevel) * (width * 0.95)

            ZStack {
                Circle()
                    .stroke(Color.gray, lineWidth: 1)
                    .frame(width: circleSize)

                Circle()
                    .fill(isMuted ? Color.disabledMic : Color.backgroundCircle)
                    .frame(width: innerCircleSize)

                if !isMuted {
                    Circle()
                        .fill(Color.micVolume)
                        .opacity(0.5)
                        .frame(width: audioCircleSize)
                        .animation(.easeInOut(duration: 0.2), value: audioLevel)
                }

                Image(systemName: isMuted ? "mic.slash.fill" : "mic.fill")
                    .resizable()
                    .scaledToFit()
                    .frame(width: width * 0.2)
                    .foregroundColor(.white)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity) // Ensures the ZStack is centered
        }
    }
}

#Preview {
    MicrophoneView(audioLevel: 1, isMuted: false)
}
