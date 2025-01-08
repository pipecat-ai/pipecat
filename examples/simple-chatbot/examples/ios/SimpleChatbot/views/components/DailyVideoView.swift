import SwiftUI

import Daily
import RTVIClientIOS
import RTVIClientIOSDaily


/// A wrapper for `VoiceClientVideoView` that exposes the video size via a `@Binding`.
struct DailyVideoView: UIViewRepresentable {
    
    /// The current size of the video being rendered by this view.
    @Binding private(set) var videoSize: CGSize

    private let voiceClientTrack: MediaTrackId?
    private let videoScaleMode: VoiceClientVideoView.VideoScaleMode

    init(
        voiceClientTrack: MediaTrackId? = nil,
        videoScaleMode: VoiceClientVideoView.VideoScaleMode = .fill,
        videoSize: Binding<CGSize> = .constant(.zero)
    ) {
        self.voiceClientTrack = voiceClientTrack
        self.videoScaleMode = videoScaleMode
        self._videoSize = videoSize
    }

    func makeUIView(context: Context) -> VoiceClientVideoView {
        let videoView = VoiceClientVideoView()
        videoView.delegate = context.coordinator
        return videoView
    }

    func updateUIView(_ videoView: VoiceClientVideoView, context: Context) {
        context.coordinator.dailyVideoView = self

        if videoView.voiceClientTrack != voiceClientTrack {
            videoView.voiceClientTrack = voiceClientTrack
        }

        if videoView.videoScaleMode != videoScaleMode {
            videoView.videoScaleMode = videoScaleMode
        }
    }
}

extension DailyVideoView {
    final class Coordinator: VideoViewDelegate {
        fileprivate var dailyVideoView: DailyVideoView

        init(_ dailyVideoView: DailyVideoView) {
            self.dailyVideoView = dailyVideoView
        }

        func videoView(_ videoView: Daily.VideoView, didChangeVideoSize size: CGSize) {
            // Update the `videoSize` binding with the current `size` value.
            DispatchQueue.main.async {
                self.dailyVideoView.videoSize = size
            }
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
}

#Preview {
    VideoView()
}
