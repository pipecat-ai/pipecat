import Foundation

struct SettingsPreference: Codable {
    var selectedMic: String?
    var enableMic: Bool
    var enableCam: Bool
    var backendURL: String
}

