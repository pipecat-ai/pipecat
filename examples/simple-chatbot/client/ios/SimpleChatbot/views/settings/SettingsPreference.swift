import Foundation

struct SettingsPreference: Codable {
    var selectedMic: String?
    var enableMic: Bool
    var backendURL: String
}

