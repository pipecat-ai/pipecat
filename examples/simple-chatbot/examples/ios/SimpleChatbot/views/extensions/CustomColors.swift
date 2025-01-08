import SwiftUI

public extension Color {
    
    static let backgroundCircle = Color(hex: "#374151")
    static let backgroundCircleNotConnected = Color(hex: "#D1D5DB")
    static let backgroundApp = Color(hex: "#F9FAFB")
    static let buttonsBorder = Color(hex: "#E5E7EB")
    static let micVolume = Color(hex: "#86EFAC")
    static let timer = Color(hex: "#E5E7EB")
    static let disabledMic = Color(hex: "#ee6b6e")
    static let disabledVision = Color(hex: "#BBF7D0")
        
    init(hex: String) {
        let scanner = Scanner(string: hex)
        _ = scanner.scanString("#")
        
        var rgb: UInt64 = 0
        scanner.scanHexInt64(&rgb)
        
        let red = Double((rgb >> 16) & 0xFF) / 255.0
        let green = Double((rgb >> 8) & 0xFF) / 255.0
        let blue = Double(rgb & 0xFF) / 255.0
        
        self.init(red: red, green: green, blue: blue)
    }
}
