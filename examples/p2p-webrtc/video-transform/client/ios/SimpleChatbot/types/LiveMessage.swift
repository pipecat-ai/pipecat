import Foundation

enum MessageType {
    case bot, user, system
}

class LiveMessage: ObservableObject, Identifiable, Equatable {
    @Published var content: String
    let type: MessageType
    let updatedAt: Date
    
    init(content: String, type: MessageType, updatedAt: Date) {
        self.content = content
        self.type = type
        self.updatedAt = updatedAt
    }
    
    static func == (lhs: LiveMessage, rhs: LiveMessage) -> Bool {
        lhs.updatedAt == rhs.updatedAt
    }
}
