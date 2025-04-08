import SwiftUI

struct ChatView: View {
    @EnvironmentObject private var model: CallContainerModel
    @State private var timer = Timer.publish(every: 0.5, on: .main, in: .common).autoconnect()
    
    var body: some View {
        VStack {
            ScrollViewReader { scrollViewProxy in
                ScrollView {
                    VStack(spacing: 10) {
                        ForEach(self.model.messages) { message in
                            MessageView(message: message)
                            .frame(maxWidth: .infinity, alignment: messageAlignment(for: message.type))
                            .padding(.horizontal)
                            .id(message.id)
                        }
                    }
                    .onChange(of: self.model.messages) { _, _ in
                        scrollToLastMessage(scrollViewProxy)
                    }
                }
                .onReceive(timer) { _ in
                    scrollToLastMessage(scrollViewProxy)
                }
                .onAppear {
                    scrollToLastMessage(scrollViewProxy)
                }
            }
        }
        .edgesIgnoringSafeArea(.bottom)
    }
    
    private func messageAlignment(for type: MessageType) -> Alignment {
        switch type {
        case .bot: return .leading
        case .user: return .trailing
        case .system: return .center
        }
    }
    
    private func scrollToLastMessage(_ scrollViewProxy: ScrollViewProxy) {
        if let lastMessageId = self.model.messages.last?.id {
            withAnimation {
                scrollViewProxy.scrollTo(lastMessageId, anchor: .bottom)
            }
        }
    }
}

struct MessageView: View {
    @ObservedObject var message: LiveMessage
    
    var body: some View {
        HStack {
            if message.type == .bot {
                Image(systemName: "gearshape")
                    .resizable()
                    .frame(width: 24, height: 24)
            }
            
            Text(message.content)
                .padding(message.type == .system ? 5 : 10)
                .foregroundColor(.white)
                .background(messageBackgroundColor(for: message.type))
                .cornerRadius(15)
                .overlay(
                    RoundedRectangle(cornerRadius: 15)
                        .stroke(Color.gray.opacity(0.5), lineWidth: 1)
                )
        }
        .padding(messagePadding(for: message.type))
    }
    
    private func messageBackgroundColor(for type: MessageType) -> Color {
        switch type {
        case .bot: return .black
        case .user: return .gray
        case .system: return .blue.opacity(0.6)
        }
    }
    
    private func messagePadding(for type: MessageType) -> EdgeInsets {
        switch type {
        case .bot: return EdgeInsets(top: 0, leading: 0, bottom: 0, trailing: 40)
        case .user: return EdgeInsets(top: 0, leading: 40, bottom: 0, trailing: 0)
        case .system: return EdgeInsets(top: 0, leading: 0, bottom: 0, trailing: 0)
        }
    }
}

#Preview {
    let mockModel = MockCallContainerModel()
    let result = ChatView().environmentObject(mockModel as CallContainerModel)
    return result
}
