import SwiftUI

struct ToastModifier: ViewModifier {
    var message: String?
    var isShowing: Bool
    
    func body(content: Content) -> some View {
        ZStack {
            content
            if isShowing, let message = message {
                VStack {
                    Text(message)
                        .padding()
                        .background(Color.black.opacity(0.7))
                        .foregroundColor(.white)
                        .cornerRadius(8)
                        .transition(.slide)
                        .padding(.top, 50)
                    Spacer()
                }
                .animation(.easeInOut(duration: 0.5), value: isShowing)
            }
        }
    }
}

extension View {
    func toast(message: String?, isShowing: Bool) -> some View {
        self.modifier(ToastModifier(message: message, isShowing: isShowing))
    }
}
