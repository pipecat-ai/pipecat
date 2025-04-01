import Foundation
import PipecatClientIOS
import PipecatClientIOSOpenAIRealtimeWebrtc

public enum ToolsFunctions: String {
    case endBot = "end_bot"
    case getWeatherTool = "getWeatherTool"
}

class OpenAIBotConfig {
    
    static let endBotTool = OpenAIFunctionTool.init(
        name: ToolsFunctions.endBot.rawValue,
        description: "Invoke this function when the user indicates that the conversation has ended and no further assistance is needed.",
        parameters: nil
    )
    
    static let getWeatherTool = OpenAIFunctionTool.init(
        name: ToolsFunctions.getWeatherTool.rawValue,
        description: "Gets the current weather for a given location.",
        parameters: .object([
            "type": .string("object"),
            "properties": .object([
                "location": .object([
                    "type": "string",
                    "description": "A city or location"
                ])
            ])
        ])
    )
    
    static func createOptions(openaiAPIKey:String) -> RTVIClientOptions {
        var tools: [Value?] = []
        do {
            tools.append(try endBotTool.convertToRtviValue())
            tools.append(try getWeatherTool.convertToRtviValue())
        } catch {
            // nothing to do here
        }
        
        let currentSettings = SettingsManager.getSettings()
        let rtviClientOptions = RTVIClientOptions.init(
            enableMic: currentSettings.enableMic,
            enableCam: false,
            params: .init(config: [
                .init(
                    service: "llm",
                    options: [
                        .init(name: "api_key", value: .string(openaiAPIKey)),
                        .init(name: "initial_messages", value: .array([
                            .object([
                                "role": .string("user"), // "user" | "system"
                                "content": .string("Start by introducing yourself.")
                            ])
                        ])),
                        .init(name: "session_config", value: .object([
                            "instructions": .string("You are Chatbot, a friendly and helpful assistant who provides useful information, including weather updates."),
                            "voice": .string("echo"),
                            "input_audio_noise_reduction": .object([
                                "type": .string("near_field")
                            ]),
                            "turn_detection": .object([
                                "type": .string("semantic_vad")
                            ]),
                            "tools": .array(tools)
                        ])),
                    ]
                )
            ])
        )
        return rtviClientOptions
    }
    
}
