package ai.pipecat.simple_chatbot_client

import ai.pipecat.client.RTVIClient
import ai.pipecat.client.RTVIClientOptions
import ai.pipecat.client.RTVIClientParams
import ai.pipecat.client.RTVIEventCallbacks
import ai.pipecat.client.daily.DailyTransport
import ai.pipecat.client.result.Future
import ai.pipecat.client.result.RTVIError
import ai.pipecat.client.result.Result
import ai.pipecat.client.types.ActionDescription
import ai.pipecat.client.types.Participant
import ai.pipecat.client.types.PipecatMetrics
import ai.pipecat.client.types.RTVIURLEndpoints
import ai.pipecat.client.types.ServiceConfig
import ai.pipecat.client.types.Tracks
import ai.pipecat.client.types.Transcript
import ai.pipecat.client.types.TransportState
import ai.pipecat.simple_chatbot_client.utils.Timestamp
import android.content.Context
import android.util.Log
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.Stable
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf

@Immutable
data class Error(val message: String)

@Stable
class VoiceClientManager(private val context: Context) {

    companion object {
        private const val TAG = "VoiceClientManager"
    }

    private val client = mutableStateOf<RTVIClient?>(null)

    val state = mutableStateOf<TransportState?>(null)

    val errors = mutableStateListOf<Error>()

    val actionDescriptions =
        mutableStateOf<Result<List<ActionDescription>, RTVIError>?>(null)

    val expiryTime = mutableStateOf<Timestamp?>(null)

    val botReady = mutableStateOf(false)
    val botIsTalking = mutableStateOf(false)
    val userIsTalking = mutableStateOf(false)
    val botAudioLevel = mutableFloatStateOf(0f)
    val userAudioLevel = mutableFloatStateOf(0f)

    val mic = mutableStateOf(false)
    val camera = mutableStateOf(false)
    val tracks = mutableStateOf<Tracks?>(null)

    private fun <E> Future<E, RTVIError>.displayErrors() = withErrorCallback {
        Log.e(TAG, "Future resolved with error: ${it.description}", it.exception)
        errors.add(Error(it.description))
    }

    fun start(baseUrl: String) {

        if (client.value != null) {
            return
        }

        val options = RTVIClientOptions(
            params = RTVIClientParams(
                baseUrl = baseUrl,
                endpoints = RTVIURLEndpoints(),
            )
        )

        state.value = TransportState.Disconnected

        val callbacks = object : RTVIEventCallbacks() {
            override fun onTransportStateChanged(state: TransportState) {
                this@VoiceClientManager.state.value = state
            }

            override fun onBackendError(message: String) {
                "Error from backend: $message".let {
                    Log.e(TAG, it)
                    errors.add(Error(it))
                }
            }

            override fun onBotReady(version: String, config: List<ServiceConfig>) {

                Log.i(TAG, "Bot ready. Version $version, config: $config")

                botReady.value = true

                client.value?.describeActions()?.withCallback {
                    actionDescriptions.value = it
                }
            }

            override fun onPipecatMetrics(data: PipecatMetrics) {
                Log.i(TAG, "Pipecat metrics: $data")
            }

            override fun onUserTranscript(data: Transcript) {
                Log.i(TAG, "User transcript: $data")
            }

            override fun onBotTranscript(text: String) {
                Log.i(TAG, "Bot transcript: $text")
            }

            override fun onBotStartedSpeaking() {
                Log.i(TAG, "Bot started speaking")
                botIsTalking.value = true
            }

            override fun onBotStoppedSpeaking() {
                Log.i(TAG, "Bot stopped speaking")
                botIsTalking.value = false
            }

            override fun onUserStartedSpeaking() {
                Log.i(TAG, "User started speaking")
                userIsTalking.value = true
            }

            override fun onUserStoppedSpeaking() {
                Log.i(TAG, "User stopped speaking")
                userIsTalking.value = false
            }

            override fun onTracksUpdated(tracks: Tracks) {
                this@VoiceClientManager.tracks.value = tracks
            }

            override fun onInputsUpdated(camera: Boolean, mic: Boolean) {
                this@VoiceClientManager.camera.value = camera
                this@VoiceClientManager.mic.value = mic
            }

            override fun onConnected() {
                expiryTime.value = client.value?.expiry?.let(Timestamp::ofEpochSecs)
            }

            override fun onDisconnected() {
                expiryTime.value = null
                actionDescriptions.value = null
                botIsTalking.value = false
                userIsTalking.value = false
                state.value = null
                actionDescriptions.value = null
                botReady.value = false
                tracks.value = null

                client.value?.release()
                client.value = null
            }

            override fun onUserAudioLevel(level: Float) {
                userAudioLevel.floatValue = level
            }

            override fun onRemoteAudioLevel(level: Float, participant: Participant) {
                botAudioLevel.floatValue = level
            }
        }

        val client = RTVIClient(DailyTransport.Factory(context), callbacks, options)

        client.connect().displayErrors().withErrorCallback {
            callbacks.onDisconnected()
        }

        this.client.value = client
    }

    fun enableCamera(enabled: Boolean) {
        client.value?.enableCam(enabled)?.displayErrors()
    }

    fun enableMic(enabled: Boolean) {
        client.value?.enableMic(enabled)?.displayErrors()
    }

    fun toggleCamera() = enableCamera(!camera.value)
    fun toggleMic() = enableMic(!mic.value)

    fun stop() {
        client.value?.disconnect()?.displayErrors()
    }
}