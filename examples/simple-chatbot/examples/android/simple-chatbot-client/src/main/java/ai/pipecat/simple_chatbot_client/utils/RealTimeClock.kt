package ai.pipecat.simple_chatbot_client.utils

import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.flow

private val rtcFlowSecs = flow {
    while(true) {
        val now = Timestamp.now().toEpochMilli()

        val rounded = ((now + 500) / 1000) * 1000
        emit(Timestamp.ofEpochMilli(rounded))

        val target = rounded + 1000
        delay(target - now)
    }
}

@Composable
fun rtcStateSecs() = rtcFlowSecs.collectAsState(initial = Timestamp.now())