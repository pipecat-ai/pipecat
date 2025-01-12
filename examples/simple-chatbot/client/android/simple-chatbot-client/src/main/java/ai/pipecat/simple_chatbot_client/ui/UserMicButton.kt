package ai.pipecat.simple_chatbot_client.ui

import ai.pipecat.simple_chatbot_client.R
import ai.pipecat.simple_chatbot_client.ui.theme.Colors
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Icon
import androidx.compose.runtime.Composable
import androidx.compose.runtime.FloatState
import androidx.compose.runtime.State
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp

@Composable
fun UserMicButton(
    onClick: () -> Unit,
    micEnabled: Boolean,
    modifier: Modifier,
    isTalking: State<Boolean>,
    audioLevel: FloatState,
) {
    Box(
        modifier = modifier.padding(15.dp),
        contentAlignment = Alignment.Center
    ) {
        val borderThickness by animateDpAsState(
            if (isTalking.value) {
                (24.dp * Math.pow(audioLevel.floatValue.toDouble(), 0.3).toFloat()) + 3.dp
            } else {
                6.dp
            }
        )

        val color by animateColorAsState(
            if (!micEnabled) {
                Colors.mutedMicBackground
            } else if (isTalking.value) {
                Color.Black
            } else {
                Colors.unmutedMicBackground
            }
        )

        Box(
            Modifier
                .shadow(3.dp, CircleShape)
                .border(borderThickness, Color.White, CircleShape)
                .border(1.dp, Colors.lightGrey, CircleShape)
                .clip(CircleShape)
                .background(color)
                .clickable(onClick = onClick)
                .padding(36.dp),
            contentAlignment = Alignment.Center,
        ) {
            Icon(
                modifier = Modifier.size(48.dp),
                painter = painterResource(
                    if (micEnabled) {
                        R.drawable.microphone
                    } else {
                        R.drawable.microphone_off
                    }
                ),
                tint = Color.White,
                contentDescription = if (micEnabled) {
                    "Mute microphone"
                } else {
                    "Unmute microphone"
                },
            )
        }
    }
}

@Composable
@Preview
fun PreviewUserMicButton() {
    UserMicButton(
        onClick = {},
        micEnabled = true,
        modifier = Modifier,
        isTalking = remember { mutableStateOf(false) },
        audioLevel = remember { mutableFloatStateOf(1.0f) }
    )
}

@Composable
@Preview
fun PreviewUserMicButtonMuted() {
    UserMicButton(
        onClick = {},
        micEnabled = false,
        modifier = Modifier,
        isTalking = remember { mutableStateOf(false) },
        audioLevel = remember { mutableFloatStateOf(1.0f) }
    )
}