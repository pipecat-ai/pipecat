package ai.pipecat.simple_chatbot_client.ui

import ai.pipecat.client.daily.VoiceClientVideoView
import ai.pipecat.client.types.MediaTrackId
import ai.pipecat.simple_chatbot_client.R
import ai.pipecat.simple_chatbot_client.ui.theme.Colors
import androidx.compose.animation.animateColorAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Icon
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView

@Composable
fun UserCamButton(
    onClick: () -> Unit,
    camEnabled: Boolean,
    camTrackId: MediaTrackId?,
    modifier: Modifier,
) {
    Box(
        modifier = modifier.padding(15.dp).size(96.dp),
        contentAlignment = Alignment.Center
    ) {
        val color by animateColorAsState(
            if (camEnabled) {
                Colors.unmutedMicBackground
            } else {
                Colors.mutedMicBackground
            }
        )

        Box(
            Modifier
                .fillMaxSize()
                .shadow(3.dp, CircleShape)
                .border(6.dp, Color.White, CircleShape)
                .border(1.dp, Colors.lightGrey, CircleShape)
                .clip(CircleShape)
                .background(color)
                .clickable(onClick = onClick),
            contentAlignment = Alignment.Center,
        ) {
            if (camTrackId != null) {
                AndroidView(
                    factory = { context ->
                        VoiceClientVideoView(context)
                    },
                    update = { view ->
                        view.voiceClientTrack = camTrackId
                    }
                )
            } else {
                Icon(
                    modifier = Modifier.size(30.dp),
                    painter = painterResource(
                        if (camEnabled) {
                            R.drawable.video
                        } else {
                            R.drawable.video_off
                        }
                    ),
                    tint = Color.White,
                    contentDescription = if (camEnabled) {
                        "Disable camera"
                    } else {
                        "Enable camera"
                    },
                )
            }
        }
    }
}

@Composable
@Preview
fun PreviewUserCamButton() {
    UserCamButton(
        onClick = {},
        camTrackId = null,
        camEnabled = true,
        modifier = Modifier,
    )
}

@Composable
@Preview
fun PreviewUserCamButtonMuted() {
    UserCamButton(
        onClick = {},
        camTrackId = null,
        camEnabled = false,
        modifier = Modifier,
    )
}