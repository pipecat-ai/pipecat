package ai.pipecat.simple_chatbot_client

import ai.pipecat.simple_chatbot_client.ui.InCallLayout
import ai.pipecat.simple_chatbot_client.ui.PermissionScreen
import ai.pipecat.simple_chatbot_client.ui.theme.Colors
import ai.pipecat.simple_chatbot_client.ui.theme.RTVIClientTheme
import ai.pipecat.simple_chatbot_client.ui.theme.TextStyles
import ai.pipecat.simple_chatbot_client.ui.theme.textFieldColors
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.annotation.DrawableRes
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp


class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        val voiceClientManager = VoiceClientManager(this)

        setContent {
            RTVIClientTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Box(
                        Modifier
                            .fillMaxSize()
                            .padding(innerPadding)
                    ) {
                        PermissionScreen()

                        val vcState = voiceClientManager.state.value

                        if (vcState != null) {
                            InCallLayout(voiceClientManager)

                        } else {
                            ConnectSettings(voiceClientManager)
                        }

                        voiceClientManager.errors.firstOrNull()?.let { errorText ->

                            val dismiss: () -> Unit = { voiceClientManager.errors.removeAt(0) }

                            AlertDialog(
                                onDismissRequest = dismiss,
                                confirmButton = {
                                    Button(onClick = dismiss) {
                                        Text(
                                            text = "OK",
                                            fontSize = 14.sp,
                                            fontWeight = FontWeight.W700,
                                            color = Color.White,
                                            style = TextStyles.base
                                        )
                                    }
                                },
                                containerColor = Color.White,
                                title = {
                                    Text(
                                        text = "Error",
                                        fontSize = 22.sp,
                                        fontWeight = FontWeight.W600,
                                        color = Color.Black,
                                        style = TextStyles.base
                                    )
                                },
                                text = {
                                    Text(
                                        text = errorText.message,
                                        fontSize = 16.sp,
                                        fontWeight = FontWeight.W400,
                                        color = Color.Black,
                                        style = TextStyles.base
                                    )
                                }
                            )
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ConnectSettings(
    voiceClientManager: VoiceClientManager,
) {
    val scrollState = rememberScrollState()

    val start = {
        val backendUrl = Preferences.backendUrl.value

        voiceClientManager.start(baseUrl = backendUrl!!)
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .imePadding()
            .padding(20.dp),
        contentAlignment = Alignment.Center
    ) {
        Box(
            Modifier
                .fillMaxWidth()
                .shadow(2.dp, RoundedCornerShape(16.dp))
                .clip(RoundedCornerShape(16.dp))
                .background(Colors.mainSurfaceBackground)
        ) {
            Column(
                Modifier
                    .fillMaxWidth()
                    .padding(
                        vertical = 24.dp,
                        horizontal = 28.dp
                    )
            ) {
                Spacer(modifier = Modifier.height(12.dp))

                Text(
                    modifier = Modifier.align(Alignment.CenterHorizontally),
                    text = "Connect to an RTVI server",
                    fontSize = 22.sp,
                    fontWeight = FontWeight.W700,
                    style = TextStyles.base
                )

                Spacer(modifier = Modifier.height(36.dp))

                Text(
                    text = "Backend URL",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.W400,
                    style = TextStyles.base
                )

                Spacer(modifier = Modifier.height(12.dp))

                TextField(
                    modifier = Modifier
                        .fillMaxWidth()
                        .border(1.dp, Colors.textFieldBorder, RoundedCornerShape(12.dp)),
                    value = Preferences.backendUrl.value ?: "",
                    onValueChange = { Preferences.backendUrl.value = it },
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Uri,
                        imeAction = ImeAction.Next
                    ),
                    colors = textFieldColors(),
                    shape = RoundedCornerShape(12.dp)
                )

                Spacer(modifier = Modifier.height(36.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    ConnectDialogButton(
                        modifier = Modifier.weight(1f),
                        onClick = start,
                        text = "Connect",
                        foreground = Color.White,
                        background = Colors.buttonNormal,
                        border = Colors.buttonNormal
                    )
                }
            }
        }
    }
}

@Composable
private fun ConnectDialogButton(
    onClick: () -> Unit,
    text: String,
    foreground: Color,
    background: Color,
    border: Color,
    modifier: Modifier = Modifier,
    @DrawableRes icon: Int? = null,
) {
    val shape = RoundedCornerShape(8.dp)

    Row(
        modifier
            .border(1.dp, border, shape)
            .clip(shape)
            .background(background)
            .clickable(onClick = onClick)
            .padding(vertical = 10.dp, horizontal = 24.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Center
    ) {
        if (icon != null) {
            Icon(
                modifier = Modifier.size(24.dp),
                painter = painterResource(icon),
                tint = foreground,
                contentDescription = null
            )

            Spacer(modifier = Modifier.width(8.dp))
        }

        Text(
            text = text,
            fontSize = 16.sp,
            fontWeight = FontWeight.W500,
            color = foreground
        )
    }
}
