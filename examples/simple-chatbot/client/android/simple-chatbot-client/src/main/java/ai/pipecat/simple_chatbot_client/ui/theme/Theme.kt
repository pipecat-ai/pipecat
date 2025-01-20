package ai.pipecat.simple_chatbot_client.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val LightColorScheme = lightColorScheme(
    primary = Colors.buttonNormal,
    secondary = Colors.buttonWarning,
    background = Colors.activityBackground,
    surface = Colors.mainSurfaceBackground
)

@Composable
fun RTVIClientTheme(
    content: @Composable () -> Unit
) {
    val colorScheme = LightColorScheme

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}

@Composable
fun textFieldColors() = TextFieldDefaults.colors().copy(
    unfocusedContainerColor = Colors.activityBackground,
    focusedContainerColor = Colors.activityBackground,
    focusedIndicatorColor = Color.Transparent,
    disabledIndicatorColor = Color.Transparent,
    unfocusedIndicatorColor = Color.Transparent,
)