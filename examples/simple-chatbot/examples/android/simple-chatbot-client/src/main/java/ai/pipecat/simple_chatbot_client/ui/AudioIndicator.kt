package ai.pipecat.simple_chatbot_client.ui

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.semantics.clearAndSetSemantics

@Composable
fun ListeningAnimation(
    modifier: Modifier,
    active: Boolean,
    level: Float,
    color: Color,
) {
    val infiniteTransition = rememberInfiniteTransition("listeningAnimation")

    val loopState by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = Math.PI.toFloat() * 2f,
        animationSpec = infiniteRepeatable(tween(durationMillis = 1000, easing = LinearEasing)),
        label = "listeningAnimationLoopState"
    )

    val activeFraction by animateFloatAsState(
        if (active) {
            Math.pow(level.toDouble(), 0.3).toFloat()
        } else {
            0f
        }
    )

    Canvas(modifier.clearAndSetSemantics { }) {

        val strokeWidthPx = size.width / 12

        val lineCount = 5

        for (i in 1..lineCount) {

            val sine = Math.sin(loopState + 0.9 * i)
            val fraction = activeFraction * ((sine + 1) / 2).toFloat()

            val x = (size.width / (lineCount + 1)) * i

            val yMax = size.height * 0.25f
            val yMin = size.height * 0.5f

            val y = yMin + (yMax - yMin) * fraction
            val yEnd = size.height - y

            this.drawLine(
                start = Offset(x, y),
                end = Offset(x, yEnd),
                color = color,
                strokeWidth = strokeWidthPx,
                cap = StrokeCap.Round
            )
        }
    }
}
