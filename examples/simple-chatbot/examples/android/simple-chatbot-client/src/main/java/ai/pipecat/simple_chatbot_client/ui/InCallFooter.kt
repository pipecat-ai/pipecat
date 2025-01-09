package ai.pipecat.simple_chatbot_client.ui

import ai.pipecat.simple_chatbot_client.R
import ai.pipecat.simple_chatbot_client.ui.theme.Colors
import androidx.annotation.DrawableRes
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
private fun FooterButton(
    modifier: Modifier,
    onClick: () -> Unit,
    @DrawableRes icon: Int,
    text: String,
    foreground: Color,
    background: Color,
    border: Color,
) {
    val shape = RoundedCornerShape(12.dp)

    Row(
        modifier
            .border(1.dp, border, shape)
            .clip(shape)
            .background(background)
            .clickable(onClick = onClick)
            .padding(vertical = 10.dp, horizontal = 18.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Center
    ) {
        Icon(
            modifier = Modifier.size(24.dp),
            painter = painterResource(icon),
            tint = foreground,
            contentDescription = null
        )

        Spacer(modifier = Modifier.width(8.dp))

        Text(
            text = text,
            fontSize = 14.sp,
            fontWeight = FontWeight.W600,
            color = foreground
        )
    }
}

@Composable
fun ColumnScope.InCallFooter(
    onClickEnd: () -> Unit,
) {
    Row(Modifier
        .fillMaxWidth(0.5f)
        .padding(15.dp)
        .align(Alignment.CenterHorizontally)
    ) {
        FooterButton(
            modifier = Modifier.weight(1f),
            onClick = onClickEnd,
            icon = R.drawable.phone_hangup,
            text = "End",
            foreground = Color.White,
            background = Colors.endButton,
            border = Colors.endButton
        )
    }
}
