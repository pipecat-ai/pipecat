package ai.pipecat.simple_chatbot_client.ui

import ai.pipecat.simple_chatbot_client.ui.theme.Colors
import ai.pipecat.simple_chatbot_client.ui.theme.TextStyles
import android.Manifest
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun PermissionScreen() {
    val cameraPermission = rememberPermissionState(Manifest.permission.CAMERA)
    val micPermission = rememberPermissionState(Manifest.permission.RECORD_AUDIO)

    val requestPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { isGranted ->
        Log.i("MainActivity", "Permissions granted: $isGranted")
    }

    if (!cameraPermission.status.isGranted || !micPermission.status.isGranted) {

        Dialog(
            onDismissRequest = {},
        ) {
            val dialogShape = RoundedCornerShape(16.dp)

            Column(
                Modifier
                    .shadow(6.dp, dialogShape)
                    .border(2.dp, Colors.logoBorder, dialogShape)
                    .clip(dialogShape)
                    .background(Color.White)
                    .padding(28.dp)
            ) {
                Text(
                    text = "Permissions",
                    fontSize = 24.sp,
                    fontWeight = FontWeight.W700,
                    style = TextStyles.base
                )

                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = "Please grant camera and mic permissions to continue",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.W400,
                    style = TextStyles.base
                )

                Spacer(modifier = Modifier.height(36.dp))

                Button(
                    modifier = Modifier.align(Alignment.End),
                    shape = RoundedCornerShape(12.dp),
                    onClick = {
                        requestPermissionLauncher.launch(
                            arrayOf(
                                Manifest.permission.CAMERA,
                                Manifest.permission.RECORD_AUDIO
                            )
                        )
                    }
                ) {
                    Text(
                        text = "Grant permissions",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.W700,
                        style = TextStyles.base
                    )
                }
            }
        }
    }
}