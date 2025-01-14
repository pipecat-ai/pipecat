package ai.pipecat.simple_chatbot_client.utils

import androidx.compose.runtime.Composable
import androidx.compose.runtime.Immutable
import java.time.Duration
import java.time.Instant
import java.time.format.DateTimeFormatter
import java.util.Date

// Wrapper for Compose stability
@Immutable
@JvmInline
value class Timestamp(
    val value: Instant
) : Comparable<Timestamp> {
    val isInPast: Boolean
        get() = value < Instant.now()

    val isInFuture: Boolean
        get() = value > Instant.now()

    fun toEpochMilli() = value.toEpochMilli()

    operator fun plus(duration: Duration) = Timestamp(value + duration)

    operator fun minus(duration: Duration) = Timestamp(value - duration)

    operator fun minus(rhs: Timestamp) = Duration.between(rhs.value, value)

    override operator fun compareTo(other: Timestamp) = value.compareTo(other.value)

    fun toISOString(): String = DateTimeFormatter.ISO_INSTANT.format(value)

    override fun toString() = toISOString()

    companion object {
        fun now() = Timestamp(Instant.now())

        fun ofEpochMilli(value: Long) = Timestamp(Instant.ofEpochMilli(value))

        fun ofEpochSecs(value: Long) = ofEpochMilli(value * 1000)

        fun parse(value: CharSequence) = Timestamp(Instant.parse(value))

        fun from(date: Date) = Timestamp(date.toInstant())
    }
}

@Composable
fun formatTimer(duration: Duration): String {

    if (duration.seconds < 0) {
        return "0s"
    }

    val mins = duration.seconds / 60
    val secs = duration.seconds % 60

    return if (mins == 0L) {
        "${secs}s"
    } else {
        "${mins}m ${secs}s"
    }
}
