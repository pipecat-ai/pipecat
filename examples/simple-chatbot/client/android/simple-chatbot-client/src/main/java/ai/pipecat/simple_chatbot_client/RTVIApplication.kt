package ai.pipecat.simple_chatbot_client

import android.app.Application

class RTVIApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        Preferences.initAppStart(this)
    }
}