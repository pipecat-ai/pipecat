Implementation Approach:

Feature: Get recording after session ends.
Solution:
    The existing solution uses DailyCO API to host meetings/sessions in a room that they host, my implementation built upon that and used DailCO's inbuilt features. When a user disconnects from the session, we stop recording and the video is saved to the DailyCO cloud. We list and fetch the latest recording from the client code and display that as it is being downloaded on the same video player pane.

    DailyCO offers 3 types of recordings: 1) Local 2) Cloud 3) Raw

    
    Performance Considerations:
    - As this is a user facing application, I chose 'Cloud' option rather than Local and Raw which utilize the user's compute / resources to store recordings.
    - 'Cloud' stores the recordings on DailyCO's provisioned S3 buckets by default and we can configure these S3 buckets to our own for faster access and higher resolution storage.

    Future Improvements:
    - The current implementation fetches the latest recording that was recorded as we are only operating on a 1-1 user sessions (1 user is our client and 1 user is the bot). If we want to have multiple users in a session(>2), we will need to have a user management module where we only stop recording when all the clients except the bot leaves. The current implementation is built for 1-1 meeting sessions.

    - Custom S3 bucket config for faster fetch timings for the recorded sessions.


Feature: Get response latency and interruption latency
Solution:
    As response latency and interruption latency is a developer facing product - we do not show the metrics on the client side (also because frontend was tedious). We use the exisitng framework that is supported by Pipecat in-built: Sentry Monitoring.

    My implementation for the response latency and interruption latency has 2 aspects to it: exporting inbuilt metrics that pipecat offers and exporting custom user facing metrics(user input -> system output).
    For the inbuilt metrics the implementation is pretty simple and is natively exported to Sentry.
    For the response latency and interruption latency we capture the following metrics:
    - Response latency: The time from when the user stops talking till the bot starts replying.
    - Interruption Latency: The time from when the user starts talking while a bot is talking, till the time the bot stops talking when the user is talking. (Simply put - how fast does the bot shut up when you start talking)

    The implementation uses the frames that are being processed and the frame states, we read what state the frame is in and then record latency based on the state change of a frame stream.

    Performance Considerations:
    - This is not a good solution as it introduces a new task to the existing pipeline rather than just observing it, ideally I would have wanted to edit / add to some of the framework(pipecat) that exists underneath by forking it. This would be a much more complex implementation. Or just do it from the client end - where you export metrics for response  interruption latency from the frontend code.

    Future Improvements:
    - Using observer threads to track time and integrate it natively into a function / service that works with Pipecat. This could be a good PR(?) for the main Pipecat repo.

Feature: Live Transcription
Solution:
    DailyCO's API offers a native live transcription configuration with every room that you make. We just initialize it during room creation. This is done through the RTVI transport class, where you set the right config for the DialyParams. Pretty straightforward with an option for live word by word transcription which I have implemented.

    Performance Considerations:
    - Changing the audio transcriber models for better syntactical recognition.
    Unsure of more performance considerations here - we could try to reduce the performance times which are observed through sentry.

    Future Improvements:
    - Try different models and play around with profanity filters / puncuations. Disabling all the profanity filter should ideally decrease the latencies but will have to test it out.