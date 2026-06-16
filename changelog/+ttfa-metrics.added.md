Added Time To First Audio (TTFA) metrics to TTS services. In addition to TTFB
(time to first byte), TTS services now report `TTFAMetricsData`, which measures
the time until the first *audible* audio sample by scanning the leading silence
many providers pad onto the start of a response. Comparing TTFA against TTFB
reveals how much perceived latency is silence padding versus service response
time.
