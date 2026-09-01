- Fixed the Rime TTS services' language mapping: Arabic, Italian, Japanese and
  Portuguese — all served by Rime's `coda` model — now resolve to Rime's
  language codes, and a region-qualified language such as `Language.EN_US`
  falls back to a base code Rime accepts instead of a BCP-47 tag that made Rime
  close the connection without synthesizing.
