- `AWSTranscribeSTTService` now maps 46 more AWS Transcribe streaming languages,
  including Turkish, Hungarian, Tamil, Telugu, Swahili, Mexican Spanish and
  Welsh. Passing one of these as a `Language` enum previously sent a bare base
  code such as `tr`, which AWS Transcribe rejects.
