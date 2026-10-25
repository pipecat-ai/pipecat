Added `TurnAnalyzerEagerUserTurnStopStrategy`, which lets a turn analyzer
drive the speculation path that `EagerUserTurnStopStrategy` gives STT
services. `BaseTurnAnalyzer` gains `pop_eager()`, which an analyzer overrides
to report an *eager* end of turn: evidence good enough to start generating a
response but not yet good enough to end the turn. The strategy starts a
speculative inference on it (the response is held by the LLM service's
`SpeculationGate`), releases the response when the analyzer commits, and
withdraws it when the user resumes speaking, the committed transcript differs
from the eager one, or the turn ends without a commit. Analyzers that never
reach an eager end of turn are unaffected.
