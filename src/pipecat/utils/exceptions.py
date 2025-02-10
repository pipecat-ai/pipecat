class PipecatBaseError(Exception):
    """Base exception class for all pipecat-related errors."""

    pass


class APIKeyNotFoundError(PipecatBaseError):
    """Raised when an API key is required but not provided."""

    pass
