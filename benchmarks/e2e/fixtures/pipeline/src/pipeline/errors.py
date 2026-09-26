"""Pipeline exception hierarchy."""


class PipelineError(Exception):
    """Base error for all pipeline failures."""


class SkipRecord(PipelineError):
    """Raised by transforms or validators to drop a record quietly."""


class ConfigurationError(PipelineError):
    """Raised when a reader/sink/transform is misconfigured."""


class ExhaustedRetries(PipelineError):
    """Raised when the retry policy gives up on an operation."""
