"""Custom exception hierarchy for Kalavu."""


class KalavuError(Exception):
    """Base exception for all Kalavu errors."""


class ConfigError(KalavuError):
    """Raised when configuration is invalid or missing."""


class AlignmentError(KalavuError):
    """Raised when transaction alignment between books fails."""


class CheckpointValidationError(KalavuError):
    """Raised when a checkpoint fails validation checks."""


class FusionError(KalavuError):
    """Raised when fusing multiple data sources fails."""


class CooperativeError(KalavuError):
    """Raised when cooperative-specific operations fail."""
