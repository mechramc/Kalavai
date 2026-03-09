"""Tests for Kalavu custom exception hierarchy."""

import pytest

from kalavu.core.exceptions import (
    AlignmentError,
    CheckpointValidationError,
    ConfigError,
    CooperativeError,
    FusionError,
    KalavuError,
)


class TestExceptionHierarchy:
    """All custom exceptions inherit from KalavuError."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            ConfigError,
            AlignmentError,
            CheckpointValidationError,
            FusionError,
            CooperativeError,
        ],
    )
    def test_subclass_of_kalavu_error(self, exc_class: type) -> None:
        assert issubclass(exc_class, KalavuError)

    @pytest.mark.parametrize(
        "exc_class",
        [
            KalavuError,
            ConfigError,
            AlignmentError,
            CheckpointValidationError,
            FusionError,
            CooperativeError,
        ],
    )
    def test_raise_and_catch(self, exc_class: type) -> None:
        with pytest.raises(exc_class, match="test message"):
            raise exc_class("test message")

    def test_catch_all_via_base(self) -> None:
        """Catching KalavuError catches any subclass."""
        with pytest.raises(KalavuError):
            raise ConfigError("caught via base")
