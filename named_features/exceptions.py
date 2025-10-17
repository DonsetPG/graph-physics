"""Custom exception types for named feature layouts."""

from __future__ import annotations


class NamedFeatureError(Exception):
    """Base class for named feature related errors."""


class LayoutValidationError(NamedFeatureError):
    """Raised when the layout does not match the tensor shape."""


class FeatureNotFoundError(NamedFeatureError, KeyError):
    """Raised when a feature name is missing from the layout."""


class FeatureAssignmentError(NamedFeatureError):
    """Raised when assignment inputs do not match expectations."""
