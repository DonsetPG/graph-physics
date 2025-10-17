"""Public exports for the named feature utilities."""
from .compat import LegacyIndexAdapter, old_indices_from_layout
from .exceptions import (
    FeatureAssignmentError,
    FeatureNotFoundError,
    LayoutValidationError,
    NamedFeatureError,
)
try:  # pragma: no cover - import guard for optional torch dependency
    from .named_data import NamedData
    _NAMED_DATA_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency missing
    NamedData = None  # type: ignore[assignment]
    _NAMED_DATA_IMPORT_ERROR = exc
from .x_layout import (
    XBlock,
    XFeatureLayout,
    ensure_contiguous_last_dim,
    make_x_layout,
    validate_x_layout,
    x_layout_from_meta_and_spec,
)

__all__ = [
    "NamedData",
    "XBlock",
    "XFeatureLayout",
    "FeatureAssignmentError",
    "FeatureNotFoundError",
    "LayoutValidationError",
    "NamedFeatureError",
    "ensure_contiguous_last_dim",
    "make_x_layout",
    "validate_x_layout",
    "x_layout_from_meta_and_spec",
    "old_indices_from_layout",
    "LegacyIndexAdapter",
]


def __getattr__(name: str):  # pragma: no cover - lazy error reporting
    if name == "NamedData" and _NAMED_DATA_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "NamedData requires the 'torch' and 'torch_geometric' packages to be installed."
        ) from _NAMED_DATA_IMPORT_ERROR
    raise AttributeError(name)
