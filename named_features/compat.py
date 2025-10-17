"""Compatibility helpers bridging named layouts and legacy index windows."""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

from .exceptions import FeatureNotFoundError
from .x_layout import XFeatureLayout


def old_indices_from_layout(
    layout: XFeatureLayout,
    targets: Sequence[str],
    node_type_name: Optional[str] = None,
) -> Dict[str, int]:
    """Return legacy index windows derived from a named layout."""

    if not targets:
        raise FeatureNotFoundError("Targets list cannot be empty when deriving indices.")
    feature_dim = layout.feature_dim()
    slices = [layout.slc(name) for name in targets]
    start = min(slc.start for slc in slices)
    end = max(slc.stop for slc in slices)
    indices = {
        "feature_index_start": 0,
        "feature_index_end": feature_dim,
        "output_index_start": start,
        "output_index_end": end,
    }
    if node_type_name is not None:
        indices["node_type_index"] = layout.slc(node_type_name).start
    return indices


class LegacyIndexAdapter:
    """Expose legacy index windows derived from a named feature layout.

    The adapter stores the mapping between feature names and their corresponding
    legacy start/end indices. It can be used during the migration period to keep
    downstream utilities that still expect positional windows functioning while
    newer code paths rely on named access.
    """

    def __init__(
        self,
        layout: XFeatureLayout,
        targets: Sequence[str],
        *,
        node_type_name: Optional[str] = None,
    ) -> None:
        if not targets:
            raise FeatureNotFoundError(
                "Targets list cannot be empty when building the legacy adapter."
            )
        missing_targets = [name for name in targets if name not in layout.sizes()]
        if missing_targets:
            raise FeatureNotFoundError(
                "Targets %s are not present in the layout." % missing_targets
            )
        if node_type_name is not None and node_type_name not in layout.sizes():
            raise FeatureNotFoundError(
                f"Node type feature '{node_type_name}' not present in layout."
            )

        self._layout = layout
        self._targets = list(targets)
        self._node_type = node_type_name

    @property
    def layout(self) -> XFeatureLayout:
        return self._layout

    @property
    def targets(self) -> Sequence[str]:
        return tuple(self._targets)

    @property
    def node_type_name(self) -> Optional[str]:
        return self._node_type

    def as_dict(self) -> Dict[str, int]:
        """Return the derived legacy indices as a dictionary."""

        return old_indices_from_layout(
            self._layout, self._targets, self._node_type
        )

    def feature_window(self, name: str) -> tuple[int, int]:
        """Return the legacy slice ``(start, end)`` for ``name``."""

        slc = self._layout.slc(name)
        return slc.start, slc.stop

    def mismatches(
        self, legacy_index_cfg: Mapping[str, int]
    ) -> Dict[str, tuple[int, int]]:
        """Compare a legacy configuration with the derived windows.

        Parameters
        ----------
        legacy_index_cfg:
            Mapping that may contain legacy window values. Only recognised keys
            are compared.

        Returns
        -------
        Dict[str, Tuple[int, int]]
            Dictionary mapping the offending key to a tuple of
            ``(configured_value, derived_value)``. An empty dictionary indicates
            no mismatches.
        """

        derived = self.as_dict()
        mismatches: Dict[str, tuple[int, int]] = {}
        for key in ("feature_index_start", "feature_index_end", "output_index_start", "output_index_end"):
            if key in legacy_index_cfg:
                configured = int(legacy_index_cfg[key])
                if configured != derived[key]:
                    mismatches[key] = (configured, derived[key])

        if self._node_type is not None and "node_type_index" in legacy_index_cfg:
            configured = int(legacy_index_cfg["node_type_index"])
            derived_value = derived["node_type_index"]
            if configured != derived_value:
                mismatches["node_type_index"] = (configured, derived_value)

        return mismatches
