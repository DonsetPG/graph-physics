"""Named feature aware :class:`torch_geometric.data.Data` subclass."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import torch
from torch_geometric.data import Batch, Data

from .exceptions import (
    FeatureAssignmentError,
    FeatureNotFoundError,
    LayoutValidationError,
)
from .x_layout import XFeatureLayout, ensure_contiguous_last_dim, validate_x_layout


class NamedData(Data):
    """PyG ``Data`` object that carries a named feature layout for ``x``."""

    def __init__(
        self,
        *args,
        x_layout: Optional[XFeatureLayout] = None,
        x_coords: Optional[Mapping[str, object]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        layout = x_layout if x_layout is not None else getattr(self, "x_layout", None)
        if layout is not None:
            self.x_layout = layout
        coords_source = (
            x_coords if x_coords is not None else getattr(self, "x_coords", {})
        )
        self.x_coords = dict(coords_source)

        x_tensor = getattr(self, "x", None)
        if x_tensor is None:
            if layout is not None:
                # Delay validation until x is populated (e.g., during batching).
                return
            # Allow PyG internals to instantiate placeholder batches without x/layout.
            return

        if layout is None:
            raise LayoutValidationError(
                "NamedData requires 'x_layout' when 'x' is provided."
            )

        validate_x_layout(x_tensor, layout)

    # ------------------------------------------------------------------
    # PyG integration helpers
    # ------------------------------------------------------------------
    def __inc__(self, key, value, *args):  # pragma: no cover - defers to base
        return super().__inc__(key, value, *args)

    def __cat_dim__(self, key, value, *args):  # pragma: no cover - defers to base
        return super().__cat_dim__(key, value, *args)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def x_names(self) -> List[str]:
        """Return the feature names in order."""

        return self.x_layout.names()

    def x_sizes(self) -> Dict[str, int]:
        """Return mapping from feature names to sizes."""

        return self.x_layout.sizes()

    def x_slice(self, name: str) -> slice:
        """Return the slice corresponding to ``name``."""

        return self.x_layout.slc(name)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def x_sel(self, names: str | Sequence[str]) -> torch.Tensor:
        """Select features by name, returning a view for single names."""

        if isinstance(names, str):
            slc = self.x_slice(names)
            return self.x[..., slc]
        if not names:
            raise FeatureNotFoundError("x_sel requires at least one feature name.")
        parts = [self.x[..., self.x_slice(name)] for name in names]
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------
    def _check_assignment_tensor(
        self, name: str, target: torch.Tensor, value: torch.Tensor
    ) -> None:
        if target.shape != value.shape:
            raise FeatureAssignmentError(
                f"Shape mismatch for feature '{name}': expected {tuple(target.shape)} got {tuple(value.shape)}."
            )
        if target.dtype != value.dtype:
            raise FeatureAssignmentError(
                f"dtype mismatch for feature '{name}': expected {target.dtype} got {value.dtype}."
            )
        if target.device != value.device:
            raise FeatureAssignmentError(
                f"Device mismatch for feature '{name}': expected {target.device} got {value.device}."
            )

    def x_assign(
        self, mapping: Mapping[str, torch.Tensor], *, inplace: bool = True
    ) -> "NamedData":
        """Assign new values to features by name."""

        target = self if inplace else self.clone()
        for name, value in mapping.items():
            slc = target.x_slice(name)
            target_tensor = target.x[..., slc]
            self._check_assignment_tensor(name, target_tensor, value)
            target_tensor.copy_(value)
        return target

    # ------------------------------------------------------------------
    # Editing operations
    # ------------------------------------------------------------------
    def x_rename(self, mapping: Mapping[str, str]) -> "NamedData":
        """Rename features and update the layout in place."""

        self.x_layout = self.x_layout.rename(mapping)
        return self

    def x_reorder(
        self, new_order: Sequence[str], *, inplace: bool = True
    ) -> "NamedData":
        """Reorder features according to ``new_order``."""

        new_layout = self.x_layout.reorder(new_order)
        reordered = torch.cat(
            [self.x[..., self.x_slice(name)] for name in new_order], dim=-1
        )
        if inplace:
            self.x = reordered
            self.x_layout = new_layout
            return self
        clone = self.clone()
        clone.x = reordered
        clone.x_layout = new_layout
        return clone

    def x_drop(self, names: Sequence[str]) -> torch.Tensor:
        """Return a tensor containing all features except ``names``."""

        drop = set(names)
        keep = [name for name in self.x_names() if name not in drop]
        if not keep:
            raise FeatureNotFoundError(
                "Dropping all features results in an empty selection."
            )
        parts = [self.x[..., self.x_slice(name)] for name in keep]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------
    # Dictionary IO helpers
    # ------------------------------------------------------------------
    def x_to_dict(
        self, names: Optional[Sequence[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """Export selected features to a dictionary."""

        names = list(names) if names is not None else self.x_names()
        result: Dict[str, torch.Tensor] = {}
        for name in names:
            slc = self.x_slice(name)
            tensor = self.x[..., slc]
            result[name] = tensor if tensor.is_contiguous() else tensor.contiguous()
        return result

    @staticmethod
    def x_from_dict(
        features: Mapping[str, torch.Tensor], order: Sequence[str]
    ) -> torch.Tensor:
        """Pack features from a dictionary according to ``order``."""

        tensors = []
        for name in order:
            if name not in features:
                raise FeatureNotFoundError(
                    f"Feature '{name}' missing from provided mapping."
                )
            tensors.append(ensure_contiguous_last_dim(features[name]))
        return torch.cat(tensors, dim=-1)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_x(self) -> None:
        """Validate that ``x`` matches the stored layout."""

        validate_x_layout(self.x, self.x_layout)

    # ------------------------------------------------------------------
    # Cloning helpers
    # ------------------------------------------------------------------
    def clone(self) -> "NamedData":  # pragma: no cover - relies on Data.clone
        out = super().clone()
        layout = getattr(self, "x_layout", None)
        if layout is not None:
            out.x_layout = layout
        coords = getattr(self, "x_coords", {})
        out.x_coords = dict(coords)
        return out

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        layout = getattr(self, "x_layout", None)
        layout_repr = layout.sizes() if layout is not None else None
        return f"NamedData({super().__repr__()}, x_layout={layout_repr})"

    # ------------------------------------------------------------------
    # DataLoader integration
    # ------------------------------------------------------------------
    @classmethod
    def from_data_list(cls, data_list):  # pragma: no cover - relies on Batch logic
        batch = Batch.from_data_list(data_list)
        batch.__class__ = cls
        first = data_list[0]
        batch.x_layout = first.x_layout
        batch.x_coords = dict(getattr(first, "x_coords", {}))
        if getattr(batch, "x", None) is not None and batch.x_layout is not None:
            batch.validate_x()
        return batch
