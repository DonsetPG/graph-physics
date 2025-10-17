"""Feature layout definitions for :mod:`torch_geometric` node features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

from .exceptions import FeatureNotFoundError, LayoutValidationError, NamedFeatureError


@dataclass(frozen=True)
class XBlock:
    """A contiguous slice within the feature dimension of ``graph.x``.

    Parameters
    ----------
    name:
        Unique identifier for the feature block.
    start:
        Inclusive index where the block begins.
    end:
        Exclusive index where the block ends.
    """

    name: str
    start: int
    end: int

    @property
    def size(self) -> int:
        """Return the width of the block along the feature dimension."""

        return self.end - self.start


class XFeatureLayout:
    """Ordered mapping from feature names to slices over ``graph.x``.

    The layout keeps ``graph.x`` contiguous while enabling ergonomic, name-based
    accessors. The layout validates input sizes, prevents duplicates, and can be
    serialised to a plain mapping for configuration files or checkpoints.
    """

    def __init__(self, blocks: Iterable[Tuple[str, int]]):
        self.blocks: List[XBlock] = []
        self._slices: Dict[str, slice] = {}
        seen: set[str] = set()
        offset = 0
        for name, size in blocks:
            if not isinstance(name, str):
                raise TypeError(f"Feature names must be str, got {type(name)!r}.")
            if name in seen:
                raise NamedFeatureError(f"Duplicate feature name '{name}'.")
            if int(size) != size or size <= 0:
                raise ValueError(
                    f"Size for '{name}' must be a positive integer, got {size!r}."
                )
            size = int(size)
            block = XBlock(name=name, start=offset, end=offset + size)
            self.blocks.append(block)
            self._slices[name] = slice(block.start, block.end)
            offset += size
            seen.add(name)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.blocks)

    def __iter__(self) -> Iterator[XBlock]:  # pragma: no cover - trivial
        return iter(self.blocks)

    def feature_dim(self) -> int:
        """Return the total feature dimension represented by the layout."""

        return sum(block.size for block in self.blocks)

    def names(self) -> List[str]:
        """Return the ordered list of feature names."""

        return [block.name for block in self.blocks]

    def sizes(self) -> Dict[str, int]:
        """Return a mapping of feature names to their sizes."""

        return {block.name: block.size for block in self.blocks}

    def slc(self, name: str) -> slice:
        """Return the slice associated with ``name``.

        Raises
        ------
        FeatureNotFoundError
            If ``name`` is not present in the layout.
        """

        try:
            return self._slices[name]
        except KeyError as exc:  # pragma: no cover - simple branch
            raise FeatureNotFoundError(
                f"Unknown x-feature '{name}'. Known features: {self.names()}"
            ) from exc

    # ------------------------------------------------------------------
    # Editing operations
    # ------------------------------------------------------------------
    def rename(self, mapping: Mapping[str, str]) -> "XFeatureLayout":
        """Return a new layout with features renamed according to ``mapping``."""

        blocks = []
        for block in self.blocks:
            new_name = mapping.get(block.name, block.name)
            blocks.append((new_name, block.size))
        return XFeatureLayout(blocks)

    def reorder(self, new_order: Sequence[str]) -> "XFeatureLayout":
        """Return a layout reordered to match ``new_order``.

        Raises
        ------
        FeatureNotFoundError
            If ``new_order`` is missing names or contains unknown entries.
        """

        current_names = self.names()
        missing = [name for name in current_names if name not in new_order]
        extras = [name for name in new_order if name not in current_names]
        if missing:
            raise FeatureNotFoundError(
                f"Reorder missing existing names: {missing}. Provided order: {list(new_order)}"
            )
        if extras:
            raise FeatureNotFoundError(
                f"Reorder contains unknown names: {extras}. Existing names: {current_names}"
            )
        sizes = self.sizes()
        return XFeatureLayout([(name, sizes[name]) for name in new_order])

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, int]:
        """Serialise the layout to an ordered ``dict`` of sizes."""

        return self.sizes()

    @staticmethod
    def from_dict(
        sizes: Mapping[str, int], order: Optional[Sequence[str]] = None
    ) -> "XFeatureLayout":
        """Construct a layout from a mapping of feature sizes.

        Parameters
        ----------
        sizes:
            Mapping from feature name to size.
        order:
            Optional explicit order. Defaults to ``sizes.keys()`` order.
        """

        if order is None:
            order = list(sizes.keys())
        return XFeatureLayout([(name, int(sizes[name])) for name in order])

    # ------------------------------------------------------------------
    # Validation utilities
    # ------------------------------------------------------------------
    def validate_tensor(self, tensor_dim: int) -> None:
        """Ensure the tensor feature dimension matches the layout."""

        expected = self.feature_dim()
        if tensor_dim != expected:
            raise LayoutValidationError(
                f"Tensor feature dimension ({tensor_dim}) does not match layout ({expected}). "
                f"Layout sizes: {self.sizes()}"
            )


# ----------------------------------------------------------------------
# Builders and helper utilities
# ----------------------------------------------------------------------
def make_x_layout(order: Sequence[str], sizes: Mapping[str, int]) -> XFeatureLayout:
    """Create a layout from ``order`` and ``sizes`` mappings."""

    return XFeatureLayout([(name, int(sizes[name])) for name in order])


def x_layout_from_meta_and_spec(
    meta: Mapping[str, object],
    order: Sequence[str],
    overrides: Optional[Mapping[str, int]] = None,
) -> XFeatureLayout:
    """Construct a layout from dataset metadata and explicit overrides."""

    overrides = dict(overrides or {})
    features_meta = meta.get("features", {}) if isinstance(meta, Mapping) else {}
    sizes: Dict[str, int] = {}
    for name in order:
        if name in overrides:
            sizes[name] = int(overrides[name])
            continue
        entry = features_meta.get(name) if isinstance(features_meta, Mapping) else None
        if isinstance(entry, Mapping):
            shape = entry.get("shape")
            if isinstance(shape, Sequence) and shape:
                sizes[name] = int(shape[-1])
                continue
        raise FeatureNotFoundError(
            f"Cannot infer size for feature '{name}'. Provide an override or ensure meta['features']['{name}']['shape'] is present."
        )
    return make_x_layout(order, sizes)


def validate_x_layout(x, layout: XFeatureLayout) -> None:
    """Validate the ``graph.x`` tensor against the layout."""

    feature_dim = x.size(-1)
    layout.validate_tensor(feature_dim)


def ensure_contiguous_last_dim(x):
    """Ensure the last dimension is contiguous for performance-sensitive ops."""

    if not x.is_contiguous():
        return x.contiguous()
    if x.stride(-1) != 1:
        return x.contiguous()
    return x
