#!/usr/bin/env python3
"""Small demo showcasing the named feature helpers."""

from __future__ import annotations

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    raise SystemExit("This demo requires the 'torch' package.") from exc

from named_features import LegacyIndexAdapter, NamedData, XFeatureLayout


def build_sample_graph() -> NamedData:
    layout = XFeatureLayout(
        [
            ("velocity_x", 1),
            ("velocity_y", 1),
            ("pressure", 1),
            ("mesh_pos_x", 1),
            ("mesh_pos_y", 1),
            ("node_type", 1),
        ]
    )
    x = torch.randn(4, layout.feature_dim())
    x[:, layout.slc("node_type")] = torch.tensor([[0.0], [1.0], [0.0], [2.0]])
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    graph = NamedData(x=x, edge_index=edge_index, x_layout=layout)
    graph.y = torch.randn(4, 2)
    return graph


def main() -> None:
    graph = build_sample_graph()
    print("Known features:", graph.x_layout.names())

    velocity = graph.x_sel(["velocity_x", "velocity_y"])
    print("Velocity block shape:", tuple(velocity.shape))

    noisy_velocity = velocity + 0.05 * torch.randn_like(velocity)
    graph.x_assign({"velocity_x": noisy_velocity[:, :1], "velocity_y": noisy_velocity[:, 1:]})
    print("Velocity updated in-place.")

    as_dict = graph.x_to_dict(["pressure", "mesh_pos_x"])
    print("Pressure sample:", as_dict["pressure"][0].item())

    adapter = LegacyIndexAdapter(
        graph.x_layout,
        targets=["velocity_x", "velocity_y"],
        node_type_name="node_type",
    )
    print("Legacy indices:", adapter.as_dict())


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
