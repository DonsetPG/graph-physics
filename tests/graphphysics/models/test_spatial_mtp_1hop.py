import unittest

import torch
import torch.nn as nn

from graphphysics.models.spatial_mtp_1hop import (
    SpatialMTP1Hop,
    _make_undirected,
    _sorted_by_src,
)


class TestSpatialMTP1Hop(unittest.TestCase):
    def test_forward_with_no_centers_returns_zero_loss(self):
        model = SpatialMTP1Hop(d_model=4)
        model.eval()

        H = torch.randn(3, 4)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        centers = torch.tensor([], dtype=torch.long)
        target = torch.zeros(3, 4)
        out_head = nn.Identity()

        loss, stats = model(
            H,
            edge_index,
            centers,
            out_head,
            target,
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(stats["sp_mtp/centers"].item(), 0.0)
        self.assertEqual(stats["sp_mtp/pairs"].item(), 0.0)

    def test_forward_cached_structure_matches_uncached(self):
        torch.manual_seed(0)
        model = SpatialMTP1Hop(d_model=4)
        model.eval()

        H = torch.randn(4, 4)
        edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
        centers = torch.tensor([0, 1], dtype=torch.long)
        target = torch.randn(4, 4)
        out_head = nn.Linear(4, 4, bias=False)

        with torch.no_grad():
            loss_uncached, stats_uncached = model(
                H,
                edge_index,
                centers,
                out_head,
                target,
            )

        e = _make_undirected(edge_index, assume_undirected=model.assume_undirected)
        _, dst_s, row_ptr = _sorted_by_src(e, num_nodes=H.size(0))

        with torch.no_grad():
            loss_cached, stats_cached = model(
                H,
                edge_index,
                centers,
                out_head,
                target,
                row_ptr=row_ptr,
                dst_sorted=dst_s,
            )

        self.assertTrue(torch.allclose(loss_uncached, loss_cached, atol=1e-6))
        for key in ("sp_mtp/centers", "sp_mtp/pairs", "sp_mtp/mean_pair_loss"):
            self.assertTrue(
                torch.allclose(stats_uncached[key], stats_cached[key], atol=1e-6)
            )

    def test_cap_neighbors_respects_max_neighbors(self):
        torch.manual_seed(42)
        model = SpatialMTP1Hop(d_model=4, max_neighbors=1)

        dst_s = torch.tensor([1, 2, 0], dtype=torch.long)
        row_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.long)
        centers = torch.tensor([0, 1], dtype=torch.long)

        with torch.no_grad():
            selected, counts = model._cap_neighbors(dst_s, row_ptr, centers)

        self.assertEqual(counts.tolist(), [1, 1])
        self.assertEqual(selected.numel(), sum(counts.tolist()))
        self.assertTrue(set(selected.tolist()).issubset(set(dst_s.tolist())))


if __name__ == "__main__":
    unittest.main()
