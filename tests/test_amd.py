"""Hardware integration checks; run with .venv/Scripts/python -m unittest discover -s tests."""
import os
import sys
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3D_Stage"))
os.environ["CHARACTERGEN_DEVICE"] = "directml"
os.environ["HF_HOME"] = str(Path(__file__).resolve().parents[1] / ".cache/huggingface")
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parents[1] / ".cache/matplotlib")
warnings.filterwarnings("error", message=".*fall back to run on the CPU.*")
import torch
import torch.nn.functional as F
from runtime import get_device, grid_sample, sliced_attention
import gpu_raster


class AMDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = get_device()

    def setUp(self):
        warnings.filterwarnings("error", message=".*fall back to run on the CPU.*")

    def test_grid_sample_matches_torch(self):
        torch.manual_seed(12)
        image = torch.rand(2, 3, 5, 7)
        grid = torch.rand(2, 4, 9, 2)*3-1.5
        for align in (True, False):
            for padding in ("zeros", "border"):
                expected = F.grid_sample(image, grid, align_corners=align, padding_mode=padding)
                actual = grid_sample(image.to(self.device), grid.to(self.device), align_corners=align, padding_mode=padding)
                torch.testing.assert_close(actual.cpu(), expected, atol=2e-6, rtol=2e-5)

    def test_attention_matches_dense(self):
        torch.manual_seed(4)
        q, k, v = torch.rand(6, 9, 8), torch.rand(6, 11, 8), torch.rand(6, 11, 8)
        mask = torch.zeros(6, 1, 11)
        mask[:, :, -2:] = -10000
        expected = ((q @ k.transpose(-1, -2))/8**0.5 + mask).softmax(-1) @ v
        for dtype in (torch.float32, torch.float16):
            actual = sliced_attention(q.to(self.device, dtype), k.to(self.device, dtype),
                                      v.to(self.device, dtype), mask.to(self.device), chunk_size=4)
            torch.testing.assert_close(actual.cpu().float(), expected, atol=1e-3, rtol=1e-3)

    def test_opengl_depth_barycentrics_and_interpolation(self):
        vertices = torch.tensor([[-1.,-1.,0.,1.], [1.,-1.,0.,1.], [-1.,1.,0.,1.],
                                 [-1.,-1.,.5,1.], [1.,-1.,.5,1.], [-1.,1.,.5,1.]])
        tri = torch.tensor([[0,1,2], [3,4,5]], dtype=torch.int32).to(self.device)
        rast, _ = gpu_raster.rasterize(gpu_raster.RasterizeCudaContext(self.device), vertices[None].to(self.device), tri, (4,4))
        r = rast.cpu()
        self.assertEqual(r[0,0,0,3], 1)
        self.assertEqual(r[0,3,3,3], 0)
        torch.testing.assert_close(r[0,0,0,:2], torch.tensor([.75,.125]))
        attr = vertices[None,:,:3].to(self.device)
        interp, _ = gpu_raster.interpolate(attr, rast, tri)
        torch.testing.assert_close(interp.cpu()[0,0,0], torch.tensor([-.75,-.75,0.]))
        unbatched, _ = gpu_raster.interpolate(attr[0], rast, tri)
        torch.testing.assert_close(unbatched.cpu(), interp.cpu())
        from lrm.utils.rasterize import NVDiffRasterizerContext
        wrapper = NVDiffRasterizerContext("gl", self.device)
        single, derivatives = wrapper.rasterize_one(vertices.to(self.device), tri, (4,4))
        torch.testing.assert_close(single.cpu(), r[0])
        self.assertIsNone(derivatives)

    def test_marching_tetrahedra_on_gpu(self):
        from lrm.models.isosurface import MarchingTetrahedraHelper
        helper = MarchingTetrahedraHelper(32, "3D_Stage/load/tets/32_tets.npz").to(self.device)
        points = helper.grid_vertices
        level = ((points-.5).square().sum(-1, keepdim=True).sqrt()-.3)
        with torch.no_grad():
            mesh = helper(level)
        self.assertGreater(len(mesh.v_pos), 0)
        self.assertGreater(len(mesh.t_pos_idx), 0)
        self.assertTrue(torch.isfinite(mesh.v_pos).all().cpu())
        self.assertEqual(mesh.v_pos.device, self.device)
        # Compare the actual surface with the original PyTorch implementation.
        cpu = MarchingTetrahedraHelper(32, "3D_Stage/load/tets/32_tets.npz")
        with torch.no_grad():
            expected_v, expected_f = cpu._forward(points.cpu(), level.cpu(), cpu.indices)
        self.assertEqual(len(mesh.t_pos_idx), len(expected_f))
        actual_centers = mesh.v_pos[mesh.t_pos_idx].mean(1).cpu()
        expected_centers = expected_v[expected_f].mean(1)
        # Atomic shader output order differs; compare sorted centroids.
        import numpy as np
        def sorted_centers(value):
            a = value.numpy().round(5)
            return a[np.lexsort((a[:,2], a[:,1], a[:,0]))]
        np.testing.assert_allclose(sorted_centers(actual_centers), sorted_centers(expected_centers), atol=2e-5)

    def test_texture_sampling_and_empty_mesh(self):
        tex = torch.arange(12, dtype=torch.float32).reshape(1,2,2,3).to(self.device)
        uv = torch.tensor([[[[.25,.25],[.75,.75],[1.25,.25]]]], device=self.device)
        sampled = gpu_raster.texture(tex, uv).cpu()
        torch.testing.assert_close(sampled[0,0,0], torch.tensor([0.,1.,2.]))
        torch.testing.assert_close(sampled[0,0,1], torch.tensor([9.,10.,11.]))
        torch.testing.assert_close(sampled[0,0,2], sampled[0,0,0])
        tri = torch.empty((0,3), device=self.device, dtype=torch.int32)
        rast, _ = gpu_raster.rasterize(None, torch.zeros((1,0,4),device=self.device), tri, (2,2))
        self.assertEqual(rast.cpu().count_nonzero(), 0)

    def test_texture_reprojection_selects_closest_view(self):
        from refine import aggregate_texture
        from unittest.mock import patch
        base = torch.full((2, 3, 3), .3, dtype=torch.float32, device=self.device)
        views = [(base + delta).unsqueeze(0) for delta in (.2, .05, .3, .1)]
        masks = [torch.ones((1, 2, 3, 1), dtype=torch.int32, device=self.device) for _ in views]
        with patch("PIL.Image.Image.save"):
            result, missing = aggregate_texture(base, views, masks, [])
        torch.testing.assert_close(result.cpu(), torch.full((2, 3, 3), .35))
        self.assertEqual(missing.cpu().count_nonzero(), 0)


if __name__ == "__main__":
    unittest.main()
