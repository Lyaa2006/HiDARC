import tempfile
import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llava.model.spectral_pca import FixedPCARouteProjector


class SpectralPCATest(unittest.TestCase):
    def test_fit_shape_and_fixed_parameters(self):
        generator = torch.Generator().manual_seed(7)
        samples = torch.randn(40, 8, generator=generator)
        projector = FixedPCARouteProjector.fit(samples, 4)
        self.assertEqual((projector.input_dim, projector.output_dim), (8, 4))
        self.assertEqual(tuple(projector(torch.randn(2, 16, 8)).shape), (2, 16, 4))
        self.assertEqual(list(projector.parameters()), [])
        self.assertTrue(torch.isfinite(projector.explained_variance).all())

    def test_save_load_and_batch_independence(self):
        samples = torch.randn(24, 6, generator=torch.Generator().manual_seed(3))
        projector = FixedPCARouteProjector.fit(samples, 3, metadata={"input_dim": 6})
        features = torch.randn(2, 5, 6, generator=torch.Generator().manual_seed(9))
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "spectral_pca.pt")
            projector.save(path)
            loaded = FixedPCARouteProjector.from_file(
                path, expected_input_dim=6, expected_output_dim=3
            )
            self.assertTrue(torch.allclose(projector(features), loaded(features)))
        self.assertTrue(torch.allclose(projector(features[:1]), projector(features)[:1]))

    def test_dtype_and_width_validation(self):
        projector = FixedPCARouteProjector(
            torch.zeros(5), torch.eye(3, 5)
        )
        output = projector(torch.ones(2, 5, dtype=torch.bfloat16))
        self.assertEqual(output.dtype, torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "width mismatch"):
            projector(torch.ones(2, 4))


if __name__ == "__main__":
    unittest.main()
