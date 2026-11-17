"""Small deterministic contract tests; these do not initialize OpenPI weights."""
import unittest

import jax.numpy as jnp
import numpy as np

from openpi.policies.robocasa_policy import RobocasaOutputs
from examples.robocasa_spatial.geometry import masked_average_pool, parallax
from openpi.models.pi0 import make_attn_mask


class SpatialContractsTest(unittest.TestCase):
    def test_identity_camera_has_zero_parallax(self):
        h, w = 32, 40
        k = np.array([[30., 0., (w-1)/2], [0., 30., (h-1)/2], [0., 0., 1.]])
        depth = np.ones((h, w), np.float32)
        flow, valid = parallax(depth, k, np.eye(4), np.eye(4), np.ones_like(depth, bool), depth)
        self.assertTrue(valid.all())
        self.assertLess(float(np.abs(flow).max()), 1e-6)

    def test_reference_camera_translation_has_expected_flow_sign_and_scale(self):
        h, w = 32, 40
        fx, baseline = 30.0, 0.1
        k = np.array([[fx, 0., (w-1)/2], [0., fx, (h-1)/2], [0., 0., 1.]])
        depth = np.ones((h, w), np.float32)
        current = np.eye(4)
        reference = np.eye(4)
        reference[0, 3] = baseline
        flow, valid = parallax(
            depth, k, current, reference, np.ones_like(depth, bool),
            reference_depth=depth, reference_valid=np.ones_like(depth, bool),
        )
        # A reference camera translated +x sees a static point shift left by
        # fx*baseline/z pixels. Flow is (reference pixel - current pixel),
        # normalized by image width.
        np.testing.assert_allclose(flow[16, 20, 0], -(fx * baseline) / w, atol=1e-6)
        self.assertTrue(valid[16, 20])

    def test_masked_pool_keeps_empty_bins_invalid(self):
        flow = np.ones((8, 8, 2), np.float32)
        mask = np.zeros((8, 8), bool)
        mask[:4, :4] = True
        pooled, valid = masked_average_pool(flow, mask, size=2)
        self.assertEqual(pooled.shape, (8,))
        self.assertEqual(valid.shape, (8,))
        self.assertEqual(int(valid.sum()), 2)
        self.assertTrue(np.allclose(pooled[:2], 1.))
        self.assertTrue(np.allclose(pooled[2:], 0.))

    def test_action_output_uses_native_12_dimensions(self):
        x = np.arange(32, dtype=np.float32)[None, :]
        out = RobocasaOutputs()({'actions': x})['actions']
        self.assertEqual(out.shape, (1, 12))
        np.testing.assert_array_equal(out[0], x[0, :12])

    def test_suffix_can_attend_to_spatial_query_prefix(self):
        # Three visual/language prefix tokens plus two query tokens; action suffix starts next.
        valid = jnp.ones((1, 7), dtype=jnp.bool_)
        ar = jnp.array([False, False, False, False, False, True, False])
        mask = np.asarray(make_attn_mask(valid, ar))
        self.assertTrue(mask[0, 5, 3])
        self.assertTrue(mask[0, 5, 4])
        self.assertTrue(mask[0, 5, 0])


if __name__ == '__main__':
    unittest.main()
