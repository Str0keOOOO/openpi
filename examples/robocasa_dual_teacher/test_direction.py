"""CPU regression checks for current-to-Anchor teacher directions."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from examples.robocasa_dual_teacher.geometry import ee_difference, pool_match


def pose(p, r):
    t = np.eye(4)
    t[:3, 3] = p
    t[:3, :3] = Rotation.from_rotvec(r).as_matrix()
    return t


class DirectionTest(unittest.TestCase):
    def delta(self, cr, ce, ar, ae):
        return ee_difference(cr, ce, ar, ae, position_scale_m=.1, rotation_scale_rad=1.)

    def test_positive_negative_and_zero(self):
        eye = np.eye(4)
        current = pose([.02, -.03, .04], [0, 0, .2])
        np.testing.assert_allclose(self.delta(eye, current, eye, eye), [-.2, .3, -.4, 0, 0, -.2], atol=1e-6)
        np.testing.assert_allclose(self.delta(eye, current, eye, current), 0, atol=1e-6)

    def test_noncommuting_rotation_reaches_anchor(self):
        cr = pose([1, 2, 3], [.4, .3, -.2])
        ar = pose([-1, 3, 2], [-.2, .1, .7])
        c = pose([.3, -.2, .5], [.3, -.4, .2])
        a = pose([.1, .4, .7], [-.5, .1, .4])
        result = self.delta(cr, cr @ c, ar, ar @ a)
        np.testing.assert_allclose(c[:3, 3] + .1 * result[:3], a[:3, 3], atol=1e-6)
        np.testing.assert_allclose(Rotation.from_rotvec(result[3:]).as_matrix() @ c[:3, :3], a[:3, :3], atol=1e-6)
        world = pose([5, -3, 2], [.5, -.2, .4])
        np.testing.assert_allclose(result, self.delta(world @ cr, world @ cr @ c, world @ ar, world @ ar @ a), atol=1e-6)
        np.testing.assert_allclose(result, -self.delta(ar, ar @ a, cr, cr @ c), atol=1e-6)

    def test_scene_keeps_current_to_anchor_sign(self):
        f = np.zeros((64, 64, 2), np.float32)
        f[:] = [4, -2]
        mask = np.ones((64, 64), bool)
        z, valid, _, _ = pool_match(f, mask, mask, -f, np.ones((64, 64)))
        self.assertTrue(valid.all())
        np.testing.assert_allclose(z.reshape(-1, 2), np.tile([4/64, -2/64], (16, 1)), atol=1e-6)
        _, empty, _, _ = pool_match(f, ~mask, mask, -f, np.ones((64, 64)))
        self.assertFalse(empty.any())


if __name__ == '__main__':
    unittest.main()
