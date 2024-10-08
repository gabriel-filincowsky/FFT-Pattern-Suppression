import unittest
import numpy as np
import cupy as cp
from processing.mask_generator import MaskGenerator

class TestMaskGenerator(unittest.TestCase):
    def setUp(self):
        self.mask_generator = MaskGenerator()
        self.shape = (256, 256)
        self.center = (128, 128)
        self.radius = 50
        self.falloff = 10
        self.aspect_ratio = 1.5
        self.orientation = 45
        self.intensity_pct = 20

    def test_create_circular_mask(self):
        mask = self.mask_generator.create_circular_mask(
            shape=self.shape,
            center=self.center,
            radius=self.radius,
            falloff=self.falloff
        )
        self.assertEqual(mask.shape, self.shape)
        self.assertTrue(cp.all((mask >= 0) & (mask <= 1)))

    def test_create_exclusion_mask(self):
        mask = self.mask_generator.create_exclusion_mask(
            shape=self.shape,
            center=self.center,
            radius=self.radius,
            aspect_ratio=self.aspect_ratio,
            orientation=self.orientation,
            falloff=self.falloff
        )
        self.assertEqual(mask.shape, self.shape)
        self.assertTrue(cp.all((mask >= 0) & (mask <= 1)))

    def test_create_antialiasing_mask(self):
        mask = self.mask_generator.create_antialiasing_mask(
            shape=self.shape,
            intensity_pct=self.intensity_pct
        )
        self.assertEqual(mask.shape, self.shape)
        self.assertTrue(cp.all((mask >= 0) & (mask <= 1)))

    def test_masks_overlap_correctly(self):
        # Create two masks and ensure their overlap behaves as expected
        mask1 = self.mask_generator.create_circular_mask(
            shape=self.shape,
            center=self.center,
            radius=60
        )
        mask2 = self.mask_generator.create_circular_mask(
            shape=self.shape,
            center=self.center,
            radius=30
        )
        combined_mask = mask1 * mask2
        # Pixels within radius 30 should remain as mask2
        self.assertTrue(cp.all(combined_mask[self.center[0], self.center[1]] == 0))
        # Pixels between 30 and 60 should be mask1
        self.assertTrue(cp.all(combined_mask[128, 128 + 45] == 0))
        # Pixels outside 60 should be 1
        self.assertTrue(combined_mask[0, 0] == 1)

if __name__ == '__main__':
    unittest.main()