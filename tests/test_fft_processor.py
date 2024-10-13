import unittest
import cupy as cp
from processing.fft_processor import FFTProcessor
import numpy as np

class TestFFTProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = FFTProcessor(padding_size=16)
        # Initialize test_image as a CuPy array
        self.test_image = cp.random.rand(256, 256).astype(cp.float32) * 255

    def test_compute_fft_ifft(self):
        fft_shifted = self.processor.compute_fft(self.test_image)
        reconstructed = self.processor.compute_ifft(fft_shifted)
        # Ensure the reconstructed image is a CuPy array and compare with original
        self.assertIsInstance(reconstructed, cp.ndarray)
        # Convert both to NumPy for comparison
        np.testing.assert_allclose(cp.asnumpy(self.test_image), 
                                   cp.asnumpy(reconstructed[:256, :256]), 
                                   rtol=1e-5, atol=1e-2)

    def test_apply_high_pass_filter(self):
        highpass = self.processor.apply_high_pass_filter(self.test_image, radius=5)
        self.assertIsInstance(highpass, cp.ndarray)
        self.assertEqual(highpass.shape, self.test_image.shape)
        self.assertTrue(cp.all((highpass >= 0) & (highpass <= 255)))

    def test_pad_to_power_of_two_even_dimensions(self):
        image = cp.ones((64, 64))
        padded_image, extra_padding = self.processor.pad_to_power_of_two(image)
        self.assertEqual(padded_image.shape, (64, 64))
        self.assertEqual(extra_padding, (0, 0))

    def test_pad_to_power_of_two_odd_dimensions(self):
        image = cp.ones((63, 64))
        padded_image, extra_padding = self.processor.pad_to_power_of_two(image)
        self.assertEqual(padded_image.shape, (64, 64))
        self.assertEqual(extra_padding, (1, 0))

        image = cp.ones((64, 63))
        padded_image, extra_padding = self.processor.pad_to_power_of_two(image)
        self.assertEqual(padded_image.shape, (64, 64))
        self.assertEqual(extra_padding, (0, 1))

        image = cp.ones((63, 63))
        padded_image, extra_padding = self.processor.pad_to_power_of_two(image)
        self.assertEqual(padded_image.shape, (64, 64))
        self.assertEqual(extra_padding, (1, 1))

    def test_crop_padding(self):
        image = cp.ones((64, 64))
        original_shape = (64, 64)
        extra_padding = (0, 0)
        cropped_image = self.processor.crop_padding(image, original_shape, extra_padding)
        self.assertEqual(cropped_image.shape, original_shape)

        image = cp.ones((65, 64))
        original_shape = (64, 64)
        extra_padding = (1, 0)
        cropped_image = self.processor.crop_padding(image, original_shape, extra_padding)
        self.assertEqual(cropped_image.shape, original_shape)

        image = cp.ones((64, 65))
        original_shape = (64, 64)
        extra_padding = (0, 1)
        cropped_image = self.processor.crop_padding(image, original_shape, extra_padding)
        self.assertEqual(cropped_image.shape, original_shape)

        image = cp.ones((65, 65))
        original_shape = (64, 64)
        extra_padding = (1, 1)
        cropped_image = self.processor.crop_padding(image, original_shape, extra_padding)
        self.assertEqual(cropped_image.shape, original_shape)

if __name__ == '__main__':
    unittest.main()