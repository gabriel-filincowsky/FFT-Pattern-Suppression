import numpy as np
from utils.cupy_handler import cp, cp_fft, cp_gaussian_filter

class FFTProcessor:
    def compute_fft(self, image):
        """Compute the FFT of an image."""
        im_fft = cp_fft.fft2(image)
        im_fft_shifted = cp_fft.fftshift(im_fft)
        return im_fft_shifted

    def compute_ifft(self, fft_shifted):
        """Compute the inverse FFT."""
        im_ifft_shifted = cp_fft.ifftshift(fft_shifted)
        im_ifft = cp_fft.ifft2(im_ifft_shifted)
        im_new = cp.abs(im_ifft)
        return im_new

    def apply_high_pass_filter(self, image, radius):
        """Apply a high-pass filter to the image using Gaussian blur."""
        blurred = cp_gaussian_filter(image, sigma=radius)
        highpass = image - blurred + 128.0
        highpass = cp.clip(highpass, 0, 255).astype(cp.float32)
        return highpass