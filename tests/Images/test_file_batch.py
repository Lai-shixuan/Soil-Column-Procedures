import os
import tempfile
import numpy as np
import cv2
import pytest

# Add the src directory to the path
import sys
sys.path.insert(0, '/home/shixuan/Soil-Column-Procedures/src')

from API_functions.Images.file_batch import read_images, get_image_names, ImageName


class TestFileBatch:
    """
    Unit tests for file_batch.py module
    """

    @pytest.fixture
    def dummy_images(self):
        """
        Create temporary directory with dummy images for testing
        """
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 10 dummy images (5x5 pixels, uint8)
            for i in range(10):
                # Create a simple image with unique color for each image
                img = np.ones((5, 5), dtype=np.uint8) * (i * 25)

                # Create both PNG and BMP files to test different formats
                cv2.imwrite(os.path.join(tmpdir, f'test_image_{i:03d}.png'), img)
                cv2.imwrite(os.path.join(tmpdir, f'test_image_{i:03d}.bmp'), img)

            # Create some images with prefix and suffix
            img = np.ones((5, 5), dtype=np.uint8) * 100
            cv2.imwrite(os.path.join(tmpdir, f'prefix_test_image_suffix.png'), img)

            yield tmpdir

    def test_read_images_single_thread(self, dummy_images):
        """Test single-threaded image reading"""
        # Get all PNG images
        image_files = get_image_names(dummy_images, None, 'png')

        # Read all images with single-threading
        images = read_images(image_files, gray='gray', read_all=True, use_threading=False)

        # Check that all images were read
        assert len(images) == 11  # 10 numbered images + 1 prefixed/suffixed

        # Check image dimensions
        for img in images:
            assert img.shape == (5, 5)

    def test_read_images_multi_thread(self, dummy_images):
        """Test multi-threaded image reading"""
        # Get all BMP images
        image_files = get_image_names(dummy_images, None, 'bmp')

        # Read all images with multi-threading
        images = read_images(image_files, gray='gray', read_all=True, use_threading=True)

        # Check that all images were read
        assert len(images) == 10  # Only BMP images

        # Check image dimensions
        for img in images:
            assert img.shape == (5, 5)

    def test_read_images_partial(self, dummy_images):
        """Test reading only a subset of images"""
        # Get all PNG images
        image_files = get_image_names(dummy_images, None, 'png')

        # Read only 5 images
        images = read_images(image_files, gray='gray', read_all=False, read_num=5)

        # Check that only 5 images were read
        assert len(images) == 5

    def test_read_images_gray_conversion(self, dummy_images):
        """Test different gray scale modes"""
        # Get the first PNG image
        image_files = get_image_names(dummy_images, None, 'png')[:1]

        # Read as gray (IMREAD_UNCHANGED)
        img_gray = read_images(image_files, gray='gray', read_all=True)[0]

        # Read and turn to gray (IMREAD_UNCHANGED + COLOR_BGR2GRAY)
        img_turn_to_gray = read_images(image_files, gray='turn to gray', read_all=True)[0]

        # Check that both have same shape (should be 5x5)
        assert img_gray.shape == (5, 5)
        assert img_turn_to_gray.shape == (5, 5)

    def test_read_images_color(self, dummy_images):
        """Test reading images in color mode"""
        # Create a color image for test
        color_img_path = os.path.join(dummy_images, 'color_image.png')
        color_img = cv2.cvtColor(np.ones((5, 5, 3), dtype=np.uint8) * 128, cv2.COLOR_RGB2BGR)
        cv2.imwrite(color_img_path, color_img)

        image_files = [color_img_path]

        # Read as color
        img_color = read_images(image_files, gray='color', read_all=True)[0]

        # Check that image has 3 channels
        assert len(img_color.shape) == 3
        assert img_color.shape[2] == 3

    def test_read_images_with_prefix_suffix(self, dummy_images):
        """Test reading images with specific prefix and suffix"""
        # Create ImageName with prefix and suffix
        img_name = ImageName('prefix', 'suffix')

        # Get images with prefix and suffix
        image_files = get_image_names(dummy_images, img_name, 'png')

        # Should get only the one with prefix and suffix
        assert len(image_files) == 1

        # Read the image
        images = read_images(image_files, gray='gray', read_all=True)
        assert len(images) == 1


if __name__ == '__main__':
    # Run the tests with pytest
    pytest.main([__file__, '-v'])
