import os
import tempfile
from pathlib import Path
import numpy as np
import cv2
import pytest

# Add the project root to path
import sys
sys.path.insert(0, '/home/shixuan/Soil-Column-Procedures')

from src.workflow_tools.database.s7get_hist import HistogramAnalyzer


class TestHistogramAnalyzer:
    """Test class for HistogramAnalyzer"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        # Create a temporary directory for testing
        self.tmpdir = tempfile.mkdtemp()

        # Create dummy column directory structure
        self.dummy_column = '0001'
        self.img_dir = os.path.join(self.tmpdir, self.dummy_column, '3.Harmonized', 'image')
        os.makedirs(self.img_dir, exist_ok=True)

        # Create 10 dummy tif images
        for i in range(10):
            img_path = os.path.join(self.img_dir, f'image_{i:04d}.tif')
            img = np.random.rand(100, 100).astype(np.float32)
            cv2.imwrite(img_path, (img * 255).astype(np.uint8))

    def teardown_method(self):
        """Clean up test fixtures after each test method"""
        import shutil
        shutil.rmtree(self.tmpdir)

    def test_generate_histogram(self):
        """Test that histogram generation works correctly"""
        # Generate histogram with exclude_background=False since our test images don't have background 0s
        hist, bin_edges = HistogramAnalyzer.generate_histogram(
            self.img_dir,
            pixel_range=(0, 1),
            exclude_background=False
        )

        # Verify results
        assert hist is not None
        assert isinstance(hist, np.ndarray)
        assert isinstance(bin_edges, np.ndarray)
        assert len(hist) == 256  # Default bins=256
        assert len(bin_edges) == 257  # bin_edges has one more element than hist
        assert np.sum(hist) > 0  # Histogram has data

    def test_plot_histogram(self):
        """Test that histogram plotting works correctly"""
        # Generate histogram first
        hist, bin_edges = HistogramAnalyzer.generate_histogram(
            self.img_dir,
            pixel_range=(0, 1)
        )

        # Create a temporary file path for the plot
        plot_path = os.path.join(self.tmpdir, 'test_histogram.png')

        # Test plotting
        HistogramAnalyzer.plot_histogram(hist, bin_edges, plot_path)

        # Verify the plot file was created
        assert os.path.exists(plot_path)
        assert os.path.isfile(plot_path)

        # Check that it's a valid image file
        try:
            img = cv2.imread(plot_path)
            assert img is not None
        except Exception as e:
            pytest.fail(f"Failed to read generated plot: {e}")

    def test_process_single_column(self):
        """Test processing a single column"""
        output_dir = os.path.join(self.tmpdir, 'histograms')

        # Process single column with Path objects
        success = HistogramAnalyzer.process_single_column(
            Path(self.img_dir),
            Path(output_dir),
            self.dummy_column,
            pixel_range=(0, 1)
        )

        assert success

        # Check that only PNG file was created, no CSV
        files = os.listdir(output_dir)
        png_files = [f for f in files if f.endswith('.png')]
        csv_files = [f for f in files if f.endswith('.csv')]

        assert len(png_files) == 1  # Should have exactly 1 PNG file
        assert len(csv_files) == 0  # Should have no CSV files

        # Verify PNG file name
        expected_png = f"{self.dummy_column}_histogram.png"
        assert expected_png in png_files

    def test_process_multiple_columns(self):
        """Test processing multiple columns"""
        # Create a second dummy column
        dummy_column2 = '0002'
        img_dir2 = os.path.join(self.tmpdir, dummy_column2, '3.Harmonized', 'image')
        os.makedirs(img_dir2, exist_ok=True)

        # Create 5 dummy images for the second column
        for i in range(5):
            img_path = os.path.join(img_dir2, f'image_{i:04d}.tif')
            img = np.random.rand(100, 100).astype(np.float32)
            cv2.imwrite(img_path, (img * 255).astype(np.uint8))

        output_dir = os.path.join(self.tmpdir, 'histograms')
        tmp_path = Path(self.tmpdir)

        # Test 1: Process multiple columns with combine=False (separated, default)
        print("\nTest: Multiple columns with combine=False")
        HistogramAnalyzer.process_multiple_columns(
            tmp_path,
            Path(output_dir),
            [self.dummy_column, dummy_column2],
            pixel_range=(0, 1),
            combine=False
        )

        # Check results
        files = os.listdir(output_dir)
        png_files = [f for f in files if f.endswith('.png')]
        csv_files = [f for f in files if f.endswith('.csv')]

        assert len(png_files) == 2  # Should have 2 PNG files
        assert len(csv_files) == 0  # Should have no CSV files

        # Verify both columns' histograms exist
        expected_png1 = f"{self.dummy_column}_histogram.png"
        expected_png2 = f"{dummy_column2}_histogram.png"
        assert expected_png1 in png_files
        assert expected_png2 in png_files

    def test_process_multiple_columns_combined(self):
        """Test processing multiple columns with combine=True"""
        # Create two more dummy columns
        dummy_column2 = '0002'
        img_dir2 = os.path.join(self.tmpdir, dummy_column2, '3.Harmonized', 'image')
        os.makedirs(img_dir2, exist_ok=True)

        dummy_column3 = '0003'
        img_dir3 = os.path.join(self.tmpdir, dummy_column3, '3.Harmonized', 'image')
        os.makedirs(img_dir3, exist_ok=True)

        # Create dummy images for each column
        for i in range(5):
            cv2.imwrite(
                os.path.join(img_dir2, f'image_{i:04d}.tif'),
                (np.random.rand(100, 100).astype(np.float32) * 255).astype(np.uint8)
            )
            cv2.imwrite(
                os.path.join(img_dir3, f'image_{i:04d}.tif'),
                (np.random.rand(100, 100).astype(np.float32) * 255).astype(np.uint8)
            )

        output_dir = os.path.join(self.tmpdir, 'histograms_combined')
        tmp_path = Path(self.tmpdir)

        # Test: Multiple columns with combine=True
        print("\nTest: Multiple columns with combine=True")
        HistogramAnalyzer.process_multiple_columns(
            tmp_path,
            Path(output_dir),
            [self.dummy_column, dummy_column2, dummy_column3],
            pixel_range=(0, 1),
            combine=True
        )

        # Check results
        files = os.listdir(output_dir)
        png_files = [f for f in files if f.endswith('.png')]
        csv_files = [f for f in files if f.endswith('.csv')]

        assert len(png_files) == 1  # Should have 1 combined PNG file
        assert len(csv_files) == 0  # Should have no CSV files

        # Verify combined histogram exists
        expected_combined = 'combined_histogram.png'
        assert expected_combined in png_files

    def test_exclude_background(self):
        """Test that exclude_background parameter works"""
        # Create images with a background of 0
        img_dir_background = os.path.join(self.tmpdir, 'background_test', '3.Harmonized', 'image')
        os.makedirs(img_dir_background, exist_ok=True)

        for i in range(5):
            img_path = os.path.join(img_dir_background, f'image_{i:04d}.tif')
            # Create an image with 50% background (0)
            img = np.random.rand(100, 100).astype(np.float32)
            img[:50, :] = 0  # Top half as background
            cv2.imwrite(img_path, (img * 255).astype(np.uint8))

        # Generate histogram with background exclusion
        hist_with_background, _ = HistogramAnalyzer.generate_histogram(
            img_dir_background,
            pixel_range=(0, 1),
            exclude_background=False
        )

        # Generate histogram without background exclusion
        hist_without_background, _ = HistogramAnalyzer.generate_histogram(
            img_dir_background,
            pixel_range=(0, 1),
            exclude_background=True
        )

        # With background exclusion, the total pixels should be about half of with background
        total_with_background = np.sum(hist_with_background)
        total_without_background = np.sum(hist_without_background)

        assert total_without_background < total_with_background
        assert abs(total_with_background - 2 * total_without_background) < 1000  # Allow small variation


if __name__ == "__main__":
    pytest.main()
