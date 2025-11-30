import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to path for module imports
sys.path.insert(0, "/home/shixuan/Soil-Column-Procedures/")

from src.API_functions.Images import file_batch as fb
from src.API_functions.Images import file_info as fi


class HistogramAnalyzer:
    """Utility class for generating histograms from image stacks"""

    @staticmethod
    def generate_histogram(path_in: Path | str, bins=256, pixel_range=(0, 1), exclude_background=True):
        """
        Generate histogram from a stack of images using file_info.calculate_hist

        Args:
            path_in: Path to folder containing images
            bins: Number of histogram bins (default: 256)
            pixel_range: Range of pixel values (default: (0, 1) for float images)
            exclude_background: Whether to exclude background 0 values

        Returns:
            Tuple of (hist, bin_edges): Combined histogram data
        """
        # Convert to string if it's a Path object
        path_in_str = str(path_in) if isinstance(path_in, Path) else path_in

        # Get image names
        images_paths = fb.get_image_names(path_in_str, None, 'tif')
        if not images_paths:
            raise ValueError(f"No images found in {path_in}")

        # Initialize combined histogram
        combined_hist = None
        # Determine if image is float by checking dtype, not just pixel_range
        img = fb.read_images([images_paths[0]], 'gray', read_all=True)[0]  # Read first image to check dtype
        img_format = img.dtype == np.float32 or img.dtype == np.float64  # Float image if dtype is float

        # Process images in batch instead of one by one
        print(f"Reading all {len(images_paths)} images...")
        imgs = fb.read_images(images_paths, 'gray', read_all=True)

        for img in imgs:
            # Apply mask to exclude background 0 values if needed
            if exclude_background:
                mask = img > 0
                img = img[mask]

            if img_format:
                # For float images, calculate directly with the correct range
                hist = fi.calculate_hist(img, bins=bins, hist_range=pixel_range)
            else:
                # For uint8 images, use 0-256 range
                hist = fi.calculate_hist(img, bins=bins, hist_range=(0, 256))

            # Convert to 1D array
            hist = hist.flatten()

            # Combine with existing histogram
            if combined_hist is None:
                combined_hist = hist
            else:
                combined_hist += hist

        # Create bin edges (since file_info.calculate_hist uses 0-255 range with 256 bins by default)
        if img_format:
            bin_edges = np.linspace(pixel_range[0], pixel_range[1], bins + 1)
        else:
            bin_edges = np.linspace(0, 255, bins + 1)  # Use the same bins parameter for consistency

        # If no valid pixels were found in any image, return an empty histogram
        if combined_hist is None:
            combined_hist = np.zeros(bins, dtype=np.float64)

        return combined_hist, bin_edges

    @staticmethod
    def plot_histogram(hist, bin_edges, output_path, title="Pixel Value Histogram", xlabel="Pixel Value", ylabel="Frequency"):
        """
        Plot and save histogram

        Args:
            hist: Histogram data
            bin_edges: Histogram bin edges
            output_path: Path to save the plot
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        plt.figure(figsize=(12, 8))
        plt.hist(bin_edges[:-1], bin_edges, weights=hist, alpha=0.7)
        plt.title(title, fontsize=14)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    @classmethod
    def process_single_column(cls, column_path: Path, output_dir: Path, column_name: str, bins=256, pixel_range=(0, 1), exclude_background=True):
        """
        Process a single soil column and generate its histogram image

        Args:
            column_path: Path to column's Harmonized/image folder
            output_dir: Path to save histogram image
            column_name: Name/ID of the column for labeling
            bins: Number of histogram bins
            pixel_range: Pixel value range
            exclude_background: Whether to exclude background 0 values
        """
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Generate histogram
            print(f"\nGenerating histogram for {column_name}...")
            hist, bin_edges = cls.generate_histogram(str(column_path), bins, pixel_range, exclude_background)
            print(f"Generated histogram with {len(hist)} bins")

            # Save as PNG image
            plot_path = output_dir / f"{column_name}_histogram.png"
            cls.plot_histogram(hist, bin_edges, str(plot_path), title=f"Histogram - {column_name}")
            print(f"Saved histogram image: {plot_path}")

            return True

        except Exception as e:
            print(f"Error processing {column_name}: {str(e)}")
            return False

    @classmethod
    def process_multiple_columns(cls, base_path: Path, output_dir: Path, columns: list, bins=256, pixel_range=(0, 1), combine=False, exclude_background=True):
        """
        Process multiple soil columns and generate histograms for each

        Args:
            base_path: Path containing soil column directories
            output_dir: Path to save all histograms
            columns: List of column identifiers to process
            bins: Number of histogram bins
            pixel_range: Pixel value range
            combine: Whether to combine all columns into a single histogram (default: False)
            exclude_background: Whether to exclude background 0 values
        """

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        if combine:
            # Collect all images from all columns first
            all_images = []
            img_format = None  # Track if images are float or uint8
            total_images = 0

            for col_id in columns:
                # Find folder using pattern matching like in s4pick_up.py
                pattern = f"*/Soil.column.{col_id}"
                matching_folders = list(base_path.glob(pattern))

                # If no matches found, try direct path without Soil.column prefix (for backward compatibility)
                if not matching_folders:
                    direct_path = base_path / col_id
                    if direct_path.exists():
                        matching_folders = [direct_path]

                if matching_folders:
                    # Construct path to the Harmonized/image folder
                    harmonized_folder = matching_folders[0] / '3.Harmonized'
                    column_path = harmonized_folder / 'image'

                    # Read all images from the folder
                    images_paths = fb.get_image_names(str(column_path), None, 'tif')
                    if not images_paths:
                        print(f"Warning: No images found for column {col_id}")
                        continue

                    print(f"\nReading {len(images_paths)} images from column {col_id}...")
                    imgs = fb.read_images(images_paths, 'gray', read_all=True)

                    # Check image format (float or uint8) using first image
                    if img_format is None:
                        img_format = imgs[0].dtype == np.float32 or imgs[0].dtype == np.float64

                    all_images.extend(imgs)
                    total_images += len(imgs)
                else:
                    print(f"Warning: No folder found for column {col_id}")

            # Process combined images if we have any
            if all_images:
                print(f"\nMerging all {total_images} images together...")

                # Apply background exclusion if needed
                combined_pixels = []
                for img in all_images:
                    if exclude_background:
                        pixels = img[img > 0]
                    else:
                        pixels = img.flatten()

                    # Skip if no valid pixels
                    if pixels.size > 0:
                        combined_pixels.append(pixels)

                if combined_pixels:
                    # Combine all pixels into a single array
                    combined_pixels = np.concatenate(combined_pixels)
                    print(f"Total valid pixels after background exclusion: {len(combined_pixels)}")

                    # Generate histogram
                    if img_format:
                        # For float images, calculate directly with the correct range
                        hist = fi.calculate_hist(combined_pixels, bins=bins, hist_range=pixel_range)
                    else:
                        # For uint8 images, use 0-256 range
                        hist = fi.calculate_hist(combined_pixels, bins=bins, hist_range=(0, 256))

                    hist = hist.flatten()

                    # Create appropriate bin edges
                    if img_format:
                        # For float images, use the specified pixel range
                        bin_edges = np.linspace(pixel_range[0], pixel_range[1], bins + 1)
                    else:
                        # For uint8 images, use 0-255 range
                        bin_edges = np.linspace(0, 255, bins + 1)

                    # Plot and save the combined histogram
                    print(f"\nGenerating and saving combined histogram with {bins} bins...")
                    combined_path = output_dir / 'combined_histogram.png'

                    plt.figure(figsize=(15, 8))
                    plt.hist(bin_edges[:-1], bin_edges, weights=hist, alpha=0.7)
                    plt.title(f"Combined Histogram - All {total_images} Images from {len(columns)} Columns", fontsize=14)
                    plt.xlabel("Pixel Value", fontsize=12)
                    plt.ylabel("Frequency", fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved combined histogram: {combined_path}")
                else:
                    print("Warning: No valid pixels found in any images after background exclusion")

        else:
            # Process each column separately
            for col_id in columns:
                # Find folder using pattern matching like in s4pick_up.py
                pattern = f"*/Soil.column.{col_id}"
                matching_folders = list(base_path.glob(pattern))

                # If no matches found, try direct path without Soil.column prefix (for backward compatibility)
                if not matching_folders:
                    direct_path = base_path / col_id
                    if direct_path.exists():
                        matching_folders = [direct_path]

                if matching_folders:
                    # Construct path to the Harmonized/image folder
                    harmonized_folder = matching_folders[0] / '3.Harmonized'
                    column_path = harmonized_folder / 'image'

                    cls.process_single_column(column_path, output_dir, col_id, bins, pixel_range, exclude_background)
                else:
                    print(f"Warning: No folder found for column {col_id}")


if __name__ == "__main__":
    """
    Example usage: Generate histograms for specified soil columns

    The column selection method is similar to s4pick_up.py,
    where columns are manually specified using range and list concatenation
    """

    config = {
        'base_input': Path(r'/mnt/f/3.Experimental_Data/Soils/'),  # Input base path
        'output_folder': Path(r'/mnt/g/DL_Data_raw/histograms/'),  # Output folder
        'mode': 'column_id',  # Selection mode

        # Manual column selection (similar to s4pick_up.py style)
        'column_ids': [f"{i:04d}" for i in range(28, 35)] + [f"{i:04d}" for i in range(16, 22)],

        'exclude_background': False,  # Exclude background 0 values
    }

    # Get Path objects directly
    base_path = config['base_input']
    output_folder = config['output_folder']
    columns = config['column_ids']

    print("Starting histogram generation...")
    print(f"Selected columns: {', '.join(columns)}")
    print(f"Input path: {base_path}")
    print(f"Output path: {output_folder}")

    # Process the columns - separated histograms
    # HistogramAnalyzer.process_multiple_columns(base_path, output_folder, columns, bins=65535)

    # Process the columns - combined histogram
    HistogramAnalyzer.process_multiple_columns(base_path, output_folder, columns, bins=65535, combine=True, exclude_background=config['exclude_background'])

    print("\nHistogram generation completed!")
