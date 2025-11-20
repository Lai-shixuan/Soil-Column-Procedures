import os
import tempfile
import numpy as np
import cv2
import time

# Add the src directory to the path
import sys
sys.path.insert(0, '/home/shixuan/Soil-Column-Procedures/src')

from API_functions.Images.file_batch import read_images, get_image_names


def create_large_dummy_images(num_images=3000, image_size=(1024, 1024)):
    """
    Create a temporary directory with a large number of dummy images
    """
    # Create a temporary directory
    tmpdir = tempfile.mkdtemp()

    print(f"Creating {num_images} dummy images of size {image_size[0]}x{image_size[1]} in {tmpdir}...")

    for i in range(num_images):
        # Create a simple image
        img = np.ones(image_size, dtype=np.uint8) * (i % 255)

        # Save as PNG
        cv2.imwrite(os.path.join(tmpdir, f'image_{i:04d}.png'), img)

    print("Image creation complete!")

    return tmpdir


def main():
    num_images = 3000
    image_size = (1024, 1024)  # 1MP images

    # Create dummy images
    tmpdir = create_large_dummy_images(num_images, image_size)

    try:
        # Get all image files
        image_files = get_image_names(tmpdir, None, 'png')

        # Test single-threaded reading
        print("\n=== Testing Single-Threaded Reading ===")
        start_time = time.time()
        images_single = read_images(image_files, gray='gray', read_all=True, use_threading=False)
        end_time = time.time()
        single_thread_time = end_time - start_time
        print(f"Single-threaded time: {single_thread_time:.2f} seconds")
        print(f"Read {len(images_single)} images")

        # Test multi-threaded reading
        print("\n=== Testing Multi-Threaded Reading ===")
        start_time = time.time()
        images_multi = read_images(image_files, gray='gray', read_all=True, use_threading=True, max_workers=16)
        end_time = time.time()
        multi_thread_time = end_time - start_time
        print(f"Multi-threaded time: {multi_thread_time:.2f} seconds")
        print(f"Read {len(images_multi)} images")

        # Calculate speedup
        if single_thread_time > 0:
            speedup = single_thread_time / multi_thread_time
            print(f"\n=== Performance Improvement ===")
            print(f"Multi-threaded is {speedup:.2f}x faster than single-threaded!")

        # Verify both methods read the same number of images
        assert len(images_single) == num_images, f"Single-threaded read {len(images_single)} images, expected {num_images}"
        assert len(images_multi) == num_images, f"Multi-threaded read {len(images_multi)} images, expected {num_images}"
        print("\n✓ Verification: Both methods read all images successfully!")

    finally:
        # Clean up
        print(f"\nCleaning up temporary files in {tmpdir}...")
        for f in os.listdir(tmpdir):
            os.unlink(os.path.join(tmpdir, f))
        os.rmdir(tmpdir)
        print("Cleanup complete!")


if __name__ == '__main__':
    main()
