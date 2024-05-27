import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
from typing import Union
import nibabel as nib
import cupy as cp
from skimage.measure import label

# User-defined library import
from .. import file_batch


# get the left and right surface of the image array
def get_left_right_surface(image_files: list[str], name: str, path: str):
    """
    Only for gray images!
    """

    if not image_files:
        raise Exception('Error: No images found')
    
    if not os.path.exists(path):
        raise Exception('Error: The path does not exist.')
    
    if os.path.exists(os.path.join(path, name + '_left_surface.png')):
        raise Exception('Error: The file has existed, please change the name.')
    
    images = file_batch.read_images(image_files, gray="gray", read_all=True)
    images = np.stack(images, axis=-1)

    # Besed on that the images are stacked in the z axis, and x means the length, y means the width
    left_surface = images[:, 0, :]
    left_surface = np.swapaxes(left_surface, 0, 1)
    right_surface = images[:, -1, :]
    right_surface = np.swapaxes(right_surface, 0, 1)

    # Save the left surface
    new_file_path = os.path.join(path, name + '_left_surface.png')
    cv2.imwrite(new_file_path, left_surface)

    # Save the right surface
    new_file_path = os.path.join(path, name + '_right_surface.png')
    cv2.imwrite(new_file_path, right_surface)

    print("\033[1;3mLeft and Right Surface Saved Completely!\033[0m")


def get_porosity_curve(image_files: list[str], scan_resolution_um: int, minimal_body_value_cm: int):
    """
    Only for gray images!
    """
    if not image_files:
        raise Exception('Error: No images found')
    
    images = file_batch.read_images(image_files, gray="gray", read_all=True)
    images = np.stack(images, axis=-1)

    # Besed on that the images are stacked in the z axis, and x means the length, y means the width
    porosity_curve = []

    minimal_image_number = int(minimal_body_value_cm * 10000 / scan_resolution_um)

    # every minmal_body_value_cm cm, calculate the porosity
    for num, i in tqdm(enumerate(range(0, images.shape[2], minimal_image_number))):
        if i + minimal_image_number < images.shape[2]:
            image = images[:, :, i:i+minimal_image_number]
            porosity = np.sum(image != 0) / (image.shape[0] * image.shape[1] * image.shape[2])
            porosity_curve.append([minimal_body_value_cm * (num + 1), porosity])
        else:
            image = images[:, :, i:]
            porosity = np.sum(image != 0) / (image.shape[0] * image.shape[1] * image.shape[2])
            porosity_curve.append([minimal_body_value_cm * num + 
                                   round((images.shape[2] - i + 1) * scan_resolution_um / 10000, 2)
                                   , porosity])

    return porosity_curve


def draw_porosity_curve(curve):
    curve = np.array(curve)
    plt.plot(curve[:, 0], curve[:, 1])
    plt.xlabel('Distance (cm)')
    plt.ylabel('Porosity')
    plt.title('Porosity Curve')
    plt.show()


def dying_color_optimized(volume_list, threshold: int, color_map_num: int):
    """
    :param volume_list: List of paths to 3D binary image files
    :param threshold: Threshold value to determine the size of the connected component
    :param color_map_num: Number of colors to be used for the connected components
    :return: 3D volume with the connected components colored based on their size
    """

    def calculate_sizes_and_color_in_chunks(labeled_volume_gpu, num_features, color_values, threshold, color_map_num, chunks=3):
        # Split the volume into chunks along the third dimension
        z_slices = labeled_volume_gpu.shape[2]
        chunk_size = z_slices // chunks
        feature_sizes = cp.zeros(num_features + 1, dtype=cp.int16)
        output_volume_gpu = cp.zeros_like(labeled_volume_gpu, dtype=cp.int16)
        feature_details = []

        for i in range(chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i != chunks - 1 else z_slices
            chunk = labeled_volume_gpu[:, :, start_idx:end_idx]

            # Apply bincount on the chunk
            chunk_counts = cp.bincount(chunk.ravel(), minlength=num_features + 1)
            feature_sizes += chunk_counts

            # Update large_components based on updated feature_sizes
            large_components = feature_sizes >= threshold
            color_assignments = cp.arange(num_features + 1) % (color_map_num - 1)

            # Find large component indices
            large_indices = cp.nonzero(large_components)[0]  # This returns a tuple, take the first element which is the array of indices

            # Apply colors directly without using nonzero to find indices
            for li in tqdm(large_indices):
                color = color_values[color_assignments[li]]
                output_volume_gpu[:, :, start_idx:end_idx][chunk == li] = color
                feature_details.append((li, int(feature_sizes[li]), int(color)))

        return output_volume_gpu, feature_details
    
    images = file_batch.read_images(volume_list, gray="gray", read_all=True)
    combined_array = np.stack(images, axis=-1)
    del images  # Free up memory after use

    # Perform connected component labeling on CPU
    labeled_volume = label(combined_array, connectivity=1)
    del combined_array  # Free up memory after use

    # Move the labeled volume to GPU
    labeled_volume_gpu = cp.asarray(labeled_volume)
    del labeled_volume  # Free up memory after use

    num_features = labeled_volume_gpu.max().item()  # Max label value corresponds to the number of features
    print(f"Number of features: {num_features}")

    # Generate color values
    color_values = cp.linspace(50, 250, color_map_num).astype(cp.int16)

    # Calculate feature sizes and apply colors in chunks
    output_volume_gpu, feature_details = calculate_sizes_and_color_in_chunks(labeled_volume_gpu, num_features, color_values, threshold, color_map_num)

    # Transfer the output volume back to CPU
    output_volume = cp.asnumpy(output_volume_gpu)
    del labeled_volume_gpu, output_volume_gpu  # Free up GPU memory

    return output_volume, feature_details


def create_nifti(image_lists: Union[list[str], np.ndarray], output_folder: str, nifti_name: str):
    """
    Only for gray images!
    It will read all images and stack them together, which will take up a lot of memory.
    """
    if isinstance(image_lists, list):
        images = file_batch.read_images(image_lists, gray="gray", read_all=True)
        combined_array = np.stack(images, axis=-1)
    elif isinstance(image_lists, np.ndarray):
        combined_array = image_lists

    nifti_img = nib.Nifti1Image(combined_array, affine=np.eye(4))
    output_path = os.path.join(output_folder, nifti_name + '.nii') 

    # Detect whether has a nib file in that path
    if os.path.exists(output_path):
        raise Exception('Error: The file has existed, please change the name.')

    nib.save(nifti_img, output_path)
    print("\033[1;3mSave Done!\033[0m")