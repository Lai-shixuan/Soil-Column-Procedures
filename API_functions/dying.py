import numpy as np
import cupy as cp
from . import file_batch as fb
from skimage.measure import label
from tqdm import tqdm
import torch


# def dying_color(volume_list, threshold: int, color_map_num: int):
#     """
#     :param volume: 3D binary image
#     :param threshold: Threshold value to determine the size of the connected component
#     :param color_map_num: Number of colors to be used for the connected components
#     :return: 3D volume with the connected components colored based on their size
#     """
    
#     images = fb.read_images(volume_list, gray="gray", read_all=True)
#     combined_array = np.stack(images, axis=-1)

#     # Perform connected component labeling
#     labeled_volume = label(combined_array, connectivity=1)

#     del combined_array

#     # Calculate the number of unique labels
#     num_features = labeled_volume.max()  # Max label value corresponds to the number of features
#     print(f"Number of features: {num_features}")

#     # Create an output volume to store the new values
#     output_volume = np.zeros_like(labeled_volume)

#     large_part_num = 0
#     index = []

#     for i in tqdm(range(1, num_features + 1)):

#         # Find the size of the current component
#         component_size = np.sum(labeled_volume == i)

#         if component_size >= threshold:
#             color = large_part_num % (color_map_num - 1) + 1
#             value = (255 / color_map_num - 1) * color

#             # Assign the value if the component size is equal to or larger than the threshold
#             output_volume[labeled_volume == i] = value
#             output_volume = output_volume.astype(np.uint8)

#             large_part_num += 1

#         index.append([i, component_size, color if component_size >= threshold else 0])

#     return output_volume, index


# pytorch and cuda version
# No chunks using

# def dying_color(volume_list, threshold: int, color_map_num: int):
#     """
#     :param volume: 3D binary image
#     :param threshold: Threshold value to determine the size of the connected component
#     :param color_map_num: Number of colors to be used for the connected components
#     :return: 3D volume with the connected components colored based on their size
#     """
    
#     images = fb.read_images(volume_list, gray="gray", read_all=True)
#     combined_array = np.stack(images, axis=-1)

#     # combined_array = np.random.randint(0, 2, (100, 100, 100), dtype=int)

#     # Perform connected component labeling on CPU
#     labeled_volume = label(combined_array, connectivity=1)
#     labeled_volume = labeled_volume.astype(np.int32)

#     # Calculate the number of unique labels
#     num_features = labeled_volume.max()  # Max label value corresponds to the number of features
#     print(f"Number of features: {num_features}")

#     # Create an output volume to store the new values
#     output_volume = np.zeros_like(labeled_volume, dtype=np.uint8)

#     # numpu to torch
#     output_volume = torch.tensor(output_volume, device='cuda', dtype=torch.short)

#     large_part_num = 0
#     index = []

#     # Move the entire labeled volume to GPU
#     labeled_volume_gpu = torch.tensor(labeled_volume, device='cuda', dtype=torch.short)

#     temp_counter = 0

#     for i in tqdm(range(1, num_features + 1)):
#         # Find the size of the current component

#         if temp_counter <= 40:
#             component_mask = (labeled_volume_gpu == i).to(torch.bool)
#             component_size = component_mask.sum().item()

#             if component_size >= threshold:
#                 color = large_part_num % (color_map_num - 1) + 1
#                 value = int(250 / (color_map_num - 1)) * color

#                 # Assign the value if the component size is equal to or larger than the threshold
#                 output_volume[component_mask] = value
#                 # output_volume[component_mask.cpu().numpy()] = value

#                 large_part_num += 1
#                 temp_counter += 1

#             index.append([i, component_size, color if component_size >= threshold else 0])
        
#             # Free GPU memory for the current component mask
#             del component_mask
#             torch.cuda.empty_cache()
    
#     # turn output_volume to numpy
#     output_volume = output_volume.cpu().numpy()

#     return output_volume, index

# Cupy version
# def dying_color(volume_list, threshold: int, color_map_num: int):
#     """
#     :param volume: 3D binary image
#     :param threshold: Threshold value to determine the size of the connected component
#     :param color_map_num: Number of colors to be used for the connected components
#     :return: 3D volume with the connected components colored based on their size
#     """
    
#     images = fb.read_images(volume_list, gray="gray", read_all=True)
#     combined_array = np.stack(images, axis=-1)

#     # Perform connected component labeling on CPU
#     labeled_volume = label(combined_array, connectivity=1)

#     del combined_array
#     del images

#     labeled_volume = labeled_volume.astype(cp.int16)

#     # Calculate the number of unique labels
#     num_features = labeled_volume.max()  # Max label value corresponds to the number of features
#     print(f"Number of features: {num_features}")

#     # Move the labeled volume to GPU
#     labeled_volume_gpu = cp.asarray(labeled_volume)

#     # Create an output volume on GPU
#     output_volume_gpu = cp.zeros_like(labeled_volume_gpu, dtype=cp.int16)

#     large_part_num = 0
#     index = []

#     temp_counter = 0

#     color_values = cp.arange(1, color_map_num) * (250 // (color_map_num - 1))   # 62, 124, 186, 248

#     for i in tqdm(range(1, num_features + 1)):
#         if temp_counter <= 60:
#             component_mask = (labeled_volume_gpu == i).astype(cp.bool_)
#             component_size = component_mask.sum()

#             if component_size >= threshold:
#                 color = large_part_num % (color_map_num - 1)    # 0, 1, 2, 3

#                 # Assign the value if the component size is equal to or larger than the threshold
#                 output_volume_gpu[component_mask] = color_values[color]

#                 large_part_num += 1
#                 temp_counter += 1

#             index.append([i, component_size.item(), color if component_size >= threshold else 0])
    
#     del labeled_volume_gpu
#     del component_mask

#     # Transfer the output volume back to CPU
#     output_volume = cp.asnumpy(output_volume_gpu)

#     del output_volume_gpu

#     return output_volume, index


# Cupy version with GPT4 modification

import numpy as np
import cupy as cp
from skimage.measure import label

def dying_color_optimized(volume_list, threshold: int, color_map_num: int):
    """
    :param volume_list: List of paths to 3D binary image files
    :param threshold: Threshold value to determine the size of the connected component
    :param color_map_num: Number of colors to be used for the connected components
    :return: 3D volume with the connected components colored based on their size
    """
    
    images = fb.read_images(volume_list, gray="gray", read_all=True)
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

