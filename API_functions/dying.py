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
def dying_color(volume_list, threshold: int, color_map_num: int):
    """
    :param volume: 3D binary image
    :param threshold: Threshold value to determine the size of the connected component
    :param color_map_num: Number of colors to be used for the connected components
    :return: 3D volume with the connected components colored based on their size
    """
    
    images = fb.read_images(volume_list, gray="gray", read_all=True)
    combined_array = np.stack(images, axis=-1)

    # Perform connected component labeling on CPU
    labeled_volume = label(combined_array, connectivity=1)

    del combined_array
    del images

    labeled_volume = labeled_volume.astype(cp.int16)

    # Calculate the number of unique labels
    num_features = labeled_volume.max()  # Max label value corresponds to the number of features
    print(f"Number of features: {num_features}")

    # Move the labeled volume to GPU
    labeled_volume_gpu = cp.asarray(labeled_volume)

    # Create an output volume on GPU
    output_volume_gpu = cp.zeros_like(labeled_volume_gpu, dtype=cp.int16)

    large_part_num = 0
    index = []

    temp_counter = 0

    color_values = cp.arange(1, color_map_num) * (250 // (color_map_num - 1))   # 62, 124, 186, 248

    for i in tqdm(range(1, num_features + 1)):
        if temp_counter <= 60:
            component_mask = (labeled_volume_gpu == i).astype(cp.bool_)
            component_size = component_mask.sum()

            if component_size >= threshold:
                color = large_part_num % (color_map_num - 1)    # 0, 1, 2, 3

                # Assign the value if the component size is equal to or larger than the threshold
                output_volume_gpu[component_mask] = color_values[color]

                large_part_num += 1
                temp_counter += 1

            index.append([i, component_size.item(), color if component_size >= threshold else 0])
    
    del labeled_volume_gpu
    del component_mask

    # Transfer the output volume back to CPU
    output_volume = cp.asnumpy(output_volume_gpu)

    del output_volume_gpu

    return output_volume, index