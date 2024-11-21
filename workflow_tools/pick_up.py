#%%
# This file is to pick up images and compose them into a database to be labeled.

# %%
import os
import shutil

#%%
def extract_images(input_folder, column_number: str, output_folder):
    """
    Extract specified images from the input folder and save them to the output folder.

    Parameters:
    - input_folder: The directory containing the set of column images (each layer is an image, ordered sequentially).
    - column_number: The number of the column (from 1 to 7). Starting from 0028
    - output_folder: The directory where the extracted images will be saved.
    """

    # Turn the column number into a valid index
    column_number = int(column_number) - 28

    # Get a sorted list of image files
    images = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    images.sort()

    column_length = len(images)
    interval = column_length // 8
    start_index = column_number * 8

    indices = []
    for g in range(8):
        group_start = start_index + g * interval
        for i in range(8):
            index = group_start + i
            if index < column_length:
                indices.append(index)
            else:
                # Skip if the index is out of range
                pass

    selected_images = [images[i] for i in indices]

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Copy the selected images to the output folder
    for image_file in selected_images:
        src_path = os.path.join(input_folder, image_file)
        dst_path = os.path.join(output_folder, image_file)
        shutil.copyfile(src_path, dst_path)

    print(f"Extracted {len(selected_images)} images to {output_folder}")

# Example usage:
# - input_folder: The path to the folder containing the column images, e.g., "path/to/input_folder"
# - column_number: The number of the column (from 1 to 7)
# - output_folder: The path to the output folder, e.g., "path/to/output_folder"

# Call the function

if __name__ == "__main__":
    input_folder = "f:/3.Experimental_Data/Soils/Quzhou_Henan/Soil.column.0028/3.Preprocess/noise1high_boost1/"
    output_folder = "g:/DL_Data_raw/version1-1119/train_val_set/"
    id = '0028'
    extract_images(input_folder, id, output_folder)