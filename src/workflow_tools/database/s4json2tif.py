import json
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path


def json_to_mask(json_path, output_dir):
    """
    Converts a LabelMe JSON annotation to a mask image.

    Args:
        json_path (Path): Path to the LabelMe JSON annotation file.
        output_dir (Path): Path to the directory where the mask image will be saved.
    """
    try:
        with json_path.open('r') as f:
            data = json.load(f)

        image_height = data['imageHeight']
        image_width = data['imageWidth']

        # Create a blank mask
        mask = np.zeros((image_height, image_width), dtype=np.float32)

        for shape in data['shapes']:
            points = np.array(shape['points'], dtype=np.float32)

            # Check if it is a polygon
            if shape['shape_type'] == 'polygon':
                # Create a binary mask for the current polygon
                polygon_mask = np.zeros((image_height, image_width), dtype=np.float32)
                
                # Reshape points for draw_polygon function
                points = points.reshape((-1, 1, 2))
                # Draw polygon using cv2 instead of labelme
                cv2.fillPoly(polygon_mask, [points.astype(np.int32)], 1)

                # Assign a unique ID for every object (you can do more sophisticated handling of labels here)
                mask = np.maximum(mask, polygon_mask) # Use np.maximum to handle overlapping

        mask = (mask * 1)
        image_name = json_path.stem + '.tif'
        output_path = output_dir / image_name
        cv2.imwrite(str(output_path), mask)
        return True # Indicate success

    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return False # Indicate failure


def process_json_files(json_dir: Path, output_dir: Path):
    """
    Processes JSON files sequentially.

    Args:
        json_dir (Path): Directory containing JSON files.
        output_dir (Path): Directory to store output masks.
    """
    json_files = list(json_dir.glob('*.json'))
    success_count = 0
    
    for json_file in tqdm(json_files):
        if json_to_mask(json_file, output_dir):
            success_count += 1
    
    print(f"Successfully converted {success_count} out of {len(json_files)} json files")


if __name__ == '__main__':
    json_dir = Path(r'/mnt/g/DL_Data_raw/version8-low-precise/4.Converted/label-json')
    output_dir = Path(r'/mnt/g/DL_Data_raw/version8-low-precise/4.Converted/label-origin')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    process_json_files(json_dir, output_dir)