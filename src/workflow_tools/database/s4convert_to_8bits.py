import cv2
import sys

# sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")
sys.path.insert(0, "/home/shixuan/Soil-Column-Procedures/")

from pathlib import Path
from tqdm import tqdm
from src.API_functions.Images import file_batch as fb


def batch_images(path_in: Path, path_out: Path):
    image_lists = fb.get_image_names(str(path_in), None, 'tif')
    
    for image_path in tqdm(image_lists):
        # Read image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = fb.windows_adjustment_one_image(image)
        
        # Invert image
        image_8bits = fb.bitconverter.binary_to_grayscale_one_image(image, 'uint8')

        newpath = path_out / Path(image_path).with_suffix('.png').name.replace('harmonized', '8bits')
        
        # Save with same name in same location
        cv2.imwrite(newpath, image_8bits)


if __name__ == "__main__":
    path_in = r'/mnt/g/DL_Data_raw/version8-low-precise/3.Harmonized/image'
    path_out = r'/mnt/g/DL_Data_raw/version8-low-precise/4.Converted/8bit'

    path_in = Path(path_in)
    path_out = Path(path_out)
    if not path_out.exists():
        path_out.mkdir(parents=True, exist_ok=True)

    batch_images(path_in, path_out)
