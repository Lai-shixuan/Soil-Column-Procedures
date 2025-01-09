from pathlib import Path
from typing import List, Union
import random
from enum import Enum

class SplitMode(Enum):
    """Enum for different splitting modes."""
    RANDOM = "random"  # Random distribution of files
    AVERAGE = "average"  # Sequential distribution of files

def calculate_file_counts(total_files: int, ratios: List[int]) -> List[int]:
    """Calculate the exact number of files for each part based on integer ratios.
    
    Args:
        total_files: Total number of files to be distributed.
        ratios: List of integer ratios (e.g. [7,2,1] for 7:2:1 split).
    
    Returns:
        A list of integers representing the number of files for each part.
    
    Example:
        >>> calculate_file_counts(100, [7,2,1])
        [70, 20, 10]
    """
    ratio_sum = sum(ratios)
    counts = []
    remaining = total_files
    
    # Calculate counts for all parts except the last one
    for ratio in ratios[:-1]:
        count = int(total_files * ratio / ratio_sum)
        counts.append(count)
        remaining -= count
    
    # Add remaining files to last part
    counts.append(remaining)
    return counts

def split_folder(
    folder_path: Union[str, Path], 
    num_parts: int = None, 
    ratios: Union[List[int], int] = None,
    mode: SplitMode = SplitMode.RANDOM,
    seed: int = None
) -> List[List[Path]]:
    """Split files in a folder into multiple parts based on integer ratios.
    
    Args:
        folder_path: Path to the folder containing files.
        num_parts: Number of parts to split into (if ratios not provided).
        ratios: List of integer ratios (e.g. [7,2,1] for 7:2:1 split).
        mode: Split mode - random or average distribution.
        seed: Random seed for shuffling (only used in RANDOM mode).
    
    Returns:
        A list of lists, where each inner list contains Path objects for that part.
    
    Example:
        >>> split_folder("./data", ratios=[7,2,1], mode=SplitMode.RANDOM)
    """
    folder_path = Path(folder_path)
    files = list(p for p in folder_path.iterdir() if p.is_file())
    total_files = len(files)
    
    # Handle ratios
    if ratios is None and num_parts:
        ratios = [1] * num_parts
    elif isinstance(ratios, int):
        ratios = [ratios, 1]  # Convert single ratio to [n,1]
        
    # Calculate exact file counts for each part
    file_counts = calculate_file_counts(total_files, ratios)
    
    # Random shuffle if in random mode
    if mode == SplitMode.RANDOM:
        random.seed(seed)
        random.shuffle(files)
    
    # Split files according to calculated counts
    parts = []
    start_idx = 0
    for count in file_counts:
        parts.append(files[start_idx:start_idx + count])
        start_idx += count
    
    return parts

def move_split_files(folder_path: Union[str, Path], split_results: List[List[Path]]) -> None:
    """Move split files into subfolders with size information in folder names.
    
    Args:
        folder_path: Base folder path.
        split_results: List of file lists from split_folder function.
    
    Example:
        For a 70-20-10 split of 100 files, creates:
        - part_0_70files/
        - part_1_20files/
        - part_2_10files/
    """
    folder_path = Path(folder_path)
    
    # Create subfolders with size information
    for i, files in enumerate(split_results):
        subfolder = folder_path / f"part_{i}_{len(files)}files"
        subfolder.mkdir(exist_ok=True)
        
        # Move files to corresponding subfolder
        for file_path in files:
            new_path = subfolder / file_path.name
            file_path.rename(new_path)

if __name__ == "__main__":
    # Default configuration
    config = {
        "folder_path": r'/mnt/g/DL_Data_raw/version8-low-precise/6.Precheck',
        "ratios": [64, 16],     # Split ratio 7:2:1:1
        "mode": SplitMode.RANDOM,
        "move": True,
        "seed": 102                   # Default random seed
    }
    
    # Execute splitting
    split_results = split_folder(
        config["folder_path"],
        ratios=config["ratios"],
        mode=config["mode"],
        seed=config["seed"]
    )
    
    # Print split results
    for i, part in enumerate(split_results):
        print(f"Part {i}: {len(part)} files")
    
    # Move files if requested
    if config["move"]:
        move_split_files(config["folder_path"], split_results)
        print("Files have been moved to subfolders with size information")
