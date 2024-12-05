import os
from typing import List, Optional
from src.API_functions.Images.file_batch import get_image_names
from PyQt5.QtWidgets import QFileDialog

class FileManager:
    def __init__(self, parent):
        self.parent = parent
        self.folder1 = None
        self.folder2 = None
        self.images1_list = []
        self.images2_list = []
        self.pattern_length = None

    def get_image_files(self, folder: str) -> List[str]:
        """Get a sorted list of TIF image files from the specified folder."""
        files = get_image_names(folder, None, 'tif')
        return sorted([os.path.splitext(os.path.basename(f))[0] for f in files])

    def _get_common_prefix_length(self, str1: str, str2: str) -> int:
        """Find the length of common prefix between two strings."""
        min_len = min(len(str1), len(str2))
        for i in range(min_len):
            if str1[i] != str2[i]:
                return i
        return min_len

    def analyze_filename_pattern(self, files1: List[str], files2: List[str]) -> Optional[int]:
        """Analyze file pairs to determine consistent naming pattern length."""
        if len(files1) != len(files2):
            return None
            
        sample_size = min(5, len(files1))
        lengths = []
        
        for i in range(sample_size):
            name1 = os.path.splitext(os.path.basename(files1[i]))[0]
            name2 = os.path.splitext(os.path.basename(files2[i]))[0]
            common_len = self._get_common_prefix_length(name1, name2)
            lengths.append(common_len)
            
        return lengths[0] if len(set(lengths)) == 1 else None

    def validate_folders(self) -> bool:
        """Validate that both folders contain matching image files."""
        if not self.folder1 or not self.folder2:
            return False
            
        files1 = sorted(get_image_names(self.folder1, None, 'tif'))
        files2 = sorted(get_image_names(self.folder2, None, 'tif'))
        
        if not files1 or not files2:
            self.parent.show_status('Warning: One or both folders are empty!', error=True)
            return False
            
        if len(files1) != len(files2):
            self.parent.show_status('Warning: Folders contain different numbers of files!', error=True)
            return False

        self.pattern_length = self.analyze_filename_pattern(files1, files2)
        if self.pattern_length is None:
            self.parent.show_status('Warning: Files do not follow a consistent naming pattern between folders!', error=True)
            return False

        for f1, f2 in zip(files1, files2):
            name1 = os.path.splitext(os.path.basename(f1))[0]
            name2 = os.path.splitext(os.path.basename(f2))[0]
            if name1[:self.pattern_length] != name2[:self.pattern_length]:
                self.parent.show_status(f'Warning: Pattern mismatch: {name1} vs {name2}', error=True)
                return False

        self.images1_list = files1
        self.images2_list = files2
        self.parent.scrollbar.setMaximum(len(files1) - 1)
        self.parent.enable_controls(True)
        self.parent.scrollbar.setValue(0)
        return True

    def select_folder(self, folder_num: int) -> None:
        """Handle folder selection dialog and validation.
        
        Args:
            folder_num (int): Which folder to select (1 or 2)
        """
        folder = QFileDialog.getExistingDirectory(
            self.parent,
            f'Select Folder {folder_num}',
            '',
            QFileDialog.ShowDirsOnly
        )
        
        if not folder:  # User cancelled
            return
            
        if folder_num == 1:
            self.folder1 = folder
            self.parent.controls.label_folder1.setText(folder)
        else:
            self.folder2 = folder
            self.parent.controls.label_folder2.setText(folder)
            
        # If both folders are selected, validate and load images
        if self.folder1 and self.folder2:
            if self.validate_folders():
                self.parent.index1 = 0
                self.parent.index2 = 0
                self.parent.display_images()
            else:
                # Reset if validation fails
                if folder_num == 1:
                    self.folder1 = None
                    self.parent.controls.label_folder1.setText('No folder selected')
                else:
                    self.folder2 = None
                    self.parent.controls.label_folder2.setText('No folder selected')