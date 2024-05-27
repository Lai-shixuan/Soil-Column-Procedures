# Conmon library import
import os
import csv

# User-defined library import
from .. import file_batch

# A class to define the soil column structure, including parts and subfolders:
class SoilColumn:
    # A class to define the part structure, including part_id and subfolders:
    class Part:
        def __init__(self, column_id, part_id, part_path, subfolders):
            self.column_id = column_id
            self.part_id = SoilColumn._format_id(self, id_number=part_id, type='part')
            self.part_path = part_path
            self.sub_folders = subfolders
            self.roi = self._load_roi()

        def _load_roi(self):
            roi_file = os.path.join(self.part_path, f"roi-{self.column_id}-{self.part_id}.csv")
            if os.path.exists(roi_file):
                with open(roi_file, "r") as f:
                    reader = csv.reader(f)
                    lines = [row for row in reader
                             if row and len(row) >= 2 and row[1].strip() != '']
                    return self._parse_roi(lines) if len(lines) == 6 else None

        def _parse_roi(self, lines):
            roi_values = {row[0].strip(): int(row[1].strip()) for row in lines}
            print(f"{self.column_id}-{self.part_id}:")
            
            # Create a roi object using the values from the second column
            my_roi = file_batch.roi_region(
                x1=roi_values['x1'],
                y1=roi_values['y1'],
                width=roi_values['width'],
                height=roi_values['height'],
                z1=roi_values['z1'],
                z2=roi_values['z2']
            )
            return my_roi
            
        def get_roi(self):
            return self.roi

        def get_subfolder_path(self, index):
            if 0 <= index < len(self.sub_folders):
                return self.sub_folders[index]
            return None

    def __init__(self, column_path, column_id, part_num):
        self.column_path = column_path
        self.column_id = self._format_id(column_id, 'column')
        self.part_num = part_num
        self.data_structure = {}
        self.parts = []
        self.initialize_structure()


    def initialize_structure(self):
        for part_id in range(self.part_num):
            part_id_formatted = self._format_id(part_id, 'part')
            part_path = os.path.join(self.column_path, f"{self.column_id}-{part_id_formatted}")
            os.makedirs(part_path, exist_ok=True)
            sub_folders = self._create_sub_folders(part_path)
            self._create_roi_file(part_path, part_id_formatted)
            self.data_structure[f"{self.column_id}-{part_id_formatted}"] = self._check_part(part_path, part_id_formatted, sub_folders)
            self.parts.append(self.Part(self.column_id, part_id, part_path, sub_folders))
        self._print_structure()

    def _format_id(self, id_number, type):
        if type == 'column':
            return f"{id_number:04d}"
        elif type == 'part':
            return f"{id_number:02d}"

    def _create_sub_folders(self, part_path):
        sub_folders = [
            "0.Origin",
            "1.Reconstruct",
            "2.ROI",
            "3.Rename",
            "4.Threshold",
            "5.Analysis"
        ]
        sub_folders = [os.path.join(part_path, folder) for folder in sub_folders]

        for folder in sub_folders:
            os.makedirs(folder, exist_ok=True)
        return sub_folders

    def _create_roi_file(self, part_path, part_id):
        roi_file = os.path.join(part_path, f"roi-{self.column_id}-{part_id}.csv")
        if not os.path.exists(roi_file):
            with open(roi_file, "w") as f:
                f.write("x1\ny1\nwidth\nheight\nz1\nz2")
    
    def _check_part(self, part_path, part_id, sub_folders):
        checks = {}

        for folder in sub_folders:
            folder_name = os.path.basename(folder)
            try:
                # List the directory and count files, ignoring subdirectories
                files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
                checks[folder_name] = len(files)
            except FileNotFoundError:
                checks[folder_name] = 0  # Folder not found or an error occurred, hence no files

        # Check the ROI file
        roi_file = os.path.join(part_path, f"roi-{self.column_id}-{part_id}.csv")
        roi_filled = self._is_roi_filled(part_path, part_id)
        checks['ROI File'] = {'exists': os.path.exists(roi_file), 'filled': roi_filled}

        return checks

    def _is_roi_filled(self, part_path, part_id):
        expected_keys = {"x1", "y1", "width", "height", "z1", "z2"}  # Expected keys in the first column
        roi_file = os.path.join(part_path, f"roi-{self.column_id}-{part_id}.csv")
        if os.path.exists(roi_file):
            with open(roi_file, "r") as file:
                reader = csv.reader(file, delimiter=',')  # Assuming tab-delimited based on your data format
                found_keys = {row[0].strip(): row[1].strip() for row in reader if row and len(row) >= 2 and row[1].strip()}
                if expected_keys == set(found_keys.keys()):  # Check if all keys are present and have values
                    return True
        return False
    
    def _print_structure(self):
        # Table headers
        print('')
        headers = ["Part ID", "0.Origin", "1.Reconstruct", "2.ROI", "3.Rename", "4.Threshold", "5.Analysis", "ROI File Exists", "ROI File Filled"]
        print("{:<10} {:<10} {:<15} {:<10} {:<10} {:<15} {:<15} {:<15} {:<15}".format(*headers))
        
        # Data rows
        for key, value in self.data_structure.items():
            data = [
                key,
                value['0.Origin'],
                value['1.Reconstruct'],
                value['2.ROI'],
                value['3.Rename'],
                value['4.Threshold'],
                value['5.Analysis'],
                'Yes' if value['ROI File']['exists'] else 'No',
                'Yes' if value['ROI File']['filled'] else 'No'
            ]
            print("{:<10} {:<10} {:<15} {:<10} {:<10} {:<15} {:<15} {:<15} {:<15}".format(*data))

    def get_part(self, part_index):
        for part in self.parts:
            if int(part.part_id) == part_index:
                return part
        return None
    
    def get_data_structure(self):
        return self.data_structure