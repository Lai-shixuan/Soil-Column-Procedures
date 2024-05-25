# Conmon library import
import os


# A class to define the soil column structure, including parts and subfolders:
class SoilColumn:

    # A class to define the part structure, including part_id and subfolders:
    class Part:
        def __init__(self, part_id, part_path):
            self.part_id = part_id
            self.path = part_path
            self.sub_folders = [
                os.path.join(part_path, "0.Origin"),
                os.path.join(part_path, "1.Reconstruct"),
                os.path.join(part_path, "2.ROI"),
                os.path.join(part_path, "3.Rename"),
                os.path.join(part_path, "4.Threshold"),
                os.path.join(part_path, "5.Analysis")
            ]

        def get_subfolder_path(self, index):
            if 0 <= index < len(self.sub_folders):
                return self.sub_folders[index]
            return None

    def __init__(self, root_path):
        self.root_path = root_path
        self.id = self._extract_id(root_path)
        self.parts = self._load_parts()

    def _extract_id(self, path):
        # Extract the last part of the path and split it by '.'
        base_name = os.path.normpath(path).split(os.sep)[-1]
        if base_name.startswith("Soil.column."):
            return base_name.split(".")[-1]
        return None

    def _load_parts(self):
        parts = []
        for part_name in os.listdir(self.root_path):
            part_path = os.path.join(self.root_path, part_name)
            if os.path.isdir(part_path) and part_name.startswith(self.id):
                part_id = part_name.split("-")[-1]
                parts.append(SoilColumn.Part(part_id, part_path))
        return parts

    def get_part(self, part_index):
        for part in self.parts:
            if int(part.part_id) == part_index:
                return part
        return None

    def get_subfolder_path(self, part_index, folder_index):
        part = self.get_part(part_index)
        if part:
            return part.get_subfolder_path(folder_index)
        return None


# Create the column structure with parts and subfolders
def create_column_structure(column_id, part_ids, base_path):
    # Ensure column_id is 4 digits
    column_id = f"Soil.column.{int(column_id):04d}"
    column_path = os.path.join(base_path, column_id)
    os.makedirs(column_path, exist_ok=True)
    
    for part_id in range(part_ids):
        # Ensure part_id is 2 digits
        part_id = f"{int(part_id):02d}"
        part_path = os.path.join(column_path, f"{column_id.split('.')[-1]}-{part_id}")
        os.makedirs(part_path, exist_ok=True)
        
        sub_folders = [
            "0.Origin",
            "1.Reconstruct",
            "2.ROI",
            "3.Rename",
            "4.Threshold",
            "5.Analysis"
        ]
        
        for folder in sub_folders:
            os.makedirs(os.path.join(part_path, folder), exist_ok=True)

    return column_path
