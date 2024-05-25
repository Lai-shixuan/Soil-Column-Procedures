import API_functions.file_batch as fb

# fb.create_column_structure(column_id=3, part_ids=10, base_path="e:/3.Experimental_Data/")

path = "e:/3.Experimental_Data/Soil.column.0003/"
column3 = fb.SoilColumn(path)
path = column3.get_subfolder_path(part_index=1, folder_index=1)
print(path)