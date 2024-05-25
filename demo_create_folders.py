import API_functions.column_batch_process as fb

fb.create_column_structure(column_id=1, part_ids=5, base_path="e:/3.Experimental_Data/")

# path = "e:/3.Experimental_Data/Soil.column.0001/"
# column3 = fb.SoilColumn(path)
# path = column3.get_subfolder_path(part_index=1, folder_index=1)
# print(path)