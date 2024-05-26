import API_functions.column_batch_process as fb

path = "e:/3.Experimental_Data/Soil.column.0008/"
column1 = fb.SoilColumn(column_path=path, column_id=8, part_num=10)
print(column1.get_data_structure())


# column3 = fb.SoilColumn(path)
# path = column3.get_subfolder_path(part_index=1, folder_index=1)
# print(path)