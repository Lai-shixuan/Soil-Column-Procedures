import API_functions.file_batch as fb
import API_functions.column_batch_process as column_bp

column_path = "e:/3.Experimental_Data/Soil.column.0001/"
column1 = column_bp.SoilColumn(column_path)

path_reconstruct = column1.get_subfolder_path(part_index=4, folder_index=1)
path_roi = column1.get_subfolder_path(part_index=4, folder_index=2)
path_rename = column1.get_subfolder_path(part_index=4, folder_index=3)
path_processed = column1.get_subfolder_path(part_index=4, folder_index=4)

image_read_name = fb.ImageName(prefix='0104_rec0', suffix='')       # Can remove the special photo by image size rather than name
image_write_name = fb.ImageName(prefix='0001_04_ou_DongYing', suffix='')
my_roi = fb.roi_region(x1=510, y1=676, width=800, height=800, z1=465, z2=3320)

fb.roi_select(path_in=path_reconstruct, path_out=path_roi, name_read=image_read_name, roi=my_roi)
fb.rename(path_in=path_roi, path_out=path_rename, new_name=image_write_name, start_index=1, reverse=True)
fb.image_process(path_in=path_rename, path_out=path_processed)
