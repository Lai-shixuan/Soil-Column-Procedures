最正确的逻辑，应该是column_batch中的函数，其输入输出特性都可以在正常的情况下被使用，比如临时我有一百张图片想截个ROI这样。
新的api改成两条线，一条正常的函数，可以被通用。另一条所有函数归在SoilColumn类下，仅能在这里被调用，并且在这里调用逻辑则更简单，输入输出更简单。

# To_do list

- [ ] add id to column class

# file_batch

+ get_list
    + show_image_names
    + get_image_names
+ read_all_into_rom
    + read_images
    + output_images
+ read_one_by_one
    + column_batch_related
        + roi_select
        + rename
        + image_process
    + transformer
        + format_transformer
        + convert_to_binary
        + binary_to_grayscale

# Column_Analysis

+ get_left_right_surface
+ get_porosity_curve
+ draw_porosity_curve
+ dying_color_optimized
+ create_nifti