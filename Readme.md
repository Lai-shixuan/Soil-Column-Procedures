# 代码仓库结构

随着前人的精神，本代码分割为，api部分，test部分，和正式的执行文件部分。由于处于毕业焦灼之际，本代码Readme不会进行快速更替，部分文件结构可能超越/落后于此时的实际代码结构。

本开源仓库，除了Readme使用中文以外（后续会用AI写英文注释），所有文件名、变量名、注释尽量全部使用英文，以便于国际化使用和传播交流。

+ API_functions
  + 图片相关，包括批量读写、格式转换等
  + 土柱相关
    + 一整套读写土柱并自动分割的函
    + 多张图片裁切、然后拼合的工具，用来组成土柱纵剖面图
  + dicom 也仅在读 dicom 类型的文件信息是使用，后来被 microDicom viewer 替代
  + visual 是为了展示土壤纵剖面用的。仅使用了一次，可能终稿前会再使用一次。此外，还有放大对照、图片比较的功能函数
  + DL
    + DL模型的简单框架，例如读取数据、随机化种子固定
    + 操作步骤相关，比如1）raw生成png，2）ROI裁剪，3）precheck，4）process，5）threshold（机器学习）等等
    + 评价用的函数，例如Dice系数，IOU和MIOU
+ Test，进行单元测试
+ doc 中，其中有一部分是一些简单的操作，可以理解为一个 demo
+ workflow_tools文件夹，是为了保存所有用到的工具性文件，是一些API函数的组合但是没有那么小到可以原子化，以及可能会被重复利用。
  + 可视化的图像比较，放大比较的工具
  + DL 的操作步骤
  - 3 大步骤，包括raw转图片，裁剪ROI和预处理，16转8位图片
  - 古老的 one-click-script 文件，用以一键对所有图片进行处理，但是随着数据库结构变化后不在启用
  - create_sqlite，将元数据转为 sqlite 文件
  - UI方面的绘制

# 关键文件的结构和功能（未来的代码库文档）

## file_batch

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

## Column_Analysis

+ get_left_right_surface
+ get_porosity_curve
+ draw_porosity_curve
+ dying_color_optimized
+ create_nifti