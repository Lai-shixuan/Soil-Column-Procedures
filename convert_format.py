from API_functions import file_batch as fb

# read_path = 'e:/3.Experimental_Data/DL_Data_raw/uploads/file_5_test\defaultannot/'
# save_path = 'e:/3.Experimental_Data/DL_Data_raw/uploads/file_5_test\defaultannott/'
read_path = '//wsl.localhost/Ubuntu/home/shixuan/DL_Algorithms/nnunet/nnUNet_raw/Dataset001_240531/labelsTr/origin/'
save_path = '//wsl.localhost/Ubuntu/home/shixuan/DL_Algorithms/nnunet/nnUNet_raw/Dataset001_240531/labelsTr/'

fb.grayscale_to_binary(read_path=read_path, output_path=save_path, format='png', read_name=None)